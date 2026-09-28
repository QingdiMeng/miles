"""The file-backed buffers behind Muon's disk-resident optimizer state.

Guards a silent failure: if the allocator stops returning file-backed storage, the offloader
keeps working against pinned host memory while the log still claims otherwise.
"""

import os

import pytest
import torch

from miles_plugins.optimizers import nvme_stream


def test_disk_buffer_matches_shape_and_dtype_and_is_not_pinned(tmp_path):
    src = torch.randn(64, 32, dtype=torch.float32)

    buf = nvme_stream._disk_backed_like(src, str(tmp_path))

    assert buf.shape == src.shape
    assert buf.dtype == src.dtype
    assert buf.device.type == "cpu"
    # The inherited offloader picks its sync/async copy path off is_pinned().
    assert not buf.is_pinned()


def test_disk_buffer_round_trip_is_bit_exact(tmp_path):
    src = torch.randn(128, 64, dtype=torch.float32)
    buf = nvme_stream._disk_backed_like(src, str(tmp_path))

    buf.copy_(src)
    out = torch.empty_like(src)
    out.copy_(buf)

    assert torch.equal(out, src)


def test_disk_buffer_leaves_no_file_behind(tmp_path):
    nvme_stream._disk_backed_like(torch.zeros(8), str(tmp_path))

    # Unlinked at creation, so a killed run leaves no residue.
    assert os.listdir(tmp_path) == []


def test_disk_buffer_is_recognized_as_already_managed(tmp_path):
    """Megatron's checkpoint adoption reallocates non-pinned CPU state; ours must be exempt."""
    buf = nvme_stream._disk_backed_like(torch.zeros(32, 8), str(tmp_path))

    assert nvme_stream._is_disk_backed(buf)
    assert not nvme_stream._is_disk_backed(torch.zeros(32, 8))


def test_flush_mapping_covers_the_buffer_and_repeats_cheaply(tmp_path):
    """Checkpointing fsyncs its own files behind the kernel's writeback of ours."""
    buf = nvme_stream._disk_backed_like(torch.zeros(1024, 256), str(tmp_path))
    nbytes = buf.numel() * buf.element_size()
    buf.fill_(1.0)

    assert nvme_stream._flush_mapping(buf) == nbytes
    # Already clean, so the repeat is the cheap case the checkpoint hook relies on.
    assert nvme_stream._flush_mapping(buf) == nbytes


def test_reserve_sizes_the_file(tmp_path):
    path = tmp_path / "f.bin"
    fd = os.open(str(path), os.O_RDWR | os.O_CREAT, 0o600)
    try:
        nvme_stream._reserve(fd, 4096)
        assert os.fstat(fd).st_size == 4096
    finally:
        os.close(fd)


@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16])
def test_disk_buffer_preserves_dtype(tmp_path, dtype):
    src = torch.zeros(16, 4, dtype=dtype)

    buf = nvme_stream._disk_backed_like(src, str(tmp_path))

    assert buf.dtype is dtype
    assert buf.numel() == src.numel()


def _staged_store(clip_grad: float):
    """A store over one bucket of two BF16 params, with a fake DistributedOptimizer around it."""
    from types import SimpleNamespace

    torch.manual_seed(0)
    entries, ranges = [], {}
    for numel in (96, 160):
        model_param = torch.nn.Parameter(torch.zeros(numel, dtype=torch.bfloat16))
        model_param.main_grad = torch.randn(numel, dtype=torch.bfloat16)
        main_param = torch.zeros(numel // 2, dtype=torch.float32)
        ranges[model_param] = SimpleNamespace(start=numel // 4, end=numel // 4 + numel // 2)
        entries.append(nvme_stream._Entry(model_param, main_param, 0))
    dist_opt = SimpleNamespace(
        config=SimpleNamespace(clip_grad=clip_grad),
        model_fp32_groups=[],
        shard_fp32_groups=[],
        _get_model_param_range_map=lambda param: {"param": ranges[param]},
    )
    store = object.__new__(nvme_stream.NVMeOptimizerStateStore)
    store.dist_opt = dist_opt
    store.buckets = [SimpleNamespace(entries=entries)]
    store._grad_views, store._grads_staged, store._clip_coeff = {}, False, 1.0
    return store, entries, ranges


def _megatron_main_grad(entry, ranges, total_norm, clip_grad):
    """DistributedOptimizer._copy_model_grads_to_main_grads + clip_grad_by_total_norm_fp32."""
    r = ranges[entry.model_param]
    grad = entry.model_param.main_grad.view(-1)[r.start : r.end].float()
    coeff = clip_grad / (total_norm + 1.0e-6)
    return grad * coeff if coeff < 1.0 else grad


@pytest.mark.parametrize("total_norm", [0.5, 40.0])
def test_streamed_grads_are_cast_and_clipped_per_bucket_like_megatron(total_norm):
    """BF16 grads stay views until their bucket steps, then match Megatron's fp32 main grads exactly."""
    clip_grad = 1.0
    store, entries, ranges = _staged_store(clip_grad)

    store.stage_model_grads()
    assert all(entry.main_param.grad is None for entry in entries)
    for entry in entries:
        view = store.grad_view(entry.main_param)
        assert view.dtype == torch.bfloat16
        assert view.data_ptr() == entry.model_param.main_grad[ranges[entry.model_param].start :].data_ptr()

    store.set_total_grad_norm(total_norm)
    store._attach_grads(entries)
    for entry in entries:
        assert entry.main_param.grad.dtype == torch.float32
        expected = _megatron_main_grad(entry, ranges, total_norm, clip_grad)
        torch.testing.assert_close(entry.main_param.grad, expected, atol=0, rtol=0)

    store._detach_grads(entries)
    assert all(entry.main_param.grad is None for entry in entries)


def test_grad_norm_filter_reads_the_staged_shards():
    """The upstream filter counts a streamed param through its staged BF16 shard."""
    from types import SimpleNamespace

    from megatron.core.optimizer.distrib_optimizer import DistributedOptimizer

    store, entries, _ = _staged_store(clip_grad=1.0)
    store.stage_model_grads()
    for entry in entries:
        entry.main_param.tensor_model_parallel = True
    fake_opt = SimpleNamespace(
        config=SimpleNamespace(
            use_precision_aware_optimizer=False, use_precision_aware_optimizer_no_fp8_or_ds_fp8=False
        )
    )

    views = [nvme_stream._GradView(e.main_param, store.grad_view(e.main_param)) for e in entries]
    grads = DistributedOptimizer._filter_grads_for_norm(fake_opt, views)

    assert [g.data_ptr() for g in grads] == [store.grad_view(e.main_param).data_ptr() for e in entries]
