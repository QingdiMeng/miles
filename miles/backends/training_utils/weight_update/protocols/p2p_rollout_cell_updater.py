import logging
from argparse import Namespace
from concurrent.futures import Future, ThreadPoolExecutor
from datetime import datetime, timezone
from typing import Any

from miles.backends.sglang_utils.sglang_api_client import SGLangApiClient
from miles.backends.training_utils.weight_update import checksum_utils
from miles.backends.training_utils.weight_update.protocols.p2p_transfer_utils import RemoteWeightInfo
from miles.backends.training_utils.weight_update.rollout_cell_updater import _RolloutCellUpdater
from miles.utils.audit_utils.event_logger.logger import get_event_logger, is_event_logger_initialized
from miles.utils.audit_utils.event_logger.models import WeightTransferFailedEvent
from miles.utils.test_utils.fault_injector.controller import fault_hook_controller, reach_fault_hook
from miles.utils.test_utils.fault_injector.models import FaultHookName


logger = logging.getLogger(__name__)


# This class, like the rest of the p2p weight-update code, is kept deliberately naive until yueming's refactor part 2 reshapes it.
class _P2PRolloutCellUpdater(_RolloutCellUpdater):
    def __init__(
        self,
        args: Namespace,
        cell_id: str,
        api_client: SGLangApiClient,
        selector: str,
    ) -> None:
        super().__init__(args=args, cell_id=cell_id, api_client=api_client)
        self.selector = selector
        self._executor = ThreadPoolExecutor(max_workers=1)
        self.targets_by_rollout_engine_rank: dict[int, RemoteWeightInfo] = {}
        self._pending_writes: list[Future[None]] = []

    def submit_write(
        self,
        rollout_engine_rank: int,
        names: list[str],
        weight_memory_registry: dict[str, tuple[int, int, int]],
        transfer_engine: Any,
        sent_checksums: dict[str, str] | None,
    ) -> None:
        if self.is_errored:
            return
        self._pending_writes.append(
            self._executor.submit(
                self._write_if_active,
                transfer_engine,
                self.targets_by_rollout_engine_rank[rollout_engine_rank],
                names,
                weight_memory_registry,
                rollout_engine_rank=rollout_engine_rank,
                sent_checksums=sent_checksums,
            )
        )

    def wait_for_pending_writes(self, timeout: float) -> None:
        if self.is_errored:
            return
        pending, self._pending_writes = self._pending_writes, []
        for i, future in enumerate(pending):
            try:
                future.result(timeout=timeout)
            except Exception as error:
                self.mark_errored(error)
                for remaining in pending[i + 1 :]:
                    remaining.cancel()
                return

    def _write_if_active(
        self,
        transfer_engine: Any,
        target: RemoteWeightInfo,
        names: list[str],
        weight_memory_registry: dict[str, tuple[int, int, int]],
        rollout_engine_rank: int,
        sent_checksums: dict[str, str] | None,
    ) -> None:
        if self.is_errored:
            logger.warning(f"[P2P-Shared] skipping a queued write to rollout cell {self.cell_id}")
            return
        _do_p2p_write_one_session(
            transfer_engine=transfer_engine,
            remote_session=target,
            names=names,
            weight_memory_registry=weight_memory_registry,
            cell_id=self.cell_id,
        )
        if sent_checksums is not None:
            _verify_transfer_checksums(
                self, rollout_engine_rank=rollout_engine_rank, names=names, sent_checksums=sent_checksums
            )


def _do_p2p_write_one_session(
    transfer_engine: Any,
    remote_session: RemoteWeightInfo,
    names: list[str],
    weight_memory_registry: dict[str, tuple[int, int, int]],
    *,
    cell_id: str,
) -> None:
    """P2P write from shared CPU pinned buffers to a single remote session.

    Used by the parallelized submission path where each session within an
    rollout engine rank is submitted as a separate task to its cell updater's thread.
    """
    source_ptrs, source_lens = [], []
    valid_names = []

    for name in names:
        cpu_reg = weight_memory_registry.get(name)
        assert cpu_reg, f"the _weight_memory_registry of {name} failed"

        data_ptr, numel, ele_size = cpu_reg
        source_ptrs.append(data_ptr)
        source_lens.append(numel * ele_size)
        valid_names.append(name)

    if not source_ptrs:
        return

    session_id = remote_session.session_id
    target_ptrs = []
    for name, source_len in zip(valid_names, source_lens, strict=True):
        if name in remote_session.weights_info:
            location = remote_session.weights_info[name]
            target_len = location.numel * location.element_size
            assert target_len == source_len, (
                f"[P2P-Shared] {name} spans {source_len} bytes here and {target_len} bytes on session "
                f"{session_id}, so writing it would run past the target buffer"
            )
            target_ptrs.append(location.address)

    assert len(target_ptrs) == len(source_ptrs), (
        f"[P2P-Shared] Pointer count mismatch for session {session_id}, "
        f"source: {len(source_ptrs)}, target: {len(target_ptrs)}"
    )

    reach_fault_hook(FaultHookName.TRAINER_WEIGHT_UPDATE_BEFORE_SEND)
    context = fault_hook_controller.current_context()
    started_at = datetime.now(timezone.utc)
    try:
        ret = transfer_engine.batch_transfer_sync_write(session_id, source_ptrs, target_ptrs, source_lens)
        if ret < 0:
            raise RuntimeError(f"[P2P-Shared] Transfer failed for session {session_id}, error: {ret}")
    except Exception as error:
        if is_event_logger_initialized() and context.debug_weight_update_id is not None:
            try:
                get_event_logger().log(
                    WeightTransferFailedEvent,
                    dict(
                        debug_weight_update_id=context.debug_weight_update_id,
                        cell_id=cell_id,
                        workers_hash=context.snapshot_cell_id_to_hashes[cell_id],
                        started_at=started_at,
                        error=repr(error),
                    ),
                    include_context=False,
                )
            except Exception:
                logger.exception("Could not record P2P transport failure for session %s", session_id)
        logger.exception("P2P transport write failed for session %s", session_id)
        raise


def _verify_transfer_checksums(
    cell_updater: _P2PRolloutCellUpdater,
    *,
    rollout_engine_rank: int,
    names: list[str],
    sent_checksums: dict[str, str],
) -> None:
    engine_body = cell_updater.submit_client_call(
        "check_weights", action="raw_checksum", names=names, selector=cell_updater.selector
    ).result()
    if engine_body is None:
        return
    checksum_utils.verify_transfer_checksums(
        sent_checksums=sent_checksums, engine_body=engine_body, cell_id=cell_updater.cell_id, rank=rollout_engine_rank
    )
