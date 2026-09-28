"""Cast the official DeepSeek-V4.1 checkpoint to the BF16 checkpoint Miles and SGLang both load.

- Dense FP8 weights (e4m3 with 32x32-block ue8m0 ``.scale``) and packed-e2m1 routed experts (int8 with
  1x32-block ue8m0 ``.scale``) become BF16 ``.weight`` tensors; their scales are dropped. Both casts are
  exact: a power-of-two scale times an e4m3 or e2m1 value fits in BF16.
- The two engram tables (``layers.N.engram.embed.weight`` / ``.scale``) stay FP8 + e8m0: the trainer maps
  them read-only and SGLang allocates them as FP8 regardless of the quantization config.
- MTP, vision, aligner and image-token tensors are dropped; ``quantization_config`` and ``vision_config``
  are removed from config.json so SGLang builds neither the quantized layers nor the vision tower.

Shard the file list over processes with ``--shard-rank/--num-shards`` (one per GPU) and run
``--finalize-only`` once to write config.json, tokenizer files and model.safetensors.index.json.

python tools/convert_dsv41_to_bf16.py --src <DeepSeek-V4.1-Flash> --dst <DeepSeek-V4.1-bf16> \
    --shard-rank 0 --num-shards 8 --device cuda:0
"""

import argparse
import json
import math
import os
import shutil
import struct
from pathlib import Path

import torch
from safetensors import safe_open
from safetensors.torch import save_file

FP4_TABLE = torch.tensor(
    [0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0, -0.0, -0.5, -1.0, -1.5, -2.0, -3.0, -4.0, -6.0],
    dtype=torch.float32,
)
FP8_BLOCK = 32
FP4_BLOCK = 32
DROPPED_PREFIXES = ("mtp.", "vision.", "aligner.", "image_start", "image_end", "image_newline")
KEPT_QUANTIZED_SUFFIXES = (".engram.embed.weight", ".engram.embed.scale")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--src", type=Path, required=True)
    parser.add_argument("--dst", type=Path, required=True)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--shard-rank", type=int, default=0)
    parser.add_argument("--num-shards", type=int, default=1)
    parser.add_argument("--finalize-only", action="store_true")
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def read_header(path: Path) -> dict:
    with path.open("rb") as file:
        (length,) = struct.unpack("<Q", file.read(8))
        header = json.loads(file.read(length))
    header.pop("__metadata__", None)
    return header


def dequant_fp8(weight: torch.Tensor, scale: torch.Tensor) -> torch.Tensor:
    out_dim, in_dim = weight.shape
    assert scale.shape == (math.ceil(out_dim / FP8_BLOCK), math.ceil(in_dim / FP8_BLOCK)), (weight.shape, scale.shape)
    expanded = scale.float().repeat_interleave(FP8_BLOCK, 0).repeat_interleave(FP8_BLOCK, 1)[:out_dim, :in_dim]
    return (weight.float() * expanded).to(torch.bfloat16)


def dequant_fp4(packed: torch.Tensor, scale: torch.Tensor) -> torch.Tensor:
    out_dim, in_dim = packed.shape[0], packed.shape[1] * 2
    assert scale.shape == (out_dim, in_dim // FP4_BLOCK), (packed.shape, scale.shape)
    table = FP4_TABLE.to(packed.device)
    packed = packed.view(torch.uint8)
    values = torch.stack([table[(packed & 0x0F).long()], table[(packed >> 4).long()]], dim=-1).reshape(out_dim, in_dim)
    return (values * scale.float().repeat_interleave(FP4_BLOCK, 1)).to(torch.bfloat16)


class TensorSource:
    """Reads any tensor of the checkpoint, whichever shard holds it."""

    def __init__(self, src: Path) -> None:
        self.src = src
        self.location: dict[str, str] = {}
        self.dtype: dict[str, str] = {}
        for path in sorted(src.glob("*.safetensors")):
            for name, info in read_header(path).items():
                assert name not in self.location, f"duplicate tensor {name}"
                self.location[name] = path.name
                self.dtype[name] = info["dtype"]
        self._handles: dict[str, object] = {}

    def get(self, name: str, device: str) -> torch.Tensor:
        shard = self.location[name]
        if shard not in self._handles:
            self._handles[shard] = safe_open(self.src / shard, framework="pt", device="cpu")
        return self._handles[shard].get_tensor(name).to(device)


def convert_shard(source: TensorSource, shard: str, dst: Path, device: str) -> dict[str, int]:
    counts = {"dequant_fp8": 0, "dequant_fp4": 0, "kept": 0, "dropped": 0}
    tensors = {}
    for name in sorted(n for n, s in source.location.items() if s == shard):
        if name.startswith(DROPPED_PREFIXES):
            counts["dropped"] += 1
            continue
        if name.endswith(KEPT_QUANTIZED_SUFFIXES):
            tensors[name] = source.get(name, "cpu")
            counts["kept"] += 1
            continue
        if name.endswith(".scale"):
            weight_name = name.removesuffix(".scale") + ".weight"
            assert weight_name in source.location, f"orphan scale {name}"
            continue
        scale_name = name.removesuffix(".weight") + ".scale"
        if name.endswith(".weight") and scale_name in source.location:
            weight = source.get(name, device)
            scale = source.get(scale_name, device)
            if source.dtype[name] == "I8":
                tensors[name] = dequant_fp4(weight, scale).cpu()
                counts["dequant_fp4"] += 1
            else:
                assert source.dtype[name] == "F8_E4M3", (name, source.dtype[name])
                tensors[name] = dequant_fp8(weight, scale).cpu()
                counts["dequant_fp8"] += 1
            continue
        assert source.dtype[name] not in ("I8", "F8_E4M3", "F8_E8M0"), f"quantized tensor without a scale: {name}"
        tensors[name] = source.get(name, "cpu")
        counts["kept"] += 1
    if tensors:
        tmp = dst / f"{shard}.tmp"
        save_file(tensors, tmp)
        os.replace(tmp, dst / shard)
    return counts


def finalize(src: Path, dst: Path) -> None:
    config = json.loads((src / "config.json").read_text())
    config.pop("quantization_config", None)
    config.pop("vision_config", None)
    if isinstance(config.get("text_config"), dict):
        config["text_config"].pop("quantization_config", None)
    config["torch_dtype"] = "bfloat16"
    (dst / "config.json").write_text(json.dumps(config, indent=2) + "\n")
    for path in src.iterdir():
        if (
            path.is_file()
            and path.suffix != ".safetensors"
            and path.name not in ("config.json",)
            and not path.name.endswith(".index.json")
        ):
            shutil.copy2(path, dst / path.name)

    dtype_sizes = {"BF16": 2, "F16": 2, "F32": 4, "I32": 4, "I64": 8, "F8_E4M3": 1, "F8_E8M0": 1}
    weight_map, total_size = {}, 0
    for path in sorted(dst.glob("*.safetensors")):
        for name, info in read_header(path).items():
            assert name not in weight_map, f"duplicate tensor {name}"
            weight_map[name] = path.name
            total_size += math.prod(info["shape"]) * dtype_sizes[info["dtype"]]
    (dst / "model.safetensors.index.json").write_text(
        json.dumps({"metadata": {"total_size": total_size}, "weight_map": weight_map}, indent=2) + "\n"
    )
    print(f"finalized {dst}: {len(weight_map)} tensors, {total_size / 2**30:.1f} GiB")


def main() -> None:
    args = parse_args()
    assert args.src.resolve() != args.dst.resolve()
    args.dst.mkdir(parents=True, exist_ok=True)
    if args.finalize_only:
        finalize(args.src, args.dst)
        return

    source = TensorSource(args.src)
    shards = sorted(set(source.location.values()))
    mine = shards[args.shard_rank :: args.num_shards]
    for shard in mine:
        if (args.dst / shard).exists() and not args.overwrite:
            print(f"skip {shard}: exists")
            continue
        counts = convert_shard(source, shard, args.dst, args.device)
        print(f"{shard}: {counts}", flush=True)


if __name__ == "__main__":
    main()
