"""Export Qwen3.6 calibration weights, retaining unchanged multimodal components."""

import json
from argparse import Namespace
from collections.abc import Iterator
from pathlib import Path

import safetensors.torch
import torch
import torch.distributed.checkpoint as dcp
from safetensors import safe_open
from tap import Tap

import tools.convert_torch_dist_to_hf as converter
from tools.convert_torch_dist_to_hf import EmptyStateDictLoadPlanner, WrappedStorageReader, copy_assets, save_tensors


class Args(Tap):
    checkpoint: Path
    model: Path
    output: Path


def packed_expert(args: Namespace, name: str, param: torch.Tensor) -> Iterator[tuple[str, torch.Tensor]]:
    if ".experts.experts.linear_fc" in name:
        if param.shape[0] != args.num_experts:
            raise ValueError("Unexpected packed expert shape")
        yield name.replace(".experts.experts.", ".experts.").removesuffix(".weight"), param
    else:
        yield name, param


def export(args: Args) -> None:
    config = json.loads((args.model / "config.json").read_text())
    text = config.get("text_config", config)
    shape = Namespace(num_layers=text["num_hidden_layers"], num_experts=text["num_experts"],
                      hidden_size=text["hidden_size"], num_attention_heads=text["num_attention_heads"],
                      num_query_groups=text["num_key_value_heads"], kv_channels=text["head_dim"], vocab_size=text["vocab_size"])
    converter.get_expert_param = packed_expert
    state = {}
    dcp.state_dict_loader._load_state_dict(state, storage_reader=WrappedStorageReader(str(args.checkpoint)),
                                         planner=EmptyStateDictLoadPlanner(), no_dist=True)
    save_tensors(shape, config["model_type"], state, str(args.output), 5 * 1024**3, text["vocab_size"])
    copy_assets(str(args.model), str(args.output))
    index_path = args.output / "model.safetensors.index.json"
    index = json.loads(index_path.read_text())
    original = json.loads((args.model / "model.safetensors.index.json").read_text())["weight_map"]
    missing = set(original) - set(index["weight_map"])
    if any(not k.startswith(("model.visual.", "mtp.")) for k in missing):
        raise ValueError(f"Missing trained weights: {sorted(missing)[:10]}")
    extra = {}
    for filename in sorted({original[k] for k in missing}):
        with safe_open(args.model / filename, framework="pt", device="cpu") as source:
            for key in sorted(missing):
                if original[key] == filename:
                    extra[key] = source.get_tensor(key)
    if extra:
        safetensors.torch.save_file(extra, args.output / "unchanged-components.safetensors")
        index["weight_map"].update({k: "unchanged-components.safetensors" for k in extra})
        index["metadata"]["total_size"] += sum(t.numel() * t.element_size() for t in extra.values())
    if set(original) != set(index["weight_map"]):
        raise ValueError("Export key mismatch")
    index_path.write_text(json.dumps(index, indent=2))
    (args.output / "export-complete.json").write_text(json.dumps({"checkpoint": str(args.checkpoint), "keys": len(original)}))
    print("EXPORT_COMPLETE", len(original), flush=True)


if __name__ == "__main__":
    export(Args(underscores_to_dashes=True).parse_args())
