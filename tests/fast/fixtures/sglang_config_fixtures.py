from argparse import Namespace

from miles.backends.sglang_utils.sglang_config import SglangConfig, SglangScalingConfig, _compute_raw_sglang_config


def resolve_sglang_config(args: Namespace) -> SglangConfig:
    config, _ = resolve_sglang_config_and_scaling(args)
    return config


def resolve_sglang_config_and_scaling(args: Namespace) -> tuple[SglangConfig, SglangScalingConfig]:
    return SglangConfig.resolve(raw=_compute_raw_sglang_config(args), args=args, base_args={})
