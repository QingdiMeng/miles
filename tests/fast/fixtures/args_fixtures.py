from __future__ import annotations

import argparse
import contextlib
import functools
import sys
from argparse import Namespace
from collections.abc import Iterator
from typing import Any
from unittest.mock import patch

from miles.backends.megatron_utils.megatron_config import resolve_megatron_config
from miles.backends.sglang_utils.sglang_config import SglangConfig
from miles.utils.arguments import get_miles_extra_args_provider
from miles.utils.run_uuid import RUN_UUID_LENGTH

# megatron's own parser adds these and miles' code reads them, but a unit test builds only the miles
# extras, so nothing else would put them on the namespace
_TRAIN_BACKEND_DEFAULTS: dict[str, Any] = dict(
    disable_param_buffers_cpu_backup=False,
    fp16=False,
    lr_warmup_iters=None,
    load=None,
)

# declared with no default and resolved after parsing, so the raw parser value is one no production
# code has ever seen; these are what the resolution settles on for a plain single-deployment run
_RESOLVED_AFTER_PARSING: dict[str, Any] = dict(
    offload_train=False,
    offload_rollout=False,
    eval_uses_snapshots=False,
    starts_inference_engines=True,
    use_critic=False,
    run_uuid="0" * RUN_UUID_LENGTH,
)


@contextlib.contextmanager
def _with_relaxed_parser_required_args(parser: argparse.ArgumentParser) -> Iterator[None]:
    required = [action for action in parser._actions if action.required]
    for action in required:
        action.required = False
    try:
        yield
    finally:
        for action in required:
            action.required = True


@functools.cache
def parser_defaults() -> dict[str, Any]:
    # a hand-written defaults dict falls behind the moment production reads a new argument, and the
    # test then dies on AttributeError somewhere deep instead of on the thing it was written for
    parser = argparse.ArgumentParser()
    get_miles_extra_args_provider()(parser)

    with _with_relaxed_parser_required_args(parser), patch.object(sys, "argv", ["test"]):
        parsed, _ = parser.parse_known_args([])
    return {**_TRAIN_BACKEND_DEFAULTS, **vars(parsed), **_RESOLVED_AFTER_PARSING}


def resolve_parse_boundary_configs(args: Namespace) -> Namespace:
    args.raw_megatron = resolve_megatron_config(args, base_args={})
    args.sglang, args.sglang_scaling = SglangConfig.parse_args(args)
    return args
