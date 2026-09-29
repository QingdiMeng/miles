# Temporary experiment (not part of the FT suite): run the baseline side of the all-gather comparison several
# times with identical arguments and report, sample by sample, whether the generated tokens differ between runs.

import shutil
from collections import Counter
from dataclasses import dataclass, field
from pathlib import Path

import torch

from tests.e2e.ft.conftest_ft.execution import get_deterministic_p2p_train_args, prepare, run_training
from tests.e2e.ft.conftest_ft.modes import resolve_mode
from tests.e2e.ft.conftest_ft.scenario_trainer_all_gather_fault import UPDATE_WEIGHTS_TIMEOUT_SECONDS
from tests.utils.soak.core.utils import create_soak_config, resolve_dump_dir

from miles.utils.audit_utils.event_logger.logger import EVENTS_DIRNAME, read_events
from miles.utils.audit_utils.event_logger.models import InferenceEngineWeightChecksumEvent, WeightUpdateResultEvent
from miles.utils.external_utils import command_utils

MODE: str = "kill_train__dp2_tp2"
NUM_RUNS: int = 5
NUM_ROLLOUTS: int = 4


@dataclass(frozen=True)
class Variant:
    name: str
    extra_env_vars: dict[str, str] = field(default_factory=dict)
    extra_train_args: str = ""


def run_experiment(variant: Variant) -> None:
    mode = resolve_mode(MODE)
    config = create_soak_config(command_utils.default_config())
    dump_dir = Path(resolve_dump_dir(f"exp_determinism_{variant.name}", run_id=config.run_id))
    print(
        f"Experiment {variant.name}: dump_dir={dump_dir} env={variant.extra_env_vars} args={variant.extra_train_args!r}"
    )
    prepare(mode, config=config)

    run_dirs: list[Path] = []
    for i in range(NUM_RUNS):
        run_dir = dump_dir / f"run{i}"
        train_args = get_deterministic_p2p_train_args(
            mode,
            dump_dir=str(run_dir),
            num_steps=NUM_ROLLOUTS,
            enable_dumper=False,
            test_name=f"exp_determinism_{variant.name}",
            extra_ft_components=("rollout",),
        )
        train_args += f"--update-weights-timeout {UPDATE_WEIGHTS_TIMEOUT_SECONDS} " + variant.extra_train_args
        run_training(
            train_args=train_args,
            mode=mode,
            dump_dir=str(run_dir),
            extra_env_vars={"MILES_EXP_TOP_LOGPROBS": "1", **variant.extra_env_vars},
            config=config,
        )
        run_dirs.append(run_dir)

    try:
        _report_weights(run_dirs)
        any_diff = False
        for i in range(1, NUM_RUNS):
            print(f"===== run0 vs run{i}")
            any_diff |= _report_diff(run_dirs[0], run_dirs[i])
        assert (
            not any_diff
        ), f"Experiment {variant.name}: identical runs produced different rollout tokens (see report)"
        print(f"Experiment {variant.name}: all {NUM_RUNS} runs produced identical rollout tokens")
    finally:
        shutil.rmtree(dump_dir, ignore_errors=True)


def _report_diff(dir_a: Path, dir_b: Path) -> bool:
    any_diff = False
    for rollout_id in range(NUM_ROLLOUTS):
        pa, pb = dir_a / "rollout_data" / f"{rollout_id}.pt", dir_b / "rollout_data" / f"{rollout_id}.pt"
        if not pa.exists() or not pb.exists():
            print(f"r{rollout_id}: missing {pa if not pa.exists() else pb}")
            any_diff = True
            continue
        sa, sb = _load(pa), _load(pb)
        by_key_b = {_key(s): s for s in sb}
        n_diff = prompt_diff = 0
        top_same: Counter = Counter()
        offsets: Counter = Counter()
        groups: Counter = Counter()
        examples: list[str] = []
        for s in sa:
            t = by_key_b.get(_key(s))
            if t is None:
                print(f"r{rollout_id}: sample {_key(s)} missing in the other run")
                any_diff = True
                continue
            prompt_len = len(s["tokens"]) - s["response_length"]
            d = _first_diff(s["tokens"], t["tokens"])
            if d is None:
                continue
            n_diff += 1
            groups[s.get("group_index")] += 1
            if d < prompt_len:
                prompt_diff += 1
                continue
            offsets[d - prompt_len] += 1
            top_same.update(_classify_top_logprobs(s, t, d - prompt_len))
            if len(examples) < 6 or (len(examples) < 10 and (s.get("metadata") or {}).get("exp_top_logprobs")):
                lp_a, lp_b = s.get("rollout_log_probs") or [], t.get("rollout_log_probs") or []
                before = d - prompt_len
                max_delta = max((abs(x - y) for x, y in zip(lp_a[:before], lp_b[:before])), default=0.0)
                examples.append(
                    f"    sample {_key(s)}: first diff at response token {before}, max |dlogprob| before it = "
                    f"{max_delta:.3e}, weight_versions equal = {str(s.get('weight_versions')) == str(t.get('weight_versions'))}"
                )
                examples.extend(_describe_top_logprobs(s, t, before))
        summary = f"r{rollout_id}: {len(sa)} samples, {n_diff} differ"
        if n_diff:
            any_diff = True
            summary += (
                f"; prompt-differs={prompt_diff}; first response offset histogram={sorted(offsets.items())[:12]}"
                f"; groups affected={len(groups)} {sorted(groups.items())[:16]}"
                f"; top-5 at pos 0 / at diff: {dict(top_same)}"
            )
        print(summary)
        for line in examples:
            print(line)
    return any_diff


def _report_weights(run_dirs: list[Path]) -> None:
    print("===== weights per version (engine digest per cell / trainer hashes digest)")
    per_run = [_collect_weight_digests(d) for d in run_dirs]
    for version in sorted(set().union(*[set(r) for r in per_run])):
        cells = [r.get(version, {}) for r in per_run]
        print(f"v{version}: " + " | ".join(f"run{i}={c}" for i, c in enumerate(cells)))
    for i, d in enumerate(run_dirs):
        for rollout_id in range(NUM_ROLLOUTS):
            if (path := d / "rollout_data" / f"{rollout_id}.pt").exists():
                versions = Counter(str(s.get("weight_versions")) for s in _load(path))
                print(f"run{i} r{rollout_id} weight_versions: {versions.most_common(3)}")


def _collect_weight_digests(run_dir: Path) -> dict[str, dict[str, str]]:
    out: dict[str, dict[str, str]] = {}
    for event in read_events(run_dir / EVENTS_DIRNAME):
        if isinstance(event, InferenceEngineWeightChecksumEvent):
            for snap in event.engine_snapshots:
                digest = f"{hash(tuple(sorted(snap.tensor_checksums.items()))) & 0xFFFFFF:06x}"
                out.setdefault(f"{event.weight_version}", {})[f"eng:{snap.cell_id}"] = digest
        if isinstance(event, WeightUpdateResultEvent):
            digest = f"{hash(tuple(sorted(event.snapshot_cell_id_to_hashes.items()))) & 0xFFFFFF:06x}"
            out.setdefault(f"{event.published_version}", {})[f"trainer@r{event.rollout_id}"] = digest
    return out


def _classify_top_logprobs(a: dict, b: dict, diff_at: int) -> list[str]:
    ta, tb = (a.get("metadata") or {}).get("exp_top_logprobs"), (b.get("metadata") or {}).get("exp_top_logprobs")
    if not ta or not tb:
        return ["no_top"]
    out = []
    for name, pos in (("pos0", 0), ("diff", diff_at)):
        if pos >= len(ta) or pos >= len(tb):
            out.append(f"{name}_beyond")
        elif ta[pos] == tb[pos]:
            out.append(f"{name}_identical")
        elif [x[1] for x in ta[pos]] == [x[1] for x in tb[pos]]:
            out.append(f"{name}_same_ids_diff_lp")
        else:
            out.append(f"{name}_diff_ids")
    return out


def _describe_top_logprobs(a: dict, b: dict, diff_at: int) -> list[str]:
    ta, tb = (a.get("metadata") or {}).get("exp_top_logprobs"), (b.get("metadata") or {}).get("exp_top_logprobs")
    if not ta or not tb:
        return ["      (no top logprobs recorded)"]
    lines = []
    for pos in sorted({0, 1, max(diff_at - 1, 0), diff_at}):
        if pos >= len(ta) or pos >= len(tb):
            continue
        same_ids = [x[1] for x in ta[pos]] == [x[1] for x in tb[pos]]
        max_lp = max((abs(x[0] - y[0]) for x, y in zip(ta[pos], tb[pos])), default=0.0)
        lines.append(
            f"      pos {pos}: same top-5 ids = {same_ids}, max |d top-5 logprob| = {max_lp:.3e}; "
            f"a = {[(round(x[0], 4), x[1]) for x in ta[pos]]}; b = {[(round(y[0], 4), y[1]) for y in tb[pos]]}"
        )
    return lines


def _load(path: Path) -> list[dict]:
    return torch.load(path, weights_only=False)["samples"]


def _key(sample: dict) -> tuple:
    return (sample.get("group_index"), sample.get("index"))


def _first_diff(a: list[int], b: list[int]) -> int | None:
    for i, (x, y) in enumerate(zip(a, b)):
        if x != y:
            return i
    return None if len(a) == len(b) else min(len(a), len(b))
