import shlex

from tests.e2e.ft.conftest_ft.modes import MODES
from tests.e2e.ft.conftest_ft.scenario_trainer_no_failure import (
    INJECT_START_ROLLOUT_ID,
    _build_baseline_args,
    _build_target_args,
)

_REAL_ROLLOUT_MODE = "kill_train__dp2_cp2__moe_5layer"
_FAKE_ROLLOUT_MODE = "kill_train__dp2_cp2_tp2_ep2__fake_rollout__moe_5layer"


def _tokens(mode_name: str, side: str) -> list[str]:
    build = _build_baseline_args if side == "baseline" else _build_target_args
    return shlex.split(build(MODES[mode_name], f"/dumps/run/{side}", enable_dumper=False))


def _option_value(tokens: list[str], option: str) -> str:
    return tokens[tokens.index(option) + 1]


def test_the_real_rollout_target_replays_the_baseline_recording_after_the_first_rollout() -> None:
    """After one update the two topologies' weights differ by ulps, so live samples would diverge."""
    tokens = _tokens(_REAL_ROLLOUT_MODE, "target")

    assert _option_value(tokens, "--ci-inject-rollout-data-path") == "/dumps/run/baseline/rollout_data/{rollout_id}.pt"
    assert _option_value(tokens, "--ci-inject-rollout-data-start-rollout-id") == str(INJECT_START_ROLLOUT_ID) == "1"
    assert _option_value(tokens, "--ci-inject-rollout-data-min-match-ratio") == "0.5"
    assert _option_value(_tokens(_REAL_ROLLOUT_MODE, "baseline"), "--save-debug-rollout-data") == (
        "/dumps/run/baseline/rollout_data/{rollout_id}.pt"
    )


def test_the_baseline_and_fake_rollout_modes_never_inject() -> None:
    """The baseline records what the target replays, and fake-rollout modes already train on fixed data."""
    assert "--ci-inject-rollout-data-path" not in _tokens(_REAL_ROLLOUT_MODE, "baseline")
    for side in ("baseline", "target"):
        assert "--ci-inject-rollout-data-path" not in _tokens(_FAKE_ROLLOUT_MODE, side)
