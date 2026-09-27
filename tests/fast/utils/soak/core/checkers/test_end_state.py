import pytest
from tests.fast.utils.soak.soak_fakes import _at, _cell_target, _observation
from tests.utils.soak.core.checkers.end_state import assert_end_state_complete
from tests.utils.soak.core.events import SoakAdmissionClosedEvent

_EXPECTED = {"actor": 2, "rollout": 1}


def _complete() -> list:
    return [_cell_target(cell_index=0), _cell_target(cell_index=1), _cell_target(kind="rollout")]


def _closed(seconds: float = 0) -> SoakAdmissionClosedEvent:
    return SoakAdmissionClosedEvent(timestamp=_at(seconds))


class TestAssertEndStateComplete:
    def test_a_final_observation_with_every_target_alive_and_ready_passes(self) -> None:
        """The run ends whole when each kind has its expected ready targets."""
        assert_end_state_complete(
            [_closed(), _observation(_complete(), at=_at(1))],
            expected_count_of_kind=_EXPECTED,
        )

    def test_a_run_without_observations_is_rejected(self) -> None:
        """Nothing observed means nothing proven about the end state."""
        with pytest.raises(AssertionError, match="never cleanly observed"):
            assert_end_state_complete([_closed()], expected_count_of_kind=_EXPECTED)

    def test_a_failed_poll_after_the_training_exit_falls_back_to_the_last_clean_one(self) -> None:
        """A finished training run takes its api server down, so the whole view is the last clean poll of the tail."""
        assert_end_state_complete(
            [_closed(), _observation(_complete(), at=_at(1)), _observation(None, at=_at(2))],
            expected_count_of_kind=_EXPECTED,
        )

    def test_a_clean_observation_before_the_tail_does_not_count(self) -> None:
        """A whole view from before the last fault proves nothing about how the run ended."""
        with pytest.raises(AssertionError, match="never cleanly observed"):
            assert_end_state_complete(
                [_observation(_complete(), at=_at(0)), _closed(1), _observation(None, at=_at(2))],
                expected_count_of_kind=_EXPECTED,
            )

    def test_an_observation_with_errors_is_not_clean(self) -> None:
        """A partial view is not proof of completeness."""
        with pytest.raises(AssertionError, match="never cleanly observed"):
            assert_end_state_complete(
                [_closed(), _observation(_complete(), at=_at(1), errors={"pods": "down"})],
                expected_count_of_kind=_EXPECTED,
            )

    @pytest.mark.parametrize(
        "targets",
        [
            [_cell_target(cell_index=0), _cell_target(kind="rollout")],
            [
                _cell_target(cell_index=0),
                _cell_target(cell_index=1),
                _cell_target(cell_index=2),
                _cell_target(kind="rollout"),
            ],
            [_cell_target(cell_index=0), _cell_target(cell_index=1)],
        ],
    )
    def test_a_kind_with_the_wrong_target_count_is_rejected(self, targets: list) -> None:
        """Missing or leftover targets of any kind fail the end state."""
        with pytest.raises(AssertionError, match="targets, expected"):
            assert_end_state_complete([_closed(), _observation(targets, at=_at(1))], expected_count_of_kind=_EXPECTED)

    @pytest.mark.parametrize("state", [{"alive": False}, {"ready": False}])
    def test_a_dead_or_unready_target_is_rejected(self, state: dict) -> None:
        """Every target must be both alive and ready at the end."""
        targets = [_cell_target(cell_index=0), _cell_target(cell_index=1, **state), _cell_target(kind="rollout")]

        with pytest.raises(AssertionError, match="not alive and ready"):
            assert_end_state_complete([_closed(), _observation(targets, at=_at(1))], expected_count_of_kind=_EXPECTED)

    def test_only_the_latest_clean_observation_is_judged(self) -> None:
        """A healthy earlier poll cannot mask a broken later one."""
        broken = [_cell_target(cell_index=0), _cell_target(cell_index=1, ready=False), _cell_target(kind="rollout")]

        with pytest.raises(AssertionError, match="not alive and ready"):
            assert_end_state_complete(
                [_closed(), _observation(_complete(), at=_at(1)), _observation(broken, at=_at(2))],
                expected_count_of_kind=_EXPECTED,
            )
