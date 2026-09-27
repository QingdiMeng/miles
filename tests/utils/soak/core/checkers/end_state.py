from datetime import datetime

from tests.utils.soak.core.events import SoakEvent, SoakObservationEvent
from tests.utils.soak.core.views import tail_started_at


def assert_end_state_complete(events: list[SoakEvent], *, expected_count_of_kind: dict[str, int]) -> None:
    observation = _latest_clean_observation(events, since=tail_started_at(events))
    assert observation is not None, "Soak never cleanly observed the whole system after its last fault"

    for kind, expected_count in expected_count_of_kind.items():
        targets = [target for target in observation.targets if target.kind == kind]
        assert (
            len(targets) == expected_count
        ), f"Soak ended with {len(targets)} {kind} targets, expected {expected_count}"
        assert all(
            target.alive and target.ready for target in targets
        ), f"Soak ended with a {kind} target that is not alive and ready"


def _latest_clean_observation(events: list[SoakEvent], *, since: datetime) -> SoakObservationEvent | None:
    return next(
        (
            event
            for event in reversed(events)
            if isinstance(event, SoakObservationEvent)
            and event.targets is not None
            and not event.errors
            and event.timestamp >= since
        ),
        None,
    )
