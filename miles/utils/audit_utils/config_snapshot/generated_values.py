import json
import os
from typing import Literal

from miles.utils.audit_utils.config_snapshot.models import ConfigSnapshotGeneratedValue
from miles.utils.test_utils.snapshot import SNAPSHOT_RECORD_DIR_ENV_VAR

GENERATED_VALUES_ENV_VAR = "MILES_SNAPSHOT_GENERATED_VALUES"


def register_generated_value(
    *,
    kind: Literal["run_id", "temporary_directory", "ci_commit_name", "wandb_run_id"],
    value: str,
    name: str | None = None,
) -> None:
    if not os.environ.get(SNAPSHOT_RECORD_DIR_ENV_VAR):
        return

    values = read_generated_values()
    if name is None:
        if any(item.kind == kind and item.value == value for item in values):
            return
        name = f"{sum(item.kind == kind for item in values):04d}"
    entry = ConfigSnapshotGeneratedValue(kind=kind, name=name, value=value)
    for item in values:
        if (item.kind, item.name) == (kind, name):
            if item != entry:
                raise ValueError(f"Conflicting generated snapshot value: {kind}/{name}")
            return
    os.environ[GENERATED_VALUES_ENV_VAR] = json.dumps([item.model_dump() for item in [*values, entry]])


def read_generated_values() -> list[ConfigSnapshotGeneratedValue]:
    return [
        ConfigSnapshotGeneratedValue.model_validate(item)
        for item in json.loads(os.environ.get(GENERATED_VALUES_ENV_VAR, "[]"))
    ]
