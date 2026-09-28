from collections.abc import Callable

import pytest

from miles.utils.audit_utils.config_snapshot.converter import ConfigSnapshotConverter
from miles.utils.audit_utils.config_snapshot.generated_values import (
    GENERATED_VALUES_ENV_VAR,
    read_generated_values,
    register_generated_value,
)
from miles.utils.audit_utils.config_snapshot.models import ConfigSnapshotGeneratedValue
from miles.utils.audit_utils.config_snapshot.normalizer import normalize_record
from miles.utils.test_utils.snapshot import SNAPSHOT_RECORD_DIR_ENV_VAR, dump_snapshot


class TestGeneratedValueRegistration:
    def test_registration_without_a_snapshot_attempt_has_no_side_effect(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """Ordinary launches do not carry snapshot normalization metadata."""
        monkeypatch.delenv(SNAPSHOT_RECORD_DIR_ENV_VAR, raising=False)
        monkeypatch.delenv(GENERATED_VALUES_ENV_VAR, raising=False)
        register_generated_value(kind="run_id", value="generated")
        assert read_generated_values() == []

    def test_repeated_values_are_idempotent_and_distinct_values_keep_distinct_tokens(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Generated identities stay distinct without token assignment depending on record order."""
        monkeypatch.setenv(SNAPSHOT_RECORD_DIR_ENV_VAR, "/snapshot/records")
        monkeypatch.delenv(GENERATED_VALUES_ENV_VAR, raising=False)
        register_generated_value(kind="run_id", value="first")
        register_generated_value(kind="run_id", value="first")
        register_generated_value(kind="run_id", value="second")
        assert [(item.name, item.value) for item in read_generated_values()] == [("0000", "first"), ("0001", "second")]


class TestGeneratedPathNormalization:
    def test_only_registered_values_in_explicit_fields_are_normalized(self, make_record: Callable) -> None:
        """Random paths stabilize while static roots, suffixes and unrelated strings remain visible."""
        outputs = []
        for run_id, temp, commit in [("260928-123456-001", "abc", "sha-a_3739"), ("260929-112233-002", "xyz", "sha-b_3739")]:
            record = make_record(config={
                "save": f"/data/{run_id}/checkpoints",
                "backend": {"load": f"/data/{run_id}/checkpoints"},
                "dump_details": f"/tmp/{temp}/bshd/dump_details",
                "wandb_group": f"prefix_{run_id}_{commit}",
                "custom_path": "/data/260928-123456-001/checkpoints",
                "hf_checkpoint": "/data/260928-123456-001/model",
            }).model_copy(update={"generated_values": [
                ConfigSnapshotGeneratedValue(kind="run_id", name="0000", value=run_id),
                ConfigSnapshotGeneratedValue(kind="temporary_directory", name="parity", value=f"/tmp/{temp}"),
                ConfigSnapshotGeneratedValue(kind="ci_commit_name", name="github", value=commit),
            ]})
            outputs.append(normalize_record(record))
        assert outputs[0] == outputs[1]
        assert outputs[0]["args"] == {
            "save": "/data/$RUN_ID_0000/checkpoints",
            "backend": {"load": "/data/$RUN_ID_0000/checkpoints"},
            "dump_details": "/tmp/$TEMPORARY_DIRECTORY_parity/bshd/dump_details",
            "wandb_group": "prefix_$RUN_ID_0000_$CI_COMMIT_NAME_github",
            "custom_path": "/data/260928-123456-001/checkpoints",
            "hf_checkpoint": "/data/260928-123456-001/model",
        }

    @pytest.mark.parametrize("path", ["/other/id/checkpoints", "/data/id/other", "/data/id-extra/checkpoints", "/data/other/checkpoints"])
    def test_path_root_suffix_and_unregistered_values_remain_distinguishable(
        self, make_record: Callable, path: str
    ) -> None:
        """Normalization cannot hide a different output root, file, or unregistered identity."""
        metadata = [ConfigSnapshotGeneratedValue(kind="run_id", name="0000", value="id")]
        expected = make_record(config={"save": "/data/id/checkpoints"}).model_copy(update={"generated_values": metadata})
        actual = make_record(config={"save": path}).model_copy(update={"generated_values": metadata})
        assert normalize_record(expected) != normalize_record(actual)

    def test_swapping_two_generated_references_remains_observable(self, make_record: Callable) -> None:
        """Two generated run IDs never collapse to one anonymous path token."""
        metadata = [ConfigSnapshotGeneratedValue(kind="run_id", name=str(i), value=value) for i,value in enumerate(["first", "second"])]
        records = [make_record(config={"save": f"/data/{value}/checkpoints"}).model_copy(update={"generated_values": metadata}) for value in ["first", "second"]]
        assert normalize_record(records[0]) != normalize_record(records[1])

    def test_rank_local_generated_identity_conflicts_cannot_be_hidden(self, make_record: Callable) -> None:
        """Equivalent ranks cannot silently normalize different values under the same identity."""
        records = [make_record(rank=rank, config={"save": f"/data/{value}/checkpoints"}).model_copy(update={"generated_values": [ConfigSnapshotGeneratedValue(kind="run_id", name="0000", value=value)]}) for rank,value in enumerate(["first", "second"])]
        with pytest.raises(ValueError, match="Conflicting generated"):
            ConfigSnapshotConverter.convert(records)

    def test_registrations_do_not_depend_on_raw_record_order(self, make_record: Callable) -> None:
        """Generated metadata produces identical snapshots when storage order changes."""
        records = [make_record(run=run, config={"save": f"/data/{value}/checkpoints"}).model_copy(update={"generated_values": [ConfigSnapshotGeneratedValue(kind="run_id", name="0000", value=value)]}) for run,value in enumerate(["first", "second"])]
        assert dump_snapshot(ConfigSnapshotConverter.convert(records)) == dump_snapshot(ConfigSnapshotConverter.convert(list(reversed(records))))
