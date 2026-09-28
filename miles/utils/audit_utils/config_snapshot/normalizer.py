from pathlib import Path

from pydantic import JsonValue

from miles.utils.audit_utils.config_snapshot.models import ConfigSnapshotGeneratedValue, ConfigSnapshotRecord
from miles.utils.audit_utils.process_identity import ProcessIdentity, TrainProcessIdentity

_RANK = "$RANK"


def normalize_record(record: ConfigSnapshotRecord) -> JsonValue:
    context = record.context
    config = _simple_replace(record.config, src_text=context.run_uuid, dst_text="$RUN_UUID")
    config = _normalize_generated_values(config, values=record.generated_values)
    if isinstance(context.source, TrainProcessIdentity):
        if not isinstance(config, dict) or not isinstance(args := config.get("args"), dict):
            raise ValueError("Training snapshots require a config.args object")
        if "rank" in args:
            assert type(args["rank"]) is int and args["rank"] >= 0, f"Unexpected args.rank: {args['rank']!r}"
            args["rank"] = _RANK
        if "backend" in args:
            backend = args["backend"]
            assert isinstance(backend, dict), "Training snapshots require a config.args.backend object"
            if "rank" in backend:
                assert (
                    type(backend["rank"]) is int and backend["rank"] >= 0
                ), f"Unexpected args.backend.rank: {backend['rank']!r}"
                backend["rank"] = _RANK
    return config


_PATH_FIELDS = frozenset(
    {
        "save", "load", "requested_load", "critic_save", "critic_load", "ref_load", "dump_details",
        "save_debug_event_data", "save_debug_train_data", "save_debug_trajectory_data", "save_debug_rollout_data",
        "load_debug_rollout_data", "ci_save_grad_norm", "te_precision_config_file",
    }
)


def _normalize_generated_values(
    config: JsonValue, *, values: list[ConfigSnapshotGeneratedValue]
) -> JsonValue:
    if not isinstance(config, dict):
        return config
    result = dict(config)
    for key, value in result.items():
        if key in {"args", "backend"} and isinstance(value, dict):
            result[key] = _normalize_generated_values(value, values=values)
        elif isinstance(value, str) and (key in _PATH_FIELDS or key == "wandb_group"):
            for entry in values:
                token = f"${entry.kind.upper()}_{entry.name}"
                if key == "wandb_group":
                    if entry.kind == "ci_commit_name" and value.endswith(f"_{entry.value}"):
                        value = value[: -len(entry.value)] + token
                    elif entry.kind == "run_id":
                        value = "_".join(token if part == entry.value else part for part in value.split("_"))
                elif entry.kind == "run_id":
                    value = "/".join(token if part == entry.value else part for part in value.split("/"))
                elif entry.kind == "temporary_directory" and (value == entry.value or value.startswith(entry.value + "/")):
                    value = str(Path(entry.value).parent / token) + value[len(entry.value):]
            result[key] = value
    return result


def normalized_source_name(source: ProcessIdentity) -> str:
    return source.to_cell_name() if isinstance(source, TrainProcessIdentity) else source.to_name()


def validate_generated_values(records: list[ConfigSnapshotRecord]) -> None:
    observed: dict[tuple[str, str, str, str], str] = {}
    for record in records:
        for entry in record.generated_values:
            if _normalize_generated_values(record.config, values=[entry]) == record.config:
                continue
            key = (record.context.run_uuid, record.context.deploy_instance_id, entry.kind, entry.name)
            if key in observed and observed[key] != entry.value:
                raise ValueError(f"Conflicting generated snapshot value for {key}")
            observed[key] = entry.value


def _simple_replace(value: JsonValue, *, src_text: str, dst_text: str) -> JsonValue:
    if isinstance(value, str):
        return value.replace(src_text, dst_text)
    if isinstance(value, dict):
        return {key: _simple_replace(item, src_text=src_text, dst_text=dst_text) for key, item in value.items()}
    if isinstance(value, list):
        return [_simple_replace(item, src_text=src_text, dst_text=dst_text) for item in value]
    return value
