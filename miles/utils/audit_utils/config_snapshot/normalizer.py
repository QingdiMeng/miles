from pathlib import Path

from pydantic import JsonValue

from miles.utils.audit_utils.config_snapshot.endpoints import SnapshotEndpointNormalizer
from miles.utils.audit_utils.config_snapshot.models import ConfigSnapshotGeneratedValue, ConfigSnapshotRecord
from miles.utils.audit_utils.process_identity import ProcessIdentity, TrainProcessIdentity

_RANK = "$RANK"


def normalize_record(
    record: ConfigSnapshotRecord,
    *,
    endpoints: SnapshotEndpointNormalizer | None = None,
    generated_values: list[ConfigSnapshotGeneratedValue] | None = None,
) -> JsonValue:
    context = record.context
    config = endpoints.normalize(record) if endpoints is not None else record.config
    config = _simple_replace(config, src_text=context.run_uuid, dst_text="$RUN_UUID")
    config = _normalize_generated_values(
        config, values=record.generated_values if generated_values is None else generated_values
    )
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
        "save",
        "load",
        "requested_load",
        "critic_save",
        "critic_load",
        "ref_load",
        "dump_details",
        "save_debug_event_data",
        "save_debug_train_data",
        "save_debug_trajectory_data",
        "save_debug_rollout_data",
        "load_debug_rollout_data",
        "ci_save_grad_norm",
        "te_precision_config_file",
    }
)


def _normalize_generated_values(config: JsonValue, *, values: list[ConfigSnapshotGeneratedValue]) -> JsonValue:
    if not isinstance(config, dict):
        return config
    result = dict(config)
    for key, value in result.items():
        if key in {"args", "backend"} and isinstance(value, dict):
            result[key] = _normalize_generated_values(value, values=values)
        elif isinstance(value, str) and (key in _PATH_FIELDS or key in {"wandb_group", "wandb_run_id"}):
            for entry in values:
                token = f"${entry.kind.upper()}_{entry.name}"
                if key == "wandb_run_id":
                    if entry.kind == "wandb_run_id" and value == entry.value:
                        value = token
                elif key == "wandb_group":
                    if entry.kind == "ci_commit_name" and value.endswith(f"_{entry.value}"):
                        value = value[: -len(entry.value)] + token
                    elif entry.kind == "run_id":
                        value = "_".join(token if part == entry.value else part for part in value.split("_"))
                elif entry.kind == "run_id":
                    value = "/".join(token if part == entry.value else part for part in value.split("/"))
                elif entry.kind == "temporary_directory" and (
                    value == entry.value or value.startswith(entry.value + "/")
                ):
                    value = str(Path(entry.value).parent / token) + value[len(entry.value) :]
            result[key] = value
    return result


def normalized_source_name(source: ProcessIdentity) -> str:
    return source.to_cell_name() if isinstance(source, TrainProcessIdentity) else source.to_name()


def collect_generated_values(
    records: list[ConfigSnapshotRecord],
) -> dict[tuple[str, str, str], list[ConfigSnapshotGeneratedValue]]:
    by_scope: dict[tuple[str, str, str], list[ConfigSnapshotRecord]] = {}
    for record in records:
        scope = (record.context.name, record.context.run_uuid, record.context.deploy_instance_id)
        by_scope.setdefault(scope, []).append(record)

    result = {}
    for scope, scoped_records in by_scope.items():
        candidates = {
            (entry.kind, entry.name, entry.value): entry
            for record in scoped_records
            for entry in record.generated_values
        }
        observed: dict[tuple[str, str], ConfigSnapshotGeneratedValue] = {}
        for entry in sorted(candidates.values(), key=lambda entry: (entry.kind, entry.name, entry.value)):
            if not any(
                _normalize_generated_values(record.config, values=[entry]) != record.config
                for record in scoped_records
            ):
                continue
            key = (entry.kind, entry.name)
            if key in observed and observed[key] != entry:
                raise ValueError(f"Conflicting generated snapshot value for {scope}/{key}")
            observed[key] = entry
        result[scope] = list(observed.values())
    return result


def _simple_replace(value: JsonValue, *, src_text: str, dst_text: str) -> JsonValue:
    if isinstance(value, str):
        return value.replace(src_text, dst_text)
    if isinstance(value, dict):
        return {key: _simple_replace(item, src_text=src_text, dst_text=dst_text) for key, item in value.items()}
    if isinstance(value, list):
        return [_simple_replace(item, src_text=src_text, dst_text=dst_text) for item in value]
    return value
