import re
from collections.abc import Callable

import pytest

from miles.utils.audit_utils.config_snapshot.models import (
    ConfigSnapshotAllocatedEndpoint,
    ConfigSnapshotContext,
    ConfigSnapshotPoint,
    ConfigSnapshotRecord,
)
from miles.utils.audit_utils.process_identity import TrainProcessIdentity


@pytest.fixture
def make_record() -> Callable[..., ConfigSnapshotRecord]:
    def create(
        *, run: int = 0, rank: int = 0, stage: str = "process_config", config: dict | None = None
    ) -> ConfigSnapshotRecord:
        return ConfigSnapshotRecord(
            context=ConfigSnapshotContext(
                name=f"test/run-{run:04d}",
                deploy_component="all",
                deploy_instance_id="default",
                source=TrainProcessIdentity(component="actor", cell_index=0, rank_within_cell=rank),
                run_uuid=f"uuid-{run}",
                capture_id=f"capture-{run}-{rank}",
            ),
            point=ConfigSnapshotPoint(stage=stage, index=0),
            config={"args": config if config is not None else {"rank": rank}},
        )

    return create


@pytest.fixture
def apply_diff() -> Callable[..., str]:
    def apply(*, base: str, delta: str) -> str:
        if not delta:
            return base
        original = base.splitlines(keepends=True)
        result: list[str] = []
        position = 0
        for line in delta.splitlines(keepends=True)[2:]:
            if line.startswith("@@"):
                match = re.match(r"@@ -(\d+)(?:,(\d+))? \+\d+(?:,\d+)? @@", line)
                assert match is not None
                start = int(match[1]) - (match[2] != "0")
                result.extend(original[position:start])
                position = start
            elif line.startswith("-"):
                assert original[position] == line[1:]
                position += 1
            elif line.startswith("+"):
                result.append(line[1:])
            else:
                raise AssertionError(f"Unexpected patch line: {line!r}")
        result.extend(original[position:])
        return "".join(result)

    return apply


@pytest.fixture
def make_endpoint_record(make_record: Callable) -> Callable:
    def create(*, host: str = "10.0.0.1", port: int = 30000, shared: bool = False, dynamic: bool = True):
        ports = [port, port if shared else port + 1]
        endpoints = [
            ConfigSnapshotAllocatedEndpoint(owner=f"session/uuid-0-{i}", host=host, port=value,
                dynamic_host=dynamic, dynamic_port=dynamic, external_host=host)
            for i,value in enumerate(ports)
        ]
        instances = [
            {"instance_id": f"uuid-0-{i}", "addr": f"{host}:{value}", "external_addr": f"{host}:{value}"}
            for i,value in enumerate(ports)
        ]
        return make_record(config={"session_server_instances": instances}).model_copy(update={"allocated_endpoints": endpoints})
    return create
