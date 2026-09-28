import re
from argparse import Namespace
from collections.abc import Callable
from pathlib import Path
from typing import Any

import pytest

from miles.utils.audit_utils.config_snapshot.models import (
    ConfigSnapshotAllocatedEndpoint,
    ConfigSnapshotContext,
    ConfigSnapshotPoint,
    ConfigSnapshotRecord,
)
from miles.utils.audit_utils.process_identity import TrainProcessIdentity
from miles.utils.audit_utils.config_snapshot.dumper import ConfigSnapshotDumper
from miles.utils.audit_utils.process_identity import SimpleProcessIdentity
from miles.utils.test_utils.snapshot import SNAPSHOT_RECORD_DIR_ENV_VAR


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
def capture_args(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> Namespace:
    monkeypatch.setenv(SNAPSHOT_RECORD_DIR_ENV_VAR, str(tmp_path))
    monkeypatch.setattr(ConfigSnapshotDumper, "_state", None)
    args = Namespace(
        ci_disable_config_snapshot=False, ci_test=True, config_snapshot_name="test/run-0000",
        deploy_component="all", deploy_instance_id=None, run_uuid="uuid-0",
        sglang=Namespace(models=[Namespace(name="actor")]),
        sglang_router_ip=None, sglang_router_port=None, sglang_model_routers=None,
        use_session_server=True, hf_checkpoint="/model", session_server_workers=2,
        session_server_ip=None, session_server_port=None, session_server_external_host=None,
        session_server_instances=None,
    )
    ConfigSnapshotDumper.configure(args=args, source=SimpleProcessIdentity(component="main"))
    ConfigSnapshotDumper.dump(stage="process_config", config={"args": args})
    return args


@pytest.fixture
def endpoint_provider() -> Any:
    from miles.utils.workers.worker_spec import HostAndPort

    class Provider:
        async def get_addrs(self, *, worker_name: str) -> dict[str, HostAndPort]:
            index = int(worker_name.rsplit("-", maxsplit=2)[-2])
            return {"primary": HostAndPort(host="10.0.0.1", port=30000 + index)}

    return Provider()


@pytest.fixture
def ready_endpoint(monkeypatch: pytest.MonkeyPatch) -> None:
    async def ready(host: str, port: int, *, timeout: float) -> None:
        return None

    monkeypatch.setattr("miles.ray.rollout.router_manager.wait_tcp_ready_async", ready)


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
            ConfigSnapshotAllocatedEndpoint(
                owner=f"session/uuid-0-{i}",
                host=host,
                port=value,
                dynamic_host=dynamic,
                dynamic_port=dynamic,
                external_host=host,
            )
            for i, value in enumerate(ports)
        ]
        instances = [
            {"instance_id": f"uuid-0-{i}", "addr": f"{host}:{value}", "external_addr": f"{host}:{value}"}
            for i, value in enumerate(ports)
        ]
        return make_record(config={"session_server_instances": instances}).model_copy(
            update={"allocated_endpoints": endpoints}
        )

    return create
