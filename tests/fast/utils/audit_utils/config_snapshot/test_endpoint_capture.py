from argparse import Namespace
from pathlib import Path
from typing import Any

import pytest

from miles.ray.rollout.router_manager import resolve_router_addrs, wait_session_server_ready
from miles.utils.audit_utils.config_snapshot.storage import ConfigSnapshotStorage


class TestEndpointCapture:
    @pytest.mark.parametrize("static", [False, True])
    async def test_router_capture_distinguishes_requested_ports(
        self, capture_args: Namespace, endpoint_provider: Any, ready_endpoint: None, tmp_path: Path, static: bool
    ) -> None:
        """Allocation provenance retains whether the router port came from explicit configuration."""
        if static:
            capture_args.sglang_router_port = 30000
        await resolve_router_addrs(capture_args, router_providers=[endpoint_provider])
        records = ConfigSnapshotStorage(directory=tmp_path).read()
        record = next(record for record in records if record.point.stage == "router_endpoints")
        assert len(record.allocated_endpoints) == 1
        endpoint = record.allocated_endpoints[0]
        assert (endpoint.owner, endpoint.host, endpoint.port) == ("router/actor", "10.0.0.1", 30000)
        assert endpoint.dynamic_port is not static
        assert endpoint.primary
        assert record.context == next(record for record in records if record.point.stage == "process_config").context

    @pytest.mark.parametrize("external_host", [None, "public.example"])
    async def test_session_capture_preserves_explicit_external_hosts(
        self, capture_args: Namespace, endpoint_provider: Any, ready_endpoint: None, tmp_path: Path, external_host: str | None
    ) -> None:
        """Session allocation metadata never treats a configured public hostname as generated."""
        capture_args.session_server_external_host = external_host
        await wait_session_server_ready(capture_args, provider=endpoint_provider)
        records = ConfigSnapshotStorage(directory=tmp_path).read()
        record = next(record for record in records if record.point.stage == "session_endpoints")
        assert [endpoint.owner for endpoint in record.allocated_endpoints] == ["session/uuid-0-0", "session/uuid-0-1"]
        assert all(endpoint.dynamic_port for endpoint in record.allocated_endpoints)
        assert [endpoint.external_host for endpoint in record.allocated_endpoints] == (["10.0.0.1"] * 2 if external_host is None else [None] * 2)
