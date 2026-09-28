from collections.abc import Callable

import pytest

from miles.utils.audit_utils.config_snapshot.converter import ConfigSnapshotConverter
from miles.utils.audit_utils.config_snapshot.endpoints import SnapshotEndpointNormalizer
from miles.utils.audit_utils.config_snapshot.models import ConfigSnapshotAllocatedEndpoint
from miles.utils.audit_utils.config_snapshot.normalizer import normalize_record
from miles.utils.test_utils.snapshot import dump_snapshot


class TestAllocatedEndpointNormalization:
    def test_dynamic_allocations_stabilize_with_their_owners(self, make_endpoint_record: Callable) -> None:
        """Automatically allocated addresses normalize independently of their concrete host and port."""
        first = make_endpoint_record(host="10.0.0.1", port=30000)
        second = make_endpoint_record(host="10.0.0.9", port=31000)
        assert dump_snapshot(ConfigSnapshotConverter.convert([first])) == dump_snapshot(
            ConfigSnapshotConverter.convert([second])
        )

    def test_endpoint_swaps_disagree_with_registered_owners(self, make_endpoint_record: Callable) -> None:
        """An address assigned to the wrong session identity cannot be hidden by normalization."""
        record = make_endpoint_record()
        instances = [dict(instance) for instance in record.config["args"]["session_server_instances"]]
        instances[0]["addr"], instances[1]["addr"] = instances[1]["addr"], instances[0]["addr"]
        record = record.model_copy(update={"config": {"args": {"session_server_instances": instances}}})
        with pytest.raises(ValueError, match="disagrees with its allocated owner"):
            ConfigSnapshotConverter.convert([record])

    @pytest.mark.parametrize("change", ["order", "missing", "identity", "sharing"])
    def test_identity_order_count_and_sharing_remain_observable(
        self, make_endpoint_record: Callable, change: str
    ) -> None:
        """Endpoint compression retains every identity, ordering, count, and sharing relationship."""
        record = make_endpoint_record()
        expected = dump_snapshot(ConfigSnapshotConverter.convert([record]))
        instances = [dict(instance) for instance in record.config["args"]["session_server_instances"]]
        if change == "order":
            instances.reverse()
        elif change == "missing":
            instances.pop()
        elif change == "identity":
            instances[0]["instance_id"] = "another-identity"
        else:
            record = make_endpoint_record(shared=True)
            instances = record.config["args"]["session_server_instances"]
        record = record.model_copy(update={"config": {"args": {"session_server_instances": instances}}})
        assert dump_snapshot(ConfigSnapshotConverter.convert([record])) != expected

    def test_static_allocations_remain_exact(self, make_endpoint_record: Callable) -> None:
        """Explicitly configured host and port changes stay visible even when provenance is present."""
        first = make_endpoint_record(dynamic=False)
        second = make_endpoint_record(host="10.0.0.9", port=31000, dynamic=False)
        assert SnapshotEndpointNormalizer.create([first]).normalize(first) == first.config
        assert dump_snapshot(ConfigSnapshotConverter.convert([first])) != dump_snapshot(
            ConfigSnapshotConverter.convert([second])
        )

    def test_unregistered_addresses_remain_exact(self, make_endpoint_record: Callable) -> None:
        """An address without allocation provenance is not inferred to be dynamic."""
        record = make_endpoint_record().model_copy(update={"allocated_endpoints": []})
        assert SnapshotEndpointNormalizer.create([record]).normalize(record) == record.config

    def test_conflicting_allocations_in_one_generation_fail(self, make_endpoint_record: Callable) -> None:
        """The same owner cannot silently acquire two addresses in one deployment generation."""
        with pytest.raises(ValueError, match="Conflicting allocated"):
            SnapshotEndpointNormalizer.create([make_endpoint_record(), make_endpoint_record(port=31000)])

    @pytest.mark.parametrize("generation_field", ["deploy_instance_id", "name"])
    def test_new_deployment_generations_can_reallocate_ports(
        self, make_endpoint_record: Callable, generation_field: str
    ) -> None:
        """A restarted deployment may legitimately allocate a different address."""
        first = make_endpoint_record()
        second = make_endpoint_record(port=31000)
        second = second.model_copy(
            update={"context": second.context.model_copy(update={generation_field: "next", "capture_id": "next"})}
        )
        assert len(ConfigSnapshotConverter.convert([first, second]).processes) == 2

    def test_static_external_host_is_preserved(self, make_endpoint_record: Callable) -> None:
        """The configured public host is retained while its automatically allocated port stabilizes."""
        record = make_endpoint_record()
        instances = [dict(instance) for instance in record.config["args"]["session_server_instances"]]
        instances[0]["external_addr"] = "public.example:30000"
        record = record.model_copy(
            update={
                "config": {"args": {"session_server_instances": instances}},
                "allocated_endpoints": [
                    endpoint.model_copy(update={"external_host": None}) for endpoint in record.allocated_endpoints
                ],
            }
        )
        actual = normalize_record(record, endpoints=SnapshotEndpointNormalizer.create([record]))
        assert actual["args"]["session_server_instances"][0]["external_addr"].startswith("public.example:$PORT_")

    def test_primary_router_and_per_model_map_keep_the_same_owner(self, make_record: Callable) -> None:
        """The primary router must still point to its declared model's endpoint."""
        endpoints = [
            ConfigSnapshotAllocatedEndpoint(
                owner=f"router/{name}",
                host="10.0.0.1",
                port=30000 + i,
                dynamic_host=True,
                dynamic_port=True,
                primary=i == 0,
            )
            for i, name in enumerate(["actor", "ref"])
        ]
        record = make_record(
            config={
                "sglang_model_routers": {
                    name: {"$tuple": ["10.0.0.1", 30000 + i]} for i, name in enumerate(["actor", "ref"])
                },
                "sglang_router_ip": "10.0.0.1",
                "sglang_router_port": 30000,
            }
        ).model_copy(update={"allocated_endpoints": endpoints})
        normalizer = SnapshotEndpointNormalizer.create([record])
        actual = normalize_record(record, endpoints=normalizer)["args"]
        assert actual["sglang_model_routers"]["actor"]["$tuple"] == [
            actual["sglang_router_ip"],
            actual["sglang_router_port"],
        ]
        wrong = record.model_copy(update={"config": {"args": {**record.config["args"], "sglang_router_port": 30001}}})
        with pytest.raises(ValueError, match="disagrees with its allocated owner"):
            normalize_record(wrong, endpoints=normalizer)
