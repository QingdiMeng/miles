from dataclasses import dataclass

from pydantic import JsonValue

from miles.utils.audit_utils.config_snapshot.models import ConfigSnapshotAllocatedEndpoint, ConfigSnapshotRecord


@dataclass(frozen=True)
class SnapshotEndpointNormalizer:
    by_generation: dict[tuple[str, str], dict[str, ConfigSnapshotAllocatedEndpoint]]

    @classmethod
    def create(cls, records: list[ConfigSnapshotRecord]) -> "SnapshotEndpointNormalizer":
        by_generation: dict[tuple[str, str], dict[str, ConfigSnapshotAllocatedEndpoint]] = {}
        for record in records:
            scope = (record.context.run_uuid, record.context.deploy_instance_id)
            endpoints = by_generation.setdefault(scope, {})
            for endpoint in record.allocated_endpoints:
                if endpoint.owner in endpoints and endpoints[endpoint.owner] != endpoint:
                    raise ValueError(f"Conflicting allocated snapshot endpoint: {scope}/{endpoint.owner}")
                endpoints[endpoint.owner] = endpoint
        return cls(by_generation=by_generation)

    def normalize(self, record: ConfigSnapshotRecord) -> JsonValue:
        config = record.config
        endpoints = self.by_generation.get((record.context.run_uuid, record.context.deploy_instance_id), {})
        if not endpoints or not isinstance(config, dict) or not isinstance(args := config.get("args"), dict):
            return config
        args = dict(args)
        if isinstance(routers := args.get("sglang_model_routers"), dict):
            routers = dict(routers)
            for model, value in routers.items():
                if (endpoint := endpoints.get(f"router/{model}")) is not None:
                    if (
                        not isinstance(value, dict)
                        or not isinstance(parts := value.get("$tuple"), list)
                        or len(parts) != 2
                    ):
                        raise ValueError(f"Invalid router endpoint snapshot: {model}")
                    host, port = self._normalize_address(
                        endpoint=endpoint, host=parts[0], port=parts[1], endpoints=endpoints
                    )
                    routers[model] = {**value, "$tuple": [host, port]}
            args["sglang_model_routers"] = routers
            primary = [endpoint for endpoint in endpoints.values() if endpoint.primary]
            if len(primary) > 1:
                raise ValueError("Multiple primary router snapshot endpoints")
            if primary and args.get("sglang_router_ip") is not None:
                args["sglang_router_ip"], args["sglang_router_port"] = self._normalize_address(
                    endpoint=primary[0],
                    host=args["sglang_router_ip"],
                    port=args["sglang_router_port"],
                    endpoints=endpoints,
                )
        if isinstance(instances := args.get("session_server_instances"), list):
            args["session_server_instances"] = [
                self._normalize_instance(instance=instance, endpoints=endpoints) for instance in instances
            ]
        return {**config, "args": args}

    def _normalize_instance(
        self, *, instance: JsonValue, endpoints: dict[str, ConfigSnapshotAllocatedEndpoint]
    ) -> JsonValue:
        if (
            not isinstance(instance, dict)
            or (endpoint := endpoints.get(f"session/{instance.get('instance_id')}")) is None
        ):
            return instance
        result = dict(instance)
        for field in ("addr", "external_addr"):
            if not isinstance(value := result.get(field), str):
                raise ValueError(f"Invalid session endpoint snapshot: {instance}")
            host, separator, raw_port = value.rpartition(":")
            if not separator or not raw_port.isdecimal():
                raise ValueError(f"Invalid session endpoint address: {value}")
            external = field == "external_addr"
            normalized_host, port = self._normalize_address(
                endpoint=endpoint, host=host, port=int(raw_port), endpoints=endpoints, external=external
            )
            result[field] = f"{normalized_host}:{port}"
        return result

    def _normalize_address(
        self,
        *,
        endpoint: ConfigSnapshotAllocatedEndpoint,
        host: JsonValue,
        port: JsonValue,
        endpoints: dict[str, ConfigSnapshotAllocatedEndpoint],
        external: bool = False,
    ) -> tuple[JsonValue, JsonValue]:
        expected_host = endpoint.external_host if external else endpoint.host
        if (expected_host is not None and host != expected_host) or port != endpoint.port:
            raise ValueError(f"Snapshot endpoint disagrees with its allocated owner: {endpoint.owner}")

        host_peers = [item for item in endpoints.values() if item.host == endpoint.host]
        port_peers = [item for item in host_peers if item.port == endpoint.port]
        if endpoint.dynamic_host and (not external or endpoint.external_host is not None):
            if all(item.dynamic_host for item in host_peers):
                host = f"$HOST_{min(item.owner for item in host_peers)}"
        if endpoint.dynamic_port and all(item.dynamic_port for item in port_peers):
            port = f"$PORT_{min(item.owner for item in port_peers)}"
        return host, port
