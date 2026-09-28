import asyncio
import logging
from collections.abc import Sequence
from dataclasses import dataclass, field

logger = logging.getLogger(__name__)

_DYNAMIC_PORT_START = 20000
_MAX_PORT = 65535
_MAX_PEER_ROUNDS = 64


@dataclass
class PortAllocator:
    _next_port_of_ip: dict[str, int] = field(default_factory=dict)
    _lock: asyncio.Lock = field(default_factory=asyncio.Lock)

    async def alloc(
        self, actor, *, node_ip: str, consecutive: int = 1, peers: Sequence[tuple[str, object]] = ()
    ) -> int:
        """Reserve ``consecutive`` free ports on ``node_ip``, probed through ``actor``.

        ``peers`` are ``(node_ip, actor)`` pairs of other nodes the block must also be free on: every
        rank of a multi-node SGLang engine checks the master ports on its own host, so a block that is
        free only on the head can already belong to another worker there.
        """
        async with self._lock:
            node_ips = [node_ip, *(ip for ip, _ in peers)]
            # use small ports to prevent ephemeral port between 32768 and 65536.
            # also, ray uses port 10002-19999, thus we avoid near-10002 to avoid racing condition
            start_port = max(self._next_port_of_ip.get(ip, _DYNAMIC_PORT_START) for ip in node_ips)
            for _ in range(_MAX_PEER_ROUNDS):
                if start_port + consecutive - 1 > _MAX_PORT:
                    start_port = _DYNAMIC_PORT_START
                port: int = await actor._get_free_port_block.remote(start_port=start_port, count=consecutive)
                peer_ports = await asyncio.gather(
                    *[peer._get_free_port_block.remote(start_port=port, count=consecutive) for _, peer in peers]
                )
                if all(peer_port == port for peer_port in peer_ports):
                    break
                start_port = max([port + 1, *(p for p in peer_ports if p > port)])
            else:
                raise RuntimeError(f"No block of {consecutive} ports is free on all of {node_ips}")
            for ip in node_ips:
                self._next_port_of_ip[ip] = port + consecutive
            return port
