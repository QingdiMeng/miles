import pytest

from miles.ray.placement_group import _sort_bundle_infos


def test_default_node_order_preserves_ip_then_gpu() -> None:
    bundles = [(0, "10.0.0.2", 1), (1, "10.0.0.1", 3), (2, "10.0.0.2", 0)]
    assert _sort_bundle_infos(bundles, "") == [bundles[1], bundles[2], bundles[0]]


def test_explicit_node_order_assigns_trainer_first_and_keeps_gpu_order() -> None:
    bundles = [(0, "10.0.0.2", 1), (1, "10.0.0.1", 3), (2, "10.0.0.2", 0)]
    assert _sort_bundle_infos(bundles, "10.0.0.2,10.0.0.1") == [bundles[2], bundles[0], bundles[1]]


@pytest.mark.parametrize("order", ["10.0.0.3", "10.0.0.1,10.0.0.1"])
def test_node_order_rejects_unknown_or_duplicate_nodes(order: str) -> None:
    with pytest.raises(ValueError, match="distinct nodes"):
        _sort_bundle_infos([(0, "10.0.0.1", 0)], order)
