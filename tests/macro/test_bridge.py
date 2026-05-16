"""B5: vendored-runner bridge contract tests.

The bridge is the single seam between FastAPI and the vendored Macro-DDQN
runner; these tests pin its public surface so a silent regression breaks CI.
"""

from __future__ import annotations

import pytest

from src.macro.bridge import bridge, resolve_algorithm

pytestmark = [pytest.mark.v2, pytest.mark.macro]


@pytest.fixture(scope="module")
def loaded_bridge():
    """A loaded Bridge singleton (idempotent — load() is safe to call again)."""
    if not bridge.is_loaded:
        bridge.load()
    return bridge


class TestFeatureNames:
    def test_first_three_feature_names(self, loaded_bridge) -> None:
        assert loaded_bridge.feature_names()[:3] == ["current_x", "current_y", "target_x"]

    def test_feature_count_is_32(self, loaded_bridge) -> None:
        # 32, not 33 — the implementation reference is authoritative; an
        # exploration agent miscounted during planning.
        assert len(loaded_bridge.feature_names()) == 32


class TestAlgorithmCatalog:
    def test_seven_entries(self, loaded_bridge) -> None:
        assert len(loaded_bridge.algorithm_catalog()) == 7

    def test_categories(self, loaded_bridge) -> None:
        cats = {e["id"]: e["category"] for e in loaded_bridge.algorithm_catalog()}
        assert cats["Macro-DQN"] == "learned"
        assert cats["Macro-DDQN-Online-Scratch"] == "learned"
        assert cats["Macro-DDQN-Offline-NoAnchor"] == "learned"
        assert cats["Macro-DP-Oracle"] == "oracle"
        assert cats["Greedy-HazardAware"] == "baseline"

    def test_entry_fields(self, loaded_bridge) -> None:
        for entry in loaded_bridge.algorithm_catalog():
            assert set(entry) == {"id", "label", "category", "requires_model", "aliases"}


class TestResolveAlgorithm:
    def test_id_resolves(self) -> None:
        assert resolve_algorithm("Macro-DDQN-Offline-NoAnchor").id == "Macro-DDQN-Offline-NoAnchor"

    def test_kebab_alias_resolves(self) -> None:
        assert resolve_algorithm("offline-no-anchor").id == "Macro-DDQN-Offline-NoAnchor"
        assert resolve_algorithm("oracle").id == "Macro-DP-Oracle"

    def test_unknown_raises(self) -> None:
        with pytest.raises(ValueError, match="Unknown algorithm"):
            resolve_algorithm("not-a-real-algorithm")


class TestReloadCycle:
    def test_survives_reload(self, loaded_bridge) -> None:
        before = loaded_bridge.feature_names()
        loaded_bridge.reload()
        assert loaded_bridge.is_loaded
        assert loaded_bridge.feature_names() == before
