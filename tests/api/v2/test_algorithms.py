"""B5: /api/v2/algorithms catalog contract — exactly the 7 thesis algorithms."""

from __future__ import annotations

import pytest
from fastapi.testclient import TestClient

from src.main import app

pytestmark = pytest.mark.v2

_EXPECTED_IDS = {
    "Macro-DQN",
    "Macro-DDQN-Online-Scratch",
    "Macro-DDQN-Offline-NoAnchor",
    "Macro-DP-Oracle",
    "Greedy-Time-Base",
    "Greedy-Time-SoftPenalty",
    "Greedy-HazardAware",
}


@pytest.fixture(scope="module")
def client():
    with TestClient(app) as test_client:
        yield test_client


class TestAlgorithmCatalog:
    def test_exactly_seven_entries(self, client) -> None:
        body = client.get("/api/v2/algorithms").json()
        assert len(body) == 7

    def test_expected_ids(self, client) -> None:
        body = client.get("/api/v2/algorithms").json()
        assert {entry["id"] for entry in body} == _EXPECTED_IDS

    def test_required_fields_present(self, client) -> None:
        body = client.get("/api/v2/algorithms").json()
        for entry in body:
            assert {"id", "label", "category", "requires_model", "aliases"} <= set(entry)
            assert entry["category"] in {"learned", "baseline", "oracle"}
            assert isinstance(entry["requires_model"], bool)
            assert isinstance(entry["aliases"], list)

    def test_three_learned_one_oracle_three_baseline(self, client) -> None:
        categories = [entry["category"] for entry in client.get("/api/v2/algorithms").json()]
        assert categories.count("learned") == 3
        assert categories.count("oracle") == 1
        assert categories.count("baseline") == 3

    def test_learned_variants_require_a_model(self, client) -> None:
        by_id = {e["id"]: e for e in client.get("/api/v2/algorithms").json()}
        assert by_id["Macro-DDQN-Offline-NoAnchor"]["requires_model"] is True
        assert by_id["Macro-DP-Oracle"]["requires_model"] is False
        assert by_id["Greedy-HazardAware"]["requires_model"] is False
