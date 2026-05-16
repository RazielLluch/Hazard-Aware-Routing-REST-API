"""B5: /api/v2/inference parity against the precomputed evaluation CSV.

This is the single most important green light for the vendored bridge: replay a
known scenario through ``POST /api/v2/inference`` and assert the live objective
reproduces the runner's precomputed ``evaluation_raw.csv`` value within 1e-3.
If this passes, the bridge replays the runner faithfully.
"""

from __future__ import annotations

import csv
import json
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from src.core.config import settings
from src.main import app

pytestmark = pytest.mark.v2

_ARTEFACT = (
    Path(settings.macro_vendor_root)
    / "macro_DDQN"
    / "artifacts"
    / settings.macro_default_artefact
)


@pytest.fixture(scope="module")
def client():
    """A TestClient whose ``with`` block triggers the real lifespan (bridge.load())."""
    with TestClient(app) as test_client:
        yield test_client


def _csv_row(algorithm: str) -> dict[str, str]:
    """First ``evaluation_raw.csv`` row for the given runner-level algorithm label."""
    csv_path = _ARTEFACT / "results" / "evaluation_raw.csv"
    with csv_path.open(newline="", encoding="utf-8") as handle:
        for row in csv.DictReader(handle):
            if row["algorithm"] == algorithm:
                return row
    raise AssertionError(f"no {algorithm!r} row in {csv_path}")


def _scenario(scenario_id: str) -> dict:
    scn_path = _ARTEFACT / "evaluation_scenarios.jsonl"
    with scn_path.open(encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                data = json.loads(line)
                if data["scenario_id"] == scenario_id:
                    return data
    raise AssertionError(f"scenario {scenario_id!r} not found in {scn_path}")


class TestInferenceParity:
    def test_learned_objective_matches_precomputed(self, client) -> None:
        # The default artefact is offline-no-anchor, so its CSV `Macro-DQN` rows
        # ARE the Macro-DDQN-Offline-NoAnchor results.
        row = _csv_row("Macro-DQN")
        scn = _scenario(row["scenario_id"])
        resp = client.post(
            "/api/v2/inference",
            json={
                "algorithm": "Macro-DDQN-Offline-NoAnchor",
                "profile": row["profile"],
                "ri_key": row["ri_key"],
                "start": scn["start"],
                "deliveries": scn["deliveries"],
            },
        )
        assert resp.status_code == 200
        body = resp.json()
        assert abs(body["metrics"]["objective"] - float(row["objective"])) < 1e-3
        assert (
            abs(body["metrics"]["common_risk_exposure"] - float(row["common_risk_exposure"]))
            < 1e-3
        )

    def test_greedy_objective_matches_precomputed(self, client) -> None:
        row = _csv_row("Greedy-HazardAware")
        scn = _scenario(row["scenario_id"])
        resp = client.post(
            "/api/v2/inference",
            json={
                "algorithm": "Greedy-HazardAware",
                "profile": row["profile"],
                "ri_key": row["ri_key"],
                "start": scn["start"],
                "deliveries": scn["deliveries"],
            },
        )
        assert resp.status_code == 200
        assert abs(resp.json()["metrics"]["objective"] - float(row["objective"])) < 1e-3


class TestInferenceShape:
    def test_first_leg_starts_at_start(self, client) -> None:
        row = _csv_row("Greedy-HazardAware")
        scn = _scenario(row["scenario_id"])
        body = client.post(
            "/api/v2/inference",
            json={
                "algorithm": "Macro-DDQN-Offline-NoAnchor",
                "profile": "balanced",
                "ri_key": scn["ri_key"],
                "start": scn["start"],
                "deliveries": scn["deliveries"],
            },
        ).json()
        assert body["legs"], "expected a non-empty legs[]"
        assert body["legs"][0]["source"] == scn["start"]
        assert body["delivered"] == len(scn["deliveries"])

    def test_response_has_no_success_field(self, client) -> None:
        row = _csv_row("Greedy-HazardAware")
        scn = _scenario(row["scenario_id"])
        resp = client.post(
            "/api/v2/inference",
            json={
                "algorithm": "Greedy-HazardAware",
                "profile": "balanced",
                "ri_key": scn["ri_key"],
                "start": scn["start"],
                "deliveries": scn["deliveries"],
            },
        )
        assert '"success"' not in resp.text

    def test_unknown_algorithm_is_422(self, client) -> None:
        row = _csv_row("Greedy-HazardAware")
        scn = _scenario(row["scenario_id"])
        resp = client.post(
            "/api/v2/inference",
            json={
                "algorithm": "Not-A-Real-Algorithm",
                "profile": "balanced",
                "ri_key": scn["ri_key"],
                "start": scn["start"],
                "deliveries": scn["deliveries"],
            },
        )
        assert resp.status_code == 422
