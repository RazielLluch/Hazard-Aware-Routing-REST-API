"""B5: /api/v2/metrics schema-hygiene + aggregation-shape tests.

The headline assertion: ``success_rate`` (and the bare ``success`` key) must be
absent from every v2 metrics response — under soft-blocking it is always 1.0 and
the manuscript reports the percent-of-means reduction instead.
"""

from __future__ import annotations

import pytest
from fastapi.testclient import TestClient

from src.main import app

pytestmark = pytest.mark.v2


@pytest.fixture(scope="module")
def client():
    with TestClient(app) as test_client:
        yield test_client


class TestMetricsSchemaHygiene:
    def test_no_success_rate_in_response_body(self, client) -> None:
        resp = client.get("/api/v2/metrics", params={"cohort": "random"})
        assert resp.status_code == 200
        assert "success_rate" not in resp.text
        assert '"success"' not in resp.text

    def test_facets_and_reductions_present(self, client) -> None:
        body = client.get("/api/v2/metrics", params={"cohort": "random"}).json()
        assert body["facets"]
        assert body["baseline_reductions"]
        # 7 algorithms x 3 profiles x 5 RI = 105 facets for the random cohort.
        assert len(body["facets"]) == 105

    def test_facet_carries_manuscript_metrics(self, client) -> None:
        body = client.get("/api/v2/metrics", params={"cohort": "random"}).json()
        facet = body["facets"][0]
        for field in ("common_risk_exposure", "travel_time_min", "blockage_exposure", "objective"):
            assert field in facet
        assert "success_rate" not in facet

    def test_reduction_formula(self, client) -> None:
        body = client.get("/api/v2/metrics", params={"cohort": "random"}).json()
        # Pick a reduction with a non-zero baseline (RI1 has zero hazard exposure).
        sample = next(
            r for r in body["baseline_reductions"] if r["mean_baseline"] != 0.0
        )
        expected = (
            100.0 * (sample["mean_baseline"] - sample["mean_model"]) / sample["mean_baseline"]
        )
        assert abs(sample["reduction_pct"] - round(expected, 4)) < 1e-6

    def test_zero_baseline_reduction_is_zero(self, client) -> None:
        # When the baseline mean is 0 (e.g. blockage exposure at RI1), reduction_pct
        # must be a safe 0.0, not a division error.
        body = client.get("/api/v2/metrics", params={"cohort": "random"}).json()
        for reduction in body["baseline_reductions"]:
            if reduction["mean_baseline"] == 0.0:
                assert reduction["reduction_pct"] == 0.0


class TestMetricsCohorts:
    def test_cohort_only_subsets_ri2_to_ri5(self, client) -> None:
        # hazard_opportunity / risk_time_tradeoff are sampled RI2-RI5 only.
        body = client.get("/api/v2/metrics", params={"cohort": "hazard_opportunity"}).json()
        assert len(body["facets"]) == 84  # 7 x 3 x 4

    def test_unknown_cohort_is_422(self, client) -> None:
        resp = client.get("/api/v2/metrics", params={"cohort": "nonexistent"})
        assert resp.status_code == 422
