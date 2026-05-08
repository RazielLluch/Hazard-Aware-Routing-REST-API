from typing import Annotated

from fastapi import APIRouter, Query

from ....data import benchmark_repo
from ....schemas.common import Page
from ....schemas.scenario import Scenario, ScenarioListItem

router = APIRouter(
    prefix="/benchmarks/{benchmark_id}/scenarios",
    tags=["scenarios"],
)


@router.get("", response_model=Page[ScenarioListItem])
def list_scenarios(
    benchmark_id: str,
    ri: Annotated[int | None, Query(ge=1, le=5)] = None,
    page: Annotated[int, Query(ge=1)] = 1,
    page_size: Annotated[int, Query(ge=1, le=200, alias="pageSize")] = 50,
) -> Page[ScenarioListItem]:
    items, total = benchmark_repo.list_scenario_summaries(
        benchmark_id, ri=ri, page=page, page_size=page_size
    )
    return Page(items=items, page=page, page_size=page_size, total=total)


@router.get("/{scenario_id}", response_model=Scenario)
def get_scenario(benchmark_id: str, scenario_id: str) -> Scenario:
    return benchmark_repo.get_scenario(benchmark_id, scenario_id)
