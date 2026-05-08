from typing import Annotated

from fastapi import Query
from pydantic import BaseModel


class PageParams(BaseModel):
    page: int
    page_size: int


def page_params(
    page: Annotated[int, Query(ge=1)] = 1,
    page_size: Annotated[int, Query(ge=1, le=200, alias="pageSize")] = 50,
) -> PageParams:
    return PageParams(page=page, page_size=page_size)
