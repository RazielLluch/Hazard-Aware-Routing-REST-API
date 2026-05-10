"""Pydantic schemas for saved node bundles.

A bundle is a named ``(depot, stops)`` tuple scoped to a specific graph.
Reusable across benchmarks: pick once, run many.

Persisted at ``data/node_bundles/<graph_id>/<name>.json``. Bundle names
must match a conservative slug pattern so they're safe as filename
components on Windows + POSIX.
"""

from datetime import datetime

from pydantic import Field

from .common import CamelModel


class Bundle(CamelModel):
    name: str = Field(min_length=1, max_length=64, pattern=r"^[a-zA-Z0-9_-]+$")
    graph_id: str
    depot: str
    stops: list[str] = Field(default_factory=list, max_length=20)
    created_at: datetime
    updated_at: datetime
    description: str | None = Field(default=None, max_length=240)


class BundleSummary(CamelModel):
    name: str
    graph_id: str
    num_stops: int
    created_at: datetime
    updated_at: datetime


class BundleCreateRequest(CamelModel):
    name: str = Field(min_length=1, max_length=64, pattern=r"^[a-zA-Z0-9_-]+$")
    depot: str
    stops: list[str] = Field(default_factory=list, max_length=20)
    description: str | None = Field(default=None, max_length=240)


class BundleUpdateRequest(CamelModel):
    depot: str | None = None
    stops: list[str] | None = Field(default=None, max_length=20)
    description: str | None = Field(default=None, max_length=240)
