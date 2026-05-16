"""GET /api/v2/algorithms — the 7-entry comparison-surface catalog.

Variant identity (the DP-pretrained ``Macro-DQN`` reference vs the two DDQN
variants) is resolved here by artefact directory — the runner's CSV collapses
every learned variant to the single label ``Macro-DQN``, so this catalog is the
only place they become distinct.
"""

from __future__ import annotations

from fastapi import APIRouter

from ...macro.bridge import bridge
from .schemas import AlgorithmEntry

router = APIRouter(prefix="/algorithms", tags=["v2:algorithms"])


@router.get("", response_model=list[AlgorithmEntry])
def list_algorithms() -> list[dict]:
    """The 7 thesis algorithms: 3 learned, the exact oracle, and 3 greedy baselines."""
    return bridge.algorithm_catalog()
