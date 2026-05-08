import time

import numpy as np

from ..data import graph_repo
from ..schemas.graph import SampleNodesRequest, SampleNodesResponse


def sample_feasible_nodes(
    graph_id: str,
    request: SampleNodesRequest,
) -> SampleNodesResponse:
    seed = request.seed if request.seed is not None else int(time.time_ns() % (2**32))

    scc_nodes = graph_repo.scc_for_rain(graph_id, request.rain_level)
    scc_size = len(scc_nodes)

    if scc_size < request.k:
        return SampleNodesResponse(
            feasible=False,
            scc_size=scc_size,
            seed=seed,
            depot=None,
            nodes=[],
        )

    rng = np.random.default_rng(seed)
    chosen_indices = rng.choice(scc_size, size=request.k, replace=False)
    chosen = [scc_nodes[int(i)] for i in chosen_indices]

    depot = graph_repo.get_node(graph_id, chosen[0])
    stops = [graph_repo.get_node(graph_id, n) for n in chosen[1:]]

    return SampleNodesResponse(
        feasible=True,
        scc_size=scc_size,
        seed=seed,
        depot=depot,
        nodes=stops,
    )
