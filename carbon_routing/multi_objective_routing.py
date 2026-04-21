"""
multi_objective_routing.py
===========================
Modified Dijkstra that minimises a weighted composite cost:

    cost(edge) = alpha * dist_norm + beta * emis_norm + gamma * time_norm

where _norm values are pre-computed normalised edge weights so that
alpha, beta, gamma in [0, 1] are directly interpretable as importance weights.

Algorithm complexity: O((V + E) log V)  - identical to standard Dijkstra.
The only difference is the composite edge weight, computed once up front.

Mathematical formulation
-------------------------
Let G = (V, E) be a directed weighted graph.
For each edge e in E define:
    w(e) = alpha*d(e) + beta*e(e) + gamma*t(e)

where d, e, t are min-max normalised distance, emission, and time.

Find path P* = argmin_{P in Paths(s,t)} Sum_{e in P} w(e)

This reduces to single-source shortest path on the composite weight graph.
"""

import heapq
import math
import networkx as nx
import numpy as np
from typing import Optional, Tuple

from carbon_routing.baseline_routing import _path_metrics as compute_path_metrics


# -- Step 1: Normalise edge attributes ----------------------------------------

def _normalise_edges(G: nx.MultiDiGraph) -> dict:
    """
    Compute min-max normalised edge attributes across the entire graph.

    Returns
    -------
    norms : dict mapping (u, v, key) -> {'dist_n', 'emis_n', 'time_n'}
    """
    # Collect raw values
    all_dist  = [d.get("length",      0.0) for _, _, d in G.edges(data=True)]
    all_emis  = [d.get("emission",    0.0) for _, _, d in G.edges(data=True)]
    all_time  = [d.get("travel_time", 0.0) for _, _, d in G.edges(data=True)]

    d_min, d_max = min(all_dist),  max(all_dist)
    e_min, e_max = min(all_emis),  max(all_emis)
    t_min, t_max = min(all_time),  max(all_time)

    # Avoid division by zero on degenerate graphs
    d_range = (d_max - d_min) or 1.0
    e_range = (e_max - e_min) or 1.0
    t_range = (t_max - t_min) or 1.0

    norms = {}
    for u, v, key, data in G.edges(keys=True, data=True):
        norms[(u, v, key)] = {
            "dist_n": (data.get("length",      0.0) - d_min) / d_range,
            "emis_n": (data.get("emission",    0.0) - e_min) / e_range,
            "time_n": (data.get("travel_time", 0.0) - t_min) / t_range,
        }
    return norms


# -- Step 2: Multi-Objective Dijkstra -----------------------------------------

def carbon_aware_path(
    G:      nx.MultiDiGraph,
    source: int,
    target: int,
    alpha:  float = 0.2,   # distance weight
    beta:   float = 0.6,   # emission weight  <- dominant
    gamma:  float = 0.2,   # time weight
) -> Optional[dict]:
    """
    Multi-objective Dijkstra minimising:
        composite = alpha*dist_norm + beta*emis_norm + gamma*time_norm

    Parameters
    ----------
    G      : annotated MultiDiGraph (must have 'length', 'emission', 'travel_time')
    source : source node ID
    target : target node ID
    alpha  : weight on distance  (0-1)
    beta   : weight on emission  (0-1)
    gamma  : weight on time      (0-1)

    Note: alpha + beta + gamma need not equal 1.0, but conventionally they do.

    Returns
    -------
    dict with: path, distance_m, travel_time_s, emission_g, composite_cost
    or None if unreachable.
    """
    if source not in G or target not in G:
        return None

    # Pre-compute composite weight for all edges
    norms = _normalise_edges(G)

    def composite_weight(u, v, data):
        """Return the composite cost for the BEST parallel edge between u->v."""
        best = math.inf
        for key in G[u][v]:
            n = norms.get((u, v, key), {"dist_n": 0, "emis_n": 0, "time_n": 0})
            w = alpha * n["dist_n"] + beta * n["emis_n"] + gamma * n["time_n"]
            if w < best:
                best = w
        return best

    try:
        # NetworkX Dijkstra with a callable weight function
        length, path = nx.single_source_dijkstra(
            G, source, target, weight=composite_weight
        )
        metrics = compute_path_metrics(G, path)
        metrics["path"]            = path
        metrics["algorithm"]       = f"Carbon-Aware (alpha={alpha}, beta={beta}, gamma={gamma})"
        metrics["composite_cost"]  = length
        return metrics
    except (nx.NetworkXNoPath, nx.NodeNotFound):
        return None


# -- Step 3: Weight sensitivity helper ---------------------------------------

def sweep_weights(
    G:      nx.MultiDiGraph,
    source: int,
    target: int,
    steps:  int = 5,
) -> list:
    """
    Run carbon_aware_path for a grid of (alpha, beta, gamma) weight combinations.
    Useful for sensitivity analysis and Pareto sweeping.

    Returns
    -------
    results : list of result dicts (one per weight combination)
    """
    results = []
    vals = np.linspace(0, 1, steps)
    for a in vals:
        for b in vals:
            g = max(0.0, 1.0 - a - b)          # enforce alpha+beta+gamma = 1
            if g < 0 or g > 1:
                continue
            r = carbon_aware_path(G, source, target, alpha=a, beta=b, gamma=g)
            if r:
                r["alpha"] = round(a, 2)
                r["beta"]  = round(b, 2)
                r["gamma"] = round(g, 2)
                results.append(r)
    return results
