"""
baseline_routing.py
====================
Baseline Dijkstra-based routing algorithms:
  1. Shortest path  -> minimise total distance  (metres)
  2. Fastest path   -> minimise total travel time (seconds)

Both use NetworkX's built-in Dijkstra for reliability.
After finding a path, we compute ALL three metrics (distance, time, emission)
so results are directly comparable with the carbon-aware router.
"""

import networkx as nx
import numpy as np
from typing import Optional


# -- Internal helper ----------------------------------------------------------

def _path_metrics(G: nx.MultiDiGraph, path: list) -> dict:
    """
    Given an ordered list of node IDs, compute the aggregate metrics
    for the full route by summing over each consecutive edge pair.

    For MultiDiGraph we always take the minimum-cost parallel edge.

    Returns
    -------
    dict with keys: distance_m, travel_time_s, emission_g, n_edges
    """
    total_dist  = 0.0
    total_time  = 0.0
    total_emis  = 0.0

    for u, v in zip(path[:-1], path[1:]):
        # Pick the best (lowest length) parallel edge
        edge_data = min(
            G[u][v].values(),
            key=lambda d: d.get("length", 9999)
        )
        total_dist += edge_data.get("length",      0.0)
        total_time += edge_data.get("travel_time", 0.0)
        total_emis += edge_data.get("emission",    0.0)

    return {
        "distance_m":    total_dist,
        "travel_time_s": total_time,
        "emission_g":    total_emis,
        "n_edges":       len(path) - 1,
    }


# -- Baseline 1: Shortest Path ------------------------------------------------

def shortest_path(
    G: nx.MultiDiGraph,
    source: int,
    target: int,
) -> Optional[dict]:
    """
    Find the distance-minimising path using Dijkstra.

    Parameters
    ----------
    G      : annotated MultiDiGraph
    source : source node ID
    target : target node ID

    Returns
    -------
    dict with keys: path (list), distance_m, travel_time_s, emission_g
    or None if no path exists.
    """
    try:
        path = nx.dijkstra_path(G, source, target, weight="length")
        metrics = _path_metrics(G, path)
        metrics["path"]      = path
        metrics["algorithm"] = "Shortest Path (Dijkstra)"
        return metrics
    except (nx.NetworkXNoPath, nx.NodeNotFound):
        return None


# -- Baseline 2: Fastest Path -------------------------------------------------

def fastest_path(
    G: nx.MultiDiGraph,
    source: int,
    target: int,
) -> Optional[dict]:
    """
    Find the time-minimising path using Dijkstra.

    Parameters
    ----------
    G      : annotated MultiDiGraph
    source : source node ID
    target : target node ID

    Returns
    -------
    dict with keys: path (list), distance_m, travel_time_s, emission_g
    or None if no path exists.
    """
    try:
        path = nx.dijkstra_path(G, source, target, weight="travel_time")
        metrics = _path_metrics(G, path)
        metrics["path"]      = path
        metrics["algorithm"] = "Fastest Path (Dijkstra)"
        return metrics
    except (nx.NetworkXNoPath, nx.NodeNotFound):
        return None


# -- Public metric extractor (re-exported for other modules) ------------------

def compute_path_metrics(G: nx.MultiDiGraph, path: list) -> dict:
    """Publicly accessible wrapper around _path_metrics."""
    return _path_metrics(G, path)
