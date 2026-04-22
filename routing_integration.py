"""Routing integration layer for the Streamlit dashboard.

This module wraps existing project routing functions and provides a stable
interface for the UI layer.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List

import networkx as nx
import pandas as pd

from carbon_routing.baseline_routing import fastest_path, shortest_path
from carbon_routing.emission_model import annotate_graph_emissions
from carbon_routing.graph_builder import build_graph
from carbon_routing.multi_objective_routing import carbon_aware_path
from carbon_routing.simulation import run_simulation


@dataclass
class RouteResult:
    """Container for route metrics exposed to the dashboard."""

    key: str
    label: str
    color: List[int]
    path: List[int]
    distance_m: float
    travel_time_s: float
    emission_g: float


def load_annotated_graph(
    place_name: str,
    cache_path: str,
    force_download: bool,
) -> nx.MultiDiGraph:
    """Build/load graph and ensure edge emissions are available."""
    graph = build_graph(
        place_name=place_name,
        cache_path=cache_path,
        force_download=force_download,
    )
    return annotate_graph_emissions(graph)


def get_routes(
    graph: nx.MultiDiGraph,
    source: int,
    destination: int,
    alpha: float,
    beta: float,
    gamma: float,
) -> Dict[str, RouteResult]:
    """Run all route algorithms and return successful results."""
    raw = {
        "shortest": shortest_path(graph, source, destination),
        "fastest": fastest_path(graph, source, destination),
        "carbon": carbon_aware_path(
            graph,
            source,
            destination,
            alpha=alpha,
            beta=beta,
            gamma=gamma,
        ),
    }

    labels = {
        "shortest": "Shortest path",
        "fastest": "Fastest path",
        "carbon": "Carbon-aware path",
    }
    colors = {
        "shortest": [225, 59, 48],
        "fastest": [48, 120, 225],
        "carbon": [34, 168, 84],
    }

    results: Dict[str, RouteResult] = {}
    for key, value in raw.items():
        if not value:
            continue
        results[key] = RouteResult(
            key=key,
            label=labels[key],
            color=colors[key],
            path=value["path"],
            distance_m=float(value["distance_m"]),
            travel_time_s=float(value["travel_time_s"]),
            emission_g=float(value["emission_g"]),
        )
    return results


def run_route_simulation(
    graph: nx.MultiDiGraph,
    n_routes: int,
    alpha: float,
    beta: float,
    gamma: float,
    seed: int,
    min_hops: int,
) -> pd.DataFrame:
    """Execute simulation and return records as a DataFrame."""
    return run_simulation(
        graph,
        n_routes=n_routes,
        alpha=alpha,
        beta=beta,
        gamma=gamma,
        seed=seed,
        min_hops=min_hops,
    )


def list_nodes(graph: nx.MultiDiGraph) -> List[int]:
    """Return graph node ids as a list for selectors."""
    return list(graph.nodes())


def graph_bounds(graph: nx.MultiDiGraph) -> Dict[str, float]:
    """Get basic bounds for camera defaults."""
    lats = [data.get("y", 0.0) for _, data in graph.nodes(data=True)]
    lons = [data.get("x", 0.0) for _, data in graph.nodes(data=True)]
    return {
        "lat_min": min(lats),
        "lat_max": max(lats),
        "lon_min": min(lons),
        "lon_max": max(lons),
    }


def node_label(graph: nx.MultiDiGraph, node_id: int) -> str:
    """Human-readable node label for dropdown controls."""
    data = graph.nodes[node_id]
    lat = data.get("y", 0.0)
    lon = data.get("x", 0.0)
    return f"{node_id}  |  ({lat:.5f}, {lon:.5f})"
