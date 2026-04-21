"""
pareto_frontier.py
==================
Pareto-optimal route finder for bi-objective (emission vs. time) routing.

Theory
------
A solution P dominates Q  iff:
    P.emission <= Q.emission  AND  P.time <= Q.time
    (with at least one strict inequality)

The Pareto frontier is the set of non-dominated solutions - routes where
you cannot reduce emissions without increasing time, or vice versa.

Algorithm
---------
We use label-setting (multi-label Dijkstra):
  - Each node holds a SET of non-dominated (time, emission) labels
  - A new label is added only if it is not dominated by any existing label
  - Labels propagate via a priority queue (ordered by time)

Complexity: O(k * (V + E) * log V)
  where k = average labels per node (bounded empirically, typically < 20)

This is a lightweight but rigorous implementation suitable for city-scale graphs.
"""

import heapq
import networkx as nx
from typing import Optional


def _dominates(label_a: tuple, label_b: tuple) -> bool:
    """
    Return True if label_a dominates label_b.
    label = (time_s, emission_g)
    Dominates means both objectives are <=, with at least one strictly <.
    """
    ta, ea = label_a
    tb, eb = label_b
    return (ta <= tb and ea <= eb) and (ta < tb or ea < eb)


def _prune_dominated(label_set: list, new_label: tuple) -> tuple[bool, list]:
    """
    Check if new_label should be added to label_set (not dominated).
    Also remove existing labels dominated by new_label.

    Returns
    -------
    (should_add, pruned_label_set)
    """
    # Is new_label dominated by any existing label?
    for existing in label_set:
        if _dominates(existing, new_label):
            return False, label_set   # reject

    # Remove labels that new_label dominates
    pruned = [ex for ex in label_set if not _dominates(new_label, ex)]
    return True, pruned


def pareto_routes(
    G:         nx.MultiDiGraph,
    source:    int,
    target:    int,
    max_paths: int = 10,
) -> list:
    """
    Find Pareto-optimal routes between source and target
    w.r.t. travel time and CO2 emission.

    Parameters
    ----------
    G         : annotated MultiDiGraph
    source    : source node ID
    target    : target node ID
    max_paths : cap on stored Pareto solutions (prevents memory blowup)

    Returns
    -------
    List of dicts, each containing:
        path, travel_time_s, emission_g, distance_m
    Sorted by emission_g ascending.
    """
    if source not in G or target not in G:
        return []

    # labels[node] = list of (time_s, emission_g, path_so_far)
    labels: dict[int, list] = {n: [] for n in G.nodes()}

    # Priority queue: (time_so_far, emission_so_far, current_node, path)
    pq = [(0.0, 0.0, source, [source])]

    pareto_solutions = []

    while pq:
        time_cur, emis_cur, node, path = heapq.heappop(pq)
        cur_label = (time_cur, emis_cur)

        # Check if this state is still non-dominated at this node
        dominated = any(_dominates(ex_label[:2], cur_label)
                        for ex_label in labels[node])
        if dominated:
            continue

        # Reached target - record as a Pareto candidate
        if node == target:
            pareto_solutions.append({
                "path":          path,
                "travel_time_s": time_cur,
                "emission_g":    emis_cur,
                "distance_m":    sum(
                    min(G[u][v][k].get("length", 0) for k in G[u][v])
                    for u, v in zip(path[:-1], path[1:])
                ),
                "algorithm":     "Pareto-Optimal",
            })
            if len(pareto_solutions) >= max_paths:
                break
            continue

        # Add label to node
        labels[node].append((time_cur, emis_cur))

        # Expand neighbours
        for neighbor in G.successors(node):
            # Use best (lowest time) parallel edge
            best_edge = min(G[node][neighbor].values(),
                            key=lambda d: d.get("travel_time", 9999))
            next_time = time_cur + best_edge.get("travel_time", 0.0)
            next_emis = emis_cur + best_edge.get("emission",    0.0)
            next_label = (next_time, next_emis)

            # Only expand if not dominated at neighbour
            add, _ = _prune_dominated(labels[neighbor], next_label)
            if add:
                heapq.heappush(pq, (next_time, next_emis, neighbor, path + [neighbor]))

    # Sort Pareto solutions by emission (ascending)
    pareto_solutions.sort(key=lambda r: r["emission_g"])
    return pareto_solutions


def print_pareto_summary(solutions: list) -> None:
    """Pretty-print the Pareto frontier solutions."""
    print("=" * 65)
    print(f"{'#':<4} {'Time (min)':<12} {'Emission (g)':<15} {'Distance (km)':<14}")
    print("-" * 65)
    for i, sol in enumerate(solutions, 1):
        t_min = sol["travel_time_s"] / 60
        d_km  = sol["distance_m"] / 1000
        print(f"{i:<4} {t_min:<12.2f} {sol['emission_g']:<15.1f} {d_km:<14.2f}")
    print("=" * 65)
