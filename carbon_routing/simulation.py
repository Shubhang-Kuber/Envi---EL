"""
simulation.py
=============
Simulation engine that:
  1. Randomly samples N src-destination pairs from the graph
  2. Runs all three routing algorithms on each pair
  3. Collects metrics into a Pandas DataFrame

Designed to be fast: skips pairs where any algorithm returns None.
Progress is shown via tqdm.
"""

import random
import warnings
import pandas as pd
import numpy as np
from tqdm import tqdm
import networkx as nx

from carbon_routing.baseline_routing       import shortest_path, fastest_path
from carbon_routing.multi_objective_routing import carbon_aware_path

warnings.filterwarnings("ignore")   # suppress minor OSMnx / NetworkX warnings


def run_simulation(
    G:          nx.MultiDiGraph,
    n_routes:   int   = 500,
    alpha:      float = 0.2,
    beta:       float = 0.6,
    gamma:      float = 0.2,
    seed:       int   = 42,
    min_hops:   int   = 5,
) -> pd.DataFrame:
    """
    Run the three routing algorithms over N randomly sampled OD pairs.

    Parameters
    ----------
    G          : annotated MultiDiGraph
    n_routes   : number of origin-destination pairs to test
    alpha/beta/gamma : weights for carbon-aware router
    seed       : random seed for reproducibility
    min_hops   : minimum path length in edges (filters trivially short routes)

    Returns
    -------
    DataFrame with columns:
        route_id, src, dst,
        sp_dist, sp_time, sp_emis,   (shortest path)
        fp_dist, fp_time, fp_emis,   (fastest path)
        ca_dist, ca_time, ca_emis    (carbon-aware path)
    """
    random.seed(seed)
    nodes = list(G.nodes())

    records = []
    attempts = 0
    max_attempts = n_routes * 10    # give up after too many failed pairs

    pbar = tqdm(total=n_routes, desc="Simulating routes", unit="route")

    while len(records) < n_routes and attempts < max_attempts:
        attempts += 1
        src, dst = random.sample(nodes, 2)

        # -- Run all three algorithms --------------------------------------
        sp = shortest_path(G, src, dst)
        if sp is None or sp["n_edges"] < min_hops:
            continue

        fp = fastest_path(G, src, dst)
        if fp is None:
            continue

        ca = carbon_aware_path(G, src, dst, alpha=alpha, beta=beta, gamma=gamma)
        if ca is None:
            continue

        records.append({
            "route_id": len(records) + 1,
            "src":      src,
            "dst":      dst,

            # Shortest path metrics
            "sp_dist_m":  sp["distance_m"],
            "sp_time_s":  sp["travel_time_s"],
            "sp_emis_g":  sp["emission_g"],

            # Fastest path metrics
            "fp_dist_m":  fp["distance_m"],
            "fp_time_s":  fp["travel_time_s"],
            "fp_emis_g":  fp["emission_g"],

            # Carbon-aware path metrics
            "ca_dist_m":  ca["distance_m"],
            "ca_time_s":  ca["travel_time_s"],
            "ca_emis_g":  ca["emission_g"],
        })
        pbar.update(1)

    pbar.close()

    df = pd.DataFrame(records)

    if df.empty:
        print("[Simulation] WARNING: No valid routes found. Check graph connectivity.")
        return df

    print(f"\n[Simulation] Completed {len(df)}/{n_routes} routes "
          f"({attempts} attempts, {attempts - len(df)} skipped).")
    return df


def simulation_summary(df: pd.DataFrame) -> pd.DataFrame:
    """
    Compute aggregate statistics from the simulation DataFrame.

    Returns
    -------
    summary_df : mean values for each algorithm across all routes
    """
    if df.empty:
        return pd.DataFrame()

    summary = pd.DataFrame({
        "Algorithm":     ["Shortest Path", "Fastest Path", "Carbon-Aware"],
        "Avg Dist (km)": [
            df["sp_dist_m"].mean() / 1000,
            df["fp_dist_m"].mean() / 1000,
            df["ca_dist_m"].mean() / 1000,
        ],
        "Avg Time (min)": [
            df["sp_time_s"].mean() / 60,
            df["fp_time_s"].mean() / 60,
            df["ca_time_s"].mean() / 60,
        ],
        "Avg Emission (g)": [
            df["sp_emis_g"].mean(),
            df["fp_emis_g"].mean(),
            df["ca_emis_g"].mean(),
        ],
        "Total Emission (kg)": [
            df["sp_emis_g"].sum() / 1000,
            df["fp_emis_g"].sum() / 1000,
            df["ca_emis_g"].sum() / 1000,
        ],
    })
    return summary
