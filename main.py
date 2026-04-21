"""
main.py
=======
Carbon-Aware Multi-Objective Routing System
===========================================
Entry point - runs the full research pipeline:

  1. Build / load graph (OSMnx, Bengaluru)
  2. Annotate edges with carbon emission model
  3. Run baseline routing (shortest + fastest)
  4. Run carbon-aware multi-objective routing
  5. Run Pareto frontier analysis
  6. Run large-scale simulation (N routes)
  7. Generate visualizations (map + charts)
  8. Print analysis report

Usage
-----
    python main.py

Optional CLI args (edit CONFIG block below):
    PLACE_NAME   : OSM query string for the city / neighbourhood
    N_ROUTES     : number of OD pairs in simulation
    ALPHA/BETA/GAMMA : objective weights
"""

import os
import sys
import time
import random
import warnings
warnings.filterwarnings("ignore")

# -- Project modules ----------------------------------------------------------
from carbon_routing.graph_builder          import build_graph, graph_summary
from carbon_routing.emission_model         import annotate_graph_emissions
from carbon_routing.baseline_routing       import shortest_path, fastest_path
from carbon_routing.multi_objective_routing import carbon_aware_path
from carbon_routing.pareto_frontier        import pareto_routes, print_pareto_summary
from carbon_routing.simulation             import run_simulation, simulation_summary
from carbon_routing.visualization          import (
    plot_routes_on_map,
    plot_comparison_bars,
    plot_emission_distribution,
    plot_pareto_scatter,
)
from carbon_routing.result_analysis        import (
    analyse_results,
    print_analysis_report,
    export_results,
)


# ==============================================================================
# CONFIGURATION - edit these to customise the run
# ==============================================================================

CONFIG = {
    # OSM place name: use a small neighbourhood for fast testing,
    # or "Bengaluru, India" for the full city (takes ~3 min to download)
    "PLACE_NAME":   "Indiranagar, Bengaluru, India",

    # Graph cache path (avoids re-downloading on subsequent runs)
    "CACHE_PATH":   "data/graph.pkl",

    # Force re-download even if cache exists
    "FORCE_DL":     False,

    # Multi-objective weights (must ideally sum to 1.0)
    # alpha = distance,  beta = emission (dominant),  gamma = time
    "ALPHA":        0.20,
    "BETA":         0.60,
    "GAMMA":        0.20,

    # Number of OD pairs in simulation
    "N_ROUTES":     300,

    # Random seed
    "SEED":         42,
}

# ==============================================================================


def banner(text: str) -> None:
    bar = "-" * 60
    print(f"\n{bar}\n  {text}\n{bar}")


def main():
    t_total = time.time()

    # -- STEP 1: Graph Construction ----------------------------------------
    banner("STEP 1 | Graph Construction")
    G = build_graph(
        place_name    = CONFIG["PLACE_NAME"],
        cache_path    = CONFIG["CACHE_PATH"],
        force_download= CONFIG["FORCE_DL"],
    )
    graph_summary(G)

    # -- STEP 2: Emission Annotation ---------------------------------------
    banner("STEP 2 | Carbon Emission Model")
    G = annotate_graph_emissions(G)
    print("[EmissionModel] Edges annotated with CO2 emission weights [DONE]")

    # -- STEP 3: Sample a demonstration OD pair ----------------------------
    banner("STEP 3 | Baseline Routing (Demo Pair)")
    random.seed(CONFIG["SEED"])
    nodes = list(G.nodes())

    # Find a valid pair with decent path length
    demo_sp = demo_fp = demo_ca = None
    for _ in range(200):
        src, dst = random.sample(nodes, 2)
        sp = shortest_path(G, src, dst)
        if sp and sp["n_edges"] >= 8:
            fp = fastest_path(G, src, dst)
            ca = carbon_aware_path(
                G, src, dst,
                alpha=CONFIG["ALPHA"],
                beta =CONFIG["BETA"],
                gamma=CONFIG["GAMMA"],
            )
            if fp and ca:
                demo_sp, demo_fp, demo_ca = sp, fp, ca
                demo_src, demo_dst = src, dst
                break

    if demo_sp is None:
        print("[main] ERROR: Could not find a valid demo pair. "
              "Try a larger PLACE_NAME or reduce min_hops.")
        sys.exit(1)

    # Print per-route comparison table
    print(f"\n  Demo Origin      : {demo_src}")
    print(f"  Demo Destination : {demo_dst}\n")
    header = f"  {'Algorithm':<30} {'Dist (km)':<12} {'Time (min)':<12} {'CO2 (g)':<12}"
    print(header)
    print("  " + "-" * 66)
    for label, result in [
        ("Shortest Path",  demo_sp),
        ("Fastest Path",   demo_fp),
        ("Carbon-Aware",   demo_ca),
    ]:
        d  = result["distance_m"]    / 1000
        t  = result["travel_time_s"] / 60
        e  = result["emission_g"]
        print(f"  {label:<30} {d:<12.3f} {t:<12.2f} {e:<12.1f}")

    # -- STEP 4: Pareto Frontier -------------------------------------------
    banner("STEP 4 | Pareto Frontier Analysis")
    print(f"  Computing Pareto-optimal routes for demo pair ...")
    pareto = pareto_routes(G, demo_src, demo_dst, max_paths=8)
    if pareto:
        print_pareto_summary(pareto)
    else:
        print("  [Pareto] No Pareto solutions found for this pair.")

    # -- STEP 5: Simulation ------------------------------------------------
    banner(f"STEP 5 | Simulation  ({CONFIG['N_ROUTES']} routes)")
    sim_df = run_simulation(
        G,
        n_routes = CONFIG["N_ROUTES"],
        alpha    = CONFIG["ALPHA"],
        beta     = CONFIG["BETA"],
        gamma    = CONFIG["GAMMA"],
        seed     = CONFIG["SEED"],
    )
    summary_df = simulation_summary(sim_df)

    if not summary_df.empty:
        print("\n" + summary_df.to_string(index=False))

    # -- STEP 6: Result Analysis -------------------------------------------
    banner("STEP 6 | Result Analysis")
    analysis = analyse_results(sim_df)
    print_analysis_report(analysis)

    # -- STEP 7: Visualizations --------------------------------------------
    banner("STEP 7 | Generating Visualizations")
    os.makedirs("outputs", exist_ok=True)

    # 7a. Interactive map (demo pair)
    plot_routes_on_map(
        G,
        sp_path    = demo_sp["path"],
        fp_path    = demo_fp["path"],
        ca_path    = demo_ca["path"],
        sp_metrics = demo_sp,
        fp_metrics = demo_fp,
        ca_metrics = demo_ca,
    )

    # 7b. Comparison bar chart
    if not summary_df.empty:
        plot_comparison_bars(summary_df)

    # 7c. Emission distribution
    if not sim_df.empty:
        plot_emission_distribution(sim_df)

    # 7d. Pareto scatter
    plot_pareto_scatter(pareto, demo_sp, demo_fp, demo_ca)

    # -- STEP 8: Export raw results ----------------------------------------
    banner("STEP 8 | Exporting Results")
    if not sim_df.empty:
        export_results(sim_df, summary_df)

    # -- Done --------------------------------------------------------------
    elapsed = time.time() - t_total
    banner(f"PIPELINE COMPLETE  ({elapsed:.1f}s)")
    print("  Outputs generated:")
    for f in ["outputs/route_map.html",
              "outputs/comparison_chart.png",
              "outputs/emission_dist.png",
              "outputs/pareto_scatter.png",
              "outputs/simulation_results.csv",
              "outputs/simulation_results_summary.csv"]:
        mark = "[DONE]" if os.path.exists(f) else "[MISSING]"
        print(f"    {mark} {f}")
    print()


if __name__ == "__main__":
    main()
