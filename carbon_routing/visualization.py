"""
visualization.py
================
All map and chart visualizations for the carbon routing project.

Outputs
-------
1. Folium interactive map  -> outputs/route_map.html
   - Shortest path  in RED
   - Fastest path   in BLUE
   - Carbon-aware   in GREEN
2. Bar chart comparison   -> outputs/comparison_chart.png
3. Emission distribution  -> outputs/emission_dist.png
4. Pareto scatter plot    -> outputs/pareto_scatter.png  (if Pareto data available)
"""

import os
import warnings
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")           # non-interactive backend for saving
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import folium
import networkx as nx

warnings.filterwarnings("ignore")

OUTPUT_DIR = "outputs"
os.makedirs(OUTPUT_DIR, exist_ok=True)


# -- Helper: extract lat/lon coordinates for a path ---------------------------

def _path_coords(G: nx.MultiDiGraph, path: list) -> list[tuple]:
    """Return list of (lat, lon) tuples for nodes in path."""
    coords = []
    for node in path:
        data = G.nodes[node]
        lat  = data.get("y", 0.0)   # OSMnx stores lat as 'y'
        lon  = data.get("x", 0.0)   # OSMnx stores lon as 'x'
        coords.append((lat, lon))
    return coords


# -- 1. Interactive Folium Map -------------------------------------------------

def plot_routes_on_map(
    G:       nx.MultiDiGraph,
    sp_path: list,
    fp_path: list,
    ca_path: list,
    sp_metrics: dict,
    fp_metrics: dict,
    ca_metrics: dict,
    output_file: str = f"{OUTPUT_DIR}/route_map.html",
) -> None:
    """
    Render three routes on an interactive Folium map.

    Parameters
    ----------
    G           : graph (for node coordinates)
    sp_path     : list of node IDs - shortest path
    fp_path     : list of node IDs - fastest path
    ca_path     : list of node IDs - carbon-aware path
    *_metrics   : dict from routing algorithms (for popup info)
    output_file : HTML output path
    """
    # Centre map on the median node location
    all_nodes = sp_path + fp_path + ca_path
    all_lats  = [G.nodes[n]["y"] for n in all_nodes]
    all_lons  = [G.nodes[n]["x"] for n in all_nodes]
    centre    = (np.median(all_lats), np.median(all_lons))

    m = folium.Map(location=centre, zoom_start=14, tiles="CartoDB dark_matter")

    # -- Draw each path ----------------------------------------------------
    route_configs = [
        (sp_path, "red",   "Shortest Path",  sp_metrics),
        (fp_path, "blue",  "Fastest Path",   fp_metrics),
        (ca_path, "green", "Carbon-Aware",   ca_metrics),
    ]

    for path, color, label, metrics in route_configs:
        coords = _path_coords(G, path)
        if len(coords) < 2:
            continue

        popup_text = (
            f"<b>{label}</b><br>"
            f"Distance: {metrics.get('distance_m', 0)/1000:.2f} km<br>"
            f"Time: {metrics.get('travel_time_s', 0)/60:.1f} min<br>"
            f"CO2: {metrics.get('emission_g', 0):.1f} g"
        )

        folium.PolyLine(
            locations=coords,
            color=color,
            weight=5,
            opacity=0.85,
            tooltip=label,
            popup=folium.Popup(popup_text, max_width=200),
        ).add_to(m)

    # -- Start / End markers -----------------------------------------------
    src_coords = _path_coords(G, [sp_path[0]])[0]
    dst_coords = _path_coords(G, [sp_path[-1]])[0]

    folium.Marker(
        location=src_coords,
        popup="<b>Origin</b>",
        icon=folium.Icon(color="white", icon="circle", prefix="fa"),
    ).add_to(m)

    folium.Marker(
        location=dst_coords,
        popup="<b>Destination</b>",
        icon=folium.Icon(color="red", icon="flag", prefix="fa"),
    ).add_to(m)

    # -- Legend ------------------------------------------------------------
    legend_html = """
    <div style="position: fixed; bottom: 30px; left: 30px; z-index: 1000;
                background-color: rgba(0,0,0,0.7); padding: 12px 18px;
                border-radius: 8px; font-family: Arial; color: white; font-size: 13px;">
      <b>Route Legend</b><br>
      <span style="color:#ff4444;">-</span> Shortest Path<br>
      <span style="color:#4488ff;">-</span> Fastest Path<br>
      <span style="color:#44ff88;">-</span> Carbon-Aware Path
    </div>
    """
    m.get_root().html.add_child(folium.Element(legend_html))

    m.save(output_file)
    print(f"[Visualization] Map saved -> {output_file}")


# -- 2. Bar Chart Comparison ---------------------------------------------------

def plot_comparison_bars(summary_df: pd.DataFrame,
                         output_file: str = f"{OUTPUT_DIR}/comparison_chart.png") -> None:
    """
    Grouped bar chart comparing avg distance, time, and emission
    across the three routing algorithms.
    """
    if summary_df.empty:
        return

    algorithms = summary_df["Algorithm"].tolist()
    metrics    = ["Avg Dist (km)", "Avg Time (min)", "Avg Emission (g)"]
    colors     = ["#E74C3C", "#3498DB", "#2ECC71"]

    fig, axes = plt.subplots(1, 3, figsize=(14, 5))
    fig.patch.set_facecolor("#0f0f1a")

    for ax, metric, color in zip(axes, metrics, colors):
        vals = summary_df[metric].values
        bars = ax.bar(algorithms, vals, color=color, alpha=0.85, width=0.5,
                      edgecolor="white", linewidth=0.5)
        ax.set_title(metric, color="white", fontsize=12, pad=10)
        ax.set_facecolor("#1a1a2e")
        ax.tick_params(colors="white")
        ax.spines["bottom"].set_color("#444")
        ax.spines["left"].set_color("#444")
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        ax.yaxis.label.set_color("white")
        for label in ax.get_xticklabels():
            label.set_color("white")
            label.set_fontsize(8)

        # Value labels on bars
        for bar, val in zip(bars, vals):
            ax.text(bar.get_x() + bar.get_width() / 2,
                    bar.get_height() + max(vals) * 0.01,
                    f"{val:.2f}", ha="center", va="bottom",
                    color="white", fontsize=8)

    plt.suptitle("Routing Algorithm Comparison (Simulation Average)",
                 color="white", fontsize=14, y=1.02)
    plt.tight_layout()
    plt.savefig(output_file, dpi=150, bbox_inches="tight",
                facecolor=fig.get_facecolor())
    plt.close()
    print(f"[Visualization] Bar chart saved -> {output_file}")


# -- 3. Emission Distribution Plot --------------------------------------------

def plot_emission_distribution(df: pd.DataFrame,
                                output_file: str = f"{OUTPUT_DIR}/emission_dist.png") -> None:
    """
    Overlapping histogram comparing emission distributions of the three algorithms.
    """
    if df.empty:
        return

    fig, ax = plt.subplots(figsize=(10, 5))
    fig.patch.set_facecolor("#0f0f1a")
    ax.set_facecolor("#1a1a2e")

    bins = 40
    ax.hist(df["sp_emis_g"], bins=bins, alpha=0.5, color="#E74C3C",
            label="Shortest Path",  edgecolor="none")
    ax.hist(df["fp_emis_g"], bins=bins, alpha=0.5, color="#3498DB",
            label="Fastest Path",   edgecolor="none")
    ax.hist(df["ca_emis_g"], bins=bins, alpha=0.7, color="#2ECC71",
            label="Carbon-Aware",   edgecolor="none")

    ax.set_xlabel("CO2 Emission per Route (g)", color="white", fontsize=11)
    ax.set_ylabel("Frequency",                  color="white", fontsize=11)
    ax.set_title("Emission Distribution Across Routing Algorithms",
                 color="white", fontsize=13)
    ax.tick_params(colors="white")
    ax.spines["bottom"].set_color("#444")
    ax.spines["left"].set_color("#444")
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.legend(facecolor="#2a2a3e", labelcolor="white", fontsize=10)

    plt.tight_layout()
    plt.savefig(output_file, dpi=150, bbox_inches="tight",
                facecolor=fig.get_facecolor())
    plt.close()
    print(f"[Visualization] Emission distribution saved -> {output_file}")


# -- 4. Pareto Scatter Plot ----------------------------------------------------

def plot_pareto_scatter(pareto_solutions: list,
                         sp_result: dict, fp_result: dict, ca_result: dict,
                         output_file: str = f"{OUTPUT_DIR}/pareto_scatter.png") -> None:
    """
    Scatter plot of Pareto-optimal solutions with baseline annotations.
    X-axis: travel time (min),  Y-axis: emissions (g CO2).
    """
    if not pareto_solutions:
        return

    fig, ax = plt.subplots(figsize=(9, 6))
    fig.patch.set_facecolor("#0f0f1a")
    ax.set_facecolor("#1a1a2e")

    # Pareto front
    p_times = [s["travel_time_s"] / 60 for s in pareto_solutions]
    p_emis  = [s["emission_g"]         for s in pareto_solutions]
    ax.plot(p_times, p_emis, "o--", color="#F39C12", markersize=7,
            linewidth=1.5, label="Pareto Frontier", zorder=5)

    # Baseline points
    def _plot_point(result, color, label, marker):
        if result:
            ax.scatter(result["travel_time_s"] / 60, result["emission_g"],
                       color=color, s=120, zorder=10, marker=marker, label=label,
                       edgecolors="white", linewidths=0.8)

    _plot_point(sp_result, "#E74C3C", "Shortest Path",  "s")
    _plot_point(fp_result, "#3498DB", "Fastest Path",   "^")
    _plot_point(ca_result, "#2ECC71", "Carbon-Aware",   "D")

    ax.set_xlabel("Travel Time (min)",    color="white", fontsize=11)
    ax.set_ylabel("CO2 Emission (g)",     color="white", fontsize=11)
    ax.set_title("Pareto Frontier: Time vs. Emission Trade-off",
                 color="white", fontsize=13)
    ax.tick_params(colors="white")
    ax.spines["bottom"].set_color("#444")
    ax.spines["left"].set_color("#444")
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.legend(facecolor="#2a2a3e", labelcolor="white", fontsize=10)

    plt.tight_layout()
    plt.savefig(output_file, dpi=150, bbox_inches="tight",
                facecolor=fig.get_facecolor())
    plt.close()
    print(f"[Visualization] Pareto scatter saved -> {output_file}")
