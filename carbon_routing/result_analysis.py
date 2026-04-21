"""
result_analysis.py
==================
Statistical analysis and interpretation of simulation results.

Computes:
  - % emission reduction of carbon-aware vs. baselines
  - % time overhead vs. baselines
  - Wilcoxon signed-rank test for statistical significance
  - Summary table (print + CSV export)
"""

import os
import pandas as pd
import numpy as np
from scipy import stats

OUTPUT_DIR = "outputs"
os.makedirs(OUTPUT_DIR, exist_ok=True)


def percentage_change(baseline: float, new: float) -> float:
    """Return signed % change: positive = increase, negative = reduction."""
    if baseline == 0:
        return 0.0
    return ((new - baseline) / baseline) * 100


def analyse_results(df: pd.DataFrame) -> dict:
    """
    Compute key performance indicators comparing carbon-aware against
    shortest-path and fastest-path baselines.

    Parameters
    ----------
    df : simulation DataFrame from simulation.run_simulation()

    Returns
    -------
    analysis : dict with KPI values and statistical test results
    """
    if df.empty:
        print("[Analysis] Empty DataFrame - nothing to analyse.")
        return {}

    # Emission KPIs
    ca_vs_sp_emis = percentage_change(
        df["sp_emis_g"].mean(), df["ca_emis_g"].mean()
    )
    ca_vs_fp_emis = percentage_change(
        df["fp_emis_g"].mean(), df["ca_emis_g"].mean()
    )

    # Time KPIs
    ca_vs_sp_time = percentage_change(
        df["sp_time_s"].mean(), df["ca_time_s"].mean()
    )
    ca_vs_fp_time = percentage_change(
        df["fp_time_s"].mean(), df["ca_time_s"].mean()
    )

    # Distance KPIs
    ca_vs_sp_dist = percentage_change(
        df["sp_dist_m"].mean(), df["ca_dist_m"].mean()
    )

    # Statistical Significance (Wilcoxon signed-rank)
    try:
        _, p_sp = stats.wilcoxon(df["sp_emis_g"], df["ca_emis_g"],
                                  alternative="greater")
        _, p_fp = stats.wilcoxon(df["fp_emis_g"], df["ca_emis_g"],
                                  alternative="greater")
    except Exception:
        p_sp = p_fp = float("nan")

    # Per-route savings
    df = df.copy()
    df["emis_saving_vs_sp_g"]  = df["sp_emis_g"] - df["ca_emis_g"]
    df["emis_saving_vs_fp_g"]  = df["fp_emis_g"] - df["ca_emis_g"]
    df["time_overhead_vs_sp_s"] = df["ca_time_s"] - df["sp_time_s"]

    n_better_sp = (df["emis_saving_vs_sp_g"] > 0).sum()
    n_better_fp = (df["emis_saving_vs_fp_g"] > 0).sum()

    analysis = {
        "ca_vs_sp_emis_pct":          ca_vs_sp_emis,
        "ca_vs_fp_emis_pct":          ca_vs_fp_emis,
        "ca_vs_sp_time_pct":          ca_vs_sp_time,
        "ca_vs_fp_time_pct":          ca_vs_fp_time,
        "ca_vs_sp_dist_pct":          ca_vs_sp_dist,
        "win_rate_vs_sp":             n_better_sp / len(df) * 100,
        "win_rate_vs_fp":             n_better_fp / len(df) * 100,
        "wilcoxon_p_vs_sp":           p_sp,
        "wilcoxon_p_vs_fp":           p_fp,
        "mean_emis_saving_vs_sp_g":   df["emis_saving_vs_sp_g"].mean(),
        "mean_time_overhead_vs_sp_s": df["time_overhead_vs_sp_s"].mean(),
        "n_routes":                   len(df),
        "df":                         df,
    }
    return analysis


def print_analysis_report(analysis: dict) -> None:
    """Pretty-print the full analysis report to console (ASCII-safe)."""
    if not analysis:
        return

    SEP  = "=" * 64
    LINE = "-" * 64

    print()
    print(SEP)
    print("    CARBON-AWARE ROUTING  --  ANALYSIS REPORT")
    print(SEP)
    print(f"  Routes evaluated : {analysis['n_routes']}")
    print(LINE)

    # --- Emission vs Shortest Path ---
    pct = analysis["ca_vs_sp_emis_pct"]
    print("  EMISSION vs Shortest Path")
    print(f"    Change   : {abs(pct):.2f}% {'REDUCTION' if pct < 0 else 'INCREASE'}")
    print(f"    Savings  : {analysis['mean_emis_saving_vs_sp_g']:.1f} g CO2/route (mean)")
    print(f"    Win rate : {analysis['win_rate_vs_sp']:.1f}% of routes emitted less")
    p = analysis["wilcoxon_p_vs_sp"]
    sig = "SIGNIFICANT (p<0.05)" if p < 0.05 else "not significant"
    print(f"    Wilcoxon : p = {p:.4f}  -> {sig}")
    print(LINE)

    # --- Emission vs Fastest Path ---
    pct2 = analysis["ca_vs_fp_emis_pct"]
    print("  EMISSION vs Fastest Path")
    print(f"    Change   : {abs(pct2):.2f}% {'REDUCTION' if pct2 < 0 else 'INCREASE'}")
    print(f"    Win rate : {analysis['win_rate_vs_fp']:.1f}% of routes emitted less")
    print(LINE)

    # --- Time overhead ---
    t_pct = analysis["ca_vs_sp_time_pct"]
    t_s   = analysis["mean_time_overhead_vs_sp_s"]
    print("  TIME OVERHEAD (vs Shortest Path)")
    print(f"    Change   : {abs(t_pct):.2f}% {'MORE' if t_pct > 0 else 'LESS'} travel time")
    print(f"    Overhead : {t_s:.1f} seconds per route (mean)")
    print(LINE)

    # --- Interpretation ---
    print("  INTERPRETATION")
    if pct < -5:
        print("    [+] Carbon-aware routing achieves significant emission")
        print("        reductions with acceptable time trade-off.")
        print("        Suitable for green navigation applications.")
    elif pct < 0:
        print("    [~] Moderate emission improvement observed.")
        print("        Try increasing beta (emission weight) in CONFIG.")
    else:
        print("    [-] No emission benefit detected on this graph/config.")
        print("        Consider a larger area or higher beta weight.")
    print(SEP)
    print()


def export_results(df:         pd.DataFrame,
                   summary_df: pd.DataFrame,
                   path:       str = f"{OUTPUT_DIR}/simulation_results.csv") -> None:
    """Export per-route DataFrame and summary to CSV."""
    df.to_csv(path, index=False)
    summary_df.to_csv(path.replace(".csv", "_summary.csv"), index=False)
    print(f"[Analysis] Results exported -> {path}")
