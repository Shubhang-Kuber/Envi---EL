# Carbon-Aware Multi-Objective Routing System

> **Research Project** | Graph-Based Multi-Objective Routing Algorithm for Carbon-Aware Navigation Systems  
> **Author:** Shubhang Kuber | 2nd Year Engineering, 4th Semester

---

## Overview

This project implements a **research-grade carbon-aware navigation system** that minimises CO2 emissions alongside distance and travel time using a modified multi-objective Dijkstra algorithm on real city road graphs (OpenStreetMap / Bengaluru).

### Key Contributions
- **VT-Micro emission model** calibrated to realistic vehicle behaviour (U-shaped EF curve)
- **Composite-weight Dijkstra** with normalised multi-objective cost: `alpha*dist + beta*emission + gamma*time`
- **Multi-label Pareto frontier** exploration (bi-objective: time vs. emission)
- **Large-scale simulation** (3001000 OD pairs) with statistical significance testing (Wilcoxon)
- **Interactive Folium map** + publication-quality charts

---

## Project Structure

```
Envi---EL/
+-- main.py                          #  Run this
+-- requirements.txt
+-- data/                            # Graph cache (auto-created)
+-- outputs/                         # All results (auto-created)
|   +-- route_map.html               # Interactive Folium map
|   +-- comparison_chart.png         # Bar chart comparison
|   +-- emission_dist.png            # Emission distribution
|   +-- pareto_scatter.png           # Pareto frontier scatter
|   +-- simulation_results.csv       # Full per-route data
|   +-- simulation_results_summary.csv
+-- carbon_routing/
    +-- __init__.py
    +-- graph_builder.py             # OSMnx graph + edge annotation
    +-- emission_model.py            # VT-Micro CO2 model
    +-- baseline_routing.py          # Shortest + Fastest path Dijkstra
    +-- multi_objective_routing.py   # Carbon-aware composite Dijkstra
    +-- pareto_frontier.py           # Multi-label Pareto Dijkstra
    +-- simulation.py                # N-route simulation engine
    +-- visualization.py             # All charts and maps
    +-- result_analysis.py           # KPIs + statistical testing
```

---

## Setup & Installation

### 1. Install dependencies

```bash
pip install -r requirements.txt
```

> Requires Python 3.10+. On Windows, if `osmnx` fails, run:  
> `pip install osmnx --pre`

### 2. Run the Dashboard / Pipeline

To run the **Streamlit Interactive UI Dashboard** (recommended):
```bash
streamlit run app.py
```
> The dashboard provides a dynamic map with a dark/light mode toggle, 10 predefined bad-traffic junctions in Bengaluru, sliders to adjust weights in real-time, and live comparisons.

To run the **Headless Analytical Pipeline** (for generating CSVs and static PNG charts):
```bash
python main.py
```

First run downloads the Bengaluru road graph (~30s for Indiranagar).  
Subsequent runs use the cached graph (`data/graph.pkl`) - starts in seconds.

---

## Configuration

Edit the `CONFIG` block in `main.py`:

| Parameter     | Default                          | Description                                |
|---------------|----------------------------------|--------------------------------------------|
| `PLACE_NAME`  | `"Indiranagar, Bengaluru, India"`| OSM place query (use larger area for more OD pairs) |
| `N_ROUTES`    | `300`                            | Simulation size (3001000 recommended)     |
| `ALPHA`       | `0.20`                           | Weight on distance objective               |
| `BETA`        | `0.60`                           | Weight on emission objective (dominant)    |
| `GAMMA`       | `0.20`                           | Weight on travel time objective            |

For full Bengaluru city: change `PLACE_NAME` to `"Bengaluru, India"` (graph download ~3 min).

---

## Emission Model

Based on the **VT-Micro** (Virginia Tech Microscopic) model:

```
EF(v) = 154 + 1200/v + 0.0025*v   [gCO2/km]
```

| Speed (km/h) | EF (gCO2/km) |
|-------------|---------------|
| 10          | ~274          |
| 30          | ~198          |
| 60          | ~183  optimum|
| 100         | ~229          |

---

## Algorithm: Multi-Objective Dijkstra

**Cost function per edge:**
```
w(e) = alpha * dist_norm(e) + beta * emis_norm(e) + gamma * time_norm(e)
```

Where `*_norm` is min-max normalised across all graph edges.  
Standard Dijkstra is then run on this composite weight. Complexity: **O((V+E) log V)**.

---

## Outputs

| File | Description |
|------|-------------|
| `route_map.html` | Open in browser - interactive map with all 3 routes |
| `comparison_chart.png` | Grouped bar chart: distance / time / emission |
| `emission_dist.png` | Emission histogram across 300 routes |
| `pareto_scatter.png` | Pareto frontier: time vs. emission trade-off |
| `simulation_results.csv` | Raw per-route data for further analysis |

---

## References

1. Rakha, H. et al. (2003). *VT-Micro: A microscopic model for predicting vehicle fuel consumption and emissions.* Transportation Research Part D.
2. Boeing, G. (2017). *OSMnx: New methods for acquiring, constructing, analyzing, and visualizing complex street networks.* Computers, Environment and Urban Systems.
3. Ehrgott, M. (2005). *Multicriteria Optimization.* Springer.
4. EMEP/EEA Air Pollutant Emission Inventory Guidebook (2023).