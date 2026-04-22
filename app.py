"""Streamlit dashboard for carbon-aware routing visualization.

Run with:
    streamlit run app.py
"""

from __future__ import annotations

import random

import pandas as pd
import streamlit as st
from streamlit_folium import st_folium

from routing_integration import (
    get_routes,
    list_nodes,
    load_annotated_graph,
    node_label,
    run_route_simulation,
)
from utils import (
    build_folium_route_map,
    build_comparison_charts,
    build_simulation_figures,
    inject_app_styles,
    metrics_cards_html,
    metrics_table_style,
    normalize_weights,
    route_dataframe,
    to_path_rows,
)


st.set_page_config(
    page_title="Carbon-Aware Navigation Dashboard",
    page_icon="C",
    layout="wide",
    initial_sidebar_state="expanded",
)
st.markdown(inject_app_styles(), unsafe_allow_html=True)


@st.cache_resource(show_spinner=False)
def cached_graph(place_name: str, cache_path: str):
    """Load graph once and keep it in memory across reruns."""
    return load_annotated_graph(place_name=place_name, cache_path=cache_path, force_download=False)


@st.cache_data(show_spinner=False)
def cached_node_labels(_graph) -> pd.DataFrame:
    """Prepare selector dataframe from graph nodes."""
    records = []
    for node_id in list_nodes(_graph):
        records.append({"node": int(node_id), "label": node_label(_graph, int(node_id))})
    return pd.DataFrame(records)


st.markdown(
    """
    <div class="dashboard-hero">
      <h2 style="margin:0;">Carbon-Aware Navigation Lab</h2>
      <p>Interactive research prototype for multi-objective routing with route-level and simulation analytics.</p>
    </div>
    """,
    unsafe_allow_html=True,
)

with st.sidebar:
    st.header("Configuration")
    place_name = st.text_input("Place", value="Indiranagar, Bengaluru, India")
    cache_path = st.text_input("Graph cache", value="data/graph.pkl")

    st.markdown("### Objective Weights")
    alpha = st.slider("alpha (distance)", 0.0, 1.0, 0.2, 0.01)
    beta = st.slider("beta (emissions)", 0.0, 1.0, 0.6, 0.01)
    gamma = st.slider("gamma (time)", 0.0, 1.0, 0.2, 0.01)

    auto_normalize = st.toggle("Normalize weights to sum=1", value=True)
    if auto_normalize:
        alpha, beta, gamma = normalize_weights(alpha, beta, gamma)
        st.caption(f"Normalized: alpha={alpha:.2f}, beta={beta:.2f}, gamma={gamma:.2f}")

    st.markdown("### Map Mode")
    map_mode = st.radio("Rendering", ["Static routes", "Animated traversal"], horizontal=False)

    st.markdown("### Route Visibility")
    selected_modes = st.multiselect(
        "Show routes",
        options=["shortest", "fastest", "carbon"],
        default=["shortest", "fastest", "carbon"],
        help="Advanced toggle for route overlays.",
    )

    st.markdown("### Simulation")
    n_routes = st.slider("Number of routes", min_value=30, max_value=1000, value=300, step=10)
    min_hops = st.slider("Minimum hops", min_value=2, max_value=20, value=5, step=1)
    seed = st.number_input("Random seed", min_value=0, value=42, step=1)

try:
    with st.spinner("Loading graph and emissions..."):
        graph = cached_graph(place_name, cache_path)
except Exception as exc:
    st.error(f"Graph loading failed: {exc}")
    st.stop()

nodes_df = cached_node_labels(graph)
if nodes_df.empty:
    st.error("No nodes found in graph.")
    st.stop()

if "src" not in st.session_state or "dst" not in st.session_state:
    sampled = nodes_df["node"].sample(n=2, random_state=11).tolist()
    st.session_state.src = int(sampled[0])
    st.session_state.dst = int(sampled[1])

# If place/cache config changes, previously selected nodes may not exist anymore.
available_nodes = set(nodes_df["node"].tolist())
if st.session_state.src not in available_nodes or st.session_state.dst not in available_nodes:
    sampled = nodes_df["node"].sample(n=2, random_state=11).tolist()
    st.session_state.src = int(sampled[0])
    st.session_state.dst = int(sampled[1])


def _safe_index(df: pd.DataFrame, node_id: int) -> int:
    matches = df.index[df["node"] == node_id].tolist()
    return max(0, matches[0]) if matches else 0

controls = st.columns([2, 2, 1])
with controls[0]:
    src = st.selectbox(
        "Source node",
        options=nodes_df["node"].tolist(),
        format_func=lambda x: node_label(graph, int(x)),
        index=_safe_index(nodes_df, st.session_state.src),
    )
with controls[1]:
    dst = st.selectbox(
        "Destination node",
        options=nodes_df["node"].tolist(),
        format_func=lambda x: node_label(graph, int(x)),
        index=_safe_index(nodes_df, st.session_state.dst),
    )
with controls[2]:
    st.write("")
    st.write("")
    randomize = st.button("Random pair")

if randomize:
    sampled = random.sample(nodes_df["node"].tolist(), 2)
    st.session_state.src = int(sampled[0])
    st.session_state.dst = int(sampled[1])
    st.rerun()

st.session_state.src = int(src)
st.session_state.dst = int(dst)

if st.session_state.src == st.session_state.dst:
    st.warning("Source and destination cannot be the same.")
    st.stop()

with st.spinner("Computing routes..."):
    routes = get_routes(
        graph,
        source=st.session_state.src,
        destination=st.session_state.dst,
        alpha=alpha,
        beta=beta,
        gamma=gamma,
    )

routes = {key: value for key, value in routes.items() if key in selected_modes}
if not routes:
    st.warning("No route available for the selected pair and mode(s).")
    st.stop()

route_df = route_dataframe(routes)
path_df, center = to_path_rows(graph, routes)

st.subheader("Interactive Route Map")
route_map = build_folium_route_map(
    graph,
    routes,
    source_node=st.session_state.src,
    destination_node=st.session_state.dst,
    mode=map_mode,
)
st_folium(route_map, width=None, height=620, returned_objects=[])

st.subheader("Metrics Panel")
card_columns = st.columns(len(route_df))
for idx, card_html in enumerate(metrics_cards_html(route_df)):
    with card_columns[idx]:
        st.markdown(card_html, unsafe_allow_html=True)

st.markdown("<div class='panel-title'>Route Comparison Table</div>", unsafe_allow_html=True)
st.dataframe(
    metrics_table_style(route_df[["algorithm", "distance_km", "time_min", "emission_g"]]),
    use_container_width=True,
    hide_index=True,
)

st.subheader("Analytics Visualization")
emis_fig, time_fig, scatter_fig = build_comparison_charts(route_df)
chart_cols = st.columns(2)
with chart_cols[0]:
    st.plotly_chart(emis_fig, use_container_width=True)
with chart_cols[1]:
    st.plotly_chart(time_fig, use_container_width=True)
st.plotly_chart(scatter_fig, use_container_width=True)

st.subheader("Simulation Visualization")
run_simulation_btn = st.button("Run Simulation", type="primary")

if run_simulation_btn:
    with st.spinner("Running simulations..."):
        sim_df = run_route_simulation(
            graph,
            n_routes=int(n_routes),
            alpha=alpha,
            beta=beta,
            gamma=gamma,
            seed=int(seed),
            min_hops=int(min_hops),
        )

    if sim_df.empty:
        st.warning("Simulation returned no valid route pairs.")
    else:
        avg_reduction = (sim_df["sp_emis_g"] - sim_df["ca_emis_g"]).mean()
        pct_reduction = (
            (sim_df["sp_emis_g"].mean() - sim_df["ca_emis_g"].mean())
            / max(sim_df["sp_emis_g"].mean(), 1e-9)
            * 100
        )

        kpi_cols = st.columns(3)
        with kpi_cols[0]:
            st.metric("Avg emission reduction", f"{avg_reduction:.1f} g")
        with kpi_cols[1]:
            st.metric("Avg reduction vs shortest", f"{pct_reduction:.2f}%")
        with kpi_cols[2]:
            st.metric("Simulation samples", f"{len(sim_df)}")

        hist_fig, box_fig, time_overhead_fig = build_simulation_figures(sim_df)
        sim_cols_top = st.columns(2)
        with sim_cols_top[0]:
            st.plotly_chart(hist_fig, use_container_width=True)
        with sim_cols_top[1]:
            st.plotly_chart(box_fig, use_container_width=True)
        st.plotly_chart(time_overhead_fig, use_container_width=True)

        with st.expander("View simulation data"):
            st.dataframe(sim_df.head(200), use_container_width=True)

st.caption(
    "Research prototype dashboard. Algorithms sourced from existing carbon_routing modules with Streamlit caching and interactive pydeck/plotly visuals."
)
