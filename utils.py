"""UI and visualization helpers for the Streamlit dashboard."""

from __future__ import annotations

from typing import Dict, List, Tuple

import folium
import networkx as nx
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import pydeck as pdk
from branca.element import MacroElement, Template
from folium.plugins import AntPath

from routing_integration import RouteResult


ROUTE_ORDER = ["shortest", "fastest", "carbon"]
ROUTE_COLOR_HEX = {
    "shortest": "#e13b30",
    "fastest": "#3078e1",
    "carbon": "#22a854",
}


def inject_app_styles() -> str:
    """Return custom CSS for a polished startup-style dashboard."""
    return """
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Space+Grotesk:wght@500;700&family=IBM+Plex+Sans:wght@400;500;600&display=swap');

    :root {
        --bg: #f4f6f2;
        --panel: #ffffff;
        --ink: #16211d;
        --muted: #5f6d67;
        --accent: #1f7a4c;
        --accent-soft: #e9f4ed;
        --danger: #e13b30;
        --primary: #3078e1;
    }

    .stApp {
        background:
            radial-gradient(circle at 20% 0%, #dceedd 0%, transparent 38%),
            radial-gradient(circle at 82% 12%, #d8e8f8 0%, transparent 34%),
            var(--bg);
        color: var(--ink);
        font-family: 'IBM Plex Sans', sans-serif;
    }

    h1, h2, h3 {
        font-family: 'Space Grotesk', sans-serif;
        letter-spacing: -0.01em;
    }

    .dashboard-hero {
        background: linear-gradient(120deg, #173b2f, #1f7a4c);
        border-radius: 18px;
        padding: 1.1rem 1.25rem;
        color: #f8fffa;
        border: 1px solid rgba(255, 255, 255, 0.1);
        box-shadow: 0 12px 28px rgba(20, 47, 36, 0.22);
        margin-bottom: 0.75rem;
        animation: rise 550ms ease-out;
    }

    .dashboard-hero p {
        color: rgba(245, 255, 250, 0.9);
        margin: 0.2rem 0 0;
    }

    .metric-card {
        background: var(--panel);
        border: 1px solid rgba(22, 33, 29, 0.08);
        border-left: 5px solid var(--accent);
        border-radius: 14px;
        padding: 0.85rem 0.95rem;
        box-shadow: 0 7px 20px rgba(0, 0, 0, 0.06);
        animation: rise 500ms ease-out;
    }

    .metric-card h4 {
        margin: 0;
        font-size: 0.85rem;
        color: var(--muted);
    }

    .metric-card .value {
        margin-top: 0.3rem;
        font-family: 'Space Grotesk', sans-serif;
        font-size: 1.35rem;
        line-height: 1.1;
    }

    .metric-card .sub {
        color: var(--muted);
        font-size: 0.78rem;
        margin-top: 0.2rem;
    }

    .panel-title {
        margin: 0.2rem 0 0.4rem;
        font-size: 1.02rem;
        color: #173b2f;
    }

    @keyframes rise {
        from { transform: translateY(8px); opacity: 0; }
        to { transform: translateY(0px); opacity: 1; }
    }
    </style>
    """


def route_dataframe(routes: Dict[str, RouteResult]) -> pd.DataFrame:
    """Convert route results to a tidy dataframe for tables/charts."""
    rows = []
    for key in ROUTE_ORDER:
        route = routes.get(key)
        if not route:
            continue
        rows.append(
            {
                "route_key": key,
                "algorithm": route.label,
                "distance_km": route.distance_m / 1000.0,
                "time_min": route.travel_time_s / 60.0,
                "emission_g": route.emission_g,
            }
        )
    return pd.DataFrame(rows)


def to_path_rows(
    graph: nx.MultiDiGraph,
    routes: Dict[str, RouteResult],
) -> Tuple[pd.DataFrame, List[float]]:
    """Build pydeck path layer records and map center."""
    path_rows = []
    all_lats: List[float] = []
    all_lons: List[float] = []

    for key in ROUTE_ORDER:
        route = routes.get(key)
        if not route:
            continue
        coords = []
        for node in route.path:
            data = graph.nodes[node]
            lat = float(data.get("y", 0.0))
            lon = float(data.get("x", 0.0))
            all_lats.append(lat)
            all_lons.append(lon)
            coords.append([lon, lat])

        path_rows.append(
            {
                "route_key": key,
                "algorithm": route.label,
                "path": coords,
                "distance_km": route.distance_m / 1000.0,
                "time_min": route.travel_time_s / 60.0,
                "emission_g": route.emission_g,
                "color": route.color,
            }
        )

    if not all_lats or not all_lons:
        return pd.DataFrame(), [0.0, 0.0]

    center = [sum(all_lats) / len(all_lats), sum(all_lons) / len(all_lons)]
    return pd.DataFrame(path_rows), center


def build_route_map(path_df: pd.DataFrame, center: List[float], mode: str) -> pdk.Deck:
    """Create either static or animated pydeck map."""
    if mode == "Animated traversal" and not path_df.empty:
        trip_df = build_trip_dataframe(path_df)
        layer = pdk.Layer(
            "TripsLayer",
            trip_df,
            get_path="path",
            get_timestamps="timestamps",
            get_color="color",
            opacity=0.9,
            width_min_pixels=5,
            rounded=True,
            trail_length=180,
            current_time=180,
        )
    else:
        layer = pdk.Layer(
            "PathLayer",
            path_df,
            get_path="path",
            get_color="color",
            width_min_pixels=5,
            pickable=True,
            auto_highlight=True,
        )

    return pdk.Deck(
        map_style="light",
        initial_view_state=pdk.ViewState(
            latitude=center[0],
            longitude=center[1],
            zoom=13,
            pitch=45 if mode == "Animated traversal" else 25,
        ),
        layers=[layer],
        tooltip={
            "html": "<b>{algorithm}</b><br/>Distance: {distance_km} km<br/>Time: {time_min} min<br/>Emissions: {emission_g} g",
            "style": {"backgroundColor": "#1f2a24", "color": "#f4fbf6"},
        },
    )


def build_trip_dataframe(path_df: pd.DataFrame) -> pd.DataFrame:
    """Add timestamps for animated traversal via TripsLayer."""
    rows = []
    for _, row in path_df.iterrows():
        path = row["path"]
        if len(path) < 2:
            continue

        n_steps = len(path)
        step = max(10, int(180 / max(n_steps - 1, 1)))
        timestamps = [idx * step for idx in range(n_steps)]

        rows.append(
            {
                "algorithm": row["algorithm"],
                "path": path,
                "timestamps": timestamps,
                "distance_km": round(float(row["distance_km"]), 2),
                "time_min": round(float(row["time_min"]), 2),
                "emission_g": round(float(row["emission_g"]), 2),
                "color": row["color"],
            }
        )
    return pd.DataFrame(rows)


def metrics_table_style(df: pd.DataFrame) -> pd.io.formats.style.Styler:
    """Apply lightweight color formatting to metric table."""

    min_emission = df["emission_g"].min()
    min_time = df["time_min"].min()
    min_distance = df["distance_km"].min()

    def highlight_row(row: pd.Series) -> List[str]:
        styles = [""] * len(row)
        if row["emission_g"] == min_emission:
            styles[df.columns.get_loc("emission_g")] = "background-color: #e9f4ed; color: #184a2f;"
        if row["time_min"] == min_time:
            styles[df.columns.get_loc("time_min")] = "background-color: #e8f1ff; color: #1f4f96;"
        if row["distance_km"] == min_distance:
            styles[df.columns.get_loc("distance_km")] = "background-color: #ffeceb; color: #8e2a24;"
        return styles

    return (
        df.style.format(
            {
                "distance_km": "{:.2f}",
                "time_min": "{:.2f}",
                "emission_g": "{:.1f}",
            }
        )
        .apply(highlight_row, axis=1)
        .hide(axis="index")
    )


def metrics_cards_html(route_df: pd.DataFrame) -> List[str]:
    """Build HTML snippets for route cards."""
    cards = []
    for _, row in route_df.iterrows():
        key = row["route_key"]
        accent = ROUTE_COLOR_HEX.get(key, "#1f7a4c")
        cards.append(
            f"""
            <div class=\"metric-card\" style=\"border-left-color: {accent};\">
                <h4>{row['algorithm']}</h4>
                <div class=\"value\">{row['emission_g']:.1f} g CO2</div>
                <div class=\"sub\">{row['distance_km']:.2f} km | {row['time_min']:.2f} min</div>
            </div>
            """
        )
    return cards


def build_comparison_charts(route_df: pd.DataFrame) -> Tuple[go.Figure, go.Figure, go.Figure]:
    """Generate required Plotly comparison visuals."""
    color_map = {
        "Shortest path": ROUTE_COLOR_HEX["shortest"],
        "Fastest path": ROUTE_COLOR_HEX["fastest"],
        "Carbon-aware path": ROUTE_COLOR_HEX["carbon"],
    }

    emis_fig = px.bar(
        route_df,
        x="algorithm",
        y="emission_g",
        color="algorithm",
        color_discrete_map=color_map,
        title="Emission Comparison",
        text_auto=".1f",
    )
    emis_fig.update_layout(showlegend=False, margin=dict(t=55, l=10, r=10, b=10))

    time_fig = px.bar(
        route_df,
        x="algorithm",
        y="time_min",
        color="algorithm",
        color_discrete_map=color_map,
        title="Travel Time Comparison",
        text_auto=".2f",
    )
    time_fig.update_layout(showlegend=False, margin=dict(t=55, l=10, r=10, b=10))

    scatter_fig = px.scatter(
        route_df,
        x="distance_km",
        y="emission_g",
        color="algorithm",
        color_discrete_map=color_map,
        size="time_min",
        title="Distance vs Emissions",
        hover_data={"time_min": ":.2f", "distance_km": ":.2f", "emission_g": ":.1f"},
    )
    scatter_fig.update_layout(margin=dict(t=55, l=10, r=10, b=10))

    return emis_fig, time_fig, scatter_fig


def build_simulation_figures(sim_df: pd.DataFrame) -> Tuple[go.Figure, go.Figure, go.Figure]:
    """Generate simulation histogram and box plots."""
    sim = sim_df.copy()
    sim["emis_reduction_vs_sp_g"] = sim["sp_emis_g"] - sim["ca_emis_g"]

    hist_fig = px.histogram(
        sim,
        x="emis_reduction_vs_sp_g",
        nbins=30,
        title="Distribution of Emission Reduction vs Shortest Path",
        color_discrete_sequence=["#1f7a4c"],
    )
    hist_fig.update_layout(margin=dict(t=55, l=10, r=10, b=10))

    box_source = pd.DataFrame(
        {
            "algorithm": ["Shortest"] * len(sim)
            + ["Fastest"] * len(sim)
            + ["Carbon-aware"] * len(sim),
            "emission_g": pd.concat(
                [sim["sp_emis_g"], sim["fp_emis_g"], sim["ca_emis_g"]],
                ignore_index=True,
            ),
        }
    )
    box_fig = px.box(
        box_source,
        x="algorithm",
        y="emission_g",
        color="algorithm",
        color_discrete_map={
            "Shortest": ROUTE_COLOR_HEX["shortest"],
            "Fastest": ROUTE_COLOR_HEX["fastest"],
            "Carbon-aware": ROUTE_COLOR_HEX["carbon"],
        },
        title="Emission Distribution by Algorithm",
    )
    box_fig.update_layout(showlegend=False, margin=dict(t=55, l=10, r=10, b=10))

    time_overhead_fig = px.histogram(
        sim.assign(time_overhead_min=(sim["ca_time_s"] - sim["sp_time_s"]) / 60.0),
        x="time_overhead_min",
        nbins=30,
        title="Carbon-aware Time Overhead vs Shortest Path (minutes)",
        color_discrete_sequence=["#3078e1"],
    )
    time_overhead_fig.update_layout(margin=dict(t=55, l=10, r=10, b=10))

    return hist_fig, box_fig, time_overhead_fig


def normalize_weights(alpha: float, beta: float, gamma: float) -> Tuple[float, float, float]:
    """Normalize weights so alpha + beta + gamma = 1."""
    total = alpha + beta + gamma
    if total <= 0:
        return 1 / 3, 1 / 3, 1 / 3
    return alpha / total, beta / total, gamma / total


def _route_latlon(graph: nx.MultiDiGraph, path: List[int]) -> List[List[float]]:
    """Convert a node-id path into folium-friendly [lat, lon] points."""
    coords: List[List[float]] = []
    for node in path:
        node_data = graph.nodes[node]
        coords.append([float(node_data.get("y", 0.0)), float(node_data.get("x", 0.0))])
    return coords


def _add_recenter_control(map_obj: folium.Map, source_lat: float, source_lon: float) -> None:
    """Add a custom recenter button below zoom controls similar to navigation apps."""
    template = Template(
        """
        {% macro script(this, kwargs) %}
        var map = {{this._parent.get_name()}};
        var SourceRecenterControl = L.Control.extend({
            options: { position: 'topleft' },
            onAdd: function() {
                var container = L.DomUtil.create('div', 'leaflet-bar leaflet-control leaflet-control-custom');
                container.style.backgroundColor = 'white';
                container.style.width = '34px';
                container.style.height = '34px';
                container.style.lineHeight = '34px';
                container.style.textAlign = 'center';
                container.style.cursor = 'pointer';
                container.style.fontSize = '18px';
                container.style.marginTop = '50px';
                container.title = 'Recenter to source';
                container.innerHTML = '&#9673;';
                L.DomEvent.disableClickPropagation(container);
                L.DomEvent.on(container, 'click', function() {
                    map.setView([{{this.source_lat}}, {{this.source_lon}}], Math.max(map.getZoom(), 14));
                });
                return container;
            }
        });
        map.addControl(new SourceRecenterControl());
        {% endmacro %}
        """
    )
    control = MacroElement()
    control._template = template
    control.source_lat = source_lat
    control.source_lon = source_lon
    map_obj.get_root().add_child(control)


def build_folium_route_map(
    graph: nx.MultiDiGraph,
    routes: Dict[str, RouteResult],
    source_node: int,
    destination_node: int,
    mode: str,
) -> folium.Map:
    """Build an interactive folium map with route overlays and source recenter control."""
    source_data = graph.nodes[source_node]
    destination_data = graph.nodes[destination_node]
    source_lat = float(source_data.get("y", 0.0))
    source_lon = float(source_data.get("x", 0.0))
    destination_lat = float(destination_data.get("y", 0.0))
    destination_lon = float(destination_data.get("x", 0.0))

    m = folium.Map(
        location=[source_lat, source_lon],
        zoom_start=14,
        control_scale=True,
        tiles=None,
    )

    # Add dynamic and realistic tile layers for different visualizations
    folium.TileLayer(
        tiles="OpenStreetMap", 
        name="Realistic Street View (Default)",
        show=True
    ).add_to(m)
    folium.TileLayer(
        tiles="https://server.arcgisonline.com/ArcGIS/rest/services/World_Imagery/MapServer/tile/{z}/{y}/{x}",
        attr="Esri",
        name="Satellite View (Realistic)",
    ).add_to(m)
    folium.TileLayer(
        tiles="CartoDB dark_matter", 
        name="Dark Mode (Dynamic)", 
    ).add_to(m)
    folium.TileLayer(
        tiles="CartoDB positron", 
        name="Light Mode (Minimal)",
    ).add_to(m)

    for key in ROUTE_ORDER:
        route = routes.get(key)
        if not route:
            continue
        coords = _route_latlon(graph, route.path)
        popup_html = (
            f"<b>{route.label}</b><br/>"
            f"Distance: {route.distance_m / 1000.0:.2f} km<br/>"
            f"Time: {route.travel_time_s / 60.0:.2f} min<br/>"
            f"Emissions: {route.emission_g:.1f} g"
        )

        if mode == "Animated traversal":
            AntPath(
                locations=coords,
                color=ROUTE_COLOR_HEX[key],
                pulse_color="#ffffff",
                weight=6,
                delay=800,
                tooltip=route.label,
            ).add_to(m)
            folium.PolyLine(
                locations=coords,
                color=ROUTE_COLOR_HEX[key],
                weight=2,
                opacity=0.55,
                popup=folium.Popup(popup_html, max_width=260),
            ).add_to(m)
        else:
            folium.PolyLine(
                locations=coords,
                color=ROUTE_COLOR_HEX[key],
                weight=6,
                opacity=0.9,
                tooltip=route.label,
                popup=folium.Popup(popup_html, max_width=260),
            ).add_to(m)

    # Source icon with pulsing dynamic ring
    folium.CircleMarker(
        location=[source_lat, source_lon],
        radius=12,
        color="white",
        weight=2,
        fill=True,
        fill_color="green",
        fill_opacity=0.7,
        tooltip="Start Location",
    ).add_to(m)
    folium.Marker(
        location=[source_lat, source_lon],
        tooltip="Source",
        popup="Source",
        icon=folium.Icon(color="green", icon="play", prefix="fa"),
    ).add_to(m)

    # Destination icon with dynamic pulsing ring
    folium.CircleMarker(
        location=[destination_lat, destination_lon],
        radius=12,
        color="white",
        weight=2,
        fill=True,
        fill_color="red",
        fill_opacity=0.7,
        tooltip="End Location",
    ).add_to(m)
    folium.Marker(
        location=[destination_lat, destination_lon],
        tooltip="Destination",
        popup="Destination",
        icon=folium.Icon(color="red", icon="flag", prefix="fa"),
    ).add_to(m)

    _add_recenter_control(m, source_lat, source_lon)

    # Add Legend to the map
    template = """
    {% macro html(this, kwargs) %}
    <div style="
        position: fixed; 
        top: 10px; 
        right: 10px; 
        width: 160px; 
        height: auto; 
        z-index:9999; 
        background-color: white; 
        color: black;
        border: 2px solid grey; 
        padding: 10px;
        border-radius: 6px;
        box-shadow: 3px 3px 5px rgba(0,0,0,0.3);
        font-family: Arial, sans-serif;
        font-size: 14px;
        ">
      <h4 style="margin-top: 0; margin-bottom: 10px; font-size: 16px; color: black;">Route Legend</h4>
      <div style="margin-bottom: 5px; color: black;"><i style="background: #e13b30; width: 14px; height: 14px; display: inline-block; border-radius: 50%; margin-right: 5px;"></i> Shortest</div>
      <div style="margin-bottom: 5px; color: black;"><i style="background: #3078e1; width: 14px; height: 14px; display: inline-block; border-radius: 50%; margin-right: 5px;"></i> Fastest</div>
      <div style="margin-bottom: 5px; color: black;"><i style="background: #22a854; width: 14px; height: 14px; display: inline-block; border-radius: 50%; margin-right: 5px;"></i> Carbon-aware</div>
    </div>
    {% endmacro %}
    """
    
    macro = MacroElement()
    macro._template = Template(template)
    m.get_root().add_child(macro)

    folium.LayerControl(position='bottomright', collapsed=True).add_to(m)

    return m
