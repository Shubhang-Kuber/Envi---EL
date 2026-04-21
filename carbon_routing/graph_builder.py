"""
graph_builder.py
================
Downloads and prepares a real-world city road network using OSMnx.
Stores per-edge attributes: distance (m), speed (km/h), travel time (s).

City default: Bengaluru, India (manageable graph ~50k nodes).
For faster testing use 'Indiranagar, Bengaluru, India' (small neighbourhood).
"""

import os
import pickle
import osmnx as ox
import networkx as nx
import numpy as np

# -- Default speed assumptions by road type (km/h) --------------------------
SPEED_MAP = {
    "motorway":       100,
    "trunk":           80,
    "primary":         60,
    "secondary":       50,
    "tertiary":        40,
    "unclassified":    30,
    "residential":     25,
    "living_street":   15,
    "service":         15,
    "pedestrian":      10,
    "track":           20,
    "road":            30,
}

DEFAULT_SPEED = 30   # km/h fallback


def _resolve_speed(highway_tag) -> float:
    """Return speed (km/h) for a given OSM highway tag value."""
    if isinstance(highway_tag, list):
        highway_tag = highway_tag[0]
    return SPEED_MAP.get(highway_tag, DEFAULT_SPEED)


def build_graph(
    place_name: str = "Indiranagar, Bengaluru, India",
    cache_path:  str = "data/bengaluru_graph.pkl",
    force_download: bool = False,
) -> nx.MultiDiGraph:
    """
    Download (or load from cache) a drive-network graph for `place_name`
    and annotate every edge with:
        - 'length'      : road segment length in metres
        - 'speed_kph'   : estimated travel speed in km/h
        - 'travel_time' : edge traversal time in seconds

    Parameters
    ----------
    place_name      : OSM place string understood by Nominatim
    cache_path      : file path to pickle the graph for reuse
    force_download  : ignore existing cache and re-download

    Returns
    -------
    G : annotated MultiDiGraph
    """
    os.makedirs(os.path.dirname(cache_path) if os.path.dirname(cache_path) else ".", exist_ok=True)

    # -- Load from cache if available --------------------------------------
    if os.path.exists(cache_path) and not force_download:
        print(f"[GraphBuilder] Loading cached graph from '{cache_path}' ...")
        with open(cache_path, "rb") as f:
            G = pickle.load(f)
        print(f"[GraphBuilder] Loaded: {G.number_of_nodes()} nodes, {G.number_of_edges()} edges.")
        return G

    # -- Download from OpenStreetMap ---------------------------------------
    print(f"[GraphBuilder] Downloading graph for '{place_name}' ...")
    # osmnx 2.x: API moved to ox.graph.from_place
    try:
        G = ox.graph.from_place(place_name, network_type="drive")
    except AttributeError:
        G = ox.graph_from_place(place_name, network_type="drive")   # osmnx 1.x fallback
    print(f"[GraphBuilder] Raw graph: {G.number_of_nodes()} nodes, {G.number_of_edges()} edges.")

    # -- Annotate edges ----------------------------------------------------
    for u, v, key, data in G.edges(keys=True, data=True):
        # Distance: OSMnx already stores 'length' in metres
        length_m = data.get("length", 0.0)

        # Speed: prefer OSM maxspeed tag, else infer from highway type
        maxspeed = data.get("maxspeed", None)
        if maxspeed:
            try:
                if isinstance(maxspeed, list):
                    maxspeed = maxspeed[0]
                speed_kph = float(str(maxspeed).split()[0])
            except (ValueError, AttributeError):
                speed_kph = _resolve_speed(data.get("highway", "road"))
        else:
            speed_kph = _resolve_speed(data.get("highway", "road"))

        # Clamp to realistic range
        speed_kph = float(np.clip(speed_kph, 5, 130))

        # Travel time in seconds
        speed_ms  = speed_kph * (1000 / 3600)          # m/s
        travel_time_s = length_m / speed_ms if speed_ms > 0 else 9999.0

        # Write back annotated attributes
        G[u][v][key]["speed_kph"]   = speed_kph
        G[u][v][key]["travel_time"] = travel_time_s     # seconds
        # 'length' already present; ensure float
        G[u][v][key]["length"]      = float(length_m)

    # -- Cache for future runs ---------------------------------------------
    with open(cache_path, "wb") as f:
        pickle.dump(G, f)
    print(f"[GraphBuilder] Graph cached to '{cache_path}'.")

    return G


def get_node_list(G: nx.MultiDiGraph) -> list:
    """Return list of all node IDs in the graph."""
    return list(G.nodes())


def graph_summary(G: nx.MultiDiGraph) -> None:
    """Print a quick summary of graph statistics."""
    lengths      = [d.get("length",      0.0) for _, _, d in G.edges(data=True)]
    speeds       = [d.get("speed_kph",   30.0) for _, _, d in G.edges(data=True)]
    travel_times = [d.get("travel_time", 0.0)  for _, _, d in G.edges(data=True)]

    print("=" * 50)
    print(" GRAPH SUMMARY")
    print("=" * 50)
    print(f"  Nodes            : {G.number_of_nodes():,}")
    print(f"  Edges            : {G.number_of_edges():,}")
    print(f"  Avg length (m)   : {np.mean(lengths):.1f}")
    print(f"  Avg speed (km/h) : {np.mean(speeds):.1f}")
    print(f"  Avg time (s)     : {np.mean(travel_times):.1f}")
    print("=" * 50)
