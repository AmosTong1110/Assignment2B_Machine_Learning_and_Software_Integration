"""
graph_builder.py  -  build the Boroondara road network: nodes + edges   (Task 3, step 1)
=======================================================================================
Run once from the project root:     python -m src.Routing.graph_builder
        (add --no-plot to skip the map picture)

Nodes   = the SCATS sites in data/Processed_Data/site_metadata.csv (lat/lon per site).
Edges   = two sites are connected if they are NEIGHBOURS ON THE SAME ROAD.

How the edges are found (no map data needed - only the raw SCATS file):
    1. Every detector has a 'Location' like  "WARRIGAL_RD N of TOORAK_RD".
       Both road names in it (WARRIGAL_RD and TOORAK_RD) pass through that intersection.
    2. Group the sites by road.  For every road with 2+ sites, sort the sites along the
       road (projection on the road's main direction) and join each site to the next one.
    3. Drop joins longer than MAX_EDGE_KM (usually two different roads with the same name),
       then apply the hand-checked fixes in EXTRA_EDGES / REMOVED_EDGES.
    4. Every road is two-way, so each join becomes two directed edges A->B and B->A.

Outputs
    data/Processed_Data/edges.csv          from, to, road, distance_km   (directed)
    Results/plots/network_map.png          picture of the graph - CHECK IT against a map

Distance = straight-line (haversine) distance between the two sites' coordinates.
"""
from __future__ import annotations

import argparse
import re
from pathlib import Path

import numpy as np
import pandas as pd

import src.Models.deep_common as dc
from src.Routing.travel_time import haversine_km

RAW_XLS = "data/Raw_Data/Scats Data October 2006.xls"
EDGE_COLUMNS = ["from", "to", "road", "distance_km"]

# Joins longer than this are treated as "same name, different road" and dropped.
MAX_EDGE_KM = 4.0

# Different spellings of the same road in the raw file  ->  one name.
ROAD_ALIASES = {
    "HARP_ST": "HARP_RD",                       # 4321: "HIGH_ST NE OF HARP_ST" (typo for Harp Rd)
    "EASTERN_FWY_W_BD_RAMPS": "EASTERN_FWY",
    "OFFRAMP_EASTERN_FWY": "EASTERN_FWY",
    "HIGH STREET_RD": "HIGH_STREET_RD",         # Glen Waverley's High Street Rd (not Kew's High St)
}

# Hand-checked corrections after looking at the map (Results/plots/network_map.png).
# Undirected pairs: one entry adds/removes BOTH directions.
EXTRA_EDGES: list[tuple[str, str, str]] = [
    # (site_a, site_b, road)  - roads that change NAME between two sites, so step 2 misses them
    ("2200", "4063", "WHITEHORSE_RD"),       # Whitehorse Rd continues east as Maroondah Hwy
    ("2846", "0970", "HIGH_ST"),             # High St (Glen Iris) continues east as High Street Rd
    ("2846", "4043", "BURKE_RD"),            # High St -> Burke Rd north to Toorak Rd
    ("4821", "3001", "BARKERS_RD"),          # Victoria St crosses the Yarra and becomes Barkers Rd
    ("4812", "4262", "BRIDGE_RD"),           # Swan St -> Burnley St -> Bridge Rd (Hawthorn Bridge)
    ("4324", "3662", "COTHAM_RD"),           # Cotham Rd runs west from Glenferrie Rd to Kew Junction
]
REMOVED_EDGES: list[tuple[str, str]] = [
    # (site_a, site_b)
]


# --------------------------------------------------------------------------
# 1. Nodes
# --------------------------------------------------------------------------
def load_nodes(cfg: dict = None) -> pd.DataFrame:
    """One row per SCATS site: scats_number (str, e.g. '0970'), latitude, longitude."""
    cfg = cfg or dc.load_config()
    meta = pd.read_csv(dc.get_path(cfg, "processed_dir") / "site_metadata.csv",
                       dtype={"scats_number": str})
    meta = meta.dropna(subset=["latitude", "longitude"])
    return meta[["scats_number", "latitude", "longitude"]].reset_index(drop=True)


# --------------------------------------------------------------------------
# 2. Road names per site
# --------------------------------------------------------------------------
def normalise_road(name: str) -> str:
    """'High_St ' -> 'HIGH_ST'; then map known aliases to one spelling."""
    name = re.sub(r"\s+", " ", str(name).strip().upper())
    return ROAD_ALIASES.get(name, name)


def parse_roads(location: str) -> tuple[str, str]:
    """
    'WARRIGAL_RD N of TOORAK_RD'  ->  ('WARRIGAL_RD', 'TOORAK_RD')
    The direction word (N, SE, ...) and 'of'/'OF' sit between the two road names.
    """
    m = re.match(r"^(.+?)\s+(?:N|S|E|W|NE|NW|SE|SW)\s+OF\s+(.+)$", location.strip(), re.IGNORECASE)
    if not m:
        raise ValueError(f"Cannot read road names from location {location!r}")
    return normalise_road(m.group(1)), normalise_road(m.group(2))


def load_site_roads(xls_path: Path) -> pd.DataFrame:
    """Unique (scats_number, road) pairs from the 'Location' column of the raw SCATS file."""
    raw = pd.read_excel(xls_path, sheet_name="Data", header=1,
                        usecols=["SCATS Number", "Location"], dtype={"SCATS Number": str})
    raw = raw.dropna().drop_duplicates()
    rows = []
    for site, loc in raw.itertuples(index=False):
        for road in parse_roads(loc):
            rows.append((str(site).strip().zfill(4), road))
    return pd.DataFrame(rows, columns=["scats_number", "road"]).drop_duplicates()


# --------------------------------------------------------------------------
# 3. Edges
# --------------------------------------------------------------------------
def order_along_road(points: pd.DataFrame) -> pd.DataFrame:
    """
    Sort sites along a road: project them on the road's main direction
    (first principal component of their coordinates) and sort by that value.
    Longitude is scaled by cos(latitude) so north-south and east-west distances match.
    """
    xy = np.column_stack([points["longitude"] * np.cos(np.radians(points["latitude"].mean())),
                          points["latitude"]])
    xy = xy - xy.mean(axis=0)
    direction = np.linalg.svd(xy, full_matrices=False)[2][0]     # main direction of the road
    return points.assign(_pos=xy @ direction).sort_values("_pos").drop(columns="_pos")


def build_edges(nodes: pd.DataFrame, site_roads: pd.DataFrame,
                max_edge_km: float = MAX_EDGE_KM) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Returns (edges, dropped):
        edges   - directed edges  from, to, road, distance_km  (both directions)
        dropped - consecutive pairs rejected because they were longer than max_edge_km
    """
    pos = nodes.set_index("scats_number")[["latitude", "longitude"]]
    sr = site_roads[site_roads["scats_number"].isin(pos.index)]

    pairs, dropped = {}, []
    for road, group in sr.groupby("road"):
        if len(group) < 2:
            continue                                            # only one site on this road
        ordered = order_along_road(pos.loc[group["scats_number"]].reset_index())
        sites = ordered["scats_number"].tolist()
        for a, b in zip(sites, sites[1:]):                      # consecutive neighbours
            d = haversine_km(*pos.loc[a], *pos.loc[b])
            if d > max_edge_km:
                dropped.append({"from": a, "to": b, "road": road, "distance_km": round(d, 3)})
                continue
            key = tuple(sorted((a, b)))
            if key not in pairs:                                # two roads may join the same pair
                pairs[key] = (road, d)

    for a, b, road in EXTRA_EDGES:
        pairs[tuple(sorted((a, b)))] = (road, haversine_km(*pos.loc[a], *pos.loc[b]))
    for a, b in REMOVED_EDGES:
        pairs.pop(tuple(sorted((a, b))), None)

    rows = []
    for (a, b), (road, d) in sorted(pairs.items()):
        rows.append((a, b, road, round(d, 4)))                  # two-way road:
        rows.append((b, a, road, round(d, 4)))                  #   A->B and B->A
    return pd.DataFrame(rows, columns=EDGE_COLUMNS), pd.DataFrame(dropped)


def load_edges(cfg: dict = None) -> pd.DataFrame:
    """Read edges.csv (written by main()).  Site numbers stay 4-char strings."""
    cfg = cfg or dc.load_config()
    path = dc.get_path(cfg, "processed_dir") / "edges.csv"
    if not path.exists():
        raise FileNotFoundError(f"{path} not found - run:  python -m src.Routing.graph_builder")
    return pd.read_csv(path, dtype={"from": str, "to": str})


# --------------------------------------------------------------------------
# 4. Picture of the network (to check the edges by eye)
# --------------------------------------------------------------------------
def plot_network(nodes: pd.DataFrame, edges: pd.DataFrame, out_path: Path) -> Path:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    pos = nodes.set_index("scats_number")
    fig, ax = plt.subplots(figsize=(10, 10))
    roads = sorted(edges["road"].unique())
    palette = list(plt.get_cmap("tab20").colors) + list(plt.get_cmap("Dark2").colors)
    cmap = lambda i: palette[i % len(palette)]                  # one colour per road (up to 28)
    seen = set()
    for row in edges.itertuples(index=False):
        key = tuple(sorted((row[0], row[1])))
        if key in seen:
            continue                                            # draw each two-way edge once
        seen.add(key)
        a, b = pos.loc[row[0]], pos.loc[row[1]]
        ax.plot([a.longitude, b.longitude], [a.latitude, b.latitude],
                color=cmap(roads.index(row.road)), lw=2, zorder=1)
    ax.scatter(pos.longitude, pos.latitude, s=60, color="black", zorder=2)
    for site, p in pos.iterrows():
        ax.annotate(site, (p.longitude, p.latitude), xytext=(4, 4),
                    textcoords="offset points", fontsize=8)
    handles = [plt.Line2D([], [], color=cmap(i), lw=2, label=r) for i, r in enumerate(roads)]
    ax.legend(handles=handles, fontsize=7, loc="upper left", bbox_to_anchor=(1.01, 1))
    ax.set_title(f"Boroondara SCATS network: {len(pos)} sites, {len(seen)} two-way roads")
    ax.set_xlabel("Longitude")
    ax.set_ylabel("Latitude")
    ax.set_aspect(1 / np.cos(np.radians(pos.latitude.mean())))  # metres look the same N-S and E-W
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    return out_path


# --------------------------------------------------------------------------
def main(cfg: dict = None, plot: bool = True) -> pd.DataFrame:
    cfg = cfg or dc.load_config()
    nodes = load_nodes(cfg)
    site_roads = load_site_roads(dc.ROOT / RAW_XLS)
    edges, dropped = build_edges(nodes, site_roads)

    out = dc.get_path(cfg, "processed_dir") / "edges.csv"
    edges.to_csv(out, index=False)
    print(f"{len(nodes)} nodes, {len(edges)} directed edges ({len(edges) // 2} two-way roads) -> {out}")

    isolated = sorted(set(nodes["scats_number"]) - set(edges["from"]))
    if isolated:
        print(f"WARNING: sites with no edge (unreachable): {isolated}")
    if len(dropped):
        print(f"Dropped {len(dropped)} joins longer than {MAX_EDGE_KM} km:")
        print(dropped.to_string(index=False))
    if plot:
        print("Map ->", plot_network(nodes, edges,
                                     dc.get_path(cfg, "results_dir") / "plots" / "network_map.png"))
    return edges


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description="Build the SCATS road network (edges.csv)")
    ap.add_argument("--config", default=None, help="path to config.yaml (default: project root)")
    ap.add_argument("--no-plot", action="store_true", help="do not draw the network map")
    args = ap.parse_args()
    main(dc.load_config(args.config), plot=not args.no_plot)
