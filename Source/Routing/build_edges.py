"""
build_edges.py  -  build a DRAFT edge list (edges.csv) for the Boroondara map
=============================================================================
The VicRoads dataset has SCATS sites (intersections) but NOT the road links
between them.  This script guesses the links with a simple rule:

    1. Every detector name says which two roads cross at the site, e.g.
       "WARRIGAL_RD N of TOORAK_RD"  ->  roads WARRIGAL_RD and TOORAK_RD.
    2. All sites that sit on the SAME road are put in order along that road
       (using their latitude / longitude).
    3. Each site is linked to the next site on that road, in both directions.

Example:  BURKE_RD has 7 sites.  Sorted north -> south they become a chain
          A - B - C - D - E - F - G, so we create edges A-B, B-C, ... F-G.

THIS IS A DRAFT.  Please check it against a real map (Google Maps / OSM) and
edit edges.csv by hand where it is wrong (see README, section "edges.csv").

Run from the project root:   python Source/Routing/build_edges.py
Output:                      Data/edges.csv
Columns: from_site, to_site, road, distance_km
"""
from __future__ import annotations

import argparse
import re
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
RAW_XLS = ROOT / "Data" / "Raw_Data" / "Scats Data October 2006.xls"
META_CSV = ROOT / "Data" / "Processed_Data" / "site_metadata.csv"
OUT_CSV = ROOT / "Data" / "edges.csv"

# "<road> <direction> of <cross road>"  (direction = N, S, E, W, NE, NW, SE, SW; any case)
LOCATION_RE = re.compile(r"^(.+?)\s+(NE|NW|SE|SW|N|S|E|W)\s+of\s+(.+)$", re.IGNORECASE)
MAX_EDGE_KM = 6.0          # a link longer than this is probably a missing site, so we drop it


def haversine_km(lat1, lon1, lat2, lon2) -> float:
    """Straight-line distance in km between two latitude/longitude points."""
    p1, p2 = np.radians(lat1), np.radians(lat2)
    a = (np.sin((p2 - p1) / 2) ** 2
         + np.cos(p1) * np.cos(p2) * np.sin(np.radians(lon2 - lon1) / 2) ** 2)
    return float(2 * 6371.0088 * np.arcsin(np.sqrt(a)))


def roads_per_site(locations: pd.DataFrame) -> dict[str, set[str]]:
    """locations has columns site, location. Returns {site: {road names}}."""
    roads: dict[str, set[str]] = {}
    for site, loc in locations[["site", "location"]].drop_duplicates().itertuples(index=False):
        m = LOCATION_RE.match(str(loc).strip())
        if m:
            roads.setdefault(site, set()).update(
                [m.group(1).strip().upper(), m.group(3).strip().upper()])
    return roads


def order_along_road(sites: pd.DataFrame) -> list[str]:
    """Sort the sites of one road along the line they form (first principal axis)."""
    xy = sites[["latitude", "longitude"]].to_numpy(dtype=float)
    xy = xy - xy.mean(axis=0)
    axis = np.linalg.svd(xy, full_matrices=False)[2][0]
    return sites.iloc[np.argsort(xy @ axis)]["scats_number"].tolist()


def build_edges(roads: dict[str, set[str]], meta: pd.DataFrame,
                max_km: float = MAX_EDGE_KM) -> pd.DataFrame:
    """Chain neighbouring sites on every road. Returns directed edges (both ways)."""
    meta = meta.set_index("scats_number", drop=False)
    by_road: dict[str, list[str]] = {}
    for site, rs in roads.items():
        if site in meta.index:
            for r in rs:
                by_road.setdefault(r, []).append(site)

    rows, seen = [], set()
    for road, sites in sorted(by_road.items()):
        if len(sites) < 2:
            continue
        chain = order_along_road(meta.loc[sorted(sites)])
        for a, b in zip(chain[:-1], chain[1:]):
            km = haversine_km(meta.at[a, "latitude"], meta.at[a, "longitude"],
                              meta.at[b, "latitude"], meta.at[b, "longitude"])
            if km > max_km or (a, b) in seen:
                continue
            seen.update({(a, b), (b, a)})
            rows += [{"from_site": a, "to_site": b, "road": road, "distance_km": round(km, 3)},
                     {"from_site": b, "to_site": a, "road": road, "distance_km": round(km, 3)}]
    return pd.DataFrame(rows, columns=["from_site", "to_site", "road", "distance_km"])


def nearest_sites(site: str, meta: pd.DataFrame, k: int = 3) -> list[tuple[str, float]]:
    """The k closest other sites (straight-line km). Helps you add missing edges by hand."""
    m = meta.set_index("scats_number")
    d = {o: haversine_km(m.at[site, "latitude"], m.at[site, "longitude"],
                         m.at[o, "latitude"], m.at[o, "longitude"])
         for o in m.index if o != site}
    return [(o, round(km, 2)) for o, km in sorted(d.items(), key=lambda t: t[1])[:k]]


def main(out: Path = OUT_CSV) -> pd.DataFrame:
    raw = pd.read_excel(RAW_XLS, sheet_name="Data", header=1)
    raw = pd.DataFrame({"site": raw["SCATS Number"].astype(int).astype(str).str.zfill(4),
                        "location": raw["Location"]})
    meta = pd.read_csv(META_CSV, dtype={"scats_number": str})
    edges = build_edges(roads_per_site(raw), meta)
    out.parent.mkdir(parents=True, exist_ok=True)
    edges.to_csv(out, index=False)

    nodes = set(meta["scats_number"])
    linked = set(edges["from_site"])
    print(f"{len(edges)} directed edges ({len(edges) // 2} roads links) -> {out}")
    alone = sorted(nodes - linked)
    print(f"Sites with NO edge (add by hand): {alone}")
    for site in alone:
        print(f"   {site}: nearest sites (km) = {nearest_sites(site, meta)}")
    return edges


if __name__ == "__main__":
    argparse.ArgumentParser(description="Build the draft edges.csv").parse_args()
    main()
