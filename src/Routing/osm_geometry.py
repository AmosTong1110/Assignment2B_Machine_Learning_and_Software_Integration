"""
osm_geometry.py  -  put the SCATS network on the real OpenStreetMap roads   (Research: map view)
================================================================================================
Run ONCE from the project root (needs internet; the result is saved, so the map works offline):

    python -m src.Routing.osm_geometry            # about 1-2 minutes (polite pauses between requests)

The brief warns that the SCATS latitude/longitude "do not map correctly to the actual
intersections".  This script fixes that and fetches the real road shapes:

1. Coordinate correction (Overpass API = OpenStreetMap database)
   For every site we download the named roads within SEARCH_RADIUS_M of its SCATS position.
   An OSM node shared by two ways whose names match two of the site's roads
   (e.g. "Warrigal Road" + "Toorak Road" for site 2000) is a real intersection of those roads.
   The site is moved to the closest such intersection (centre of the cluster of nodes, because
   a divided road crosses with several nodes).  If no match is found the site keeps its
   SCATS position, so nothing is ever lost.

2. Road geometry (OSRM = open-source router on OSM data)
   For every road link A-B of edges.csv we ask OSRM for the driving path between the corrected
   positions, so the map draws the link along the real streets instead of a straight line.
   A path that is much longer than the straight line (MAX_DETOUR) means OSRM went round
   another way (e.g. a turn ban), so that link is drawn straight instead.

Outputs (read by src/GUI/web_map.py)
    data/Processed_Data/site_coords_osm.csv    scats_number, raw lat/lon, osm lat/lon, offset_m, method, matched_roads
    data/Processed_Data/edge_geometry_osm.json {"A|B": {"coords": [[lat, lon], ...], "road_km": .., "straight_km": .., "source": ..}}

The numbers printed at the end (average offset, road vs straight-line distance) are used in
the Research section of the report.
"""
from __future__ import annotations

import argparse
import json
import math
import re
import time
import urllib.parse
import urllib.request
from pathlib import Path

import pandas as pd

import src.Models.deep_common as dc
from src.Routing.graph_builder import load_edges, load_nodes, SITE_ROADS_FILE
from src.Routing.travel_time import haversine_km

# Overpass servers, tried in order (public servers are sometimes busy or refuse a burst of requests)
OVERPASS_URLS = ["https://overpass-api.de/api/interpreter",
                 "https://maps.mail.ru/osm/tools/overpass/api/interpreter",
                 "https://overpass.kumi.systems/api/interpreter"]
OSRM_URL = "https://router.project-osrm.org/route/v1/driving/"
USER_AGENT = "TBRGS-student-project/1.0 (COS30019 assignment)"

SITE_COORDS_FILE = "site_coords_osm.csv"
EDGE_GEOMETRY_FILE = "edge_geometry_osm.json"

SEARCH_RADIUS_M = 400        # how far from the SCATS position we look for the real intersection
CLUSTER_M = 60               # nodes this close to the best one belong to the same intersection
MAX_DETOUR = 1.6             # OSRM path longer than 1.6 x straight line -> draw the link straight
PAUSE_S = 1.0                # pause between requests (fair use of the free public servers)

# SCATS abbreviations  <-  OSM full words (last word of the road name)
SUFFIXES = {"ROAD": "RD", "STREET": "ST", "HIGHWAY": "HWY", "GROVE": "GV", "FREEWAY": "FWY",
            "PARADE": "PDE", "AVENUE": "AV", "AVE": "AV", "DRIVE": "DR", "CRESCENT": "CR"}
# SCATS names that are called something else in OSM today
OSM_NAME_ALIASES = {"S.E.ARTERIAL": "MONASHFWY", "SEARTERIAL": "MONASHFWY"}


# --------------------------------------------------------------------------
# Road names:  'WARRIGAL_RD' (SCATS)  and  'Warrigal Road' (OSM)  ->  'WARRIGALRD'
# --------------------------------------------------------------------------
def road_key(name: str) -> str:
    """One comparable spelling for SCATS and OSM road names."""
    upper = str(name).upper().strip()
    if upper.replace(" ", "") in OSM_NAME_ALIASES:
        return OSM_NAME_ALIASES[upper.replace(" ", "")]
    words = [w for w in re.split(r"[\s_]+", re.sub(r"[^A-Z0-9_ ]", "", upper)) if w]
    if words and words[-1] in SUFFIXES:
        words[-1] = SUFFIXES[words[-1]]
    return "".join(words)


def site_road_keys(label: str) -> set[str]:
    """'BURWOOD_HWY / TOORAK_RD / WARRIGAL_RD' -> {'BURWOODHWY', 'TOORAKRD', 'WARRIGALRD'}"""
    return {road_key(r) for r in str(label).split("/") if r.strip()}


def metres(lat1, lon1, lat2, lon2) -> float:
    return 1000.0 * haversine_km(lat1, lon1, lat2, lon2)


# --------------------------------------------------------------------------
# 1. Coordinate correction
# --------------------------------------------------------------------------
def _get_json(url: str, data: bytes | None = None, retries: int = 3) -> dict:
    """GET/POST with a user agent and a few retries."""
    for attempt in range(retries):
        try:
            req = urllib.request.Request(url, data=data, headers={"User-Agent": USER_AGENT})
            with urllib.request.urlopen(req, timeout=60) as resp:
                return json.loads(resp.read().decode("utf-8"))
        except Exception:
            if attempt == retries - 1:
                raise
            time.sleep(5 * (attempt + 1))
    return {}


def fetch_ways_near(lat: float, lon: float, radius_m: int = SEARCH_RADIUS_M) -> list[dict]:
    """Named roads (OSM ways with their node ids and coordinates) around one point."""
    query = (f'[out:json][timeout:60];way(around:{radius_m},{lat},{lon})["highway"]["name"];'
             f'out tags geom;')
    body = urllib.parse.urlencode({"data": query}).encode()
    for i, url in enumerate(OVERPASS_URLS):
        try:
            data = _get_json(url, body, retries=2)
            return [w for w in data.get("elements", []) if w.get("type") == "way"]
        except Exception:
            if i == len(OVERPASS_URLS) - 1:
                raise
    return []


def find_intersection(lat: float, lon: float, road_keys: set[str], ways: list[dict]) -> dict:
    """
    Best real position for a site from the OSM ways around it.
    Returns {latitude, longitude, method, matched_roads}; method is
        'intersection'  two of the site's roads meet there (the normal case)
        'on_road'       only one road matched - nearest point on that road
        'unchanged'     nothing matched - SCATS position kept
    """
    node_roads: dict[tuple[float, float], set[str]] = {}       # (lat, lon) -> matching road keys
    for way in ways:
        key = road_key(way.get("tags", {}).get("name", ""))
        if key not in road_keys:
            continue
        for p in way.get("geometry", []):
            node_roads.setdefault((p["lat"], p["lon"]), set()).add(key)

    crossings = [(pt, r) for pt, r in node_roads.items() if len(r) >= 2]
    if crossings:
        best, _ = min(crossings, key=lambda c: metres(lat, lon, *c[0]))
        # A divided road gives several crossing nodes: average the ones near the best one.
        cluster = [(pt, r) for pt, r in crossings if metres(*best, *pt) <= CLUSTER_M]
        roads = sorted(set().union(*(r for _, r in cluster)))
        return {"latitude": sum(p[0] for p, _ in cluster) / len(cluster),
                "longitude": sum(p[1] for p, _ in cluster) / len(cluster),
                "method": "intersection", "matched_roads": " / ".join(roads)}
    if node_roads:
        pt = min(node_roads, key=lambda p: metres(lat, lon, *p))
        return {"latitude": pt[0], "longitude": pt[1], "method": "on_road",
                "matched_roads": " / ".join(sorted(node_roads[pt]))}
    return {"latitude": lat, "longitude": lon, "method": "unchanged", "matched_roads": ""}


def correct_sites(nodes: pd.DataFrame, labels: dict[str, str]) -> pd.DataFrame:
    """One Overpass request per site -> table of raw vs corrected coordinates."""
    rows = []
    for site, lat, lon in nodes[["scats_number", "latitude", "longitude"]].itertuples(index=False):
        try:
            ways = fetch_ways_near(lat, lon)
        except Exception as e:                                   # keep going, keep the raw point
            print(f"  {site}: Overpass failed ({e}); keeping the SCATS position")
            ways = []
        fix = find_intersection(lat, lon, site_road_keys(labels.get(site, "")), ways)
        offset = metres(lat, lon, fix["latitude"], fix["longitude"])
        rows.append({"scats_number": site, "raw_latitude": lat, "raw_longitude": lon,
                     "latitude": round(fix["latitude"], 7), "longitude": round(fix["longitude"], 7),
                     "offset_m": round(offset, 1), "method": fix["method"],
                     "matched_roads": fix["matched_roads"]})
        print(f"  {site:>5} {fix['method']:<12} moved {offset:6.0f} m   {fix['matched_roads']}", flush=True)
        time.sleep(PAUSE_S)
    return pd.DataFrame(rows)


# --------------------------------------------------------------------------
# 2. Road geometry for every link
# --------------------------------------------------------------------------
def osrm_path(a: tuple[float, float], b: tuple[float, float]) -> tuple[list[list[float]], float]:
    """Driving path a -> b from OSRM: ([[lat, lon], ...], length_km)."""
    url = (f"{OSRM_URL}{a[1]},{a[0]};{b[1]},{b[0]}"
           f"?overview=full&geometries=geojson&continue_straight=true")
    data = _get_json(url)
    route = data["routes"][0]
    coords = [[round(lat, 6), round(lon, 6)] for lon, lat in route["geometry"]["coordinates"]]
    return coords, route["distance"] / 1000.0


def edge_geometries(edges: pd.DataFrame, pos: dict[str, tuple[float, float]]) -> dict[str, dict]:
    """{'A|B': {...}} for every directed edge (each direction routed on its own carriageway)."""
    out = {}
    for a, b in zip(edges["from"], edges["to"]):
        straight = haversine_km(*pos[a], *pos[b])
        entry = {"coords": [list(pos[a]), list(pos[b])], "road_km": round(straight, 3),
                 "straight_km": round(straight, 3), "source": "straight"}
        try:
            coords, km = osrm_path(pos[a], pos[b])
            if km <= MAX_DETOUR * straight + 0.2:                 # +0.2 km: very short links
                entry.update(coords=coords, road_km=round(km, 3), source="osrm")
            else:
                print(f"  {a}->{b}: OSRM path {km:.2f} km vs {straight:.2f} km straight - drawn straight")
        except Exception as e:
            print(f"  {a}->{b}: OSRM failed ({e}) - drawn straight")
        out[f"{a}|{b}"] = entry
        time.sleep(PAUSE_S / 2)
    return out


# --------------------------------------------------------------------------
# Loading (used by the map; never needs the internet)
# --------------------------------------------------------------------------
def load_site_coords(cfg: dict = None) -> pd.DataFrame | None:
    """Corrected coordinates, or None if osm_geometry has not been run."""
    path = dc.get_path(cfg or dc.load_config(), "processed_dir") / SITE_COORDS_FILE
    if not path.exists():
        return None
    return pd.read_csv(path, dtype={"scats_number": str})


def load_edge_geometry(cfg: dict = None) -> dict[str, dict]:
    """{'A|B': {coords, road_km, straight_km, source}}, or {} if not built yet."""
    path = dc.get_path(cfg or dc.load_config(), "processed_dir") / EDGE_GEOMETRY_FILE
    if not path.exists():
        return {}
    return json.loads(path.read_text())


def summary(sites: pd.DataFrame, geometry: dict[str, dict]) -> str:
    """Numbers for the Research section of the report."""
    lines = ["Coordinate correction:",
             f"  {len(sites)} sites  " + ", ".join(f"{m}: {n}" for m, n in sites['method'].value_counts().items()),
             f"  offset SCATS -> OSM intersection: mean {sites['offset_m'].mean():.0f} m, "
             f"median {sites['offset_m'].median():.0f} m, max {sites['offset_m'].max():.0f} m"]
    moved = sites[sites["method"] == "intersection"]
    if len(moved):
        dlat = (moved["latitude"] - moved["raw_latitude"]).mean() * 111_320
        dlon = (moved["longitude"] - moved["raw_longitude"]).mean() * 111_320 * math.cos(math.radians(-37.83))
        lines.append(f"  average shift: {dlat:+.0f} m north, {dlon:+.0f} m east  (a systematic offset)")
    g = pd.DataFrame(geometry.values())
    if len(g):
        osrm = g[g["source"] == "osrm"]
        ratio = (osrm["road_km"] / osrm["straight_km"]).replace([float("inf")], float("nan")).dropna()
        lines += ["Road geometry:",
                  f"  {len(osrm)}/{len(g)} directed links follow real roads (OSRM), rest drawn straight",
                  f"  road distance / straight-line distance: mean {ratio.mean():.2f}, max {ratio.max():.2f}"]
    return "\n".join(lines)


def main(cfg: dict = None) -> None:
    ap = argparse.ArgumentParser(description="Correct SCATS coordinates and fetch road shapes from OSM")
    ap.add_argument("--skip-geometry", action="store_true", help="only correct the site coordinates")
    ap.add_argument("--retry", action="store_true",
                    help="keep the saved positions; only re-query sites left 'unchanged' (e.g. after a timeout)")
    args = ap.parse_args()

    cfg = cfg or dc.load_config()
    processed = dc.get_path(cfg, "processed_dir")
    nodes, edges = load_nodes(cfg), load_edges(cfg)
    labels = pd.read_csv(processed / SITE_ROADS_FILE, dtype=str).set_index("scats_number")["label"].to_dict()

    old = load_site_coords(cfg) if args.retry else None
    todo = nodes if old is None else nodes[nodes["scats_number"].isin(old.loc[old["method"] == "unchanged", "scats_number"])]
    print(f"1. Correcting {len(todo)} site positions with OpenStreetMap (Overpass)...")
    sites = correct_sites(todo, labels)
    if old is not None:
        sites = pd.concat([old[~old["scats_number"].isin(sites["scats_number"])], sites])
        sites = sites.sort_values("scats_number").reset_index(drop=True)
    sites.to_csv(processed / SITE_COORDS_FILE, index=False)

    geometry = load_edge_geometry(cfg)
    if not args.skip_geometry:
        print(f"2. Fetching road shapes for {len(edges)} directed links (OSRM)...")
        pos = {s: (la, lo) for s, la, lo in sites[["scats_number", "latitude", "longitude"]].itertuples(index=False)}
        geometry = edge_geometries(edges, pos)
        (processed / EDGE_GEOMETRY_FILE).write_text(json.dumps(geometry))

    print("\n" + summary(sites, geometry))
    print(f"\nSaved {processed / SITE_COORDS_FILE}" + ("" if args.skip_geometry else f" and {EDGE_GEOMETRY_FILE}"))


if __name__ == "__main__":
    main()
