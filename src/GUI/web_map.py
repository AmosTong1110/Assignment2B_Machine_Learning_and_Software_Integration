"""
web_map.py  -  "Google Maps Traffic"-style view on OpenStreetMap   (Research: visualisation)
============================================================================================
Writes ONE self-contained HTML page (Leaflet + OpenStreetMap tiles) and opens it in the browser:

    * every road link coloured by the PREDICTED speed, each direction on its own side
      (green fast ... dark red at capacity), like the Google Maps traffic layer
    * the up-to-5 routes from route_finder, click one to highlight it and see its legs
    * a time slider: predicted traffic for every hour of the trip day (optional, --timeline)
    * SCATS sites at their corrected OSM positions; a toggle shows the original SCATS
      positions and how far each was moved (the coordinate problem from the brief)

Run from the project root:
    python -m src.GUI.web_map -o 2000 -d 3002 -t "2006-10-26 08:00" --model gru --timeline
The GUI has an "Open traffic map" button that does the same for the current search.

Needs  python -m src.Routing.osm_geometry  to have been run once (corrected positions and road
shapes).  Without it the map still works: SCATS positions and straight links.
The page loads Leaflet and the map tiles from the internet when it is opened.
"""
from __future__ import annotations

import argparse
import json
import webbrowser
from pathlib import Path

import pandas as pd

import src.Models.deep_common as dc
from src.Routing import edge_costs as ec
from src.Routing import route_finder as rf
from src.Routing.graph_builder import load_nodes
from src.Routing.osm_geometry import load_edge_geometry, load_site_coords
from src.Search import SEARCH_METHODS

TEMPLATE = Path(__file__).with_name("web_map_template.html")
MODEL_NAMES = {"gru": "GRU", "lstm": "LSTM", "linear": "Linear regression", "actual": "Measured flow"}


# --------------------------------------------------------------------------
# Data for the page
# --------------------------------------------------------------------------
def site_positions(cfg: dict = None, use_osm: bool = True) -> pd.DataFrame:
    """scats_number, latitude, longitude (corrected if available), raw_latitude, raw_longitude, offset_m, method."""
    nodes = load_nodes(cfg).rename(columns={"latitude": "raw_latitude", "longitude": "raw_longitude"})
    fixed = load_site_coords(cfg) if use_osm else None
    if fixed is None:
        out = nodes.assign(latitude=nodes["raw_latitude"], longitude=nodes["raw_longitude"],
                           offset_m=0.0, method="unchanged")
    else:
        out = nodes.merge(fixed[["scats_number", "latitude", "longitude", "offset_m", "method"]],
                          on="scats_number", how="left")
        missing = out["latitude"].isna()
        out.loc[missing, ["latitude", "longitude"]] = out.loc[missing, ["raw_latitude", "raw_longitude"]].to_numpy()
        out["offset_m"] = out["offset_m"].fillna(0.0)
        out["method"] = out["method"].fillna("unchanged")
    return out


def timeline_tables(when: pd.Timestamp, model: str, cfg: dict, step_hours: int = 1,
                    progress=None) -> dict[pd.Timestamp, pd.DataFrame]:
    """edge_costs() for every `step_hours` of the trip day (skips times without 3 h of history)."""
    tables = {}
    day = when.normalize()
    times = sorted({*pd.date_range(day, day + pd.Timedelta(hours=23), freq=f"{step_hours}h"), when})
    for i, t in enumerate(times):
        if progress:
            progress(i, len(times))
        try:
            tables[t] = ec.edge_costs(t, model, cfg)
        except ValueError:                       # e.g. 1 Oct before 03:00 - no history for the models
            continue
    return tables


def page_data(result: rf.RouteResult, cfg: dict = None, frames: dict | None = None,
              use_osm: bool = True) -> dict:
    """
    Everything the HTML page needs, as plain JSON-friendly Python.
    use_osm=False ignores the corrected positions and road shapes (the 'before' picture).
    """
    cfg = cfg or dc.load_config()
    labels = rf.site_labels()
    sites = site_positions(cfg, use_osm)
    pos = {s: (la, lo) for s, la, lo in sites[["scats_number", "latitude", "longitude"]].itertuples(index=False)}
    geometry = load_edge_geometry(cfg) if use_osm else {}

    table = result.edge_table.reset_index(drop=True)
    edges = []
    for a, b, road, km in table[["from", "to", "road", "distance_km"]].itertuples(index=False):
        g = geometry.get(f"{a}|{b}")
        coords = g["coords"] if g else [list(pos[a]), list(pos[b])]
        edges.append({"from": a, "to": b, "road": road, "km": round(float(km), 3), "coords": coords,
                      "road_km": g["road_km"] if g else None})
    index = {(e["from"], e["to"]): i for i, e in enumerate(edges)}

    frames = dict(frames or {})
    frames[result.when] = result.edge_table                     # the departure time is always there
    frame_list = []
    for t in sorted(frames):
        f = frames[t].set_index(["from", "to"]).loc[list(zip(table["from"], table["to"]))]
        frame_list.append({"time": f"{t:%H:%M}", "speed": f["speed_kmh"].round(1).tolist(),
                           "flow": f["flow_vph"].round(0).tolist(), "tt": f["travel_time_s"].round(0).tolist()})

    routes = [{"rank": r.rank, "minutes": round(r.minutes, 1), "km": round(r.distance_km, 2),
               "path": r.path, "roads": r.roads,
               "edges": [index[(a, b)] for a, b in zip(r.path, r.path[1:])]}
              for r in result.routes]

    params = ec.settings(cfg)
    return {
        "meta": {"origin": result.origin, "destination": result.destination,
                 "originLabel": labels.get(result.origin, ""), "destinationLabel": labels.get(result.destination, ""),
                 "when": f"{result.when:%a %d %b %Y %H:%M}", "departure": f"{result.when:%H:%M}",
                 "model": MODEL_NAMES.get(result.model, result.model),
                 "method": SEARCH_METHODS[result.method][1], "nodesCreated": result.nodes_created,
                 "speedLimit": float(params["speed_limit_kmh"]), "capacitySpeed": float(params["capacity_speed_kmh"]),
                 "corrected": bool((sites["method"] != "unchanged").any()),
                 "roadShapes": bool(geometry)},
        "sites": [{"id": s.scats_number, "label": labels.get(s.scats_number, ""),
                   "lat": s.latitude, "lon": s.longitude, "rawLat": s.raw_latitude, "rawLon": s.raw_longitude,
                   "offset": round(float(s.offset_m)), "method": s.method}
                  for s in sites.itertuples(index=False)],
        "edges": edges,
        "frames": frame_list,
        "routes": routes,
    }


# --------------------------------------------------------------------------
# HTML
# --------------------------------------------------------------------------
def render_html(data: dict) -> str:
    """The template with the data inserted (one file, nothing else to copy)."""
    payload = json.dumps(data, separators=(",", ":")).replace("</", "<\\/")
    title = f"TBRGS traffic map - {data['meta']['origin']} to {data['meta']['destination']}"
    return TEMPLATE.read_text(encoding="utf-8").replace("__TITLE__", title).replace("__DATA__", payload)


def write_map(result: rf.RouteResult, cfg: dict = None, frames: dict | None = None,
              out_path: Path | None = None, use_osm: bool = True) -> Path:
    """Write the HTML for a route search and return its path (results/maps/...)."""
    cfg = cfg or dc.load_config()
    if out_path is None:
        out_dir = dc.get_path(cfg, "results_dir") / "maps"
        out_dir.mkdir(parents=True, exist_ok=True)
        raw = "" if use_osm else "_raw"
        out_path = out_dir / f"map_{result.origin}_{result.destination}_{result.when:%m%d_%H%M}_{result.model}{raw}.html"
    out_path.write_text(render_html(page_data(result, cfg, frames, use_osm)), encoding="utf-8")
    return out_path


def open_in_browser(path: Path) -> None:
    webbrowser.open(Path(path).resolve().as_uri())


def main() -> None:
    ap = argparse.ArgumentParser(description="Open the route search on an OpenStreetMap traffic map")
    ap.add_argument("-o", "--origin", required=True)
    ap.add_argument("-d", "--destination", required=True)
    ap.add_argument("-t", "--time", required=True, help='e.g. "2006-10-26 08:00"')
    ap.add_argument("--model", default=None, choices=ec.FLOW_SOURCES)
    ap.add_argument("--method", default="AS", choices=list(SEARCH_METHODS))
    ap.add_argument("-k", type=int, default=rf.MAX_ROUTES)
    ap.add_argument("--timeline", action="store_true", help="also predict every hour of the day (time slider)")
    ap.add_argument("--raw-positions", action="store_true",
                    help="use the original SCATS positions and straight links (no OSM correction)")
    ap.add_argument("--no-open", action="store_true", help="only write the file")
    args = ap.parse_args()

    cfg = dc.load_config()
    result = rf.find_routes(args.origin, args.destination, args.time, args.model, args.method, args.k, cfg)
    frames = None
    if args.timeline:
        frames = timeline_tables(result.when, result.model, cfg,
                                 progress=lambda i, n: print(f"\r  predicting hour {i + 1}/{n}", end=""))
        print()
    path = write_map(result, cfg, frames, use_osm=not args.raw_positions)
    print(rf.format_result(result))
    print(f"\nMap written to {path}")
    if not args.no_open:
        open_in_browser(path)


if __name__ == "__main__":
    main()
