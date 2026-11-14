"""
report_figures.py  -  save what the GUI shows, as files for the report / backup demo
====================================================================================
Run from the project root:
    python -m src.GUI.report_figures                         # brief example 2000 -> 3002, Thu 08:00, GRU
    python -m src.GUI.report_figures -o 4821 -d 2827 -t "2006-10-26 17:00" --model linear

Writes to results/routes/ (names start with the trip, e.g. route_2000_3002_0800_gru_*):
    *_route1.png    map with the fastest route highlighted (same drawing as the GUI)
    *_traffic.png   same map with roads coloured by predicted speed
    *_routes.csv    the route table: rank, minutes, km, intersections, path, roads
"""
from __future__ import annotations

import argparse
import tkinter as tk

import pandas as pd

import src.Models.deep_common as dc
from src.GUI.app import RouteApp
from src.Routing import route_finder as rf


def save_figures(origin: str, destination: str, when: str, model: str, method: str = "AS") -> list:
    cfg = dc.load_config()
    result = rf.find_routes(origin, destination, when, model, method, cfg=cfg)
    out = dc.get_path(cfg, "results_dir") / "routes"
    out.mkdir(parents=True, exist_ok=True)
    stem = out / f"route_{result.origin}_{result.destination}_{result.when:%H%M}_{model}"

    root = tk.Tk()
    root.withdraw()                                      # no window needed, only the drawing
    app = RouteApp(root)
    app.show_result(result)                              # selects route 1 and draws the map
    files = [f"{stem}_route1.png", f"{stem}_traffic.png", f"{stem}_routes.csv"]
    app.fig.savefig(files[0], dpi=150, bbox_inches="tight")
    app.traffic.set(True)
    app.draw_map()
    app.fig.savefig(files[1], dpi=150, bbox_inches="tight")
    root.destroy()

    pd.DataFrame([{"rank": r.rank, "minutes": round(r.minutes, 1), "km": round(r.distance_km, 2),
                   "intersections": len(r.path) - 1, "path": " -> ".join(r.path),
                   "roads": " -> ".join(r.roads)} for r in result.routes]).to_csv(files[2], index=False)
    return files


def main() -> None:
    ap = argparse.ArgumentParser(description="Save GUI route maps and the route table for the report")
    ap.add_argument("-o", "--origin", default="2000")
    ap.add_argument("-d", "--destination", default="3002")
    ap.add_argument("-t", "--time", default="2006-10-26 08:00")
    ap.add_argument("--model", default="gru")
    ap.add_argument("--method", default="AS")
    args = ap.parse_args()
    for f in save_figures(args.origin, args.destination, args.time, args.model, args.method):
        print("Saved ->", f)


if __name__ == "__main__":
    main()
