"""
map_screenshots.py  -  report pictures of the OpenStreetMap traffic map   (Research section)
==========================================================================================
Run from the project root (after  python -m src.Routing.osm_geometry ):

    python -m src.GUI.map_screenshots                      # GRU, 2000 -> 3002, Thu 26 Oct 2006
    python -m src.GUI.map_screenshots -o 4821 -d 2827 --model lstm

Builds the maps with src/GUI/web_map.py and photographs them with headless Google Chrome
(installed on most machines; set CHROME=/path/to/chrome otherwise).  Needs internet for the tiles.

Writes to results/maps/screenshots/
    traffic_0800.png              predicted traffic + 5 routes, leaving 08:00 (morning peak)
    traffic_1700.png              the same trip leaving 17:00 (evening peak)
    positions_before.png          original SCATS positions, straight links (the problem in the brief)
    positions_after.png           corrected positions on real roads, original positions in purple
    positions_zoom_<site>.png     close-up of one site: SCATS position vs real intersection
and prints the offset numbers (also saved as site_offsets.csv) for the report.
"""
from __future__ import annotations

import argparse
import os
import shutil
import subprocess
from pathlib import Path

import pandas as pd

import src.Models.deep_common as dc
from src.GUI import web_map as wm
from src.Routing import route_finder as rf
from src.Routing.osm_geometry import load_edge_geometry, load_site_coords, summary

CHROME_CANDIDATES = [
    "/Applications/Google Chrome.app/Contents/MacOS/Google Chrome",
    r"C:\Program Files\Google\Chrome\Application\chrome.exe",
    r"C:\Program Files (x86)\Google\Chrome\Application\chrome.exe",
    "google-chrome", "chromium", "chromium-browser",
]

def find_chrome() -> str:
    for c in [os.environ.get("CHROME", "")] + CHROME_CANDIDATES:
        if c and (Path(c).exists() or shutil.which(c)):
            return c
    raise RuntimeError("Google Chrome not found. Set CHROME=/path/to/chrome and run again.")

def screenshot(chrome: str, html: Path, out_png: Path, view: str = "", size=(1600, 1000)) -> Path:
    """Open html#view in headless Chrome, wait for the tiles, save a PNG."""
    url = html.resolve().as_uri() + (f"#{view}" if view else "")
    subprocess.run([chrome, "--headless=new", "--disable-gpu", "--hide-scrollbars",
                    f"--window-size={size[0]},{size[1]}", "--virtual-time-budget=20000",
                    f"--screenshot={out_png}", url],
                   check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, timeout=120)
    print(f"  saved {out_png}")
    return out_png

def main() -> None:
    ap = argparse.ArgumentParser(description="Screenshots of the OSM traffic map for the report")
    ap.add_argument("-o", "--origin", default="2000")
    ap.add_argument("-d", "--destination", default="3002")
    ap.add_argument("--day", default="2006-10-26", help="trip day (default Thu 26 Oct 2006, test week)")
    ap.add_argument("--times", nargs="+", default=["08:00", "17:00"])
    ap.add_argument("--model", default="gru")
    ap.add_argument("--zoom-site", default=None, help="site for the close-up (default: the origin)")
    args = ap.parse_args()

    cfg = dc.load_config()
    if load_site_coords(cfg) is None or not load_edge_geometry(cfg):
        print("NOTE: run  python -m src.Routing.osm_geometry  first - without it the 'after' "
              "pictures still show the original SCATS positions.")
    out = dc.get_path(cfg, "results_dir") / "maps" / "screenshots"
    out.mkdir(parents=True, exist_ok=True)
    chrome = find_chrome()

    first_html = None
    for t in args.times:
        result = rf.find_routes(args.origin, args.destination, f"{args.day} {t}", args.model, "AS", rf.MAX_ROUTES, cfg)
        html = wm.write_map(result, cfg, out_path=out / f"map_{t.replace(':', '')}.html")
        print(f"{t}: route 1 {result.routes[0].minutes:.1f} min via {', '.join(result.routes[0].roads)}")
        screenshot(chrome, html, out / f"traffic_{t.replace(':', '')}.png")
        if first_html is None:
            first_html, first_result = html, result

    # Original vs corrected positions (same trip, first time)
    raw_html = wm.write_map(first_result, cfg, out_path=out / "map_raw.html", use_osm=False)
    screenshot(chrome, raw_html, out / "positions_before.png", "panel=0&view=all")
    screenshot(chrome, first_html, out / "positions_after.png", "panel=0&view=all&raw=1")
    site = (args.zoom_site or args.origin).zfill(4)
    screenshot(chrome, first_html, out / f"positions_zoom_{site}.png",
               f"panel=0&raw=1&traffic=0&focus={site}&zoom=17", size=(1200, 800))

    sites = load_site_coords(cfg)
    if sites is not None:
        sites.to_csv(out / "site_offsets.csv", index=False)
        print("\n" + summary(sites, load_edge_geometry(cfg)))
        print(f"\nOffsets per site: {out / 'site_offsets.csv'}")

if __name__ == "__main__":
    main()
