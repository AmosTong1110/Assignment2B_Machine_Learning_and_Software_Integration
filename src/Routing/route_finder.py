"""
route_finder.py  -  up to 5 routes from origin O to destination D   (Task 4)
===========================================================================
Ties everything together:

    ML flow prediction at time T  ->  edge_costs()  ->  travel time of every edge   (Task 2 + 3)
    edges + travel times          ->  Part A Problem  (nodes = SCATS sites, cost = seconds)
    Part A search + Yen's top-k   ->  up to k routes, fastest first                  (Task 4)

Run from the project root, e.g.:
    python -m src.Routing.route_finder --origin 2000 --destination 3002 --time "2006-10-26 08:00"
    python -m src.Routing.route_finder -o 2000 -d 3002 -t "2006-10-26 17:30" --model lstm --method BFS

Use from Python (the GUI does this):
    result = find_routes("2000", "3002", "2006-10-26 08:00", model="gru", method="AS", k=5)
    for r in result.routes:  print(r.rank, r.minutes, " -> ".join(r.path))
"""
from __future__ import annotations

import argparse
import copy
from dataclasses import dataclass
from functools import lru_cache

import pandas as pd

import src.Models.deep_common as dc
from src.Routing import edge_costs as ec
from src.Routing import travel_time as tt
from src.Routing.graph_builder import RAW_XLS, load_edges, load_nodes, load_site_roads
from src.Search import SEARCH_METHODS
from src.Search.problem import Problem
from src.Search.top_k import k_routes

MAX_ROUTES = 5          # the brief asks for "up to five (5) routes"


# --------------------------------------------------------------------------
# Results
# --------------------------------------------------------------------------
@dataclass
class Route:
    rank: int                  # 1 = fastest
    path: list[str]            # SCATS sites, origin first
    travel_time_s: float       # sum of the edge travel times (driving + 30 s per intersection)
    legs: pd.DataFrame         # one row per edge: from, to, road, distance_km, flow_vph, speed_kmh, travel_time_s

    @property
    def minutes(self) -> float:
        return self.travel_time_s / 60.0

    @property
    def distance_km(self) -> float:
        return float(self.legs["distance_km"].sum())

    @property
    def roads(self) -> list[str]:
        """Road names in driving order, with repeats merged: ['WARRIGAL_RD', 'TOORAK_RD', ...]."""
        out = []
        for road in self.legs["road"]:
            if not out or out[-1] != road:
                out.append(road)
        return out


@dataclass
class RouteResult:
    origin: str
    destination: str
    when: pd.Timestamp
    model: str
    method: str
    routes: list[Route]
    nodes_created: int         # search effort, summed over every Yen's iteration
    edge_table: pd.DataFrame   # edge_costs() output used for this search (for the map / report)


# --------------------------------------------------------------------------
# Helpers
# --------------------------------------------------------------------------
def config_with(cfg: dict = None, **travel_time) -> dict:
    """A copy of config.yaml with some travel_time: values replaced (e.g. flow_mode='total')."""
    cfg = copy.deepcopy(cfg or dc.load_config())
    cfg.setdefault("travel_time", {}).update({k: v for k, v in travel_time.items() if v is not None})
    return cfg


@lru_cache(maxsize=1)
def site_labels() -> dict[str, str]:
    """{'2000': 'BURWOOD_HWY / TOORAK_RD / WARRIGAL_RD', ...}  - road names at every site."""
    try:
        roads = load_site_roads(dc.ROOT / RAW_XLS)
    except (OSError, ValueError):
        return {}
    return roads.groupby("scats_number")["road"].apply(lambda r: " / ".join(sorted(r))).to_dict()


def build_problem(edge_table: pd.DataFrame, nodes: pd.DataFrame, origin: str, destination: str,
                  speed_limit_kmh: float = tt.SPEED_LIMIT_KMH) -> Problem:
    """
    Part A Problem for the Boroondara map:
        coordinates  {site: (lat, lon)}
        edges        {site: [(neighbour, travel_time_s), ...]}   from cost_lookup(edge_costs(...))
    The heuristic is the straight-line time at the speed limit, so A* stays optimal.
    """
    edges: dict[str, list[tuple[str, float]]] = {}
    for (a, b), seconds in ec.cost_lookup(edge_table).items():
        edges.setdefault(a, []).append((b, seconds))
    for nbrs in edges.values():
        nbrs.sort(key=lambda e: e[0])                       # same tie-breaking rule as Part A

    coordinates = {s: (lat, lon) for s, lat, lon in nodes[["scats_number", "latitude", "longitude"]]
                   .itertuples(index=False)}
    return Problem(coordinates, edges, origin, [destination],
                   distance=lambda p, q: tt.haversine_km(p[0], p[1], q[0], q[1]),
                   cost_per_distance=3600.0 / speed_limit_kmh)


def validate_sites(origin: str, destination: str, nodes: pd.DataFrame) -> tuple[str, str]:
    """'970' -> '0970'; raise a clear error for unknown or identical sites."""
    known = set(nodes["scats_number"])
    origin, destination = str(origin).strip().zfill(4), str(destination).strip().zfill(4)
    for name, site in (("Origin", origin), ("Destination", destination)):
        if site not in known:
            raise ValueError(f"{name} {site} is not a SCATS site on the map. Known sites: {', '.join(sorted(known))}")
    if origin == destination:
        raise ValueError("Origin and destination are the same site.")
    return origin, destination


# --------------------------------------------------------------------------
# Main entry point
# --------------------------------------------------------------------------
def find_routes(origin: str, destination: str, when, model: str = None, method: str = "AS",
                k: int = MAX_ROUTES, cfg: dict = None, edge_table: pd.DataFrame = None) -> RouteResult:
    """
    Up to k routes from origin to destination for a trip starting at `when`.

    model       linear | lstm | gru | actual   (default: config travel_time.default_model)
    method      AS | BFS | DFS | GBFS | CUS1 | CUS2   (Part A method codes)
    edge_table  reuse an edge_costs() table (skips the ML prediction - the GUI caches it)
    """
    cfg = cfg or dc.load_config()
    model = model or ec.settings(cfg)["default_model"]
    method = method.upper()
    if method not in SEARCH_METHODS:
        raise ValueError(f"method must be one of {list(SEARCH_METHODS)}, got {method!r}")
    if not 1 <= int(k) <= MAX_ROUTES:
        raise ValueError(f"number of routes must be between 1 and {MAX_ROUTES}, got {k}")

    nodes = load_nodes(cfg)
    origin, destination = validate_sites(origin, destination, nodes)
    when = pd.Timestamp(when).floor("15min")          # predictions are per 15-min slot

    if edge_table is None:
        edge_table = ec.edge_costs(when, model, cfg)
    params = tt.params_from_config(cfg)
    problem = build_problem(edge_table, nodes, origin, destination, params["speed_limit"])

    search, _ = SEARCH_METHODS[method]
    found, nodes_created = k_routes(problem, search, int(k))

    by_edge = edge_table.set_index(["from", "to"])
    leg_cols = ["road", "distance_km", "flow_vph", "speed_kmh", "travel_time_s"]
    routes = []
    for rank, (path, cost) in enumerate(found, start=1):
        legs = by_edge.loc[list(zip(path, path[1:])), leg_cols].reset_index()
        routes.append(Route(rank, path, float(cost), legs))
    return RouteResult(origin, destination, when, model, method, routes, nodes_created, edge_table)


def format_result(result: RouteResult) -> str:
    """Plain-text summary (used by the command line)."""
    labels = site_labels()
    lines = [f"{result.origin} ({labels.get(result.origin, '?')})  ->  "
             f"{result.destination} ({labels.get(result.destination, '?')})",
             f"Departure {result.when:%a %d %b %Y %H:%M}, flow model {result.model}, "
             f"search {SEARCH_METHODS[result.method][1]}, {result.nodes_created} nodes created", ""]
    if not result.routes:
        lines.append("No route found.")
    for r in result.routes:
        lines.append(f"Route {r.rank}: {r.minutes:5.1f} min  {r.distance_km:5.2f} km  "
                     f"{len(r.path) - 1} intersections")
        lines.append("   " + " -> ".join(r.path))
        lines.append("   via " + ", ".join(r.roads))
    return "\n".join(lines)


def main() -> None:
    ap = argparse.ArgumentParser(description="Find up to 5 routes between two SCATS sites")
    ap.add_argument("-o", "--origin", required=True, help="origin SCATS site, e.g. 2000")
    ap.add_argument("-d", "--destination", required=True, help="destination SCATS site, e.g. 3002")
    ap.add_argument("-t", "--time", required=True, help='departure time in Oct 2006, e.g. "2006-10-26 08:00"')
    ap.add_argument("--model", default=None, choices=ec.FLOW_SOURCES, help="flow model (default: config)")
    ap.add_argument("--method", default="AS", choices=list(SEARCH_METHODS), help="Part A search method")
    ap.add_argument("-k", type=int, default=MAX_ROUTES, help="number of routes (1-5)")
    ap.add_argument("--config", default=None, help="path to config.yaml (default: project root)")
    args = ap.parse_args()

    result = find_routes(args.origin, args.destination, args.time, args.model, args.method,
                         args.k, dc.load_config(args.config))
    print(format_result(result))


if __name__ == "__main__":
    main()
