"""
experiments.py  -  the four routing experiments for the report   (Task 4 insights)
==================================================================================
Run from the project root:
    python -m src.Routing.experiments                 # all four, about 1 minute (GRU/LSTM load once)
    python -m src.Routing.experiments --only methods  # just one: methods | time | flow_mode | models

Every experiment uses Thursday 26 Oct 2006, which is in the TEST week (the last 7 days that no
model was trained on), so the predictions are genuine forecasts.

Writes to results/experiments/:
    1_search_methods.csv   6 Part A methods x 4 trips: route quality vs search effort
    2_time_of_day.csv      same trips at 6 departure times: how predicted traffic changes routes
    3_flow_mode.csv        per_detector vs total flow: why per_detector was chosen
    4_model_choice.csv     GRU / LSTM / Linear vs measured flow: does the model change the route?
    time_of_day.png        fastest-route time through the day, one line per trip
"""
from __future__ import annotations

import argparse
import time

import pandas as pd

import src.Models.deep_common as dc
from src.Routing import edge_costs as ec
from src.Routing import route_finder as rf
from src.Search import SEARCH_METHODS

DAY = "2006-10-26"                                        # Thursday, test week
TRIPS = [("2000", "3002"),                                # the brief's example (south-east -> west)
         ("4821", "2827"),                                # west -> north-east, longest trip
         ("0970", "4051"),                                # south -> north
         ("3001", "2200")]                                # west -> east
TIMES = ["03:00", "07:00", "08:00", "12:00", "17:00", "20:00"]


class CostCache:
    """edge_costs() once per (time, model, flow_mode); every search reuses the table."""

    def __init__(self, cfg: dict):
        self.cfg, self.tables = cfg, {}

    def get(self, when: str, model: str, flow_mode: str = None) -> pd.DataFrame:
        flow_mode = flow_mode or ec.settings(self.cfg)["flow_mode"]
        key = (when, model, flow_mode)
        if key not in self.tables:
            self.tables[key] = ec.edge_costs(when, model, rf.config_with(self.cfg, flow_mode=flow_mode))
        return self.tables[key]


def _route(o, d, when, model, cache, method="AS", flow_mode=None, k=5):
    cfg = rf.config_with(cache.cfg, flow_mode=flow_mode)
    return rf.find_routes(o, d, when, model, method, k, cfg, edge_table=cache.get(when, model, flow_mode))


# --------------------------------------------------------------------------
# 1. Search methods
# --------------------------------------------------------------------------
def search_methods(cache: CostCache, model: str) -> pd.DataFrame:
    """Same trip, same travel times: only the search method changes."""
    when, rows = f"{DAY} 08:00", []
    cache.get(when, model)                                        # predict first: time only the search
    for o, d in TRIPS:
        optimal = None
        for method, (_, name) in SEARCH_METHODS.items():
            t0 = time.perf_counter()
            res = _route(o, d, when, model, cache, method)
            ms = (time.perf_counter() - t0) * 1000
            mins = [r.minutes for r in res.routes]
            if method == "AS":
                optimal = mins                                    # A* + Yen = true 5 fastest
            rows.append({
                "trip": f"{o}->{d}", "method": method, "method_name": name,
                "routes_found": len(mins),
                "route1_min": round(mins[0], 2),
                "mean_of_5_min": round(sum(mins) / len(mins), 2),
                "extra_vs_astar_min": round(sum(mins) / len(mins) - sum(optimal) / len(optimal), 2),
                "nodes_created": res.nodes_created,
                "runtime_ms": round(ms, 1),
            })
    return pd.DataFrame(rows)


# --------------------------------------------------------------------------
# 2. Time of day
# --------------------------------------------------------------------------
def time_of_day(cache: CostCache, model: str) -> pd.DataFrame:
    rows = []
    for t in TIMES:
        when = f"{DAY} {t}"
        table = cache.get(when, model)
        for o, d in TRIPS:
            r = _route(o, d, when, model, cache).routes[0]
            rows.append({
                "time": t, "trip": f"{o}->{d}",
                "route1_min": round(r.minutes, 2), "route1_km": round(r.distance_km, 2),
                "route1_roads": " > ".join(r.roads),
                "network_mean_speed_kmh": round(table["speed_kmh"].mean(), 1),
                "network_mean_flow_vph": round(table["flow_vph"].mean(), 0),
            })
    df = pd.DataFrame(rows)
    # did the fastest route change compared with free-flowing traffic at 03:00?
    night = df[df["time"] == TIMES[0]].set_index("trip")["route1_roads"]
    df["route_changed_vs_0300"] = [roads != night[trip] for trip, roads in zip(df["trip"], df["route1_roads"])]
    return df


# --------------------------------------------------------------------------
# 3. Flow mode
# --------------------------------------------------------------------------
def flow_mode(cache: CostCache, model: str) -> pd.DataFrame:
    capacity_speed = float(cache.cfg["travel_time"].get("capacity_speed_kmh", 32))
    rows = []
    for t in ["03:00", "08:00", "12:00", "17:00"]:
        when = f"{DAY} {t}"
        for mode in ["per_detector", "total"]:
            table = cache.get(when, model, mode)
            for o, d in TRIPS:
                res = _route(o, d, when, model, cache, flow_mode=mode)
                rows.append({
                    "time": t, "flow_mode": mode, "trip": f"{o}->{d}",
                    "edges_at_capacity": int((table["speed_kmh"] <= capacity_speed + 1e-9).sum()),
                    "edges_total": len(table),
                    "network_mean_speed_kmh": round(table["speed_kmh"].mean(), 1),
                    "route1_min": round(res.routes[0].minutes, 2),
                    "route1_roads": " > ".join(res.routes[0].roads),
                    "spread_route1_to_5_min": round(res.routes[-1].minutes - res.routes[0].minutes, 2),
                })
    return pd.DataFrame(rows)


# --------------------------------------------------------------------------
# 4. Model choice
# --------------------------------------------------------------------------
def model_choice(cache: CostCache, models: list[str]) -> pd.DataFrame:
    """Each model's routes vs the routes from the MEASURED flow (the 'truth' for that time)."""
    rows = []
    for t in ["08:00", "17:00"]:
        when = f"{DAY} {t}"
        truth = cache.get(when, "actual").set_index(["from", "to"])
        for o, d in TRIPS:
            real = _route(o, d, when, "actual", cache).routes[0]
            for model in [*models, "actual"]:
                table = cache.get(when, model).set_index(["from", "to"])
                r = _route(o, d, when, model, cache).routes[0]
                # the chosen route's time if you actually drove it in the measured traffic
                real_time_of_choice = truth.loc[list(zip(r.path, r.path[1:])), "travel_time_s"].sum() / 60
                rows.append({
                    "time": t, "trip": f"{o}->{d}", "model": model,
                    "predicted_route1_min": round(r.minutes, 2),
                    "same_route_as_measured": r.path == real.path,
                    "chosen_route_in_real_traffic_min": round(real_time_of_choice, 2),
                    "best_possible_min": round(real.minutes, 2),
                    "edge_time_mae_s": round((table["travel_time_s"] - truth["travel_time_s"]).abs().mean(), 1),
                })
    return pd.DataFrame(rows)


def plot_time_of_day(df: pd.DataFrame, path) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(7, 4))
    for trip, g in df.groupby("trip"):
        ax.plot(g["time"], g["route1_min"], marker="o", label=trip)
    ax.set_xlabel(f"Departure time ({pd.Timestamp(DAY):%a %d %b %Y})")
    ax.set_ylabel("Fastest route (min)")
    ax.set_title("Predicted travel time of the fastest route through the day")
    ax.grid(alpha=0.3)
    ax.legend(title="Trip", fontsize=8)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


# --------------------------------------------------------------------------
EXPERIMENTS = ["methods", "time", "flow_mode", "models"]


def main() -> None:
    ap = argparse.ArgumentParser(description="Run the routing experiments for the report")
    ap.add_argument("--only", nargs="+", choices=EXPERIMENTS, default=EXPERIMENTS)
    ap.add_argument("--model", default=None, help="flow model for experiments 1-3 (default: config)")
    ap.add_argument("--config", default=None, help="path to config.yaml (default: project root)")
    args = ap.parse_args()

    cfg = dc.load_config(args.config)
    model = args.model or ec.settings(cfg)["default_model"]
    out = dc.get_path(cfg, "results_dir") / "experiments"
    out.mkdir(parents=True, exist_ok=True)
    cache = CostCache(cfg)

    def save(df: pd.DataFrame, name: str, title: str) -> None:
        df.to_csv(out / name, index=False)
        print(f"\n=== {title} ===  -> {out / name}")
        with pd.option_context("display.width", 200, "display.max_columns", None, "display.max_colwidth", 50):
            print(df.to_string(index=False))

    if "methods" in args.only:
        save(search_methods(cache, model), "1_search_methods.csv", f"1. Search methods (08:00, {model})")
    if "time" in args.only:
        df = time_of_day(cache, model)
        save(df, "2_time_of_day.csv", f"2. Time of day ({model}, A*)")
        plot_time_of_day(df, out / "time_of_day.png")
    if "flow_mode" in args.only:
        save(flow_mode(cache, model), "3_flow_mode.csv", f"3. Flow mode ({model}, A*)")
    if "models" in args.only:
        models = [m for m in ("gru", "lstm", "linear") if m in ec.MODEL_MODULES]
        save(model_choice(cache, models), "4_model_choice.csv", "4. Model choice (A*)")
    print(f"\nDone. Tables in {out}")


if __name__ == "__main__":
    main()
