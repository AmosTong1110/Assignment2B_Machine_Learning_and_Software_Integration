"""
edge_costs.py  -  travel time of EVERY edge at a given time, from ML predictions  (Task 3)
=========================================================================================
Glue between Task 2 (flow prediction) and Task 4 (route search):

    edges.csv  +  predicted flow at time T  ->  travel time (s) of every edge A->B

Run from the project root, e.g.:
    python -m src.Routing.edge_costs --time "2006-10-26 08:00" --model gru
    python -m src.Routing.edge_costs --time "2006-10-26 08:00" --model lstm --save

Use from Python (Task 4):
    from src.Routing.edge_costs import edge_costs, cost_lookup
    table = edge_costs("2006-10-26 08:00", model="gru")   # DataFrame, one row per edge
    costs = cost_lookup(table)                            # {('2000', '3682'): 151.3, ...}

Steps for time T
    1. predict the hourly flow at every SCATS site ONCE (38 models, not one per edge)
    2. for every edge A->B pick the flow of the site set by config (flow_site: origin|destination)
    3. optionally turn the intersection total into a per-approach flow (flow_mode)
    4. travel time = distance / speed(flow) * 3600 + 30 s          (travel_time.py)

Sites without a trained model (low-quality sites, e.g. 3685 and 4264) fall back to the
historical average flow for the same weekday and 15-minute slot.  The 'source' column
says which flows were predicted and which were fallbacks.
"""
from __future__ import annotations

import argparse
import importlib
from functools import lru_cache

import pandas as pd

import src.Models.deep_common as dc
from src.Routing import travel_time as tt
from src.Routing.graph_builder import load_edges

# model name -> module with predict_hourly(site, when)   (same interface for every model)
MODEL_MODULES = {
    "linear": "src.Models.linear_regression_model",
    "lstm": "src.Models.lstm_model",
    "gru": "src.Models.gru_model",
}
# 'actual' is not a model: it uses the measured flow (handy for testing / comparing)
FLOW_SOURCES = [*MODEL_MODULES, "actual"]

DEFAULTS = {"flow_site": "origin",          # origin | destination  - whose flow sets the speed
            "flow_mode": "per_detector",    # per_detector | total
            "default_model": "gru"}


def settings(cfg: dict) -> dict:
    """config.yaml travel_time: section merged over DEFAULTS."""
    return {**DEFAULTS, **(cfg.get("travel_time") or {})}


# --------------------------------------------------------------------------
# Data helpers (cached: they are read once per program run)
# --------------------------------------------------------------------------
@lru_cache(maxsize=1)
def _clean(processed_dir: str) -> pd.DataFrame:
    return pd.read_csv(f"{processed_dir}/scats_clean_15min.csv",
                       dtype={"scats_number": str}, parse_dates=["datetime"])


@lru_cache(maxsize=1)
def _metadata(processed_dir: str) -> pd.DataFrame:
    return pd.read_csv(f"{processed_dir}/site_metadata.csv",
                       dtype={"scats_number": str}).set_index("scats_number")


def _has_model(model: str, site: str, cfg: dict) -> bool:
    folder = dc.get_path(cfg, "models_dir") / model
    return any(folder.glob(f"{site}.*"))                     # .keras or .joblib


def historical_flow_hourly(site: str, when: pd.Timestamp, clean: pd.DataFrame) -> float:
    """Average flow (veh/h) at `site` for the same weekday and 15-min slot as `when`."""
    g = clean[clean["scats_number"] == site]
    same = g[(g["datetime"].dt.dayofweek == when.dayofweek) & (g["datetime"].dt.time == when.time())]
    if same.empty:
        raise ValueError(f"No data for site {site} at {when:%a %H:%M}")
    return 4.0 * float(same["flow"].mean())


# --------------------------------------------------------------------------
# 1. Flow at every site
# --------------------------------------------------------------------------
def site_flows(when, model: str = "gru", sites=None, cfg: dict = None) -> pd.DataFrame:
    """
    Hourly flow at each site for the 15-min interval starting at `when`.
    Returns a DataFrame indexed by scats_number with columns  flow_vph, source.
    """
    cfg = cfg or dc.load_config()
    if model not in FLOW_SOURCES:
        raise ValueError(f"model must be one of {FLOW_SOURCES}, got {model!r}")
    when = pd.Timestamp(when)
    processed = str(dc.get_path(cfg, "processed_dir"))
    clean = _clean(processed)
    sites = sorted(sites if sites is not None else _metadata(processed).index)
    module = importlib.import_module(MODEL_MODULES[model]) if model in MODEL_MODULES else None

    rows = {}
    for site in sites:
        if model == "actual":
            hit = clean[(clean["scats_number"] == site) & (clean["datetime"] == when)]
            if hit.empty:
                raise ValueError(f"No measured flow for site {site} at {when}")
            rows[site] = (4.0 * float(hit["flow"].iloc[0]), "actual")
        elif _has_model(model, site, cfg):
            rows[site] = (module.predict_hourly(site, when), model)
        else:
            rows[site] = (historical_flow_hourly(site, when, clean), "historical_avg")
    out = pd.DataFrame.from_dict(rows, orient="index", columns=["flow_vph", "source"])
    out.index.name = "scats_number"
    return out


def link_flow(site_flow_vph: float, n_detectors: int, flow_mode: str) -> float:
    """
    The SCATS flow of a site is the SUM over all its detectors (all approaches).
    'per_detector' divides by the number of detectors -> a rough flow for ONE road link,
    which is what the 1500 veh/h capacity in the conversion PDF is meant for.
    'total' uses the intersection total as-is (most busy sites are then 'at capacity').
    """
    if flow_mode == "total":
        return site_flow_vph
    if flow_mode == "per_detector":
        return site_flow_vph / max(int(n_detectors), 1)
    raise ValueError(f"flow_mode must be 'per_detector' or 'total', got {flow_mode!r}")


# --------------------------------------------------------------------------
# 2. Travel time of every edge
# --------------------------------------------------------------------------
def edge_costs(when, model: str = None, cfg: dict = None, edges: pd.DataFrame = None) -> pd.DataFrame:
    """
    One row per directed edge:
        from, to, road, distance_km, flow_site, flow_vph, speed_kmh, travel_time_s, source
    """
    cfg = cfg or dc.load_config()
    s = settings(cfg)
    model = model or s["default_model"]
    if s["flow_site"] not in ("origin", "destination"):
        raise ValueError(f"flow_site must be 'origin' or 'destination', got {s['flow_site']!r}")
    edges = load_edges(cfg) if edges is None else edges
    params = tt.params_from_config(cfg)
    meta = _metadata(str(dc.get_path(cfg, "processed_dir")))

    flow_col = "from" if s["flow_site"] == "origin" else "to"
    flows = site_flows(when, model, sites=set(edges[flow_col]), cfg=cfg)

    out = edges.copy()
    out["flow_site"] = out[flow_col]
    out["flow_vph"] = [link_flow(flows.at[site, "flow_vph"], meta.at[site, "n_detectors"], s["flow_mode"])
                       for site in out["flow_site"]]
    out["speed_kmh"] = [tt.flow_to_speed(f, params["speed_limit"], params["capacity_flow"],
                                         params["capacity_speed"]) for f in out["flow_vph"]]
    out["travel_time_s"] = [tt.edge_travel_time_s(d, f, **params)
                            for d, f in zip(out["distance_km"], out["flow_vph"])]
    out["source"] = [flows.at[site, "source"] for site in out["flow_site"]]
    return out.round({"flow_vph": 1, "speed_kmh": 2, "travel_time_s": 1})


def cost_lookup(table: pd.DataFrame) -> dict[tuple[str, str], float]:
    """{(from, to): travel_time_s}  - the edge cost dictionary for the Task 4 search."""
    return {(a, b): float(t) for a, b, t in zip(table["from"], table["to"], table["travel_time_s"])}


# --------------------------------------------------------------------------
def main() -> None:
    ap = argparse.ArgumentParser(description="Travel time of every edge at a given time")
    ap.add_argument("--time", required=True, help='start of a 15-min slot in Oct 2006, e.g. "2006-10-26 08:00"')
    ap.add_argument("--model", default=None, choices=FLOW_SOURCES, help="flow source (default: config)")
    ap.add_argument("--config", default=None, help="path to config.yaml (default: project root)")
    ap.add_argument("--save", action="store_true", help="also write Results/edge_costs_<model>.csv")
    args = ap.parse_args()

    cfg = dc.load_config(args.config)
    model = args.model or settings(cfg)["default_model"]
    table = edge_costs(args.time, model, cfg)
    with pd.option_context("display.max_rows", None, "display.width", 140):
        print(table.to_string(index=False))
    print(f"\n{len(table)} edges at {args.time} ({model}): mean speed {table.speed_kmh.mean():.1f} km/h, "
          f"mean edge time {table.travel_time_s.mean():.0f} s, fallback flows: "
          f"{sorted(set(table.loc[table.source == 'historical_avg', 'flow_site']))}")
    if args.save:
        out = dc.get_path(cfg, "results_dir") / f"edge_costs_{model}.csv"
        table.to_csv(out, index=False)
        print("Saved ->", out)


if __name__ == "__main__":
    main()
