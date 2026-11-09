"""
evaluate.py  -  SHARED by every model (Linear, LSTM, GRU, ...)
==============================================================
One referee, one stopwatch: every model is scored on EXACTLY the same test
rows, with the SAME metrics, in the SAME units, and results are written in the
SAME file format.  That is what makes the model comparison fair.

Rules (do not change them without telling the team):
  * Data         : data/processed/scats_clean_15min.csv  (from data_processing.py)
  * Test set     : the LAST 7 days of every site (672 rows of 15 minutes)
  * Sites        : all sites except those marked low_quality in site_metadata.csv
  * Metrics      : MAE and RMSE in VEHICLES PER 15 MINUTES (not scaled 0-1!)
  * Scored on    : real measurements only (imputed == 0)
  * Result file  : columns  model, scats_number, mae, rmse, n_eval, train_seconds

How a model owner uses this file (example for a PyTorch model):

    import evaluate as ev
    clean, meta = dp.load_clean("data/processed")
    rows = []
    for site in ev.get_eval_sites(meta):
        truth = ev.get_test_truth(clean, site)       # the answer key (672 rows)
        y_pred = ...your model predictions, 672 numbers, in vehicles/15min...
        rows.append(ev.evaluate_site("LSTM", site, y_pred, truth, train_seconds))
    ev.save_results(rows, "results/lstm_metrics.csv")

Compare all models:
    python src/models/evaluate.py results/linear_metrics.csv results/lstm_metrics.csv \
                                  results/gru_metrics.csv results/baseline_metrics.csv
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

SLOTS_PER_DAY = 96
TEST_DAYS = 7
TEST_SIZE = TEST_DAYS * SLOTS_PER_DAY          # 672 test rows per site
RESULT_COLUMNS = ["model", "scats_number", "mae", "rmse", "n_eval", "train_seconds"]


# --------------------------------------------------------------------------
# The shared "answer key"
# --------------------------------------------------------------------------
def get_eval_sites(meta: pd.DataFrame) -> list[str]:
    """Sites every model is evaluated on (low-quality sites are excluded)."""
    return meta.loc[~meta["low_quality"], "scats_number"].tolist()


def get_test_truth(clean: pd.DataFrame, site: str) -> pd.DataFrame:
    """
    The last 7 days of one site = the test set (672 rows).
    Columns: datetime, y_true (vehicles/15min), imputed (0 real / 1 repaired),
             prev_flow (flow one step earlier -> used by the naive baseline).
    Your predictions must be in the SAME ORDER as these rows.
    """
    g = clean[clean["scats_number"] == site].sort_values("datetime").reset_index(drop=True)
    g["prev_flow"] = g["flow"].shift(1)
    truth = g.iloc[-TEST_SIZE:][["datetime", "flow", "imputed", "prev_flow"]]
    truth = truth.rename(columns={"flow": "y_true"}).reset_index(drop=True)
    if len(truth) != TEST_SIZE:
        raise ValueError(f"Site {site}: expected {TEST_SIZE} test rows, got {len(truth)}")
    return truth


# --------------------------------------------------------------------------
# Metrics
# --------------------------------------------------------------------------
def mae(y_true, y_pred) -> float:
    """Mean Absolute Error: the average miss, in vehicles per 15 minutes."""
    return float(np.mean(np.abs(np.asarray(y_true) - np.asarray(y_pred))))


def rmse(y_true, y_pred) -> float:
    """Root Mean Squared Error: like MAE but punishes big misses more."""
    return float(np.sqrt(np.mean((np.asarray(y_true) - np.asarray(y_pred)) ** 2)))


def evaluate_site(model: str, site: str, y_pred, truth: pd.DataFrame,
                  train_seconds: float = np.nan) -> dict:
    """
    Score ONE site. Returns one result row (a dict) to collect in a list.
    y_pred : 672 predictions, vehicles per 15 min (inverse-scaled!), same order as truth.
    """
    y_pred = np.asarray(y_pred, dtype=float).reshape(-1)
    y_true = truth["y_true"].to_numpy()

    if len(y_pred) != len(y_true):
        raise ValueError(f"{model}/{site}: {len(y_pred)} predictions but {len(y_true)} test rows")
    if np.isnan(y_pred).any():
        raise ValueError(f"{model}/{site}: predictions contain NaN")
    if y_true.max() > 20 and np.abs(y_pred).max() <= 2:
        raise ValueError(f"{model}/{site}: predictions look SCALED (0-1). "
                         "Use scaler.inverse(...) to convert back to vehicles first.")

    keep = truth["imputed"].to_numpy() == 0          # score real measurements only
    if keep.sum() == 0:
        raise ValueError(f"{model}/{site}: no real (non-imputed) test rows to score")

    return {
        "model": model,
        "scats_number": site,
        "mae": mae(y_true[keep], y_pred[keep]),
        "rmse": rmse(y_true[keep], y_pred[keep]),
        "n_eval": int(keep.sum()),
        "train_seconds": float(train_seconds),
    }


def persistence_results(clean: pd.DataFrame, sites: list[str]) -> list[dict]:
    """Naive baseline: 'next value = last value'. Every model should beat this."""
    rows = []
    for site in sites:
        truth = get_test_truth(clean, site)
        rows.append(evaluate_site("Persistence (baseline)", site, truth["prev_flow"], truth, 0.0))
    return rows


# --------------------------------------------------------------------------
# Save / compare
# --------------------------------------------------------------------------
def save_results(rows: list[dict], path: str) -> pd.DataFrame:
    """Write result rows to CSV in the shared format."""
    df = pd.DataFrame(rows, columns=RESULT_COLUMNS).round(4)
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path, index=False)
    return df


def compare(paths: list[str]) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Merge several result files into one comparison.
    Only sites that EVERY model has results for are used (fair comparison).
    Returns (summary per model, per-site table of MAE).
    """
    df = pd.concat([pd.read_csv(p, dtype={"scats_number": str}) for p in paths], ignore_index=True)
    common = set.intersection(*[set(g["scats_number"]) for _, g in df.groupby("model")])
    df = df[df["scats_number"].isin(common)]

    summary = (df.groupby("model")
                 .agg(sites=("scats_number", "nunique"), mae=("mae", "mean"),
                      rmse=("rmse", "mean"), train_seconds=("train_seconds", "sum"))
                 .sort_values("mae").round(2))
    per_site = df.pivot(index="scats_number", columns="model", values="mae").round(2)
    summary["best_on_n_sites"] = per_site.idxmin(axis=1).value_counts().reindex(summary.index).fillna(0).astype(int)
    return summary, per_site


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description="Compare model result CSV files")
    ap.add_argument("files", nargs="+", help="result CSVs, e.g. results/linear_metrics.csv ...")
    ap.add_argument("--out", default="results/model_comparison.csv")
    args = ap.parse_args()
    summary, per_site = compare(args.files)
    print(summary.to_string())
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    summary.to_csv(args.out)
    per_site.to_csv(Path(args.out).with_name("per_site_mae.csv"))
    print(f"\nSaved {args.out} and per_site_mae.csv")
