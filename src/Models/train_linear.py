"""
train_linear.py  -  train + evaluate Linear Regression for every site
======================================================================
Run from the project root:      python -m src.Models.train_linear

Paths come from config.yaml (processed_dir, models_dir, results_dir).

Outputs
    saved_models/linear/<site>.joblib     one trained model per site
    results/linear_metrics.csv            MAE / RMSE per site (shared format)
    results/baseline_metrics.csv          "next = last value" baseline (shared format)
    results/linear_predictions.csv        predictions vs truth (for plots in the report)
"""
from __future__ import annotations

import argparse
import time

import pandas as pd

import src.Models.data_processing as dp
import src.Models.deep_common as dc
import src.Models.evaluate as ev
import src.Models.linear_regression_model as lm

MODEL_NAME = "Linear Regression"


def main(processed_dir=None, models_dir=None, results_dir=None, cfg: dict = None) -> pd.DataFrame:
    """Train + evaluate on every evaluation site. Any folder left as None comes from config.yaml."""
    cfg = cfg or dc.load_config()
    processed_dir = processed_dir or dc.get_path(cfg, "processed_dir")
    models_dir = models_dir or dc.get_path(cfg, "models_dir") / "linear"
    results_dir = results_dir or dc.get_path(cfg, "results_dir")
    lags = cfg["data"]["lags"]

    clean, meta = dp.load_clean(str(processed_dir))
    sites = ev.get_eval_sites(meta)
    print(f"Training {MODEL_NAME} on {len(sites)} sites "
          f"(skipping low-quality: {sorted(set(meta.scats_number) - set(sites))})")

    rows, preds = [], []
    for site in sites:
        g = clean[clean["scats_number"] == site]
        X, y, times, _ = lm.build_features(g, lags)

        # time-based split: the last 672 rows are the test set (same as evaluate.py)
        X_tr, y_tr = X[:-ev.TEST_SIZE], y[:-ev.TEST_SIZE]
        X_te = X[-ev.TEST_SIZE:]

        t0 = time.time()
        model = lm.fit(X_tr, y_tr)
        seconds = time.time() - t0

        y_pred = model.predict(X_te).clip(min=0)           # flow cannot be negative
        truth = ev.get_test_truth(clean, site)
        assert (times.iloc[-ev.TEST_SIZE:].to_numpy() == truth["datetime"].to_numpy()).all(), \
            "test rows are not aligned with the shared answer key"

        rows.append(ev.evaluate_site(MODEL_NAME, site, y_pred, truth, seconds))
        lm.save_model(model, site, models_dir)
        preds.append(pd.DataFrame({"model": MODEL_NAME, "scats_number": site,
                                   "datetime": truth["datetime"], "y_true": truth["y_true"],
                                   "y_pred": y_pred.round(2), "imputed": truth["imputed"]}))

    results = ev.save_results(rows, results_dir / "linear_metrics.csv")
    ev.save_results(ev.persistence_results(clean, sites), results_dir / "baseline_metrics.csv")
    pd.concat(preds).to_csv(results_dir / "linear_predictions.csv", index=False)

    summary, _ = ev.compare([str(results_dir / "linear_metrics.csv"),
                             str(results_dir / "baseline_metrics.csv")])
    print("\n" + summary.to_string())
    print(f"\nSaved models to {models_dir} and results to {results_dir}")
    return results


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=f"Train and evaluate {MODEL_NAME} for every SCATS site")
    ap.add_argument("--config", default=None, help="path to config.yaml (default: project root)")
    main(cfg=dc.load_config(ap.parse_args().config))