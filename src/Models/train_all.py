"""
train_all.py  -  train every model, compare them, and draw the plots (one command)
=================================================================================
Run from the project root:      python Source/Models/train_all.py

    python Source/Models/train_all.py --only gru            # just one model
    python Source/Models/train_all.py --sites 2000 3002 --epochs 5   # quick smoke test
    python Source/Models/train_all.py --skip-training       # only compare + plot existing results

Steps
    1. Linear Regression (+ the 'next = last value' baseline)   -> train_linear.py
    2. LSTM                                                     -> lstm_model.py
    3. GRU                                                      -> gru_model.py
    4. Compare all models (same test rows, same metrics)        -> Results/model_comparison.csv
    5. Plots for the report                                     -> Results/plots/*.png
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
for _p in (ROOT, ROOT / "src", ROOT / "src" / "Models", ROOT / "Source", ROOT / "Source" / "Models"):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

try:
    import src.Models.deep_common as dc
    import src.Models.evaluate as ev
    import src.Models.gru_model as gru
    import src.Models.lstm_model as lstm
    import src.Models.plot_results as plots
    import src.Models.train_linear as train_linear
except ImportError:
    import deep_common as dc
    import evaluate as ev
    import gru_model as gru
    import lstm_model as lstm
    import plot_results as plots
    import train_linear as train_linear

MODELS = ["linear", "lstm", "gru"]


def main() -> None:
    ap = argparse.ArgumentParser(description="Train, compare and plot all models")
    ap.add_argument("--config", default=None, help="path to config.yaml")
    ap.add_argument("--only", choices=MODELS, nargs="+", default=MODELS, help="models to train")
    ap.add_argument("--sites", nargs="+", default=None, help="only these sites (LSTM/GRU quick test)")
    ap.add_argument("--epochs", type=int, default=None, help="override max epochs (LSTM/GRU)")
    ap.add_argument("--skip-training", action="store_true", help="only compare + plot")
    args = ap.parse_args()

    cfg = dc.load_config(args.config)
    if args.epochs:
        cfg["training"]["epochs"] = args.epochs
    results = dc.get_path(cfg, "results_dir")

    if not args.skip_training:
        if "linear" in args.only:
            print("\n=== Linear Regression ===")
            train_linear.main(processed_dir=dc.get_path(cfg, "processed_dir"),
                              models_dir=dc.get_path(cfg, "models_dir") / "linear",
                              results_dir=results)
        if "lstm" in args.only:
            print("\n=== LSTM ===")
            lstm.train(cfg, args.sites)
        if "gru" in args.only:
            print("\n=== GRU ===")
            gru.train(cfg, args.sites)

    # ---- compare every result file that exists ----
    files = [results / f for f in ("baseline_metrics.csv", "linear_metrics.csv",
                                   "lstm_metrics.csv", "gru_metrics.csv") if (results / f).exists()]
    if len(files) < 2:
        print("Not enough result files to compare yet.")
        return
    summary, per_site = ev.compare([str(f) for f in files])
    summary.to_csv(results / "model_comparison.csv")
    per_site.to_csv(results / "per_site_mae.csv")
    print("\n=== Model comparison (MAE/RMSE in vehicles per 15 min, lower is better) ===")
    print(summary.to_string())

    print("\n=== Plots ===")
    plots.main(cfg)


if __name__ == "__main__":
    main()
