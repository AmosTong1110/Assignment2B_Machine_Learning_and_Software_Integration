"""
plot_results.py  -  figures for the report (Insights / Testing sections)
=======================================================================
Run from the project root:      python -m src.Models.plot_results
(train_all.py also calls it at the end.)  Pictures are saved in Results/plots/.

    1_model_error_comparison.png   mean MAE and RMSE of every model
    2_mae_per_site_boxplot.png     spread of the MAE across sites
    3_actual_vs_predicted.png      real flow vs each model's prediction for one site
    4_training_time.png            how long each model took to train (all sites)
    5_learning_curves.png          LSTM vs GRU training / validation loss for one site
"""
from __future__ import annotations

import argparse

import matplotlib
matplotlib.use("Agg")                       # draw to files, no window needed
import matplotlib.pyplot as plt
import pandas as pd

import src.Models.deep_common as dc

COLORS = {"Persistence (baseline)": "#9e9e9e", "Linear Regression": "#1f77b4",
          "LSTM": "#d62728", "GRU": "#2ca02c"}
ORDER = list(COLORS)


def _read(path, **kw):
    return pd.read_csv(path, dtype={"scats_number": str}, **kw) if path.exists() else None


def _models_in(df):
    return [m for m in ORDER if m in set(df["model"])] + \
           [m for m in sorted(set(df["model"])) if m not in ORDER]


def main(cfg: dict = None) -> None:
    cfg = cfg or dc.load_config()
    results = dc.get_path(cfg, "results_dir")
    out = results / "plots"
    out.mkdir(parents=True, exist_ok=True)

    # ---------- load metrics (only sites that every model has -> fair) ----------
    frames = [_read(results / f) for f in ("baseline_metrics.csv", "linear_metrics.csv",
                                           "lstm_metrics.csv", "gru_metrics.csv")]
    metrics = pd.concat([f for f in frames if f is not None], ignore_index=True)
    common = set.intersection(*[set(g["scats_number"]) for _, g in metrics.groupby("model")])
    metrics = metrics[metrics["scats_number"].isin(common)]
    models = _models_in(metrics)
    colors = [COLORS.get(m, "#7f7f7f") for m in models]

    # ---------- 1. mean MAE / RMSE ----------
    mean = metrics.groupby("model")[["mae", "rmse"]].mean().reindex(models)
    ax = mean.plot.bar(figsize=(8, 5), color=["#4c72b0", "#dd8452"], rot=15)
    for c in ax.containers:
        ax.bar_label(c, fmt="%.1f", padding=2)
    ax.set_ylabel("error (vehicles per 15 min)")
    ax.set_xlabel("")
    ax.set_title(f"Mean prediction error on the last 7 days ({len(common)} sites, lower is better)")
    plt.tight_layout(); plt.savefig(out / "1_model_error_comparison.png", dpi=150); plt.close()

    # ---------- 2. MAE spread across sites ----------
    fig, ax = plt.subplots(figsize=(8, 5))
    bp = ax.boxplot([metrics.loc[metrics["model"] == m, "mae"] for m in models],
                    patch_artist=True)
    ax.set_xticks(range(1, len(models) + 1), labels=models)   # works on every matplotlib >= 3.5
    for patch, c in zip(bp["boxes"], colors):
        patch.set_facecolor(c); patch.set_alpha(0.7)
    ax.set_ylabel("MAE per site (vehicles per 15 min)")
    ax.set_title("How the error varies between SCATS sites")
    plt.xticks(rotation=15); plt.tight_layout()
    plt.savefig(out / "2_mae_per_site_boxplot.png", dpi=150); plt.close()

    # ---------- 3. actual vs predicted for one site ----------
    preds = [_read(results / f, parse_dates=["datetime"]) for f in
             ("linear_predictions.csv", "lstm_predictions.csv", "gru_predictions.csv")]
    preds = pd.concat([p for p in preds if p is not None], ignore_index=True)
    site = str(cfg["plots"]["site"])
    if site not in set(preds["scats_number"]):
        site = sorted(set(preds["scats_number"]))[0]
    one = preds[preds["scats_number"] == site]
    n = cfg["plots"]["days"] * 96
    fig, ax = plt.subplots(figsize=(11, 5))
    truth = one[one["model"] == one["model"].iloc[0]].iloc[:n]
    ax.plot(truth["datetime"], truth["y_true"], color="black", lw=2, label="Actual")
    for m in _models_in(one):
        d = one[one["model"] == m].iloc[:n]
        ax.plot(d["datetime"], d["y_pred"], color=COLORS.get(m), lw=1.3, label=m)
    ax.set_ylabel("flow (vehicles per 15 min)")
    ax.set_title(f"Actual vs predicted traffic flow - SCATS site {site}")
    ax.legend(); fig.autofmt_xdate(); plt.tight_layout()
    plt.savefig(out / "3_actual_vs_predicted.png", dpi=150); plt.close()

    # ---------- 4. training time ----------
    t = (metrics[metrics["model"] != "Persistence (baseline)"]
         .groupby("model")["train_seconds"].sum())
    fig, ax = plt.subplots(figsize=(7, 4.5))
    ax.bar(t.index, t.values, color=[COLORS.get(m, "#7f7f7f") for m in t.index])
    for i, v in enumerate(t.values):
        ax.text(i, v, f"{v:.1f}s", ha="center", va="bottom")
    ax.set_ylabel(f"total training time, {len(common)} sites (seconds)")
    ax.set_title("Training cost")
    plt.tight_layout(); plt.savefig(out / "4_training_time.png", dpi=150); plt.close()

    # ---------- 5. learning curves (LSTM vs GRU) ----------
    hist = [_read(results / f) for f in ("lstm_history.csv", "gru_history.csv")]
    hist = pd.concat([h for h in hist if h is not None], ignore_index=True) if any(
        h is not None for h in hist) else None
    if hist is not None:
        hs = hist[hist["scats_number"] == site]
        fig, ax = plt.subplots(figsize=(8, 5))
        for m in _models_in(hs):
            d = hs[hs["model"] == m]
            ax.plot(d["epoch"], d["loss"], color=COLORS.get(m), label=f"{m} training")
            ax.plot(d["epoch"], d["val_loss"], color=COLORS.get(m), ls="--", label=f"{m} validation")
        ax.set_xlabel("epoch"); ax.set_ylabel("loss (MSE, scaled 0-1)")
        ax.set_title(f"Learning curves - site {site}")
        ax.legend(); plt.tight_layout()
        plt.savefig(out / "5_learning_curves.png", dpi=150); plt.close()

    print(f"Saved plots to {out}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description="Draw report figures from Results/*.csv")
    ap.add_argument("--config", default=None)
    main(dc.load_config(ap.parse_args().config))