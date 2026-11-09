"""
deep_common.py  -  shared code for the deep-learning models (LSTM and GRU)
==========================================================================
LSTM and GRU are almost the same network, so everything they share lives here
and lstm_model.py / gru_model.py are only thin wrappers.  This keeps the
comparison between them fair: the ONLY difference is the recurrent layer.

What the network sees (the same information as the Linear Regression model):
    * seq : the last 12 flow values (3 hours), scaled to 0-1      shape (12, 1)
    * aux : time of day as sin/cos + weekend flag of the target   shape (3,)
Network:  seq -> LSTM/GRU -> (+ aux) -> Dense(32, relu) -> Dense(1)  = next 15-min flow

Fair-evaluation rules (same as evaluate.py / train_linear.py):
    * one model per SCATS site
    * time-based split: the LAST 672 rows (7 days) are the test set
    * the scaler is fitted on TRAINING data only (no peeking at the test set)
    * predictions are converted back to vehicles per 15 min before scoring
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from functools import lru_cache
from pathlib import Path

os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")      # hide TensorFlow info spam

import numpy as np
import pandas as pd
import yaml

ROOT = Path(__file__).resolve().parents[2]              # project root
for _p in (ROOT, ROOT / "Source"):                      # so the imports below work
    if str(_p) not in sys.path:                         # from any working directory
        sys.path.insert(0, str(_p))

import data_processing as dp                            # noqa: E402
import Source.Models.evaluate as ev                     # noqa: E402

SLOTS_PER_DAY = 96

# --------------------------------------------------------------------------
# Config
# --------------------------------------------------------------------------
DEFAULTS = {
    "paths": {"processed_dir": "Data/Processed_Data", "models_dir": "saved_models",
              "results_dir": "Results"},
    "data": {"lags": 12},
    "training": {"seed": 42, "epochs": 60, "batch_size": 32, "learning_rate": 0.001,
                 "val_days": 3, "patience": 8},
    "model": {"units": 64, "dense_units": 32, "dropout": 0.1},
    "plots": {"site": "2000", "days": 3},
}


def load_config(path=None) -> dict:
    """Read config.yaml and fill any missing value with the default above."""
    cfg = {k: dict(v) for k, v in DEFAULTS.items()}
    path = Path(path) if path else ROOT / "config.yaml"
    if path.exists():
        user = yaml.safe_load(path.read_text()) or {}
        for section, values in user.items():
            cfg.setdefault(section, {}).update(values or {})
    return cfg


def get_path(cfg: dict, name: str) -> Path:
    """Absolute path of one entry in cfg['paths'] (relative to the project root)."""
    p = Path(cfg["paths"][name])
    return p if p.is_absolute() else ROOT / p


# --------------------------------------------------------------------------
# Features and scaling
# --------------------------------------------------------------------------
def time_features(times: pd.Series) -> np.ndarray:
    """sin/cos of the time of day + weekend flag, shape (n, 3). (Same as the linear model.)"""
    t = pd.to_datetime(times)
    slot = (t.dt.hour * 4 + t.dt.minute // 15).to_numpy()
    angle = 2 * np.pi * slot / SLOTS_PER_DAY
    weekend = (t.dt.dayofweek >= 5).to_numpy().astype(float)
    return np.column_stack([np.sin(angle), np.cos(angle), weekend])


def build_windows(site_df: pd.DataFrame, lags: int):
    """
    Turn one site's time series into supervised samples.
    Sample i uses the `lags` values BEFORE time[i] to predict flow[i].
    Returns seq (n, lags), aux (n, 3), y (n,), times (n,), flow (all values, sorted).
    """
    g = site_df.sort_values("datetime").reset_index(drop=True)
    v = g["flow"].to_numpy(dtype=float)
    seq = np.stack([v[i - lags:i] for i in range(lags, len(v))])
    aux = time_features(g["datetime"].iloc[lags:])
    return seq, aux, v[lags:], g["datetime"].iloc[lags:].reset_index(drop=True), v


def fit_scaler(train_flow: np.ndarray) -> dict:
    """Min-max scaler learned from TRAINING flow only."""
    return {"min": float(np.min(train_flow)), "max": float(np.max(train_flow))}


def scale(x, sc: dict):
    return (np.asarray(x, dtype=float) - sc["min"]) / max(sc["max"] - sc["min"], 1e-9)


def unscale(x, sc: dict):
    return np.asarray(x, dtype=float) * (sc["max"] - sc["min"]) + sc["min"]


# --------------------------------------------------------------------------
# The network
# --------------------------------------------------------------------------
def build_model(layer_name: str, cfg: dict):
    """layer_name is 'LSTM' or 'GRU' - the only difference between the two models."""
    from tensorflow import keras
    lags, m, t = cfg["data"]["lags"], cfg["model"], cfg["training"]
    rnn = getattr(keras.layers, layer_name)

    seq_in = keras.Input(shape=(lags, 1), name="seq")
    aux_in = keras.Input(shape=(3,), name="aux")
    x = rnn(m["units"])(seq_in)
    if m["dropout"]:
        x = keras.layers.Dropout(m["dropout"])(x)
    x = keras.layers.Concatenate()([x, aux_in])
    x = keras.layers.Dense(m["dense_units"], activation="relu")(x)
    out = keras.layers.Dense(1)(x)

    model = keras.Model([seq_in, aux_in], out, name=layer_name.lower())
    model.compile(optimizer=keras.optimizers.Adam(t["learning_rate"]), loss="mse")
    return model


def fit(layer_name: str, seq, aux, y, cfg: dict):
    """
    Train on (seq, aux, y) - all already scaled / in time order.
    The last `val_days` of the data are held out for early stopping.
    Returns (model, history dict).
    """
    from tensorflow import keras
    t = cfg["training"]
    keras.utils.set_random_seed(t["seed"])              # reproducible results
    model = build_model(layer_name, cfg)

    n_val = t["val_days"] * SLOTS_PER_DAY
    tr, va = slice(0, len(y) - n_val), slice(len(y) - n_val, len(y))
    stop = keras.callbacks.EarlyStopping(monitor="val_loss", patience=t["patience"],
                                         restore_best_weights=True)
    hist = model.fit([seq[tr], aux[tr]], y[tr],
                     validation_data=([seq[va], aux[va]], y[va]),
                     epochs=t["epochs"], batch_size=t["batch_size"],
                     shuffle=True, callbacks=[stop], verbose=0)
    return model, hist.history


# --------------------------------------------------------------------------
# Save / load
# --------------------------------------------------------------------------
def save_model(model, scaler: dict, key: str, site: str, models_dir) -> Path:
    path = Path(models_dir) / key / f"{site}.keras"
    path.parent.mkdir(parents=True, exist_ok=True)
    model.save(path)
    (path.parent / f"{site}_scaler.json").write_text(json.dumps(scaler))
    return path


@lru_cache(maxsize=64)
def _load_cached(key: str, site: str, models_dir: str):
    from tensorflow import keras
    folder = Path(models_dir) / key
    model = keras.models.load_model(folder / f"{site}.keras")
    scaler = json.loads((folder / f"{site}_scaler.json").read_text())
    return model, scaler


def load_model(key: str, site: str, models_dir):
    """Returns (keras model, scaler dict)."""
    return _load_cached(key, site, str(models_dir))


# --------------------------------------------------------------------------
# Training loop for all sites (called by train_lstm.py / train_all.py)
# --------------------------------------------------------------------------
def train_deep(name: str, key: str, layer_name: str, cfg: dict = None,
               sites: list[str] = None) -> pd.DataFrame:
    """
    Train + evaluate one model type on every evaluation site.
    Writes  <results>/<key>_metrics.csv, <key>_predictions.csv, <key>_history.csv
    and     <models_dir>/<key>/<site>.keras (+ scaler).
    """
    cfg = cfg or load_config()
    lags = cfg["data"]["lags"]
    results_dir, models_dir = get_path(cfg, "results_dir"), get_path(cfg, "models_dir")

    clean, meta = dp.load_clean(str(get_path(cfg, "processed_dir")))
    all_sites = ev.get_eval_sites(meta)
    sites = [s for s in all_sites if sites is None or s in sites]
    print(f"Training {name} on {len(sites)} sites "
          f"(epochs<={cfg['training']['epochs']}, units={cfg['model']['units']})")

    rows, preds, history = [], [], []
    for n, site in enumerate(sites, 1):
        g = clean[clean["scats_number"] == site]
        seq, aux, y, times, flow = build_windows(g, lags)
        n_test = ev.TEST_SIZE

        sc = fit_scaler(flow[:-n_test])                  # training data only
        seq_s, y_s = scale(seq, sc)[..., None], scale(y, sc)
        X_tr, A_tr, y_tr = seq_s[:-n_test], aux[:-n_test], y_s[:-n_test]
        X_te, A_te = seq_s[-n_test:], aux[-n_test:]

        t0 = time.time()
        model, h = fit(layer_name, X_tr, A_tr, y_tr, cfg)
        seconds = time.time() - t0

        y_pred = unscale(model.predict([X_te, A_te], verbose=0).reshape(-1), sc).clip(min=0)
        truth = ev.get_test_truth(clean, site)
        assert (times.iloc[-n_test:].to_numpy() == truth["datetime"].to_numpy()).all(), \
            "test rows are not aligned with the shared answer key"

        row = ev.evaluate_site(name, site, y_pred, truth, seconds)
        rows.append(row)
        save_model(model, sc, key, site, models_dir)
        preds.append(pd.DataFrame({"model": name, "scats_number": site,
                                   "datetime": truth["datetime"], "y_true": truth["y_true"],
                                   "y_pred": y_pred.round(2), "imputed": truth["imputed"]}))
        history.append(pd.DataFrame({"model": name, "scats_number": site,
                                     "epoch": np.arange(1, len(h["loss"]) + 1),
                                     "loss": h["loss"], "val_loss": h["val_loss"]}))
        print(f"  [{n:>2}/{len(sites)}] site {site}: MAE {row['mae']:.2f}  RMSE {row['rmse']:.2f}  "
              f"({len(h['loss'])} epochs, {seconds:.1f}s)")

    metrics = ev.save_results(rows, results_dir / f"{key}_metrics.csv")
    pd.concat(preds).to_csv(results_dir / f"{key}_predictions.csv", index=False)
    pd.concat(history).to_csv(results_dir / f"{key}_history.csv", index=False)
    print(f"{name}: mean MAE {metrics['mae'].mean():.2f}, mean RMSE {metrics['rmse'].mean():.2f}")
    return metrics


# --------------------------------------------------------------------------
# Predict (used later by the route finder, Task 3 / 4)
# --------------------------------------------------------------------------
@lru_cache(maxsize=1)
def _load_clean(processed_dir: str) -> pd.DataFrame:
    return pd.read_csv(Path(processed_dir) / "scats_clean_15min.csv",
                       dtype={"scats_number": str}, parse_dates=["datetime"])


def predict(key: str, site: str, when, clean: pd.DataFrame = None, cfg: dict = None) -> float:
    """
    Predicted flow (vehicles per 15 min) at SCATS site `site` for the interval starting
    at `when`.  Needs the 12 previous intervals to exist in the data.
    """
    cfg = cfg or load_config()
    lags, when = cfg["data"]["lags"], pd.Timestamp(when)
    clean = _load_clean(str(get_path(cfg, "processed_dir"))) if clean is None else clean
    g = clean[(clean["scats_number"] == site) & (clean["datetime"] < when)].sort_values("datetime")
    hist = g.tail(lags)
    if len(hist) < lags or hist["datetime"].iloc[-1] != when - pd.Timedelta("15min"):
        raise ValueError(f"Site {site}: need the {lags} intervals right before {when} in the data")
    model, sc = load_model(key, site, get_path(cfg, "models_dir"))
    seq = scale(hist["flow"].to_numpy(), sc).reshape(1, lags, 1)
    aux = time_features(pd.Series([when]))
    y = unscale(model.predict([seq, aux], verbose=0).reshape(-1), sc)[0]
    return float(max(0.0, y))


def predict_hourly(key: str, site: str, when, **kwargs) -> float:
    """Same as predict(), but in vehicles per HOUR (15-min count x 4)."""
    return 4.0 * predict(key, site, when, **kwargs)


# --------------------------------------------------------------------------
# Command-line helper shared by train_lstm.py (and train_gru.py if you add it)
# --------------------------------------------------------------------------
def run_cli(name: str, key: str, layer_name: str) -> None:
    ap = argparse.ArgumentParser(description=f"Train and evaluate {name} for every SCATS site")
    ap.add_argument("--config", default=None, help="path to config.yaml (default: project root)")
    ap.add_argument("--sites", nargs="+", default=None, help="only these SCATS sites (quick test)")
    ap.add_argument("--epochs", type=int, default=None, help="override max epochs")
    args = ap.parse_args()
    cfg = load_config(args.config)
    if args.epochs:
        cfg["training"]["epochs"] = args.epochs
    train_deep(name, key, layer_name, cfg, args.sites)
