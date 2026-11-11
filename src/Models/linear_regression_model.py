"""
linear_model.py  -  Linear Regression traffic-flow predictor (one model per SCATS site)
======================================================================================
Idea: predict the flow of the NEXT 15 minutes from
    * the last 12 flow values (the past 3 hours)                    -> 12 features
    * the time of day, as sin/cos so 23:45 is "close to" 00:00       ->  2 features
    * a weekend flag (Sat/Sun traffic looks different)               ->  1 feature
Linear Regression then finds one weight for each of the 15 features.

Public interface (same idea for every model, so Task 4 can swap models):
    predict(site, when)         -> predicted vehicles per 15 minutes
    predict_hourly(site, when)  -> predicted vehicles per HOUR (x4, for the
                                   flow -> speed conversion in Task 3)
"""
from __future__ import annotations

from functools import lru_cache
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression

LAGS = 12                                   # past 12 slots = 3 hours
SLOTS_PER_DAY = 96
ROOT = Path(__file__).resolve().parents[2]  # project root (the folder above src/)
DEFAULT_MODEL_DIR = ROOT / "saved_models" / "linear"
DEFAULT_DATA_DIR = ROOT / "data" / "Processed_Data" if (ROOT / "data" / "Processed_Data").exists() else ROOT / "data" / "processed"
FEATURE_NAMES = [f"lag_{i}" for i in range(LAGS, 0, -1)] + ["slot_sin", "slot_cos", "is_weekend"]


# --------------------------------------------------------------------------
# Features
# --------------------------------------------------------------------------
def _time_features(times: pd.Series) -> np.ndarray:
    """sin/cos of the time of day + weekend flag, shape (n, 3)."""
    t = pd.to_datetime(times)
    slot = (t.dt.hour * 4 + t.dt.minute // 15).to_numpy()
    angle = 2 * np.pi * slot / SLOTS_PER_DAY
    weekend = (t.dt.dayofweek >= 5).to_numpy().astype(float)
    return np.column_stack([np.sin(angle), np.cos(angle), weekend])


def build_features(site_df: pd.DataFrame):
    """
    site_df : the rows of ONE site from scats_clean_15min.csv.
    Returns X (n, 15), y (n,), times (n,), imputed (n,).  Row i predicts time[i]
    using only values BEFORE time[i] (no peeking at the future).
    """
    g = site_df.sort_values("datetime").reset_index(drop=True)
    v = g["flow"].to_numpy(dtype=float)
    lags = np.stack([v[i - LAGS:i] for i in range(LAGS, len(v))])      # (n, 12)
    X = np.hstack([lags, _time_features(g["datetime"].iloc[LAGS:])])
    return X, v[LAGS:], g["datetime"].iloc[LAGS:].reset_index(drop=True), g["imputed"].to_numpy()[LAGS:]


# --------------------------------------------------------------------------
# Train / save / load
# --------------------------------------------------------------------------
def fit(X_train: np.ndarray, y_train: np.ndarray) -> LinearRegression:
    """Fit a plain Linear Regression."""
    return LinearRegression().fit(X_train, y_train)


def save_model(model: LinearRegression, site: str, model_dir=DEFAULT_MODEL_DIR) -> Path:
    path = Path(model_dir) / f"{site}.joblib"
    path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(model, path)
    return path


def load_model(site: str, model_dir=DEFAULT_MODEL_DIR) -> LinearRegression:
    return joblib.load(Path(model_dir) / f"{site}.joblib")


# --------------------------------------------------------------------------
# Predict (used later by the route finder)
# --------------------------------------------------------------------------
@lru_cache(maxsize=1)
def _load_clean(data_dir: str) -> pd.DataFrame:
    return pd.read_csv(Path(data_dir) / "scats_clean_15min.csv",
                       dtype={"scats_number": str}, parse_dates=["datetime"])


def predict(site: str, when, clean: pd.DataFrame = None,
            model_dir=DEFAULT_MODEL_DIR, data_dir=DEFAULT_DATA_DIR) -> float:
    """
    Predicted flow (vehicles per 15 min) at SCATS site `site` for the 15-minute
    interval that starts at `when`.  Needs the 12 previous intervals in the data.
    """
    when = pd.Timestamp(when)
    clean = _load_clean(str(data_dir)) if clean is None else clean
    g = clean[(clean["scats_number"] == site) & (clean["datetime"] < when)].sort_values("datetime")
    hist = g.tail(LAGS)
    if len(hist) < LAGS or hist["datetime"].iloc[-1] != when - pd.Timedelta("15min"):
        raise ValueError(f"Site {site}: need the 12 intervals right before {when} in the data")
    x = np.concatenate([hist["flow"].to_numpy(), _time_features(pd.Series([when]))[0]]).reshape(1, -1)
    return float(max(0.0, load_model(site, model_dir).predict(x)[0]))


def predict_hourly(site: str, when, **kwargs) -> float:
    """Same as predict(), but in vehicles per HOUR (15-min count x 4)."""
    return 4.0 * predict(site, when, **kwargs)
