"""
linear_regression_model.py  -  Linear Regression traffic-flow predictor (one model per SCATS site)
===================================================================================================
Idea: predict the flow of the NEXT 15 minutes from
    * the last 12 flow values (the past 3 hours)                    -> 12 features
    * the time of day, as sin/cos so 23:45 is "close to" 00:00       ->  2 features
    * a weekend flag (Sat/Sun traffic looks different)               ->  1 feature
Linear Regression then finds one weight for each of the 15 features.

Windows and time features come from data_processing.py (shared with LSTM / GRU).
Paths come from config.yaml (via deep_common), so nothing is hard-coded here.

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

import src.Models.data_processing as dp
import src.Models.deep_common as dc   # only for load_config() / get_path()

LAGS = dc.DEFAULTS["data"]["lags"]    # past 12 slots = 3 hours (same default as config.yaml)
FEATURE_NAMES = [f"lag_{i}" for i in range(LAGS, 0, -1)] + ["slot_sin", "slot_cos", "is_weekend"]


def _model_dir(model_dir=None) -> Path:
    """saved_models/linear unless a folder is given."""
    return Path(model_dir) if model_dir else dc.get_path(dc.load_config(), "models_dir") / "linear"


def _data_dir(data_dir=None) -> Path:
    """Processed data folder from config.yaml unless a folder is given."""
    return Path(data_dir) if data_dir else dc.get_path(dc.load_config(), "processed_dir")


# --------------------------------------------------------------------------
# Features
# --------------------------------------------------------------------------
def build_features(site_df: pd.DataFrame, lags: int = LAGS):
    """
    site_df : the rows of ONE site from scats_clean_15min.csv.
    Returns X (n, lags+3), y (n,), times (n,), imputed (n,).  Row i predicts time[i]
    using only values BEFORE time[i] (no peeking at the future).
    """
    seq, aux, y, times, _ = dp.build_supervised(site_df, lags)
    g = site_df.sort_values("datetime").reset_index(drop=True)
    return np.hstack([seq, aux]), y, times, g["imputed"].to_numpy()[lags:]


# --------------------------------------------------------------------------
# Train / save / load
# --------------------------------------------------------------------------
def fit(X_train: np.ndarray, y_train: np.ndarray) -> LinearRegression:
    """Fit a plain Linear Regression."""
    return LinearRegression().fit(X_train, y_train)


def save_model(model: LinearRegression, site: str, model_dir=None) -> Path:
    path = _model_dir(model_dir) / f"{site}.joblib"
    path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(model, path)
    return path


@lru_cache(maxsize=64)
def _load_cached(site: str, model_dir: str) -> LinearRegression:
    return joblib.load(Path(model_dir) / f"{site}.joblib")


def load_model(site: str, model_dir=None) -> LinearRegression:
    return _load_cached(site, str(_model_dir(model_dir)))


# --------------------------------------------------------------------------
# Predict (used later by the route finder)
# --------------------------------------------------------------------------
@lru_cache(maxsize=1)
def _load_clean(data_dir: str) -> pd.DataFrame:
    return pd.read_csv(Path(data_dir) / "scats_clean_15min.csv",
                       dtype={"scats_number": str}, parse_dates=["datetime"])


def predict(site: str, when, clean: pd.DataFrame = None,
            model_dir=None, data_dir=None) -> float:
    """
    Predicted flow (vehicles per 15 min) at SCATS site `site` for the 15-minute
    interval that starts at `when`.  Needs the 12 previous intervals in the data.
    """
    when = pd.Timestamp(when)
    clean = _load_clean(str(_data_dir(data_dir))) if clean is None else clean
    g = clean[(clean["scats_number"] == site) & (clean["datetime"] < when)].sort_values("datetime")
    hist = g.tail(LAGS)
    if len(hist) < LAGS or hist["datetime"].iloc[-1] != when - pd.Timedelta("15min"):
        raise ValueError(f"Site {site}: need the {LAGS} intervals right before {when} in the data")
    x = np.concatenate([hist["flow"].to_numpy(),
                        dp.time_features(pd.DatetimeIndex([when]))[0]]).reshape(1, -1)
    return float(max(0.0, load_model(site, model_dir).predict(x)[0]))


def predict_hourly(site: str, when, **kwargs) -> float:
    """Same as predict(), but in vehicles per HOUR (15-min count x 4)."""
    return 4.0 * predict(site, when, **kwargs)