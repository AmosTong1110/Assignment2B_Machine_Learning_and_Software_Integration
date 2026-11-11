"""
lstm_model.py  -  LSTM traffic-flow predictor (one model per SCATS site)
========================================================================
LSTM (Long Short-Term Memory) reads the last 12 flow values IN ORDER and keeps a
"memory" of the pattern (e.g. flow is rising towards the morning peak).
All shared code is in deep_common.py; this file only says "use the LSTM layer".

Public interface (same as linear_regression_model.py, so Task 4 can swap models):
    predict(site, when)         -> predicted vehicles per 15 minutes
    predict_hourly(site, when)  -> predicted vehicles per HOUR (x4, for Task 3)
"""
from __future__ import annotations

try:
    import src.Models.deep_common as dc
except ImportError:
    import Source.Models.deep_common as dc

MODEL_NAME = "LSTM"      # name that appears in the results files / plots
MODEL_KEY = "lstm"       # used for file and folder names
LAYER = "LSTM"           # the Keras layer


def train(cfg=None, sites=None):
    """Train + evaluate on all sites; writes Results/lstm_*.csv and saved_models/lstm/."""
    return dc.train_deep(MODEL_NAME, MODEL_KEY, LAYER, cfg, sites)


def load_model(site, cfg=None):
    cfg = cfg or dc.load_config()
    return dc.load_model(MODEL_KEY, site, dc.get_path(cfg, "models_dir"))


def predict(site, when, **kwargs):
    return dc.predict(MODEL_KEY, site, when, **kwargs)


def predict_hourly(site, when, **kwargs):
    return dc.predict_hourly(MODEL_KEY, site, when, **kwargs)
