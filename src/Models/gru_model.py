"""
gru_model.py  -  GRU traffic-flow predictor (one model per SCATS site)
========================================================================
GRU (Gated Recurrent Unit) is a lighter cousin of the LSTM: it also reads the last 12
flow values IN ORDER but uses fewer gates, so it trains a bit faster.
All shared code is in deep_common.py; this file only says "use the GRU layer".

Public interface (same as linear_regression_model.py, so Task 4 can swap models):
    predict(site, when)         -> predicted vehicles per 15 minutes
    predict_hourly(site, when)  -> predicted vehicles per HOUR (x4, for Task 3)
"""
from __future__ import annotations

import Source.Models.deep_common as dc

MODEL_NAME = "GRU"       # name that appears in the results files / plots
MODEL_KEY = "gru"        # used for file and folder names
LAYER = "GRU"            # the Keras layer


def train(cfg=None, sites=None):
    """Train + evaluate on all sites; writes Results/gru_*.csv and saved_models/gru/."""
    return dc.train_deep(MODEL_NAME, MODEL_KEY, LAYER, cfg, sites)


def load_model(site, cfg=None):
    cfg = cfg or dc.load_config()
    return dc.load_model(MODEL_KEY, site, dc.get_path(cfg, "models_dir"))


def predict(site, when, **kwargs):
    return dc.predict(MODEL_KEY, site, when, **kwargs)


def predict_hourly(site, when, **kwargs):
    return dc.predict_hourly(MODEL_KEY, site, when, **kwargs)
