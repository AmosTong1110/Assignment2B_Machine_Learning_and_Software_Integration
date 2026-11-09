"""Tests for feature building, scaling and the Linear Regression model. TC-13 ... TC-20."""
import numpy as np
import pandas as pd
import pytest

import Source.Models.deep_common as dc
import Source.Models.linear_regression_model as lm
from conftest import make_site


def test_tc13_linear_features_have_no_future_leakage():
    """TC-13: row 0 uses flow[0..11] to predict flow[12]; X has 12 lags + 3 time features."""
    X, y, times, _ = lm.build_features(make_site(100))
    assert X.shape == (88, 15)
    assert list(X[0, :12]) == list(range(12))       # the 12 values BEFORE the target
    assert y[0] == 12
    assert times.iloc[0] == pd.Timestamp("2006-10-02 03:00")


def test_tc14_time_features_wrap_around_midnight_and_flag_weekend():
    """TC-14: 23:45 is 'close to' 00:00 on the circle; Sunday=1, Monday=0."""
    t = pd.Series(pd.to_datetime(["2006-10-01 00:00", "2006-10-01 23:45",
                                  "2006-10-01 12:00", "2006-10-02 09:00"]))
    f = lm._time_features(t)
    assert np.hypot(f[0, 0] - f[1, 0], f[0, 1] - f[1, 1]) < 0.1     # midnight ~ 23:45
    assert np.hypot(f[0, 0] - f[2, 0], f[0, 1] - f[2, 1]) > 1.9     # midnight far from noon
    assert list(f[[0, 3], 2]) == [1.0, 0.0]                         # Sunday, Monday


def test_tc15_linear_model_learns_a_simple_trend():
    """TC-15: on a perfect ramp (0,1,2,...) the model predicts the next value almost exactly."""
    X, y, _, _ = lm.build_features(make_site(200))
    model = lm.fit(X[:150], y[:150])
    assert np.abs(model.predict(X[150:]) - y[150:]).max() < 0.5


def test_tc16_predict_hourly_is_4_times_predict(tmp_path):
    """TC-16: predict_hourly = 15-min prediction x 4 (needed for the flow -> travel-time step)."""
    clean = make_site(100)
    X, y, _, _ = lm.build_features(clean)
    lm.save_model(lm.fit(X, y), "9999", tmp_path)
    when = clean["datetime"].iloc[-1] + pd.Timedelta("15min")
    p15 = lm.predict("9999", when, clean=clean, model_dir=tmp_path)
    assert lm.predict_hourly("9999", when, clean=clean, model_dir=tmp_path) == pytest.approx(4 * p15)
    assert p15 == pytest.approx(100, abs=0.5)       # next value of the ramp 0..99


def test_tc17_predict_without_enough_history_raises_error(tmp_path):
    """TC-17: asking for a time with fewer than 12 previous intervals gives a clear ValueError."""
    clean = make_site(5)
    with pytest.raises(ValueError, match="need the 12"):
        lm.predict("9999", "2006-10-02 01:15", clean=clean, model_dir=tmp_path)


def test_tc18_deep_windows_match_linear_windows_and_do_not_leak():
    """TC-18: LSTM/GRU windows hold the same 12 past values and the same target as Linear's."""
    seq, aux, y, times, flow = dc.build_windows(make_site(100), lags=12)
    X, y_lin, _, _ = lm.build_features(make_site(100))
    assert seq.shape == (88, 12) and aux.shape == (88, 3)
    assert np.array_equal(seq, X[:, :12]) and np.array_equal(y, y_lin)
    assert np.allclose(aux, X[:, 12:])              # same time-of-day / weekend features


def test_tc19_scaler_roundtrip_and_fitted_on_training_data_only():
    """TC-19: unscale(scale(x)) = x; min/max come from the training part only (no peeking)."""
    flow = np.arange(100, dtype=float)
    sc = dc.fit_scaler(flow[:-20])                   # training part = 0..79
    assert (sc["min"], sc["max"]) == (0.0, 79.0)
    assert dc.unscale(dc.scale(flow, sc), sc) == pytest.approx(flow)
    assert dc.scale(79.0, sc) == pytest.approx(1.0)


def test_tc20_config_defaults_and_overrides(tmp_path):
    """TC-20: a missing config file gives defaults; a partial file only overrides what it sets."""
    default = dc.load_config(tmp_path / "does_not_exist.yaml")
    assert default["data"]["lags"] == 12 and default["training"]["seed"] == 42
    f = tmp_path / "c.yaml"
    f.write_text("training:\n  epochs: 5\n")
    cfg = dc.load_config(f)
    assert cfg["training"]["epochs"] == 5 and cfg["training"]["batch_size"] == 32
