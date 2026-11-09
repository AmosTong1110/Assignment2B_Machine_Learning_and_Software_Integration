"""Tests for the shared evaluation code (Source/Models/evaluate.py). TC-06 ... TC-12."""
import numpy as np
import pandas as pd
import pytest

import Source.Models.evaluate as ev
from conftest import RESULTS


def _truth(y_true, imputed=None):
    n = len(y_true)
    return pd.DataFrame({"datetime": pd.date_range("2006-10-25", periods=n, freq="15min"),
                         "y_true": np.asarray(y_true, dtype=float),
                         "imputed": np.zeros(n, dtype=int) if imputed is None else imputed,
                         "prev_flow": np.asarray(y_true, dtype=float)})


def test_tc06_mae_and_rmse_known_values():
    """TC-06: errors of 2, 4 -> MAE = 3, RMSE = sqrt((4+16)/2) = 3.162."""
    assert ev.mae([10, 10], [12, 6]) == pytest.approx(3.0)
    assert ev.rmse([10, 10], [12, 6]) == pytest.approx(np.sqrt(10))


def test_tc07_rmse_is_never_smaller_than_mae():
    """TC-07: RMSE >= MAE for any data (RMSE punishes big misses more)."""
    rng = np.random.default_rng(0)
    a, b = rng.uniform(0, 300, 200), rng.uniform(0, 300, 200)
    assert ev.rmse(a, b) >= ev.mae(a, b)


def test_tc08_test_set_is_the_last_7_days(clean):
    """TC-08: test truth = 672 rows from 25 Oct 00:00 to 31 Oct 23:45."""
    t = ev.get_test_truth(clean, "2000")
    assert len(t) == ev.TEST_SIZE == 672
    assert t["datetime"].iloc[0] == pd.Timestamp("2006-10-25 00:00")
    assert t["datetime"].iloc[-1] == pd.Timestamp("2006-10-31 23:45")


def test_tc09_scaled_predictions_are_rejected():
    """TC-09: predictions that look like 0-1 scaled values raise an error (common bug)."""
    truth = _truth([100.0] * 10)
    with pytest.raises(ValueError, match="SCALED"):
        ev.evaluate_site("X", "1", np.full(10, 0.5), truth)


def test_tc10_wrong_length_or_nan_predictions_are_rejected():
    """TC-10: wrong number of predictions, or NaN inside them, raises ValueError."""
    truth = _truth([10.0] * 5)
    with pytest.raises(ValueError):
        ev.evaluate_site("X", "1", [10.0] * 4, truth)
    with pytest.raises(ValueError):
        ev.evaluate_site("X", "1", [10.0, 10.0, np.nan, 10.0, 10.0], truth)


def test_tc11_imputed_rows_are_not_scored():
    """TC-11: a terrible prediction on an imputed (repaired) row must not change the score."""
    truth = _truth([10, 10, 10, 10], imputed=[0, 0, 0, 1])
    row = ev.evaluate_site("X", "1", [10, 10, 10, 999], truth)
    assert row["mae"] == 0.0 and row["n_eval"] == 3


def test_tc12_linear_beats_persistence_baseline_on_saved_results():
    """TC-12: on the saved results, Linear Regression has lower mean MAE than 'next = last'."""
    summary, per_site = ev.compare([str(RESULTS / "baseline_metrics.csv"),
                                    str(RESULTS / "linear_metrics.csv")])
    assert summary.loc["Linear Regression", "mae"] < summary.loc["Persistence (baseline)", "mae"]
    assert summary["sites"].nunique() == 1          # both models scored on the SAME sites
