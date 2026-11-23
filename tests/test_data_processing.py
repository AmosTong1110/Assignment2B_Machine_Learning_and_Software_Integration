"""
Unit tests for Tasks 1-2 - data processing, shared ML helpers, evaluation and model output.
Run from the project root:      python -m unittest discover tests -v

DP = data_processing.py   (cleaning pipeline + helpers every model uses)
EV = evaluate.py          (test split and metrics shared by all models)
LR = linear_regression_model.py

The cleaning steps are tested on SMALL HAND-MADE tables, so each rule can be checked exactly.
The last group checks the real saved files (data/Processed_Data, saved_models, results).
No TensorFlow needed.
"""
from __future__ import annotations

import unittest

import numpy as np
import pandas as pd

import src.Models.data_processing as dp
import src.Models.evaluate as ev
import src.Models.linear_regression_model as lr

DAYS = pd.date_range("2006-10-02", periods=21, freq="D")       # three weeks: the spike rule needs 3+ of each weekday


def raw_rows(site: str, location: str, days, value=50.0) -> pd.DataFrame:
    """Raw-file-shaped rows: one per day for one detector, every 15-min slot = value."""
    rows = pd.DataFrame({"SCATS Number": site, "Location": location, "day": list(days)})
    flows = pd.DataFrame(np.full((len(days), dp.SLOTS_PER_DAY), value), columns=dp.FLOW_COLS)
    return pd.concat([rows, flows], axis=1)


def daily_pattern() -> np.ndarray:
    """(21 days, 96) flows with a normal rush hour: 40 at night, 200 between 07:00 and 09:00."""
    arr = np.full((len(DAYS), dp.SLOTS_PER_DAY), 40.0)
    arr[:, 28:36] = 200.0
    return arr


# --------------------------------------------------------------------------
# Task 1 - cleaning pipeline
# --------------------------------------------------------------------------
class TestCleaning(unittest.TestCase):
    def test_zero_runs_only_long_runs(self):                            # DP-01
        z = np.array([1, 1, 0, 1, 1, 1, 1, 0, 1], dtype=bool)
        np.testing.assert_array_equal(dp._zero_runs(z, 3),
                                      [0, 0, 0, 1, 1, 1, 1, 0, 0])

    def test_faulty_detector_is_dropped(self):                          # DP-02
        good = raw_rows("0970", "A_RD N OF B_RD", DAYS)
        dead = raw_rows("0970", "C_ST S OF B_RD", DAYS, value=0.0)        # 100% zeros
        kept, faulty = dp.drop_faulty_detectors(pd.concat([good, dead], ignore_index=True))
        self.assertEqual(set(kept["Location"]), {"A_RD N OF B_RD"})
        self.assertEqual(faulty["Location"].tolist(), ["C_ST S OF B_RD"])

    def test_site_series_sums_detectors_and_blanks_incomplete_days(self):# DP-03
        a = raw_rows("2000", "A", DAYS, 30.0)
        b = raw_rows("2000", "B", DAYS[1:], 20.0)                          # B missing on day 0
        series, days, incomplete = dp.build_site_series(pd.concat([a, b], ignore_index=True))
        arr = series["2000"]
        self.assertEqual(arr.shape, (21, 96))
        self.assertTrue(np.isnan(arr[0]).all())                            # 1 of 2 detectors -> unreliable
        self.assertTrue((arr[1:] == 50.0).all())                           # 30 + 20
        self.assertEqual(int(incomplete["2000"].sum()), 1)

    def test_outage_and_spike_flagged_but_rush_hour_kept(self):         # DP-04
        arr = daily_pattern()
        arr[3, 50:60] = 0.0                                                # 2.5 h of zeros = outage
        arr[5, 70] = 900.0                                                 # one absurd value = spike
        series, log = dp.flag_bad_values({"2000": arr}, DAYS)
        out = series["2000"]
        self.assertTrue(np.isnan(out[3, 50:60]).all())
        self.assertTrue(np.isnan(out[5, 70]))
        self.assertFalse(np.isnan(out[:, 28:36]).any())                    # real peaks survive
        self.assertEqual(log["2000"], {"outage_values": 10, "spike_values": 1})

    def test_short_gap_interpolated_long_gap_profile_filled(self):      # DP-05
        arr = daily_pattern()
        arr[2, 10:12] = np.nan                                             # 30 min gap
        arr[4, 40:60] = np.nan                                             # 5 h gap
        arr[2, 9], arr[2, 12] = 30.0, 60.0
        clean = dp.fill_gaps({"2000": arr}, DAYS)
        flow = clean["flow"].to_numpy().reshape(21, 96)
        np.testing.assert_allclose(flow[2, 10:12], [40.0, 50.0])           # straight line 30 -> 60
        np.testing.assert_allclose(flow[4, 40:60], 40.0)                   # same weekday+slot profile
        self.assertEqual(int(clean["imputed"].sum()), 22)                  # every repair is flagged
        self.assertFalse(clean["flow"].isna().any())

    def test_validate_rejects_broken_data(self):                        # DP-06
        clean, meta = dp.load_clean()
        dp.validate(clean, meta)                                           # the real data passes
        broken = clean.copy()
        broken.loc[5, "flow"] = np.nan
        with self.assertRaises(AssertionError):
            dp.validate(broken, meta)
        dup = pd.concat([clean, clean.head(1)], ignore_index=True)
        with self.assertRaises(AssertionError):
            dp.validate(dup, meta)


# --------------------------------------------------------------------------
# Task 2 - helpers shared by Linear / LSTM / GRU
# --------------------------------------------------------------------------
class TestModelHelpers(unittest.TestCase):
    def test_minmax_round_trip_and_json(self):                          # ML-01
        x = np.array([10.0, 55.0, 100.0])
        s = dp.MinMax().fit(x)
        np.testing.assert_allclose(s.transform(x), [0.0, 0.5, 1.0], atol=1e-6)
        s2 = dp.MinMax.from_dict(s.to_dict())
        np.testing.assert_allclose(s2.inverse(s2.transform(x)), x)

    def test_time_features_cyclic_and_weekend(self):                    # ML-02
        t = pd.DatetimeIndex(["2006-10-27 23:45", "2006-10-28 00:00", "2006-10-27 12:00"])  # Fri, Sat
        f = dp.time_features(t)
        self.assertLess(np.linalg.norm(f[0, :2] - f[1, :2]), 0.1)          # 23:45 is next to 00:00
        self.assertGreater(np.linalg.norm(f[1, :2] - f[2, :2]), 1.9)       # midnight vs noon: opposite
        np.testing.assert_array_equal(f[:, 2], [0, 1, 0])

    def test_supervised_windows_never_see_the_future(self):             # ML-03
        times = pd.date_range("2006-10-01", periods=20, freq="15min")
        site = pd.DataFrame({"datetime": times[::-1], "flow": np.arange(20.0)[::-1]})   # unsorted on purpose
        seq, aux, y, t, flow = dp.build_supervised(site, lags=4)
        self.assertEqual(seq.shape, (16, 4))
        np.testing.assert_array_equal(seq[0], [0, 1, 2, 3])
        self.assertEqual(y[0], 4)
        self.assertTrue((seq.max(axis=1) < y).all())                        # inputs are all earlier
        self.assertEqual(t.iloc[0], times[4])


# --------------------------------------------------------------------------
# Evaluation - one test set and one set of metrics for every model
# --------------------------------------------------------------------------
class TestEvaluation(unittest.TestCase):
    def test_metrics_known_values(self):                                # EV-01
        y, p = [10, 20, 30, 40], [12, 18, 30, 44]
        self.assertAlmostEqual(ev.mae(y, p), 2.0)
        self.assertAlmostEqual(ev.rmse(y, p), np.sqrt(6.0))
        self.assertAlmostEqual(ev.nrmse(y, p), np.sqrt(6.0) / 25.0)

    def test_test_set_is_last_seven_days(self):                         # EV-02
        clean, _ = dp.load_clean()
        truth = ev.get_test_truth(clean, "2000")
        self.assertEqual(len(truth), 672)
        self.assertEqual(truth["datetime"].iloc[0], pd.Timestamp("2006-10-25 00:00"))
        self.assertEqual(truth["datetime"].iloc[-1], pd.Timestamp("2006-10-31 23:45"))

    def test_evaluate_site_catches_common_mistakes(self):               # EV-03
        clean, _ = dp.load_clean()
        truth = ev.get_test_truth(clean, "2000")
        y = truth["y_true"].to_numpy()
        with self.assertRaises(ValueError):
            ev.evaluate_site("m", "2000", y[:-1], truth)                    # wrong length
        with self.assertRaises(ValueError):
            ev.evaluate_site("m", "2000", y / y.max(), truth)               # forgot to inverse-scale
        row = ev.evaluate_site("m", "2000", y, truth)
        self.assertEqual(row["mae"], 0.0)
        self.assertEqual(row["n_eval"], int((truth["imputed"] == 0).sum()))  # imputed rows not scored


# --------------------------------------------------------------------------
# Saved data, trained model and results
# --------------------------------------------------------------------------
class TestSavedOutputs(unittest.TestCase):
    def test_clean_dataset_shape(self):                                 # DP-07
        clean, meta = dp.load_clean()
        self.assertEqual(clean["scats_number"].nunique(), 40)
        self.assertTrue((clean.groupby("scats_number").size() == 31 * 96).all())
        self.assertIn("0970", set(meta["scats_number"]))                   # leading zero kept

    def test_linear_prediction_needs_history_and_is_sensible(self):     # ML-04
        with self.assertRaises(ValueError):
            lr.predict("2000", "2006-10-01 01:00")                         # only 4 earlier slots
        clean, _ = dp.load_clean()
        when = pd.Timestamp("2006-10-26 08:00")
        actual = clean[(clean["scats_number"] == "2000") & (clean["datetime"] == when)]["flow"].iloc[0]
        pred = lr.predict("2000", when, clean=clean)
        self.assertGreaterEqual(pred, 0.0)
        self.assertLess(abs(pred - actual) / actual, 0.3)                  # within 30% at the peak

    def test_every_model_beats_the_naive_baseline(self):                # ML-05
        comp = pd.read_csv("results/model_comparison.csv").set_index("model")
        base = comp.loc["Persistence (baseline)", "mae"]
        for model in ("GRU", "LSTM", "Linear Regression"):
            self.assertLess(comp.loc[model, "mae"], base, model)


if __name__ == "__main__":
    unittest.main()
