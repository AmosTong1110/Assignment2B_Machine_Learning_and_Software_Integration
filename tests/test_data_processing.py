"""
Unit tests TC1-TC8 for Tasks 1-2 - data processing, shared ML helpers, evaluation and model results.
Run from the project root:      python -m unittest discover tests -v

TC1-TC4  data_processing.py  (cleaning pipeline and quality checks)
TC5      data_processing.py  (training windows used by every model)
TC6-TC7  evaluate.py         (test split and metrics shared by all models)
TC8      results/model_comparison.csv (every model beats the naive baseline)

The cleaning steps are tested on SMALL HAND-MADE tables, so each rule can be checked exactly.
TC4, TC6, TC7 and TC8 also use the real saved files (data/Processed_Data, results).
No TensorFlow needed.
"""
from __future__ import annotations

import unittest

import numpy as np
import pandas as pd

import src.Models.data_processing as dp
import src.Models.evaluate as ev

DAYS = pd.date_range("2006-10-02", periods=21, freq="D")    # three weeks: the spike rule needs 3+ of each weekday

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

# Task 1 - cleaning pipeline
class TestCleaning(unittest.TestCase):

    def test_tc01_faulty_detector_is_dropped(self):  # TC1
        good = raw_rows("0970", "A_RD N OF B_RD", DAYS)
        dead = raw_rows("0970", "C_ST S OF B_RD", DAYS, value=0.0)  # 100% zeros
        kept, faulty = dp.drop_faulty_detectors(pd.concat([good, dead], ignore_index=True))
        self.assertEqual(set(kept["Location"]), {"A_RD N OF B_RD"})
        self.assertEqual(faulty["Location"].tolist(), ["C_ST S OF B_RD"])

    def test_tc02_outage_and_spike_flagged_but_rush_hour_kept(self):  # TC2
        arr = daily_pattern()
        arr[3, 50:60] = 0.0  # 2.5 h of zeros = outage
        arr[5, 70] = 900.0  # one absurd value = spike
        series, log = dp.flag_bad_values({"2000": arr}, DAYS)
        out = series["2000"]
        self.assertTrue(np.isnan(out[3, 50:60]).all())
        self.assertTrue(np.isnan(out[5, 70]))
        self.assertFalse(np.isnan(out[:, 28:36]).any())  # real peaks survive
        self.assertEqual(log["2000"], {"outage_values": 10, "spike_values": 1})

    def test_tc03_short_gap_interpolated_long_gap_profile_filled(self):  # TC3
        arr = daily_pattern()
        arr[2, 10:12] = np.nan  # 30 min gap
        arr[4, 40:60] = np.nan  # 5 h gap
        arr[2, 9], arr[2, 12] = 30.0, 60.0
        clean = dp.fill_gaps({"2000": arr}, DAYS)
        flow = clean["flow"].to_numpy().reshape(21, 96)
        np.testing.assert_allclose(flow[2, 10:12], [40.0, 50.0])  # straight line 30 -> 60
        np.testing.assert_allclose(flow[4, 40:60], 40.0)  # same weekday+slot profile
        self.assertEqual(int(clean["imputed"].sum()), 22)  # every repair is flagged
        self.assertFalse(clean["flow"].isna().any())

    def test_tc04_validate_rejects_broken_data(self):  # TC4
        clean, meta = dp.load_clean()
        dp.validate(clean, meta)  # the real data passes
        broken = clean.copy()
        broken.loc[5, "flow"] = np.nan
        with self.assertRaises(AssertionError):
            dp.validate(broken, meta)
        dup = pd.concat([clean, clean.head(1)], ignore_index=True)
        with self.assertRaises(AssertionError):
            dp.validate(dup, meta)

# Task 2 - helpers shared by Linear / LSTM / GRU
class TestModelHelpers(unittest.TestCase):

    def test_tc05_supervised_windows_never_see_the_future(self):  # TC5
        times = pd.date_range("2006-10-01", periods=20, freq="15min")
        site = pd.DataFrame({"datetime": times[::-1], "flow": np.arange(20.0)[::-1]})  # unsorted on purpose
        seq, aux, y, t, flow = dp.build_supervised(site, lags=4)
        self.assertEqual(seq.shape, (16, 4))
        np.testing.assert_array_equal(seq[0], [0, 1, 2, 3])
        self.assertEqual(y[0], 4)
        self.assertTrue((seq.max(axis=1) < y).all())  # inputs are all earlier
        self.assertEqual(t.iloc[0], times[4])

# Evaluation - one test set and one set of metrics for every model
class TestEvaluation(unittest.TestCase):

    def test_tc06_test_set_is_last_seven_days(self):  # TC6
        clean, _ = dp.load_clean()
        truth = ev.get_test_truth(clean, "2000")
        self.assertEqual(len(truth), 672)
        self.assertEqual(truth["datetime"].iloc[0], pd.Timestamp("2006-10-25 00:00"))
        self.assertEqual(truth["datetime"].iloc[-1], pd.Timestamp("2006-10-31 23:45"))

    def test_tc07_evaluate_site_catches_common_mistakes(self):  # TC7
        clean, _ = dp.load_clean()
        truth = ev.get_test_truth(clean, "2000")
        y = truth["y_true"].to_numpy()
        with self.assertRaises(ValueError):
            ev.evaluate_site("m", "2000", y[:-1], truth)  # wrong length
        with self.assertRaises(ValueError):
            ev.evaluate_site("m", "2000", y / y.max(), truth)  # forgot to inverse-scale
        row = ev.evaluate_site("m", "2000", y, truth)
        self.assertEqual(row["mae"], 0.0)
        self.assertEqual(row["n_eval"], int((truth["imputed"] == 0).sum()))  # imputed rows not scored

# Saved data, trained model and results
class TestSavedOutputs(unittest.TestCase):

    def test_tc08_every_model_beats_the_naive_baseline(self):  # TC8
        comp = pd.read_csv("results/model_comparison.csv").set_index("model")
        base = comp.loc["Persistence (baseline)", "mae"]
        for model in ("GRU", "LSTM", "Linear Regression"):
            self.assertLess(comp.loc[model, "mae"], base, model)

if __name__ == "__main__":
    unittest.main()