"""
Unit tests TC9-TC11 for Task 3 - travel time estimation.
Run from the project root:      python -m unittest discover tests -v

TC9-TC10  travel_time.py  (flow -> speed -> seconds; pure maths)
TC11      edge_costs.py   (glue; tested with fake data and a fake model, so no TensorFlow needed)
"""
from __future__ import annotations

import sys
import types
import unittest
from unittest import mock

import pandas as pd

from src.Routing import edge_costs as ec
from src.Routing import travel_time as tt

# travel_time.py
class TestTravelTime(unittest.TestCase):
    def test_tc09_flow_to_speed_key_points(self):  # TC9
        """Parabola coefficients, speed limit cap, capacity point, over-capacity clamp, bad input."""
        a, b = tt.parabola_coefficients()
        self.assertAlmostEqual(a, -1.46484375)  # PDF graph (-1500/32^2)
        self.assertAlmostEqual(b, 93.75)
        self.assertEqual(tt.flow_to_speed(0), 60.0)  # empty road: speed limit
        self.assertEqual(tt.flow_to_speed(200), 60.0)  # low flow is capped at 60
        self.assertEqual(tt.flow_to_speed(351), 60.0)
        self.assertAlmostEqual(tt.flow_to_speed(351.5625), 60.0, places=6)  # PDF point (351.56, 60)
        self.assertAlmostEqual(tt.flow_to_speed(1500), 32.0)  # PDF point (1500, 32)
        self.assertEqual(tt.flow_to_speed(3000), 32.0)  # over capacity is clamped
        with self.assertRaises(ValueError):
            tt.flow_to_speed(-1)  # negative flow rejected

    def test_tc10_edge_travel_time(self):  # TC10
        """time = distance / speed + 30 s delay; more traffic is never faster."""
        self.assertAlmostEqual(tt.edge_travel_time_s(1.0, 0), 90.0)  # 1 km at 60 km/h = 60 s + 30 s
        self.assertAlmostEqual(tt.edge_travel_time_s(1.0, 1500), 142.5)  # 1 km at 32 km/h = 112.5 s + 30 s
        self.assertEqual(tt.edge_travel_time_s(0.0, 900), 30.0)  # zero distance: only the delay
        times = [tt.edge_travel_time_s(2.0, f) for f in (0, 500, 1000, 1500, 2500)]
        self.assertEqual(times, sorted(times))

# edge_costs.py   (fake data + fake model -> fast, no TensorFlow)
WHEN = pd.Timestamp("2006-10-26 08:00")

def _fake_world():
    clean = pd.DataFrame({"scats_number": ["A", "B", "A", "B"],
                          "datetime": [WHEN, WHEN, WHEN - pd.Timedelta("7D"), WHEN - pd.Timedelta("7D")],
                          "flow": [100.0, 300.0, 100.0, 200.0]})
    meta = pd.DataFrame({"n_detectors": [2, 4]}, index=pd.Index(["A", "B"], name="scats_number"))
    edges = pd.DataFrame({"from": ["A", "B"], "to": ["B", "A"], "road": ["X_RD"] * 2,
                          "distance_km": [1.0, 1.0]})
    return clean, meta, edges

class TestEdgeCosts(unittest.TestCase):
    def setUp(self):
        self.clean, self.meta, self.edges = _fake_world()
        self.patches = [mock.patch.object(ec, "_clean", lambda _: self.clean),
                        mock.patch.object(ec, "_metadata", lambda _: self.meta)]
        for p in self.patches:
            p.start()

    def tearDown(self):
        for p in self.patches:
            p.stop()

    def _cfg(self, **tt_settings):
        return {"paths": {"processed_dir": "x", "models_dir": "x"}, "travel_time": tt_settings}

    def test_tc11_model_predictions_and_fallback(self):  # TC11
        fake = types.ModuleType("fake_model")
        fake.predict_hourly = lambda site, when: 1500.0  # always 'at capacity'
        cfg = self._cfg(flow_site="origin", flow_mode="total")
        with mock.patch.dict(sys.modules, {"src.Models.gru_model": fake}), \
             mock.patch.object(ec, "_has_model", lambda m, site, c: site == "A"):
            t = ec.edge_costs(WHEN, "gru", cfg, self.edges).set_index(["from", "to"])
        self.assertEqual(t.at[("A", "B"), "source"], "gru")
        self.assertAlmostEqual(t.at[("A", "B"), "travel_time_s"], 142.5)  # 1 km at 32 km/h + 30 s
        self.assertEqual(t.at[("B", "A"), "source"], "historical_avg")  # B has no model
        self.assertEqual(t.at[("B", "A"), "flow_vph"], 4 * (300 + 200) / 2)

if __name__ == "__main__":
    unittest.main()