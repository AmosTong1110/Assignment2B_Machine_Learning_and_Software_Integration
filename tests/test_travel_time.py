"""
Unit tests for Task 3 - travel time estimation.
Run from the project root:      python -m unittest discover tests -v

TT  = travel_time.py   (flow -> speed -> seconds; pure maths)
GB  = graph_builder.py (road names -> edges)
EC  = edge_costs.py    (glue; tested with fake data and a fake model, so no TensorFlow needed)
"""
from __future__ import annotations

import math
import sys
import types
import unittest
from unittest import mock

import pandas as pd

from src.Routing import edge_costs as ec
from src.Routing import graph_builder as gb
from src.Routing import travel_time as tt


# ==========================================================================
# travel_time.py
# ==========================================================================
class TestFlowToSpeed(unittest.TestCase):
    def test_coefficients_match_pdf_graph(self):                       # TT-01
        a, b = tt.parabola_coefficients()
        self.assertAlmostEqual(a, -1.46484375)
        self.assertAlmostEqual(b, 93.75)

    def test_zero_flow_is_speed_limit(self):                            # TT-02
        self.assertEqual(tt.flow_to_speed(0), 60.0)

    def test_low_flow_capped_at_speed_limit(self):                      # TT-03
        # the parabola gives ~60 km/h at ~351.56 veh/h; below that it would be > 60
        self.assertEqual(tt.flow_to_speed(351), 60.0)
        self.assertEqual(tt.flow_to_speed(200), 60.0)

    def test_pdf_point_351_56_gives_60(self):                           # TT-04
        self.assertAlmostEqual(tt.flow_to_speed(351.5625), 60.0, places=6)

    def test_capacity_gives_32(self):                                   # TT-05
        self.assertAlmostEqual(tt.flow_to_speed(1500), 32.0)

    def test_over_capacity_is_clamped(self):                            # TT-06
        self.assertEqual(tt.flow_to_speed(3000), 32.0)

    def test_speed_falls_as_flow_rises(self):                           # TT-07
        speeds = [tt.flow_to_speed(f) for f in range(400, 1500, 100)]
        self.assertTrue(all(s1 > s2 for s1, s2 in zip(speeds, speeds[1:])))

    def test_round_trip_with_pdf_formula(self):                         # TT-08
        for flow in (400, 800, 1000, 1499):
            self.assertAlmostEqual(tt.speed_to_flow(tt.flow_to_speed(flow)), flow, places=6)

    def test_congested_branch_is_slower(self):                          # TT-09
        green = tt.flow_to_speed(1000)
        red = tt.flow_to_speed(1000, congested=True)
        self.assertLess(red, 32.0)
        self.assertGreater(green, 32.0)
        self.assertAlmostEqual(tt.speed_to_flow(red), 1000, places=6)

    def test_negative_flow_rejected(self):                              # TT-10
        with self.assertRaises(ValueError):
            tt.flow_to_speed(-1)


class TestEdgeTravelTime(unittest.TestCase):
    def test_1km_free_flow_is_90s(self):                                # TT-11
        # 1 km at 60 km/h = 60 s, plus 30 s intersection delay
        self.assertAlmostEqual(tt.edge_travel_time_s(1.0, 0), 90.0)

    def test_1km_at_capacity(self):                                     # TT-12
        # 1 km at 32 km/h = 112.5 s, plus 30 s
        self.assertAlmostEqual(tt.edge_travel_time_s(1.0, 1500), 142.5)

    def test_zero_distance_is_only_delay(self):                         # TT-13
        self.assertEqual(tt.edge_travel_time_s(0.0, 900), 30.0)

    def test_more_traffic_never_faster(self):                           # TT-14
        times = [tt.edge_travel_time_s(2.0, f) for f in (0, 500, 1000, 1500, 2500)]
        self.assertEqual(times, sorted(times))

    def test_config_overrides(self):                                    # TT-15
        p = tt.params_from_config({"travel_time": {"intersection_delay_s": 0, "speed_limit_kmh": 50}})
        self.assertAlmostEqual(tt.edge_travel_time_s(1.0, 0, **p), 72.0)   # 1 km at 50 km/h

    def test_haversine_known_distance(self):                            # TT-16
        # 0.01 degree of latitude is ~1.112 km everywhere
        self.assertAlmostEqual(tt.haversine_km(-37.80, 145.0, -37.81, 145.0), 1.112, places=2)
        self.assertEqual(tt.haversine_km(-37.8, 145.0, -37.8, 145.0), 0.0)


# ==========================================================================
# graph_builder.py
# ==========================================================================
class TestGraphBuilder(unittest.TestCase):
    def test_parse_roads(self):                                         # GB-01
        self.assertEqual(gb.parse_roads("WARRIGAL_RD N of TOORAK_RD"), ("WARRIGAL_RD", "TOORAK_RD"))
        self.assertEqual(gb.parse_roads("HIGH_ST NE OF HARP_ST"), ("HIGH_ST", "HARP_RD"))   # alias
        self.assertEqual(gb.parse_roads("STUDLEY PARK_RD W of PRINCESS_ST"),
                         ("STUDLEY PARK_RD", "PRINCESS_ST"))

    def test_parse_roads_rejects_bad_text(self):                        # GB-02
        with self.assertRaises(ValueError):
            gb.parse_roads("NOT A LOCATION")

    def test_consecutive_sites_on_a_road_are_joined(self):              # GB-03
        nodes = pd.DataFrame({"scats_number": ["A", "B", "C"],
                              "latitude": [-37.80, -37.81, -37.82],
                              "longitude": [145.0, 145.0, 145.0]})
        roads = pd.DataFrame({"scats_number": ["C", "A", "B"], "road": ["X_RD"] * 3})
        with mock.patch.object(gb, "EXTRA_EDGES", []), mock.patch.object(gb, "REMOVED_EDGES", []):
            edges, dropped = gb.build_edges(nodes, roads)
        pairs = set(zip(edges["from"], edges["to"]))
        # A-B and B-C both ways, but NOT A-C (C is not A's neighbour on the road)
        self.assertEqual(pairs, {("A", "B"), ("B", "A"), ("B", "C"), ("C", "B")})
        self.assertTrue(dropped.empty)

    def test_long_joins_are_dropped(self):                              # GB-04
        nodes = pd.DataFrame({"scats_number": ["A", "B"], "latitude": [-37.80, -37.90],
                              "longitude": [145.0, 145.0]})                    # ~11 km apart
        roads = pd.DataFrame({"scats_number": ["A", "B"], "road": ["HIGH_ST"] * 2})
        with mock.patch.object(gb, "EXTRA_EDGES", []), mock.patch.object(gb, "REMOVED_EDGES", []):
            edges, dropped = gb.build_edges(nodes, roads)
        self.assertTrue(edges.empty)
        self.assertEqual(len(dropped), 1)

    def test_real_network_is_connected(self):                           # GB-05
        """Every site can reach every other site in the real edges.csv."""
        try:
            edges = gb.load_edges()
            nodes = gb.load_nodes()
        except FileNotFoundError:
            self.skipTest("edges.csv not built yet")
        adj = {}
        for a, b in zip(edges["from"], edges["to"]):
            adj.setdefault(a, set()).add(b)
        start = nodes["scats_number"].iloc[0]
        seen, stack = {start}, [start]
        while stack:
            for nxt in adj.get(stack.pop(), ()):
                if nxt not in seen:
                    seen.add(nxt)
                    stack.append(nxt)
        self.assertEqual(seen, set(nodes["scats_number"]))
        self.assertTrue((edges["distance_km"] > 0).all())


# ==========================================================================
# edge_costs.py   (fake data + fake model -> fast, no TensorFlow)
# ==========================================================================
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

    def test_actual_flow_origin_per_detector(self):                     # EC-01
        cfg = self._cfg(flow_site="origin", flow_mode="per_detector")
        t = ec.edge_costs(WHEN, "actual", cfg, self.edges).set_index(["from", "to"])
        self.assertEqual(t.at[("A", "B"), "flow_vph"], 100 * 4 / 2)        # site A, 2 detectors
        self.assertEqual(t.at[("B", "A"), "flow_vph"], 300 * 4 / 4)        # site B, 4 detectors
        self.assertAlmostEqual(t.at[("A", "B"), "travel_time_s"], 90.0)    # 200 veh/h -> 60 km/h

    def test_destination_and_total_mode(self):                          # EC-02
        cfg = self._cfg(flow_site="destination", flow_mode="total")
        t = ec.edge_costs(WHEN, "actual", cfg, self.edges).set_index(["from", "to"])
        self.assertEqual(t.at[("A", "B"), "flow_vph"], 1200.0)             # B's total flow
        self.assertAlmostEqual(t.at[("A", "B"), "speed_kmh"], round(tt.flow_to_speed(1200), 2))

    def test_model_predictions_and_fallback(self):                      # EC-03
        fake = types.ModuleType("fake_model")
        fake.predict_hourly = lambda site, when: 1500.0                    # always 'at capacity'
        cfg = self._cfg(flow_site="origin", flow_mode="total")
        with mock.patch.dict(sys.modules, {"src.Models.gru_model": fake}), \
             mock.patch.object(ec, "_has_model", lambda m, site, c: site == "A"):
            t = ec.edge_costs(WHEN, "gru", cfg, self.edges).set_index(["from", "to"])
        self.assertEqual(t.at[("A", "B"), "source"], "gru")
        self.assertAlmostEqual(t.at[("A", "B"), "travel_time_s"], 142.5)   # 1 km at 32 km/h + 30 s
        self.assertEqual(t.at[("B", "A"), "source"], "historical_avg")     # B has no model
        self.assertEqual(t.at[("B", "A"), "flow_vph"], 4 * (300 + 200) / 2)

    def test_cost_lookup(self):                                         # EC-04
        t = ec.edge_costs(WHEN, "actual", self._cfg(), self.edges)
        costs = ec.cost_lookup(t)
        self.assertEqual(set(costs), {("A", "B"), ("B", "A")})
        self.assertTrue(all(isinstance(v, float) and v >= 30 for v in costs.values()))

    def test_bad_settings_rejected(self):                               # EC-05
        with self.assertRaises(ValueError):
            ec.edge_costs(WHEN, "actual", self._cfg(flow_site="middle"), self.edges)
        with self.assertRaises(ValueError):
            ec.edge_costs(WHEN, "actual", self._cfg(flow_mode="average"), self.edges)
        with self.assertRaises(ValueError):
            ec.site_flows(WHEN, "svm", ["A"], self._cfg())


if __name__ == "__main__":
    unittest.main()
