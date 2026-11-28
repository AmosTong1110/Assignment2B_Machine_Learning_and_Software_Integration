"""
Unit tests for the Research map view (src/Routing/osm_geometry.py, src/GUI/web_map.py).
Run from the project root:      python -m unittest discover tests -v

No internet and no TensorFlow needed: the OSM ways are small hand-made examples and the
edge costs are fake (same flow on every road).
"""
from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

import pandas as pd

from src.GUI import web_map as wm
from src.Routing import osm_geometry as og
from src.Routing import route_finder as rf
from src.Routing.graph_builder import load_edges

def way(name, points):
    return {"type": "way", "tags": {"name": name}, "geometry": [{"lat": a, "lon": b} for a, b in points]}

def fake_edge_table(flow_vph: float = 600.0) -> pd.DataFrame:
    t = load_edges().copy()
    t["flow_site"], t["flow_vph"], t["source"] = t["from"], flow_vph, "test"
    t["speed_kmh"] = 50.0
    t["travel_time_s"] = t["distance_km"] / 50.0 * 3600 + 30
    return t

class RoadNames(unittest.TestCase):
    def test_scats_and_osm_names_match(self):  # MAP-01
        self.assertEqual(og.road_key("WARRIGAL_RD"), og.road_key("Warrigal Road"))
        self.assertEqual(og.road_key("BURWOOD_HWY"), og.road_key("Burwood Highway"))
        self.assertEqual(og.road_key("STUDLEY PARK_RD"), og.road_key("Studley Park Road"))
        self.assertEqual(og.road_key("SEYMOUR_GV"), og.road_key("Seymour Grove"))

    def test_different_roads_differ(self):  # MAP-02
        self.assertNotEqual(og.road_key("HIGH_ST"), og.road_key("High Street Road"))

    def test_site_label_split(self):  # MAP-03
        self.assertEqual(og.site_road_keys("TOORAK_RD / WARRIGAL_RD"), {"TOORAKRD", "WARRIGALRD"})

class Intersection(unittest.TestCase):
    # A cross: "Main Road" runs east-west, "Side Street" north-south; they share node (0, 0).
    WAYS = [way("Main Road", [(0.0, -0.002), (0.0, 0.0), (0.0, 0.002)]),
            way("Side Street", [(-0.002, 0.0), (0.0, 0.0), (0.002, 0.0)]),
            way("Other Road", [(0.001, 0.001), (0.001, 0.0015)])]

    def test_snaps_to_shared_node(self):                                # MAP-04
        fix = og.find_intersection(0.0007, 0.0006, {"MAINRD", "SIDEST"}, self.WAYS)
        self.assertEqual(fix["method"], "intersection")
        self.assertAlmostEqual(fix["latitude"], 0.0)
        self.assertAlmostEqual(fix["longitude"], 0.0)

    def test_one_road_only_snaps_onto_road(self):                       # MAP-05
        fix = og.find_intersection(0.0005, 0.0019, {"MAINRD", "NOWHERERD"}, self.WAYS)
        self.assertEqual(fix["method"], "on_road")
        self.assertEqual((fix["latitude"], fix["longitude"]), (0.0, 0.002))

    def test_no_match_keeps_position(self):                             # MAP-06
        fix = og.find_intersection(0.0005, 0.0005, {"NOWHERERD"}, self.WAYS)
        self.assertEqual(fix["method"], "unchanged")
        self.assertEqual((fix["latitude"], fix["longitude"]), (0.0005, 0.0005))

    def test_divided_road_uses_cluster_centre(self):                    # MAP-07
        ways = self.WAYS + [way("Main Road", [(0.0002, -0.002), (0.0002, 0.0), (0.0002, 0.002)]),
                            way("Side Street", [(-0.002, 0.0), (0.0002, 0.0)])]
        fix = og.find_intersection(0.0007, 0.0, {"MAINRD", "SIDEST"}, ways)
        self.assertAlmostEqual(fix["latitude"], 0.0001)

class WebMap(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.result = rf.find_routes("2000", "3002", "2006-10-26 08:00", "actual", "AS", 3,
                                    edge_table=fake_edge_table())

    def test_page_data_has_routes_and_edges(self):  # MAP-08
        data = wm.page_data(self.result)
        self.assertEqual(len(data["routes"]), 3)
        self.assertEqual(len(data["edges"]), len(self.result.edge_table))
        first = data["routes"][0]
        # route edges join up: each edge starts where the previous ended
        ends = [(data["edges"][i]["from"], data["edges"][i]["to"]) for i in first["edges"]]
        self.assertEqual([a for a, _ in ends] + [ends[-1][1]], first["path"])

    def test_extra_frames_become_slider_steps(self):  # MAP-09
        frames = {pd.Timestamp("2006-10-26 07:00"): fake_edge_table(300.0)}
        data = wm.page_data(self.result, frames=frames)
        self.assertEqual([f["time"] for f in data["frames"]], ["07:00", "08:00"])

    def test_html_is_written_with_data(self):  # MAP-10
        with tempfile.TemporaryDirectory() as tmp:
            path = wm.write_map(self.result, out_path=Path(tmp) / "m.html")
            html = path.read_text()
        self.assertNotIn("__DATA__", html)
        self.assertIn("tile.openstreetmap.de", html)  # default background (works from file://)
        self.assertIn("L.control.layers", html)  # menu to switch background if one is blocked
        self.assertNotIn("unpkg.com", html)  # no extra plugin to fail loading
        payload = html.split("const DATA = ", 1)[1].split(";\n", 1)[0]
        self.assertEqual(json.loads(payload)["meta"]["origin"], "2000")

if __name__ == "__main__":
    unittest.main()
