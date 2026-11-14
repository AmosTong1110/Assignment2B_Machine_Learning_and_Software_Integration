"""
Unit tests for Task 4 - Part A search + top-k routes on the Boroondara map.
Run from the project root:      python -m unittest discover tests -v

TK = top_k.py         (Yen's algorithm on top of the Part A searches)
RF = route_finder.py  (edge_costs table -> Problem -> up to 5 routes)

Edge costs here are FAKE (distance-based) so the tests need no TensorFlow and run in a second.
"""
from __future__ import annotations

import unittest

import pandas as pd

from src.Routing import route_finder as rf
from src.Routing import travel_time as tt
from src.Routing.graph_builder import load_edges, load_nodes
from src.Search import SEARCH_METHODS
from src.Search.astar import astar
from src.Search.heuristic import cost_heuristic
from src.Search.problem import Problem
from src.Search.top_k import k_routes, route_cost


def toy_problem(origin="C", destination="H"):
    """The classic example from Yen's algorithm (Wikipedia). All nodes at (0, 0) -> h = 0."""
    pairs = {("C", "D"): 3, ("C", "E"): 2, ("D", "F"): 4, ("E", "D"): 1, ("E", "F"): 2,
             ("E", "G"): 3, ("F", "G"): 2, ("F", "H"): 1, ("G", "H"): 2}
    edges = {}
    for (a, b), c in pairs.items():
        edges.setdefault(a, []).append((b, c))
    coords = {n: (0, 0) for n in "CDEFGH"}
    return Problem(coords, edges, origin, [destination])


def fake_edge_table(flow_vph: float = 600.0) -> pd.DataFrame:
    """edge_costs()-shaped table with the same flow on every road."""
    t = load_edges().copy()
    t["flow_site"] = t["from"]
    t["flow_vph"] = flow_vph
    t["speed_kmh"] = tt.flow_to_speed(flow_vph)
    t["travel_time_s"] = [tt.edge_travel_time_s(d, flow_vph) for d in t["distance_km"]]
    t["source"] = "fake"
    return t


def all_simple_path_costs(problem, origin, destination):
    """Brute force: cost of EVERY loop-free path (small graph, so this is fine)."""
    out, stack = [], [(origin, (origin,), 0.0)]
    while stack:
        node, path, cost = stack.pop()
        if node == destination:
            out.append(cost)
            continue
        for nxt, c in problem.edges.get(node, []):
            if nxt not in path:
                stack.append((nxt, path + (nxt,), cost + c))
    return sorted(out)


# ==========================================================================
# top_k.py
# ==========================================================================
class TestYen(unittest.TestCase):
    def test_classic_example_and_route_properties(self):                    # TK-01
        routes, _ = k_routes(toy_problem(), astar, k=3)
        self.assertEqual(routes[0], (["C", "E", "F", "H"], 5))
        self.assertEqual(routes[1], (["C", "E", "G", "H"], 7))
        self.assertEqual(routes[2][1], 8)                                    # C-D-F-H or C-E-D-F-H (tie)
        # asking for more routes than exist returns all of them, distinct and loop-free
        routes, _ = k_routes(toy_problem(), astar, k=50)
        self.assertEqual(len(routes), len(all_simple_path_costs(toy_problem(), "C", "H")))
        paths = [tuple(p) for p, _ in routes]
        self.assertEqual(len(paths), len(set(paths)))
        for p in paths:
            self.assertEqual(len(p), len(set(p)))

    def test_no_path_and_bad_k(self):                                       # TK-02
        p = toy_problem(origin="H", destination="C")                        # all edges point away from C
        routes, created = k_routes(p, astar, k=5)
        self.assertEqual(routes, [])
        self.assertGreaterEqual(created, 1)
        with self.assertRaises(ValueError):
            k_routes(toy_problem(), astar, k=0)

    def test_every_method_gives_valid_sorted_routes(self):                  # TK-03
        prob = toy_problem()
        for code, (search, _) in SEARCH_METHODS.items():
            with self.subTest(method=code):
                routes, _ = k_routes(prob, search, k=4)
                self.assertEqual(len(routes), 4)
                costs = [c for _, c in routes]
                self.assertEqual(costs, sorted(costs))
                for path, cost in routes:
                    self.assertEqual((path[0], path[-1]), ("C", "H"))
                    self.assertEqual(route_cost(prob.edges, path), cost)


# ==========================================================================
# route_finder.py  (real Boroondara graph, fake travel times)
# ==========================================================================
class TestRouteFinder(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.table = fake_edge_table()
        cls.nodes = load_nodes()

    def find(self, o="2000", d="3002", method="AS", k=5):
        return rf.find_routes(o, d, "2006-10-26 08:00", model="actual", method=method, k=k,
                              edge_table=self.table)

    def test_brief_example_with_every_method(self):                         # RF-01
        res = self.find()
        self.assertEqual(len(res.routes), 5)                                 # brief: up to 5 routes
        for r in res.routes:
            self.assertEqual((r.path[0], r.path[-1]), ("2000", "3002"))
        for code in SEARCH_METHODS:                                          # all six Part A methods work
            with self.subTest(method=code):
                res = self.find(method=code)
                self.assertTrue(1 <= len(res.routes) <= 5)
                self.assertGreater(res.nodes_created, 0)
        self.assertEqual(len(self.find(k=2).routes), 2)                      # k limits the number of routes

    def test_astar_gives_the_true_fastest_routes(self):                     # RF-02
        for o, d in [("2000", "3002"), ("4821", "2827"), ("0970", "4051")]:
            with self.subTest(origin=o, destination=d):
                prob = rf.build_problem(self.table, self.nodes, o, d)
                best = all_simple_path_costs(prob, o, d)[:5]                 # brute force
                got = [r.travel_time_s for r in self.find(o, d).routes]
                for g, b in zip(got, best):
                    self.assertAlmostEqual(g, b, places=6)
        # the heuristic never overestimates the real remaining cost (so A* stays optimal)
        prob = rf.build_problem(self.table, self.nodes, "2000", "3002")
        for site in self.nodes["scats_number"]:
            if site == "3002":
                continue
            p = prob.with_changes(origin=site)
            goal, _ = astar(p)
            if goal is not None:
                self.assertLessEqual(cost_heuristic(p, site), goal.path_cost + 1e-9)

    def test_route_times(self):                                             # RF-03
        driving_per_km = 3600 / tt.flow_to_speed(600.0)
        for r in self.find().routes:
            self.assertAlmostEqual(r.travel_time_s, r.legs["travel_time_s"].sum(), places=6)
            self.assertEqual(len(r.legs), len(r.path) - 1)
            # driving time + 30 s delay for every intersection passed
            self.assertAlmostEqual(r.travel_time_s, r.distance_km * driving_per_km + 30.0 * len(r.legs), places=3)
        light = rf.find_routes("2000", "3002", "2006-10-26 08:00", "actual", edge_table=fake_edge_table(200))
        heavy = rf.find_routes("2000", "3002", "2006-10-26 08:00", "actual", edge_table=fake_edge_table(1400))
        self.assertLess(light.routes[0].travel_time_s, heavy.routes[0].travel_time_s)   # traffic slows routes

    def test_input_validation(self):                                        # RF-04
        self.assertEqual(self.find("970", "2000").origin, "0970")            # site numbers are zero-padded
        for kwargs in ({"k": 6},                                             # brief: up to 5
                       {"o": "2000", "d": "2000"},                           # same origin and destination
                       {"o": "9999"},                                        # unknown site
                       {"method": "DIJKSTRA"}):                              # unknown search method
            with self.subTest(**kwargs), self.assertRaises(ValueError):
                self.find(**kwargs)


if __name__ == "__main__":
    unittest.main()
