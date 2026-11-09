"""Tests for the road-link (edges) builder and Data/edges.csv. TC-21 ... TC-26."""
import pandas as pd
import pytest

import build_edges as be
from conftest import ROOT


@pytest.fixture(scope="module")
def edges() -> pd.DataFrame:
    return pd.read_csv(ROOT / "Data" / "edges.csv", dtype={"from_site": str, "to_site": str})


def _toy_meta():
    """Three sites in a straight line (A -> B -> C), plus a site F that is very far away."""
    return pd.DataFrame({"scats_number": ["A", "B", "C", "F"],
                         "latitude": [-37.80, -37.81, -37.82, -36.00],
                         "longitude": [145.0] * 4})


def test_tc21_haversine_distance():
    """TC-21: same point = 0 km; one degree of latitude = about 111.2 km."""
    assert be.haversine_km(-37.8, 145.0, -37.8, 145.0) == 0.0
    assert be.haversine_km(-37.0, 145.0, -38.0, 145.0) == pytest.approx(111.2, abs=0.3)


def test_tc22_road_names_are_parsed_from_detector_locations():
    """TC-22: 'WARRIGAL_RD N of TOORAK_RD' gives both roads; upper-case 'OF' also works."""
    locs = pd.DataFrame({"site": ["2000", "2000", "3682"],
                         "location": ["WARRIGAL_RD N of TOORAK_RD", "TOORAK_RD W of WARRIGAL_RD",
                                      "RIVERSDALE_RD E OF WARRIGAL_RD"]})
    roads = be.roads_per_site(locs)
    assert roads["2000"] == {"WARRIGAL_RD", "TOORAK_RD"}
    assert roads["3682"] == {"RIVERSDALE_RD", "WARRIGAL_RD"}


def test_tc23_sites_on_one_road_are_chained_in_order():
    """TC-23: A, B, C on one road -> links A-B and B-C (both ways), but NOT A-C."""
    roads = {"A": {"R1"}, "B": {"R1"}, "C": {"R1"}}
    e = be.build_edges(roads, _toy_meta())
    pairs = set(zip(e["from_site"], e["to_site"]))
    assert pairs == {("A", "B"), ("B", "A"), ("B", "C"), ("C", "B")}


def test_tc24_too_long_links_are_dropped():
    """TC-24: a link longer than the limit (a missing site in between) is not created."""
    e = be.build_edges({"C": {"R1"}, "F": {"R1"}}, _toy_meta())
    assert e.empty


def test_tc25_edges_csv_is_consistent(edges, meta):
    """TC-25: no self-loops, no duplicates, positive distances, known sites, links go both ways."""
    assert (edges["from_site"] != edges["to_site"]).all()
    assert not edges.duplicated(["from_site", "to_site"]).any()
    assert (edges["distance_km"] > 0).all()
    assert set(edges["from_site"]) | set(edges["to_site"]) <= set(meta["scats_number"])
    pairs = set(zip(edges["from_site"], edges["to_site"]))
    assert all((b, a) in pairs for a, b in pairs)


def test_tc26_route_example_from_the_brief_exists(edges):
    """TC-26: the brief's example trip 2000 -> 3002 is possible (a path exists in the graph)."""
    adj = {}
    for a, b in zip(edges["from_site"], edges["to_site"]):
        adj.setdefault(a, set()).add(b)
    seen, todo = set(), ["2000"]
    while todo:
        n = todo.pop()
        if n not in seen:
            seen.add(n)
            todo += list(adj.get(n, set()) - seen)
    assert "3002" in seen
