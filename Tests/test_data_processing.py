"""Tests for the data-processing output (Data/Processed_Data). TC-01 ... TC-05."""
import json

import numpy as np

from conftest import PROCESSED

EXPECTED_SITES = 40
ROWS_PER_SITE = 31 * 96            # October = 31 days x 96 slots of 15 minutes


def test_tc01_no_missing_or_negative_flow(clean):
    """TC-01: every flow value exists and is >= 0 (a car count cannot be negative)."""
    assert not clean["flow"].isna().any()
    assert (clean["flow"] >= 0).all()


def test_tc02_every_site_has_a_full_month(clean):
    """TC-02: 40 sites, each with exactly 2976 rows (31 days x 96 slots)."""
    sizes = clean.groupby("scats_number").size()
    assert len(sizes) == EXPECTED_SITES
    assert (sizes == ROWS_PER_SITE).all()


def test_tc03_time_steps_are_exactly_15_minutes(clean):
    """TC-03: no gaps and no duplicate timestamps inside any site."""
    for site, g in clean.groupby("scats_number"):
        steps = g.sort_values("datetime")["datetime"].diff().dropna().unique()
        assert list(steps) == [np.timedelta64(15, "m")], f"site {site} has irregular steps"


def test_tc04_metadata_matches_data_and_map_area(clean, meta):
    """TC-04: same sites in both files, and all coordinates lie in the Boroondara area."""
    assert set(meta["scats_number"]) == set(clean["scats_number"])
    assert meta["latitude"].between(-37.90, -37.75).all()
    assert meta["longitude"].between(144.95, 145.15).all()


def test_tc05_quality_report_passed_and_bad_sites_flagged(meta):
    """TC-05: all validation checks are True; the 2 low-quality sites are flagged."""
    report = json.loads((PROCESSED / "quality_report.json").read_text())
    assert all(report["validation_checks"].values())
    assert sorted(meta.loc[meta["low_quality"], "scats_number"]) == sorted(report["low_quality_sites"])
