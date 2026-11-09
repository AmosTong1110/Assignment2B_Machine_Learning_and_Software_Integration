"""
conftest.py  -  shared setup for all tests (pytest loads this file automatically)
===============================================================================
Run the tests from the project root:   python -m pytest -v
"""
import sys
from pathlib import Path

import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parents[1]
# 'data_processing.py' currently lives in Data/. Put the folders on the import path so the
# tests work from any working directory (move the file into Source/ later and delete one line).
for p in (ROOT, ROOT / "Data", ROOT / "Source" / "Routing"):
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))

PROCESSED = ROOT / "Data" / "Processed_Data"
RESULTS = ROOT / "Results"


@pytest.fixture(scope="session")
def clean() -> pd.DataFrame:
    """The real cleaned dataset (40 sites x 2976 rows)."""
    return pd.read_csv(PROCESSED / "scats_clean_15min.csv",
                       dtype={"scats_number": str}, parse_dates=["datetime"])


@pytest.fixture(scope="session")
def meta() -> pd.DataFrame:
    """One row per SCATS site (coordinates + quality flags)."""
    return pd.read_csv(PROCESSED / "site_metadata.csv", dtype={"scats_number": str})


def make_site(n: int = 100, site: str = "9999", flow=None) -> pd.DataFrame:
    """A small FAKE site (flow = 0, 1, 2, ...) so tests know the exact right answer."""
    return pd.DataFrame({
        "scats_number": site,
        "datetime": pd.date_range("2006-10-02 00:00", periods=n, freq="15min"),  # a Monday
        "flow": list(range(n)) if flow is None else flow,
        "imputed": 0,
    })
