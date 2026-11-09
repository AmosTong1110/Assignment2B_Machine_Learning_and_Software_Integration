"""
data_processing.py  -  COS30019 Assignment 2B (TBRGS)
=====================================================
Turns the raw VicRoads file  Scats_Data_October_2006.xls  into a clean,
validated, ML-ready dataset.

Pipeline (each step is its own function so it can be tested and explained):

    load_raw()              read the 'Data' sheet, fix types
    drop_faulty_detectors() remove detectors that are mostly dead (zeros)
    build_site_series()     sum detectors -> one 15-min flow series per site
    flag_bad_values()       mark outages (long zero runs) and spikes as NaN
    fill_gaps()             interpolate short gaps, profile-fill long gaps
    build_site_metadata()   one clean lat/lon row per site
    validate()              assert the data meets the quality checklist
    make_datasets()         chronological split + MinMax scaling + windows

Run:   python data_processing.py --xls Scats_Data_October_2006.xls --out clean
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

# --------------------------------------------------------------------------
# Configuration (keep every "magic number" here, so the report can cite it)
# --------------------------------------------------------------------------
CONFIG = {
    "sheet": "Data",
    "header_row": 1,              # real header is on the 2nd row of the sheet
    "slots_per_day": 96,          # 24h x 4 (15-minute intervals)
    "faulty_zero_share": 0.30,    # detector with >30% zeros all month = faulty
    "outage_zero_run": 8,         # site total = 0 for >= 8 slots (2h) = outage
    "spike_mads": 8.0,            # spike = > median + 8 robust SDs (same slot)
    "spike_min_excess": 60,       # ...and at least 60 veh/15min above median
    "spike_min_ratio": 1.5,       # ...and at least 50% above the typical value
    "max_interp_gap": 4,          # interpolate gaps up to 4 slots (1h)
    "max_imputed_share": 0.25,    # site with >25% imputed values = low quality
    "window": 12,                 # use past 12 slots (3h) as model input
    "test_days": 7,               # last 7 days = test set
}
FLOW_COLS = [f"V{i:02d}" for i in range(CONFIG["slots_per_day"])]


# --------------------------------------------------------------------------
# 1. Load
# --------------------------------------------------------------------------
def load_raw(xls_path: str) -> pd.DataFrame:
    """Read the 'Data' sheet. SCATS numbers stay as 4-char strings ('0970')."""
    df = pd.read_excel(
        xls_path,
        sheet_name=CONFIG["sheet"],
        header=CONFIG["header_row"],
        dtype={"SCATS Number": str},
    )
    df["SCATS Number"] = df["SCATS Number"].str.strip().str.zfill(4)
    df["Location"] = df["Location"].str.strip()
    df["day"] = pd.to_datetime(df["Date"]).dt.normalize()
    df[FLOW_COLS] = df[FLOW_COLS].apply(pd.to_numeric, errors="coerce")
    return df


# --------------------------------------------------------------------------
# 2. Remove faulty detectors
# --------------------------------------------------------------------------
def drop_faulty_detectors(df: pd.DataFrame):
    """
    A detector (one approach of an intersection) that reads 0 for a large share
    of the month is treated as broken. Dropping it for ALL days keeps the
    intersection total consistent (a model hates a sensor that switches on/off).
    """
    zero_share = (
        df.groupby(["SCATS Number", "Location"])[FLOW_COLS]
        .apply(lambda g: float((g.to_numpy() == 0).mean()))
        .rename("zero_share")
    )
    faulty = zero_share[zero_share > CONFIG["faulty_zero_share"]].reset_index()
    if len(faulty):
        keys = set(zip(faulty["SCATS Number"], faulty["Location"]))
        mask = [(s, l) in keys for s, l in zip(df["SCATS Number"], df["Location"])]
        df = df.loc[~np.array(mask)].copy()
    return df, faulty


# --------------------------------------------------------------------------
# 3. One time series per site
# --------------------------------------------------------------------------
def build_site_series(df: pd.DataFrame):
    """
    Sum all detector rows of a site for each day -> intersection volume.
    Days where fewer detectors reported than usual are INCOMPLETE (the sum would
    be too low), so they are set to NaN and repaired later.
    Returns: wide (site -> array[days, 96]), days index, incomplete-day table.
    """
    days = pd.date_range(df["day"].min(), df["day"].max(), freq="D")
    n_rows = df.groupby(["SCATS Number", "day"]).size().unstack(0).reindex(days)
    usual = n_rows.apply(lambda s: s.mode().iloc[0])          # modal #detectors
    incomplete = n_rows.lt(usual, axis=1) | n_rows.isna()     # days x sites

    series = {}
    for site, g in df.groupby("SCATS Number"):
        daily = g.groupby("day")[FLOW_COLS].sum(min_count=1).reindex(days)
        daily.loc[incomplete[site].to_numpy()] = np.nan
        series[site] = daily.to_numpy(dtype=float)            # (days, 96)
    return series, days, incomplete


# --------------------------------------------------------------------------
# 4. Flag bad values
# --------------------------------------------------------------------------
def _zero_runs(is_zero: np.ndarray, min_len: int) -> np.ndarray:
    """Boolean mask of positions inside runs of >= min_len consecutive True."""
    out = np.zeros_like(is_zero, dtype=bool)
    start = None
    for i, z in enumerate(np.append(is_zero, False)):
        if z and start is None:
            start = i
        elif not z and start is not None:
            if i - start >= min_len:
                out[start:i] = True
            start = None
    return out


def flag_bad_values(series: dict, days: pd.DatetimeIndex):
    """Set outages and spikes to NaN. Returns cleaned dict + counts per site."""
    dow = days.dayofweek.to_numpy()
    log = {}
    for site, arr in series.items():
        flat = arr.reshape(-1)
        # (a) outage: total flow exactly 0 for >= 2 hours in a row
        outage = _zero_runs(flat == 0, CONFIG["outage_zero_run"]).reshape(arr.shape)
        arr = np.where(outage, np.nan, arr)

        # (b) spike: compare with the SAME weekday and SAME time-of-day.
        #     Same slot -> real rush-hour peaks are not flagged.
        #     Same weekday -> busy Saturday nights are not compared with quiet Sundays.
        spike = np.zeros(arr.shape, dtype=bool)
        for d in range(7):
            mask = dow == d
            block = arr[mask]
            if block.shape[0] < 3:
                continue
            med = np.nanmedian(block, axis=0)
            mad = np.nanmedian(np.abs(block - med), axis=0) * 1.4826
            limit = med + np.maximum(CONFIG["spike_mads"] * mad, CONFIG["spike_min_excess"])
            spike[mask] = (block > limit) & (block > CONFIG["spike_min_ratio"] * med)
        arr = np.where(spike, np.nan, arr)

        series[site] = arr
        log[site] = {"outage_values": int(outage.sum()), "spike_values": int(spike.sum())}
    return series, log


# --------------------------------------------------------------------------
# 5. Fill gaps
# --------------------------------------------------------------------------
def fill_gaps(series: dict, days: pd.DatetimeIndex) -> pd.DataFrame:
    """
    Short gaps (<= 1h)  -> linear interpolation.
    Long gaps / missing days -> average of the same site, same weekday,
    same time-of-day on other weeks (a typical-day "profile").
    Every repaired value is flagged with imputed=1 so it stays transparent.
    """
    frames = []
    index = pd.date_range(days[0], periods=len(days) * CONFIG["slots_per_day"], freq="15min")
    for site, arr in series.items():
        s = pd.Series(arr.reshape(-1), index=index)
        was_nan = s.isna()
        s = s.interpolate(method="linear", limit=CONFIG["max_interp_gap"], limit_area="inside")

        still = s.isna()
        if still.any():
            keys = [s.index.dayofweek, s.index.hour * 4 + s.index.minute // 15]
            profile = s.groupby(keys).transform("mean")            # weekday+slot mean
            s = s.fillna(profile)
            s = s.fillna(s.groupby(keys[1]).transform("mean"))      # fallback: slot mean

        frames.append(pd.DataFrame({
            "scats_number": site,
            "datetime": index,
            "flow": s.round(1).to_numpy(),
            "imputed": was_nan.astype(int).to_numpy(),
        }))
    return pd.concat(frames, ignore_index=True)


# --------------------------------------------------------------------------
# 6. Site metadata (coordinates)
# --------------------------------------------------------------------------
def build_site_metadata(raw: pd.DataFrame, clean: pd.DataFrame, log: dict) -> pd.DataFrame:
    """One row per site. Lat/lon = median of the non-zero detector coordinates."""
    coords = raw[(raw["NB_LATITUDE"] != 0) & (raw["NB_LONGITUDE"] != 0)]
    meta = (coords.groupby("SCATS Number")
            .agg(latitude=("NB_LATITUDE", "median"), longitude=("NB_LONGITUDE", "median"))
            .reindex(sorted(raw["SCATS Number"].unique())))
    meta["n_detectors"] = raw.groupby("SCATS Number")["Location"].nunique()
    meta["imputed_share"] = clean.groupby("scats_number")["imputed"].mean().round(4)
    meta["outage_values"] = pd.Series({k: v["outage_values"] for k, v in log.items()})
    meta["spike_values"] = pd.Series({k: v["spike_values"] for k, v in log.items()})
    meta["low_quality"] = meta["imputed_share"] > CONFIG["max_imputed_share"]
    meta["missing_coords"] = meta[["latitude", "longitude"]].isna().any(axis=1)
    meta.index.name = "scats_number"
    return meta.reset_index()


# --------------------------------------------------------------------------
# 7. Validate (the quality checklist, as code)
# --------------------------------------------------------------------------
def validate(clean: pd.DataFrame, meta: pd.DataFrame) -> dict:
    """Raise AssertionError if the cleaned data breaks any quality rule."""
    n_days = clean["datetime"].dt.normalize().nunique()
    expected = n_days * CONFIG["slots_per_day"]
    checks = {}

    checks["completeness_no_nan"] = not clean["flow"].isna().any()
    checks["validity_non_negative"] = bool((clean["flow"] >= 0).all())
    checks["uniqueness_no_duplicates"] = not clean.duplicated(["scats_number", "datetime"]).any()
    checks["rows_per_site_correct"] = bool((clean.groupby("scats_number").size() == expected).all())
    gaps = clean.sort_values(["scats_number", "datetime"]).groupby("scats_number")["datetime"].diff().dropna()
    checks["regular_15min_spacing"] = bool((gaps == pd.Timedelta("15min")).all())
    checks["sites_match_metadata"] = set(clean["scats_number"]) == set(meta["scats_number"])
    checks["plausible_max_flow"] = bool(clean["flow"].max() < 3000)   # veh / 15 min

    # plausibility: daytime busier than night for most sites (daily rhythm)
    hour = clean["datetime"].dt.hour
    day_mean = clean[hour.between(7, 18)].groupby("scats_number")["flow"].mean()
    night_mean = clean[hour.between(1, 4)].groupby("scats_number")["flow"].mean()
    checks["daily_rhythm_present"] = bool((day_mean > night_mean).mean() >= 0.95)

    failed = [k for k, v in checks.items() if not v]
    assert not failed, f"Data quality checks failed: {failed}"
    return checks


def load_clean(out_dir: str = "clean"):
    """Reload the saved files with correct types (keeps the '0970' leading zero)."""
    clean = pd.read_csv(Path(out_dir) / "scats_clean_15min.csv",
                        dtype={"scats_number": str}, parse_dates=["datetime"])
    meta = pd.read_csv(Path(out_dir) / "site_metadata.csv", dtype={"scats_number": str})
    return clean, meta


# --------------------------------------------------------------------------
# 8. ML-ready arrays  (split by TIME, scale on TRAIN only, sliding windows)
# --------------------------------------------------------------------------
class MinMax:
    """Tiny MinMax scaler (fitted on train data only -> no data leakage)."""
    def fit(self, x):
        self.lo, self.hi = float(np.min(x)), float(np.max(x))
        return self
    def transform(self, x):
        return (np.asarray(x) - self.lo) / (self.hi - self.lo + 1e-9)
    def inverse(self, x):
        return np.asarray(x) * (self.hi - self.lo + 1e-9) + self.lo


def make_windows(values: np.ndarray, window: int):
    """values[t-window : t] -> values[t].  Returns X (n, window, 1), y (n,)."""
    X = np.stack([values[i:i + window] for i in range(len(values) - window)])
    y = values[window:]
    return X[..., None].astype("float32"), y.astype("float32")


def make_datasets(clean: pd.DataFrame, window: int = None, test_days: int = None) -> dict:
    """Per-site train/test arrays, ready for LSTM / GRU (PyTorch or Keras)."""
    window = window or CONFIG["window"]
    test_days = test_days or CONFIG["test_days"]
    out = {}
    for site, g in clean.sort_values("datetime").groupby("scats_number"):
        v = g["flow"].to_numpy()
        cut = len(v) - test_days * CONFIG["slots_per_day"]          # time-based split
        scaler = MinMax().fit(v[:cut])                               # train only!
        s = scaler.transform(v)
        X_tr, y_tr = make_windows(s[:cut], window)
        X_te, y_te = make_windows(s[cut - window:], window)          # keep context
        out[site] = {"X_train": X_tr, "y_train": y_tr, "X_test": X_te, "y_test": y_te, "scaler": scaler}
    return out


# --------------------------------------------------------------------------
# Main: run the whole pipeline
# --------------------------------------------------------------------------
def run_pipeline(xls_path: str, out_dir: str = "clean") -> dict:
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)

    raw = load_raw(xls_path)
    n_raw_rows = len(raw)
    raw, faulty = drop_faulty_detectors(raw)
    series, days, incomplete = build_site_series(raw)
    raw_missing = {s: float(np.isnan(a).mean()) for s, a in series.items()}
    series, log = flag_bad_values(series, days)
    clean = fill_gaps(series, days)
    meta = build_site_metadata(raw, clean, log)
    checks = validate(clean, meta)

    clean.to_csv(out / "scats_clean_15min.csv", index=False)
    meta.to_csv(out / "site_metadata.csv", index=False)
    summary = {
        "raw_rows": n_raw_rows,
        "faulty_detectors_dropped": faulty[["SCATS Number", "Location", "zero_share"]].round(3).to_dict("records"),
        "incomplete_site_days": int(incomplete.to_numpy().sum()),
        "outage_values_flagged": int(sum(v["outage_values"] for v in log.values())),
        "spike_values_flagged": int(sum(v["spike_values"] for v in log.values())),
        "total_values": int(len(clean)),
        "imputed_values": int(clean["imputed"].sum()),
        "imputed_share_pct": round(100 * clean["imputed"].mean(), 2),
        "sites": int(clean["scats_number"].nunique()),
        "low_quality_sites": meta.loc[meta["low_quality"], "scats_number"].tolist(),
        "sites_missing_coords": meta.loc[meta["missing_coords"], "scats_number"].tolist(),
        "validation_checks": checks,
    }
    (out / "quality_report.json").write_text(json.dumps(summary, indent=2))
    return summary


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description="Clean the SCATS October 2006 dataset")
    ap.add_argument("--xls", default="Scats_Data_October_2006.xls")
    ap.add_argument("--out", default="clean")
    args = ap.parse_args()
    print(json.dumps(run_pipeline(args.xls, args.out), indent=2))
