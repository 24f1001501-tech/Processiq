"""
ProcessIQ - Step 2: Cleaning & Consolidation

Strategy: CONSERVATIVE. Nothing is silently deleted.
  - Physically impossible values are corrected (and logged).
  - Exact duplicates are dropped (and logged).
  - Everything else suspicious is FLAGGED in a quality column, never removed,
    so downstream analysis can choose to include or exclude it.

This preserves a full audit trail, which is the point: an analyst should be
able to reproduce every decision made to this dataset.

Run:  python src/clean_data.py
Out:  data/processed/turbine_clean.parquet
"""

from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
RAW_DIR = ROOT / "data" / "raw"
OUT_DIR = ROOT / "data" / "processed"
YEARS = [2011, 2012, 2013, 2014, 2015]

SENSOR_COLS = ["AT", "AP", "AH", "AFDP", "GTEP", "TIT", "TAT", "TEY", "CDP", "CO", "NOX"]

# Minimum consecutive identical readings to call a sensor "stuck".
# 24 = a full day of an unchanging float, which no real analogue sensor does.
STUCK_RUN_THRESHOLD = 24

# Outlier detection uses 3x IQR (extreme only, not the usual 1.5x).
IQR_MULTIPLIER = 3.0

# Indicative EU industrial-emissions limits for gas turbines (mg/m3).
# Documented as an assumption, not a measured plant permit value.
EMISSION_LIMITS = {"NOX": 70.0, "CO": 20.0}


class CleaningLog:
    """Accumulates every transformation so it can be printed and saved."""

    def __init__(self):
        self.entries = []

    def add(self, step, detail, n_affected):
        self.entries.append({"step": step, "detail": detail, "rows_affected": n_affected})
        print(f"  [{step}] {detail} -> {n_affected:,} rows affected")

    def to_frame(self):
        return pd.DataFrame(self.entries)


def load_and_tag(log):
    """Load each year, tag it, and build a sequential operating-hour index."""
    print("\n1. LOADING RAW DATA")
    frames = []
    for year in YEARS:
        df = pd.read_csv(RAW_DIR / f"gt_{year}.csv")
        df["year"] = year
        # Sequential position within the year. The UCI data is ordered hourly but
        # has fewer rows than hours in a year (plant downtime / gaps), so this is
        # an OPERATING-hour index, not a wall-clock timestamp.
        df["operating_hour"] = np.arange(len(df))
        frames.append(df)
        print(f"  {year}: {len(df):,} readings")

    df = pd.concat(frames, ignore_index=True)
    log.add("load", f"Concatenated {len(YEARS)} yearly files", len(df))
    return df


def add_synthetic_timestamp(df, log):
    """
    Build an approximate datetime for plotting and time-range filtering.

    IMPORTANT ASSUMPTION: the raw data has no timestamp column and contains
    fewer rows than hours-in-year, so true wall-clock time is unrecoverable.
    We lay operating hours sequentially from Jan 1 of each year. This is valid
    for ordering and for within-year trends, but gaps mean these timestamps
    drift from real calendar dates. Documented, not hidden.
    """
    print("\n2. SYNTHETIC TIMESTAMP")
    df["timestamp"] = pd.to_datetime(df["year"].astype(str) + "-01-01") + pd.to_timedelta(
        df["operating_hour"], unit="h"
    )
    log.add(
        "timestamp",
        "Derived approximate timestamp from sequential operating hours (see docstring)",
        len(df),
    )
    return df


def fix_impossible_values(df, log):
    """Correct values that violate physical law. Currently: relative humidity > 100%."""
    print("\n3. PHYSICALLY IMPOSSIBLE VALUES")
    n_bad = (df["AH"] > 100.0).sum()
    if n_bad:
        max_before = df["AH"].max()
        df["AH_was_clipped"] = df["AH"] > 100.0
        df.loc[df["AH"] > 100.0, "AH"] = 100.0
        log.add(
            "clip_AH",
            f"Relative humidity clipped to 100% (max observed was {max_before:.2f}%, "
            "indicating sensor calibration drift)",
            n_bad,
        )
    else:
        df["AH_was_clipped"] = False
        print("  No impossible values found.")
    return df


def drop_exact_duplicates(df, log):
    """Remove rows where every sensor reading is byte-identical."""
    print("\n4. EXACT DUPLICATES")
    before = len(df)
    df = df.drop_duplicates(subset=SENSOR_COLS, keep="first").reset_index(drop=True)
    removed = before - len(df)
    log.add("drop_duplicates", "Dropped rows identical across all 11 sensor tags", removed)
    return df


def flag_stuck_sensors(df, log):
    """
    Flag (do not remove) windows where a sensor reports an unchanging value
    for an implausibly long run - the signature of a frozen/failed tag.
    """
    print("\n5. STUCK-SENSOR WINDOWS (flagged, not removed)")
    df["stuck_sensor_flag"] = False
    df["stuck_sensor_tags"] = ""

    for col in SENSOR_COLS:
        vals = df[col].values
        # Identify runs of identical consecutive values
        changes = np.concatenate(([True], vals[1:] != vals[:-1]))
        run_id = np.cumsum(changes)
        run_len = pd.Series(run_id).map(pd.Series(run_id).value_counts())

        stuck = (run_len >= STUCK_RUN_THRESHOLD).values
        n_stuck = int(stuck.sum())
        if n_stuck:
            df.loc[stuck, "stuck_sensor_flag"] = True
            df.loc[stuck, "stuck_sensor_tags"] = (
                df.loc[stuck, "stuck_sensor_tags"] + col + ";"
            )
            log.add(
                "flag_stuck",
                f"{col} held a constant value for >={STUCK_RUN_THRESHOLD} consecutive readings",
                n_stuck,
            )

    if not df["stuck_sensor_flag"].any():
        print("  No stuck-sensor windows detected.")
    return df


def flag_outliers(df, log):
    """Flag extreme statistical outliers (3x IQR). Kept - often genuine transients."""
    print("\n6. EXTREME OUTLIERS (flagged, not removed)")
    df["outlier_flag"] = False
    df["outlier_tags"] = ""

    for col in SENSOR_COLS:
        q1, q3 = df[col].quantile([0.25, 0.75])
        iqr = q3 - q1
        lo, hi = q1 - IQR_MULTIPLIER * iqr, q3 + IQR_MULTIPLIER * iqr
        mask = (df[col] < lo) | (df[col] > hi)
        n = int(mask.sum())
        if n:
            df.loc[mask, "outlier_flag"] = True
            df.loc[mask, "outlier_tags"] = df.loc[mask, "outlier_tags"] + col + ";"
            log.add("flag_outlier", f"{col} beyond {IQR_MULTIPLIER}x IQR fence", n)

    return df


def add_engineered_features(df, log):
    """
    Derive the features the agent will actually reason about.

    These are domain-driven, not automatic: each one answers a question a
    process engineer would ask about turbine performance.
    """
    print("\n7. FEATURE ENGINEERING")

    # Thermal spread across the turbine: inlet minus exhaust temperature.
    # A narrowing spread at constant load suggests degrading expansion efficiency.
    df["thermal_spread"] = df["TIT"] - df["TAT"]

    # Compression ratio proxy: discharge pressure over ambient pressure.
    df["pressure_ratio"] = df["CDP"] / (df["AP"] / 1000.0)

    # Specific yield: energy produced per unit of inlet temperature.
    # A crude heat-rate proxy - the lower this is, the more fuel-derived heat
    # is needed per MWh, which is exactly the MRPL heat-rate deviation question.
    df["specific_yield"] = df["TEY"] / df["TIT"]

    # Total regulated emissions burden, normalised against each limit.
    # 1.0 means "at the limit"; >1.0 means in breach.
    df["nox_utilisation"] = df["NOX"] / EMISSION_LIMITS["NOX"]
    df["co_utilisation"] = df["CO"] / EMISSION_LIMITS["CO"]

    # Binary compliance state per reading.
    df["nox_breach"] = df["NOX"] > EMISSION_LIMITS["NOX"]
    df["co_breach"] = df["CO"] > EMISSION_LIMITS["CO"]

    # Combustion-tuning indicator. The NOx-CO tradeoff means these move in
    # opposite directions when flame temperature is adjusted. Their ratio is a
    # compact signal for "how is this unit tuned right now?".
    df["nox_co_ratio"] = df["NOX"] / df["CO"].replace(0, np.nan)

    new_cols = [
        "thermal_spread", "pressure_ratio", "specific_yield",
        "nox_utilisation", "co_utilisation", "nox_breach", "co_breach", "nox_co_ratio",
    ]
    log.add("feature_engineering", f"Derived {len(new_cols)} features: {', '.join(new_cols)}", len(df))
    return df


def summarise(df):
    print("\n" + "=" * 78)
    print("CLEANING SUMMARY")
    print("=" * 78)
    print(f"Final dataset: {len(df):,} rows x {df.shape[1]} columns")
    print(f"Date span:     {df['timestamp'].min().date()} -> {df['timestamp'].max().date()} (approx.)")
    print(f"\nQuality flags (rows retained, flagged for optional exclusion):")
    print(f"  AH clipped to 100%:      {df['AH_was_clipped'].sum():,}")
    print(f"  Stuck-sensor windows:    {df['stuck_sensor_flag'].sum():,}")
    print(f"  Extreme outliers:        {df['outlier_flag'].sum():,}")
    clean_rows = (~df["stuck_sensor_flag"] & ~df["outlier_flag"]).sum()
    print(f"  Fully unflagged rows:    {clean_rows:,} ({100*clean_rows/len(df):.1f}%)")

    print(f"\nCompliance state (against indicative limits):")
    print(f"  NOx breaches: {df['nox_breach'].sum():,} hours ({100*df['nox_breach'].mean():.2f}%)")
    print(f"  CO breaches:  {df['co_breach'].sum():,} hours ({100*df['co_breach'].mean():.2f}%)")


def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    log = CleaningLog()

    df = load_and_tag(log)
    df = add_synthetic_timestamp(df, log)
    df = fix_impossible_values(df, log)
    df = drop_exact_duplicates(df, log)
    df = flag_stuck_sensors(df, log)
    df = flag_outliers(df, log)
    df = add_engineered_features(df, log)

    summarise(df)

    out_path = OUT_DIR / "turbine_clean.parquet"
    df.to_parquet(out_path, index=False)

    log_path = OUT_DIR / "cleaning_log.csv"
    log.to_frame().to_csv(log_path, index=False)

    print(f"\nWrote {out_path.relative_to(ROOT)}  ({out_path.stat().st_size / 1e6:.2f} MB)")
    print(f"Wrote {log_path.relative_to(ROOT)}  (full audit trail)")


if __name__ == "__main__":
    main()
