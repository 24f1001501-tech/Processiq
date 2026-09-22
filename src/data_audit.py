"""
ProcessIQ - Step 1: Data Quality Audit

Validates the UCI Gas Turbine CO and NOx Emission dataset before any modelling.
Answers one question: is this data fit for purpose?

Run:  python src/data_audit.py
"""

from pathlib import Path

import numpy as np
import pandas as pd

RAW_DIR = Path(__file__).resolve().parents[1] / "data" / "raw"
YEARS = [2011, 2012, 2013, 2014, 2015]

# Physically valid ranges for a combined-cycle gas turbine.
# Sourced from the UCI dataset documentation and general CCGT operating envelopes.
PHYSICAL_RANGES = {
    "AT":   (-10.0, 50.0,    "Ambient Temperature",           "C"),
    "AP":   (980.0, 1040.0,  "Ambient Pressure",              "mbar"),
    "AH":   (0.0,   100.0,   "Ambient Humidity",              "%"),
    "AFDP": (0.0,   10.0,    "Air Filter Differential Press", "mbar"),
    "GTEP": (0.0,   45.0,    "Gas Turbine Exhaust Pressure",  "mbar"),
    "TIT":  (900.0, 1110.0,  "Turbine Inlet Temperature",     "C"),
    "TAT":  (500.0, 560.0,   "Turbine After Temperature",     "C"),
    "TEY":  (0.0,   200.0,   "Turbine Energy Yield",          "MWH"),
    "CDP":  (0.0,   20.0,    "Compressor Discharge Pressure", "mbar"),
    "CO":   (0.0,   100.0,   "Carbon Monoxide",               "mg/m3"),
    "NOX":  (0.0,   200.0,   "Nitrogen Oxides",               "mg/m3"),
}


def banner(text):
    print(f"\n{'=' * 78}\n{text}\n{'=' * 78}")


def load_raw():
    """Load each year separately so we can compare them before combining."""
    frames = {}
    for year in YEARS:
        path = RAW_DIR / f"gt_{year}.csv"
        frames[year] = pd.read_csv(path)
    return frames


def check_schema(frames):
    banner("1. SCHEMA CONSISTENCY")
    reference = list(frames[YEARS[0]].columns)
    print(f"Reference columns ({YEARS[0]}): {reference}")

    all_match = True
    for year, df in frames.items():
        match = list(df.columns) == reference
        all_match &= match
        status = "OK" if match else "MISMATCH"
        print(f"  {year}: {df.shape[0]:>5} rows x {df.shape[1]} cols   [{status}]")

    print(f"\nVerdict: {'All years share an identical schema.' if all_match else 'SCHEMA DRIFT DETECTED.'}")
    return all_match


def check_missing(df):
    banner("2. MISSING VALUES")
    missing = df.isnull().sum()
    total = missing.sum()
    if total == 0:
        print("No missing values in any column.")
    else:
        print(missing[missing > 0].to_string())
    print(f"\nVerdict: {total} missing cells across {len(df):,} rows x {df.shape[1]} cols.")
    return total


def check_duplicates(df):
    banner("3. DUPLICATE ROWS")
    # Exclude the synthetic 'year' column - we care about identical sensor readings.
    sensor_cols = [c for c in df.columns if c != "year"]
    dupes = df.duplicated(subset=sensor_cols).sum()
    print(f"Fully identical sensor rows: {dupes}")
    if dupes:
        pct = 100 * dupes / len(df)
        print(f"  ({pct:.3f}% of the dataset)")
        print("  Note: on hourly plant data, a small number of identical rows is")
        print("  plausible during steady-state operation - not necessarily an error.")
    print(f"\nVerdict: {dupes} duplicate rows.")
    return dupes


def check_physical_ranges(df):
    banner("4. PHYSICAL PLAUSIBILITY (domain range checks)")
    print(f"{'Col':<6} {'Description':<32} {'Min':>9} {'Max':>9} {'Violations':>11}")
    print("-" * 78)

    violations = {}
    for col, (lo, hi, desc, unit) in PHYSICAL_RANGES.items():
        series = df[col]
        bad = ((series < lo) | (series > hi)).sum()
        violations[col] = bad
        flag = "" if bad == 0 else "  <-- CHECK"
        print(f"{col:<6} {desc:<32} {series.min():>9.2f} {series.max():>9.2f} {bad:>11}{flag}")

    total = sum(violations.values())
    print(f"\nVerdict: {total} readings fall outside expected physical envelopes.")
    return violations


def check_outliers(df):
    banner("5. STATISTICAL OUTLIERS (IQR method)")
    print(f"{'Col':<6} {'Q1':>10} {'Q3':>10} {'IQR':>10} {'Outliers':>10} {'% of data':>10}")
    print("-" * 78)

    sensor_cols = [c for c in df.columns if c != "year"]
    results = {}
    for col in sensor_cols:
        q1, q3 = df[col].quantile([0.25, 0.75])
        iqr = q3 - q1
        lo, hi = q1 - 3 * iqr, q3 + 3 * iqr  # 3x IQR = extreme outliers only
        n_out = ((df[col] < lo) | (df[col] > hi)).sum()
        results[col] = n_out
        print(f"{col:<6} {q1:>10.2f} {q3:>10.2f} {iqr:>10.2f} {n_out:>10} {100*n_out/len(df):>9.2f}%")

    print("\nNote: 3x IQR flags only EXTREME outliers. On real sensor data these are")
    print("often genuine transients (startup/shutdown), not errors - inspect before removing.")
    return results


def check_stuck_sensors(df):
    banner("6. STUCK-SENSOR DETECTION (consecutive identical readings)")
    print("A sensor repeating the exact same float for many hours suggests a frozen tag.")
    print(f"\n{'Col':<6} {'Longest run':>12} {'Interpretation':<40}")
    print("-" * 78)

    sensor_cols = [c for c in df.columns if c != "year"]
    for col in sensor_cols:
        # Length of the longest run of identical consecutive values
        vals = df[col].values
        changes = np.concatenate(([True], vals[1:] != vals[:-1]))
        run_ids = np.cumsum(changes)
        longest = pd.Series(run_ids).value_counts().max()

        if longest >= 24:
            interp = "SUSPICIOUS - flat for >=24 consecutive readings"
        elif longest >= 6:
            interp = "Minor - plausible steady-state"
        else:
            interp = "Healthy - sensor varies normally"
        print(f"{col:<6} {longest:>12} {interp:<40}")


def check_year_drift(frames):
    banner("7. YEAR-OVER-YEAR DRIFT (is the plant changing over time?)")
    print("This is the signal that makes a degradation/diagnosis story possible.\n")

    key_cols = ["TEY", "TIT", "NOX", "CO", "AFDP"]
    summary = pd.DataFrame(
        {year: frames[year][key_cols].mean() for year in YEARS}
    ).T
    summary.index.name = "year"
    print(summary.round(3).to_string())

    print("\nRelative change, 2011 -> 2015:")
    for col in key_cols:
        first, last = summary[col].iloc[0], summary[col].iloc[-1]
        pct = 100 * (last - first) / first
        direction = "up" if pct > 0 else "down"
        print(f"  {col:<5} {first:>9.3f} -> {last:>9.3f}   ({pct:+.1f}%, {direction})")


def check_compliance_headroom(df):
    banner("8. EMISSIONS COMPLIANCE HEADROOM")
    print("The regulatory hook for the agent. Thresholds below are indicative EU")
    print("industrial-emissions limits for gas turbines; documented as assumptions.\n")

    thresholds = {"NOX": 70.0, "CO": 20.0}
    for col, limit in thresholds.items():
        series = df[col]
        breaches = (series > limit).sum()
        pct = 100 * breaches / len(series)
        print(f"{col}: limit {limit} mg/m3")
        print(f"   mean {series.mean():.2f} | p95 {series.quantile(0.95):.2f} | max {series.max():.2f}")
        print(f"   readings above limit: {breaches:,} ({pct:.2f}% of all hours)")
        headroom = 100 * (limit - series.mean()) / limit
        print(f"   average headroom to limit: {headroom:.1f}%\n")


def verdict(df, n_missing, n_dupes, range_violations):
    banner("OVERALL VERDICT")
    issues = []
    if n_missing:
        issues.append(f"{n_missing} missing cells")
    if n_dupes:
        issues.append(f"{n_dupes} duplicate rows")
    bad_ranges = {k: v for k, v in range_violations.items() if v > 0}
    if bad_ranges:
        issues.append(f"range violations in {list(bad_ranges.keys())}")

    print(f"Dataset: {len(df):,} hourly readings x {df.shape[1]-1} sensor tags, 2011-2015")
    if not issues:
        print("\nStatus: CLEAN. No missing values, no duplicates, all readings")
        print("within physical envelopes. Safe to proceed to feature engineering.")
    else:
        print(f"\nStatus: USABLE with noted caveats -> {'; '.join(issues)}")
        print("None of these are blocking; handling documented in the cleaning step.")


def main():
    frames = load_raw()
    check_schema(frames)

    # Combine with a year tag
    df = pd.concat(
        [f.assign(year=y) for y, f in frames.items()],
        ignore_index=True,
    )

    n_missing = check_missing(df)
    n_dupes = check_duplicates(df)
    range_violations = check_physical_ranges(df)
    check_outliers(df)
    check_stuck_sensors(df)
    check_year_drift(frames)
    check_compliance_headroom(df)
    verdict(df, n_missing, n_dupes, range_violations)


if __name__ == "__main__":
    main()
