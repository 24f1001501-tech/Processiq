"""
ProcessIQ - Step 3: Core Analysis

Tests the hypotheses that came out of the data audit, so the agent's later
claims rest on verified findings rather than plausible-sounding narrative.

H1: The plant underwent a combustion retune between 2013 and 2014
    (NOx dropped sharply, CO rose - the classic NOx-CO tradeoff).
H2: Turbine Inlet Temperature is the dominant driver of energy yield.
H3: Ambient temperature materially suppresses output (air density effect).

Run:  python src/analysis.py
"""

from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score, mean_absolute_percentage_error

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data" / "processed" / "turbine_clean.parquet"

OPERATING_PARAMS = ["AT", "AP", "AH", "AFDP", "GTEP", "TIT", "TAT", "CDP"]


def banner(text):
    print(f"\n{'=' * 78}\n{text}\n{'=' * 78}")


def load():
    df = pd.read_parquet(DATA)
    # Analysis uses unflagged rows only - flagged data stays available to the
    # agent, but conclusions should rest on trustworthy readings.
    clean = df[~df["stuck_sensor_flag"] & ~df["outlier_flag"]].copy()
    print(f"Loaded {len(df):,} rows; analysing {len(clean):,} unflagged rows "
          f"({100*len(clean)/len(df):.1f}%)")
    return df, clean


def h1_combustion_retune(clean):
    banner("H1: COMBUSTION RETUNE BETWEEN 2013 AND 2014")

    yearly = clean.groupby("year").agg(
        NOX=("NOX", "mean"),
        CO=("CO", "mean"),
        TEY=("TEY", "mean"),
        TIT=("TIT", "mean"),
        nox_breach_pct=("nox_breach", "mean"),
    )
    yearly["nox_breach_pct"] *= 100
    print(yearly.round(3).to_string())

    # Year-on-year step changes
    print("\nYear-on-year change in NOx and CO:")
    print(f"{'Transition':<14} {'dNOx':>10} {'dNOx %':>9} {'dCO':>10} {'dCO %':>9}")
    print("-" * 78)
    years = sorted(yearly.index)
    steps = []
    for a, b in zip(years[:-1], years[1:]):
        d_nox = yearly.loc[b, "NOX"] - yearly.loc[a, "NOX"]
        d_co = yearly.loc[b, "CO"] - yearly.loc[a, "CO"]
        p_nox = 100 * d_nox / yearly.loc[a, "NOX"]
        p_co = 100 * d_co / yearly.loc[a, "CO"]
        steps.append((f"{a}->{b}", d_nox, p_nox, d_co, p_co))
        print(f"{a}->{b:<9} {d_nox:>10.2f} {p_nox:>8.1f}% {d_co:>10.3f} {p_co:>8.1f}%")

    # The retune signature: largest NOx drop, in the opposite direction to CO.
    biggest = min(steps, key=lambda s: s[1])
    print(f"\nLargest NOx reduction: {biggest[0]} ({biggest[1]:.2f} mg/m3, {biggest[2]:.1f}%)")
    opposing = biggest[1] < 0 and biggest[3] > 0
    print(f"CO moved in the opposite direction over the same step: {opposing}")

    # Correlation of the tradeoff across all years
    corr = clean["NOX"].corr(clean["CO"])
    print(f"\nNOx-CO correlation across full dataset: {corr:+.3f}")

    print("\nVERDICT:")
    if opposing and biggest[2] < -8:
        print(f"  H1 SUPPORTED. The {biggest[0]} transition shows a {abs(biggest[2]):.1f}% NOx")
        print(f"  reduction alongside a {biggest[4]:+.1f}% CO change - the expected")
        print("  signature of lowering flame temperature to suppress thermal NOx.")
        print(f"  NOx breach rate fell from {yearly.loc[2013, 'nox_breach_pct']:.1f}% to "
              f"{yearly.loc[2014, 'nox_breach_pct']:.1f}% of operating hours.")
    else:
        print("  H1 NOT CLEARLY SUPPORTED - the step change is weaker than expected.")

    return yearly


def h2_h3_drivers(clean):
    banner("H2/H3: WHAT DRIVES TURBINE ENERGY YIELD?")

    X = clean[OPERATING_PARAMS]
    y = clean["TEY"]
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    model = RandomForestRegressor(
        n_estimators=200, max_depth=14, random_state=42, n_jobs=-1
    )
    model.fit(X_train, y_train)
    pred = model.predict(X_test)

    r2 = r2_score(y_test, pred)
    mape = mean_absolute_percentage_error(y_test, pred) * 100
    print(f"Random Forest on {len(X_train):,} train / {len(X_test):,} test rows")
    print(f"  R2:   {r2:.4f}")
    print(f"  MAPE: {mape:.3f}%")

    importance = (
        pd.Series(model.feature_importances_, index=OPERATING_PARAMS)
        .sort_values(ascending=False)
    )
    print("\nDriver ranking (Random Forest feature importance):")
    for rank, (feat, imp) in enumerate(importance.items(), 1):
        bar = "#" * int(imp * 60)
        print(f"  {rank}. {feat:<5} {imp:>7.4f}  {bar}")

    print(f"\nH2 VERDICT: top driver is {importance.index[0]} "
          f"({importance.iloc[0]:.1%} of explained importance).")
    print("  " + ("SUPPORTED - TIT dominates as expected."
                  if importance.index[0] == "TIT"
                  else f"NOT as hypothesised - {importance.index[0]} leads, not TIT."))

    # H3: ambient temperature effect
    at_corr = clean["AT"].corr(clean["TEY"])
    cold = clean[clean["AT"] < clean["AT"].quantile(0.25)]["TEY"].mean()
    hot = clean[clean["AT"] > clean["AT"].quantile(0.75)]["TEY"].mean()
    delta_pct = 100 * (hot - cold) / cold
    print(f"\nH3 VERDICT: AT-TEY correlation {at_corr:+.3f}")
    print(f"  Coldest quartile mean TEY: {cold:.2f} MWH")
    print(f"  Hottest quartile mean TEY: {hot:.2f} MWH")
    print(f"  Output on hot days is {delta_pct:+.1f}% vs cold days.")
    print("  " + ("SUPPORTED - higher ambient temperature reduces output "
                  "(lower air density = less mass flow)."
                  if delta_pct < 0 else "NOT SUPPORTED."))

    return model, importance, {"r2": r2, "mape": mape}


def h2_corrected_drivers(clean):
    """
    The naive driver ranking is misleading, and it is worth showing why.

    CDP, GTEP and TEY all rise and fall together with unit LOAD. Feeding CDP
    into a model predicting TEY therefore doesn't identify a driver - it just
    finds the best available proxy for load and reports "load predicts load"
    at 97.6% importance. The model is accurate (R2 0.998) and useless.

    To get an actionable answer we restrict the inputs to variables that are
    genuinely upstream of output:
      - ambient conditions the plant cannot control (AT, AP, AH)
      - equipment condition (AFDP - air filter fouling)
      - the operator's firing setpoint (TIT)
    and exclude the co-symptoms of load (CDP, GTEP, TAT).
    """
    banner("H2 CORRECTED: CONTROLLING FOR LOAD MULTICOLLINEARITY")

    exogenous = ["AT", "AP", "AH", "AFDP", "TIT"]
    co_symptoms = ["CDP", "GTEP", "TAT"]

    print("Excluded as co-symptoms of load (not causes of it):")
    for c in co_symptoms:
        r = clean[c].corr(clean["TEY"])
        print(f"  {c:<5} correlation with TEY = {r:+.3f}")

    X = clean[exogenous]
    y = clean["TEY"]
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )
    model = RandomForestRegressor(
        n_estimators=200, max_depth=14, random_state=42, n_jobs=-1
    )
    model.fit(X_train, y_train)
    pred = model.predict(X_test)

    r2 = r2_score(y_test, pred)
    mape = mean_absolute_percentage_error(y_test, pred) * 100
    print(f"\nModel on exogenous inputs only:")
    print(f"  R2:   {r2:.4f}   (vs 0.9980 with co-symptoms included)")
    print(f"  MAPE: {mape:.3f}%")
    print("  Lower R2 is EXPECTED and correct - we removed the leakage.")

    importance = (
        pd.Series(model.feature_importances_, index=exogenous)
        .sort_values(ascending=False)
    )
    print("\nActionable driver ranking:")
    for rank, (feat, imp) in enumerate(importance.items(), 1):
        bar = "#" * int(imp * 60)
        print(f"  {rank}. {feat:<5} {imp:>7.4f}  {bar}")

    print(f"\nCORRECTED VERDICT: {importance.index[0]} is the dominant actionable driver "
          f"({importance.iloc[0]:.1%}).")
    return model, importance, {"r2": r2, "mape": mape}


def h3_corrected_ambient(clean):
    """
    The raw AT-TEY correlation (-0.03) hides the ambient effect because load
    swamps it. Comparing hot vs cold hours WITHIN a narrow load band isolates it.
    """
    banner("H3 CORRECTED: AMBIENT EFFECT WITHIN A FIXED LOAD BAND")

    # Restrict to a narrow band of compressor discharge pressure = near-constant load
    lo, hi = clean["CDP"].quantile([0.45, 0.55])
    band = clean[(clean["CDP"] >= lo) & (clean["CDP"] <= hi)]
    print(f"Load band: CDP in [{lo:.2f}, {hi:.2f}] mbar -> {len(band):,} readings")

    cold_t, hot_t = band["AT"].quantile([0.25, 0.75])
    cold = band[band["AT"] <= cold_t]["TEY"].mean()
    hot = band[band["AT"] >= hot_t]["TEY"].mean()
    delta = 100 * (hot - cold) / cold

    print(f"  Cold hours (AT <= {cold_t:.1f} C): mean TEY {cold:.2f} MWH")
    print(f"  Hot  hours (AT >= {hot_t:.1f} C): mean TEY {hot:.2f} MWH")
    print(f"  Ambient penalty at constant load: {delta:+.2f}%")
    print(f"\n  Within-band AT-TEY correlation: {band['AT'].corr(band['TEY']):+.3f}")
    print("  (compare to the misleading whole-dataset figure of "
          f"{clean['AT'].corr(clean['TEY']):+.3f})")


def compliance_risk(clean):
    banner("COMPLIANCE RISK PROFILE")

    for year, grp in clean.groupby("year"):
        breach = 100 * grp["nox_breach"].mean()
        mean_nox = grp["NOX"].mean()
        p95 = grp["NOX"].quantile(0.95)
        headroom = 100 * (70.0 - mean_nox) / 70.0
        print(f"  {year}: mean NOx {mean_nox:>6.2f} | p95 {p95:>6.2f} | "
              f"breach {breach:>5.1f}% of hours | headroom {headroom:>5.1f}%")

    worst = clean.groupby("year")["nox_breach"].mean().idxmax()
    print(f"\nWorst compliance year: {worst} "
          f"({100*clean[clean['year']==worst]['nox_breach'].mean():.1f}% of hours in breach)")


def correlation_structure(clean):
    banner("CORRELATION STRUCTURE (top pairs)")
    cols = OPERATING_PARAMS + ["TEY", "CO", "NOX"]
    corr = clean[cols].corr()

    pairs = (
        corr.where(np.triu(np.ones(corr.shape), k=1).astype(bool))
        .stack()
        .sort_values(key=abs, ascending=False)
    )
    print("Strongest 12 relationships:")
    for (a, b), v in pairs.head(12).items():
        print(f"  {a:<5} <-> {b:<5} {v:+.3f}")


def main():
    df, clean = load()
    h1_combustion_retune(clean)
    h2_h3_drivers(clean)
    h2_corrected_drivers(clean)
    h3_corrected_ambient(clean)
    compliance_risk(clean)
    correlation_structure(clean)

    banner("ANALYSIS COMPLETE")
    print("Verified findings are now safe to encode into the agent's tools.")


if __name__ == "__main__":
    main()
