"""
ProcessIQ - Agent tools

Every tool here is DETERMINISTIC. No tool calls a language model. Given the
same arguments they return the same numbers, and they run whether or not an
API key is configured.

That split is deliberate. The LLM's job is narrow: pick a tool, fill in its
arguments, and write prose around whatever numbers come back. It never
computes anything itself, so it cannot fabricate a statistic. If the LLM is
unavailable the tools still run and the UI shows the raw result.

Each tool returns a ToolResult carrying:
  - a headline (one sentence of plain English)
  - facts (the numbers, for the LLM to narrate and for the UI to render)
  - a dataframe and/or figure where relevant
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import duckdb
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_percentage_error, r2_score
from sklearn.model_selection import train_test_split

ROOT = Path(__file__).resolve().parents[2]
DATA_PATH = ROOT / "data" / "processed" / "turbine_clean.parquet"

NOX_LIMIT = 70.0
CO_LIMIT = 20.0

# Variables that are genuinely upstream of output. CDP, GTEP and TAT are
# excluded on purpose: they are co-symptoms of load, and including them
# produces an accurate but meaningless model. See semantic_layer.yaml.
EXOGENOUS = ["AT", "AP", "AH", "AFDP", "TIT"]
LOAD_SYMPTOMS = ["CDP", "GTEP", "TAT"]
ALL_SENSORS = ["AT", "AP", "AH", "AFDP", "GTEP", "TIT", "TAT", "TEY", "CDP", "CO", "NOX"]


@dataclass
class ToolResult:
    tool: str
    headline: str
    facts: dict[str, Any] = field(default_factory=dict)
    table: pd.DataFrame | None = None
    figure_spec: dict | None = None
    caveats: list[str] = field(default_factory=list)

    def summary_for_llm(self) -> str:
        """Compact text rendering the LLM narrates. Numbers only, no adjectives."""
        lines = [f"TOOL: {self.tool}", f"HEADLINE: {self.headline}", "FACTS:"]
        for k, v in self.facts.items():
            if isinstance(v, float):
                lines.append(f"  {k}: {v:,.4g}")
            else:
                lines.append(f"  {k}: {v}")
        if self.table is not None and len(self.table) <= 25:
            lines.append("TABLE:")
            lines.append(self.table.to_string(index=False))
        if self.caveats:
            lines.append("CAVEATS:")
            lines.extend(f"  - {c}" for c in self.caveats)
        return "\n".join(lines)


# --------------------------------------------------------------------------
# Data access
# --------------------------------------------------------------------------

_df_cache: pd.DataFrame | None = None


def load_data() -> pd.DataFrame:
    global _df_cache
    if _df_cache is None:
        if not DATA_PATH.exists():
            raise FileNotFoundError(
                f"{DATA_PATH} not found. Run: python src/clean_data.py"
            )
        _df_cache = pd.read_parquet(DATA_PATH)
    return _df_cache


def clean_subset(df: pd.DataFrame | None = None) -> pd.DataFrame:
    """Rows with no quality flags - the basis for any stated conclusion."""
    df = load_data() if df is None else df
    return df[~df["stuck_sensor_flag"] & ~df["outlier_flag"]]


# --------------------------------------------------------------------------
# Tool 1 - SQL over the plant history
# --------------------------------------------------------------------------

def query_turbine_data(sql: str, limit: int = 500) -> ToolResult:
    """
    Execute read-only SQL against the turbine table.

    Guardrails: only SELECT/WITH statements are permitted, and results are
    capped. This is the tool an LLM writes SQL into, so it must assume the
    input is untrusted.
    """
    stripped = sql.strip().rstrip(";")
    lowered = stripped.lower()

    if not (lowered.startswith("select") or lowered.startswith("with")):
        return ToolResult(
            tool="query_turbine_data",
            headline="Rejected: only SELECT queries are allowed.",
            facts={"submitted_sql": sql},
        )

    forbidden = ("insert", "update", "delete", "drop", "alter", "create", "attach", "copy")
    if any(f" {word} " in f" {lowered} " for word in forbidden):
        return ToolResult(
            tool="query_turbine_data",
            headline="Rejected: query contains a write or DDL keyword.",
            facts={"submitted_sql": sql},
        )

    df = load_data()
    con = duckdb.connect()
    con.register("turbine", df)
    try:
        result = con.execute(f"SELECT * FROM ({stripped}) LIMIT {limit}").fetchdf()
    except Exception as e:
        return ToolResult(
            tool="query_turbine_data",
            headline=f"SQL error: {e}",
            facts={"submitted_sql": sql, "error": str(e)},
        )
    finally:
        con.close()

    return ToolResult(
        tool="query_turbine_data",
        headline=f"Query returned {len(result):,} rows.",
        facts={"rows": len(result), "columns": list(result.columns), "sql": stripped},
        table=result,
    )


# --------------------------------------------------------------------------
# Tool 2 - Driver ranking (the multicollinearity-aware one)
# --------------------------------------------------------------------------

def rank_efficiency_drivers(
    target: str = "TEY",
    include_load_symptoms: bool = False,
    year: int | None = None,
) -> ToolResult:
    """
    Rank what actually drives a target variable.

    By default this EXCLUDES the load co-symptoms (CDP, GTEP, TAT). Setting
    include_load_symptoms=True reproduces the naive analysis, which is useful
    for demonstrating exactly how misleading it is.
    """
    df = clean_subset()
    if year is not None:
        df = df[df["year"] == year]
        if df.empty:
            return ToolResult(
                tool="rank_efficiency_drivers",
                headline=f"No data for year {year}.",
                facts={"year": year},
            )

    features = EXOGENOUS + (LOAD_SYMPTOMS if include_load_symptoms else [])
    features = [f for f in features if f != target]

    X, y = df[features], df[target]
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )
    model = RandomForestRegressor(
        n_estimators=200, max_depth=14, random_state=42, n_jobs=-1
    )
    model.fit(X_train, y_train)
    pred = model.predict(X_test)

    r2 = float(r2_score(y_test, pred))
    mape = float(mean_absolute_percentage_error(y_test, pred) * 100)

    ranking = (
        pd.Series(model.feature_importances_, index=features)
        .sort_values(ascending=False)
        .rename("importance")
        .reset_index()
        .rename(columns={"index": "parameter"})
    )
    ranking["importance_pct"] = (ranking["importance"] * 100).round(2)

    top = ranking.iloc[0]
    mode = "naive (load symptoms included)" if include_load_symptoms else "load-corrected"

    caveats = []
    if include_load_symptoms:
        caveats.append(
            "This ranking includes CDP/GTEP/TAT, which are co-symptoms of load "
            "rather than causes of output. The result is statistically accurate "
            "but not actionable - it essentially reports that load predicts load."
        )
    else:
        caveats.append(
            "CDP, GTEP and TAT were excluded as load co-symptoms. R2 is lower "
            "than the naive model by design: the leakage has been removed."
        )

    return ToolResult(
        tool="rank_efficiency_drivers",
        headline=(
            f"{top['parameter']} is the leading driver of {target} "
            f"({top['importance_pct']:.1f}% importance, {mode})."
        ),
        facts={
            "target": target,
            "mode": mode,
            "top_driver": top["parameter"],
            "top_driver_importance_pct": float(top["importance_pct"]),
            "model_r2": r2,
            "model_mape_pct": mape,
            "features_used": features,
            "year": year or "all",
            "n_rows": len(df),
        },
        table=ranking[["parameter", "importance_pct"]],
        figure_spec={"kind": "driver_bar", "target": target},
        caveats=caveats,
    )


# --------------------------------------------------------------------------
# Tool 3 - Emissions compliance and forecast
# --------------------------------------------------------------------------

def analyse_emissions_compliance(
    pollutant: str = "NOX", year: int | None = None
) -> ToolResult:
    """Quantify exposure against the reference emissions limit."""
    pollutant = pollutant.upper()
    if pollutant not in ("NOX", "CO"):
        return ToolResult(
            tool="analyse_emissions_compliance",
            headline=f"Unknown pollutant {pollutant!r}. Use NOX or CO.",
            facts={},
        )

    limit = NOX_LIMIT if pollutant == "NOX" else CO_LIMIT
    df = clean_subset()
    scope = "2011-2015"
    if year is not None:
        df = df[df["year"] == year]
        scope = str(year)

    series = df[pollutant]
    breaches = int((series > limit).sum())
    breach_pct = 100 * breaches / len(series)
    headroom_pct = 100 * (limit - series.mean()) / limit

    by_year = (
        clean_subset()
        .groupby("year")
        .agg(
            mean=(pollutant, "mean"),
            p95=(pollutant, lambda s: s.quantile(0.95)),
            breach_pct=(pollutant, lambda s: 100 * (s > limit).mean()),
        )
        .round(2)
        .reset_index()
    )

    return ToolResult(
        tool="analyse_emissions_compliance",
        headline=(
            f"{pollutant}: {breach_pct:.1f}% of operating hours in {scope} exceed "
            f"the {limit:g} mg/m³ reference limit."
        ),
        facts={
            "pollutant": pollutant,
            "limit_mg_m3": limit,
            "scope": scope,
            "mean": float(series.mean()),
            "p95": float(series.quantile(0.95)),
            "max": float(series.max()),
            "breach_hours": breaches,
            "breach_pct": float(breach_pct),
            "mean_headroom_pct": float(headroom_pct),
            "worst_year": int(by_year.loc[by_year["breach_pct"].idxmax(), "year"]),
        },
        table=by_year,
        figure_spec={"kind": "compliance_distribution", "pollutant": pollutant},
        caveats=[
            "The limit is an indicative EU industrial-emissions reference value, "
            "not this plant's actual permit condition.",
        ],
    )


def forecast_emissions(pollutant: str = "NOX", horizon_hours: int = 720) -> ToolResult:
    """
    Project the recent emissions trend forward and estimate when it would
    cross the compliance limit.

    Deliberately a simple OLS trend on the most recent year rather than a
    heavy time-series model: the data has synthetic timestamps and gaps, so a
    sophisticated model would imply a precision the data cannot support.
    """
    pollutant = pollutant.upper()
    limit = NOX_LIMIT if pollutant == "NOX" else CO_LIMIT

    df = clean_subset()
    latest_year = int(df["year"].max())
    recent = df[df["year"] == latest_year].sort_values("operating_hour")

    x = recent["operating_hour"].values.astype(float)
    y = recent[pollutant].values.astype(float)
    slope, intercept = np.polyfit(x, y, 1)

    current = float(y[-200:].mean())          # smoothed current level
    projected = float(slope * (x[-1] + horizon_hours) + intercept)
    per_1000h = float(slope * 1000)

    if slope > 0 and current < limit:
        hours_to_limit = (limit - current) / slope
        crossing = f"{hours_to_limit:,.0f} operating hours at the current trend"
    elif current >= limit:
        crossing = "already above the limit"
    else:
        crossing = "not on a trajectory to breach (trend is flat or improving)"

    return ToolResult(
        tool="forecast_emissions",
        headline=(
            f"{pollutant} is trending {per_1000h:+.2f} mg/m³ per 1,000 operating "
            f"hours; projected {projected:.1f} mg/m³ in {horizon_hours:,}h "
            f"against a {limit:g} limit."
        ),
        facts={
            "pollutant": pollutant,
            "basis_year": latest_year,
            "current_level": current,
            "trend_per_1000h": per_1000h,
            "horizon_hours": horizon_hours,
            "projected_level": projected,
            "limit": limit,
            "time_to_limit": crossing,
            "n_points": len(recent),
        },
        figure_spec={"kind": "emissions_forecast", "pollutant": pollutant},
        caveats=[
            "Linear trend extrapolation on synthetic operating-hour timestamps. "
            "Indicative of direction, not a calibrated forecast.",
        ],
    )


# --------------------------------------------------------------------------
# Tool 4 - Performance / degradation diagnosis
# --------------------------------------------------------------------------

def diagnose_performance_change(
    metric: str = "TEY", from_year: int | None = None, to_year: int | None = None
) -> ToolResult:
    """
    Compare two periods and attribute the change across parameters.

    This is the automated version of screening plant parameters by hand to
    find what moved.
    """
    df = clean_subset()
    years = sorted(df["year"].unique())
    from_year = int(from_year or years[0])
    to_year = int(to_year or years[-1])

    a = df[df["year"] == from_year]
    b = df[df["year"] == to_year]
    if a.empty or b.empty:
        return ToolResult(
            tool="diagnose_performance_change",
            headline=f"No data for {from_year} or {to_year}.",
            facts={"available_years": [int(y) for y in years]},
        )

    rows = []
    for col in ALL_SENSORS + ["specific_yield", "thermal_spread"]:
        m1, m2 = a[col].mean(), b[col].mean()
        pct = 100 * (m2 - m1) / m1 if m1 else np.nan
        # Standardised effect size, so tags with different units are comparable
        pooled_sd = np.sqrt((a[col].var() + b[col].var()) / 2)
        effect = (m2 - m1) / pooled_sd if pooled_sd else 0.0
        rows.append(
            {
                "parameter": col,
                f"mean_{from_year}": round(float(m1), 3),
                f"mean_{to_year}": round(float(m2), 3),
                "change_pct": round(float(pct), 2),
                "effect_size": round(float(effect), 3),
            }
        )

    table = pd.DataFrame(rows).sort_values("effect_size", key=abs, ascending=False)
    target_row = table[table["parameter"] == metric].iloc[0]
    biggest = table.iloc[0]

    return ToolResult(
        tool="diagnose_performance_change",
        headline=(
            f"{metric} moved {target_row['change_pct']:+.2f}% from {from_year} to "
            f"{to_year}; the largest shift across all tags was {biggest['parameter']} "
            f"({biggest['change_pct']:+.2f}%)."
        ),
        facts={
            "metric": metric,
            "from_year": from_year,
            "to_year": to_year,
            f"{metric}_change_pct": float(target_row["change_pct"]),
            "largest_mover": biggest["parameter"],
            "largest_mover_change_pct": float(biggest["change_pct"]),
            "largest_mover_effect_size": float(biggest["effect_size"]),
        },
        table=table.head(12),
        figure_spec={"kind": "year_comparison", "metric": metric,
                     "from_year": from_year, "to_year": to_year},
        caveats=[
            "Effect size is a standardised mean difference, used to compare tags "
            "that have different units. It does not establish causation.",
        ],
    )


# --------------------------------------------------------------------------
# Tool 5 - Ambient effect at constant load
# --------------------------------------------------------------------------

def ambient_effect_at_constant_load(band_width: float = 0.10) -> ToolResult:
    """
    Isolate the ambient-temperature penalty by holding load roughly constant.

    Across the whole dataset the AT-TEY correlation is near zero, which wrongly
    suggests ambient conditions don't matter. Load variation swamps the effect.
    Restricting to a narrow load band reveals its true size.
    """
    df = clean_subset()
    half = band_width / 2
    lo, hi = df["CDP"].quantile([0.5 - half, 0.5 + half])
    band = df[(df["CDP"] >= lo) & (df["CDP"] <= hi)]

    cold_t, hot_t = band["AT"].quantile([0.25, 0.75])
    cold = float(band[band["AT"] <= cold_t]["TEY"].mean())
    hot = float(band[band["AT"] >= hot_t]["TEY"].mean())
    penalty_pct = 100 * (hot - cold) / cold

    naive_corr = float(df["AT"].corr(df["TEY"]))
    band_corr = float(band["AT"].corr(band["TEY"]))

    return ToolResult(
        tool="ambient_effect_at_constant_load",
        headline=(
            f"At constant load, output on hot hours is {penalty_pct:+.2f}% versus "
            f"cold hours. The whole-dataset correlation ({naive_corr:+.3f}) hides this; "
            f"within-band it is {band_corr:+.3f}."
        ),
        facts={
            "load_band_cdp": [round(float(lo), 2), round(float(hi), 2)],
            "band_rows": len(band),
            "cold_threshold_c": round(float(cold_t), 1),
            "hot_threshold_c": round(float(hot_t), 1),
            "mean_tey_cold": round(cold, 2),
            "mean_tey_hot": round(hot, 2),
            "ambient_penalty_pct": round(float(penalty_pct), 2),
            "naive_correlation": round(naive_corr, 3),
            "within_band_correlation": round(band_corr, 3),
            "correlation_understatement_factor": round(abs(band_corr / naive_corr), 1)
            if naive_corr else None,
        },
        figure_spec={"kind": "ambient_effect"},
        caveats=[
            "Holding CDP constant is a proxy for holding load constant, not a "
            "controlled experiment.",
        ],
    )


# --------------------------------------------------------------------------
# Tool 6 - Data quality report
# --------------------------------------------------------------------------

def data_quality_report() -> ToolResult:
    """Surface what was cleaned, flagged and retained - the governance view."""
    df = load_data()
    flagged_stuck = int(df["stuck_sensor_flag"].sum())
    flagged_outlier = int(df["outlier_flag"].sum())
    clipped = int(df["AH_was_clipped"].sum())
    fully_clean = int((~df["stuck_sensor_flag"] & ~df["outlier_flag"]).sum())

    stuck_tags = (
        df.loc[df["stuck_sensor_flag"], "stuck_sensor_tags"]
        .str.rstrip(";")
        .str.split(";")
        .explode()
        .value_counts()
    )
    outlier_tags = (
        df.loc[df["outlier_flag"], "outlier_tags"]
        .str.rstrip(";")
        .str.split(";")
        .explode()
        .value_counts()
    )

    table = pd.DataFrame(
        {
            "issue": (
                ["Humidity clipped to 100%"]
                + [f"Stuck sensor: {t}" for t in stuck_tags.index]
                + [f"Extreme outlier: {t}" for t in outlier_tags.index]
            ),
            "rows": (
                [clipped] + stuck_tags.tolist() + outlier_tags.tolist()
            ),
        }
    )

    return ToolResult(
        tool="data_quality_report",
        headline=(
            f"{fully_clean:,} of {len(df):,} rows ({100*fully_clean/len(df):.1f}%) carry "
            f"no quality flag. Nothing was silently discarded."
        ),
        facts={
            "total_rows": len(df),
            "fully_unflagged": fully_clean,
            "humidity_clipped": clipped,
            "stuck_sensor_rows": flagged_stuck,
            "outlier_rows": flagged_outlier,
            "duplicates_removed": 7,
        },
        table=table,
        figure_spec={"kind": "quality_summary"},
        caveats=[
            "Flagged rows are retained and excluded only from stated conclusions, "
            "so every cleaning decision remains auditable and reversible.",
        ],
    )


# --------------------------------------------------------------------------
# Registry - what the router can choose from
# --------------------------------------------------------------------------

TOOL_REGISTRY = {
    "query_turbine_data": {
        "fn": query_turbine_data,
        "description": (
            "Run read-only SQL against the turbine table for specific lookups, "
            "filters or aggregations not covered by another tool."
        ),
        "args": {"sql": "A DuckDB SELECT statement over the table `turbine`."},
    },
    "rank_efficiency_drivers": {
        "fn": rank_efficiency_drivers,
        "description": (
            "Rank which parameters drive a target variable such as energy yield. "
            "Use for 'what affects/drives/causes X' questions."
        ),
        "args": {
            "target": "Column to explain (default TEY).",
            "include_load_symptoms": "true to reproduce the naive, misleading ranking.",
            "year": "Optional year filter.",
        },
    },
    "analyse_emissions_compliance": {
        "fn": analyse_emissions_compliance,
        "description": (
            "Quantify emissions exposure against the regulatory limit. Use for "
            "compliance, breach, limit or regulatory questions."
        ),
        "args": {"pollutant": "NOX or CO.", "year": "Optional year filter."},
    },
    "forecast_emissions": {
        "fn": forecast_emissions,
        "description": (
            "Project the emissions trend forward and estimate time to breach. "
            "Use for 'will we breach', 'forecast', 'future' questions."
        ),
        "args": {"pollutant": "NOX or CO.", "horizon_hours": "Projection horizon."},
    },
    "diagnose_performance_change": {
        "fn": diagnose_performance_change,
        "description": (
            "Compare two years and attribute what changed across all tags. Use "
            "for 'why did X change', 'what happened between', degradation questions."
        ),
        "args": {
            "metric": "Metric of interest (default TEY).",
            "from_year": "Start year.",
            "to_year": "End year.",
        },
    },
    "ambient_effect_at_constant_load": {
        "fn": ambient_effect_at_constant_load,
        "description": (
            "Isolate the ambient-temperature effect on output by holding load "
            "constant. Use for weather, ambient or seasonality questions."
        ),
        "args": {"band_width": "Width of the load band as a quantile fraction."},
    },
    "data_quality_report": {
        "fn": data_quality_report,
        "description": (
            "Report what was cleaned, flagged and retained. Use for data quality, "
            "trust, sensor reliability or governance questions."
        ),
        "args": {},
    },
}


if __name__ == "__main__":
    print("Smoke-testing all tools\n" + "=" * 78)
    checks = [
        ("data_quality_report", {}),
        ("rank_efficiency_drivers", {}),
        ("rank_efficiency_drivers", {"include_load_symptoms": True}),
        ("analyse_emissions_compliance", {"pollutant": "NOX"}),
        ("forecast_emissions", {"pollutant": "NOX"}),
        ("diagnose_performance_change", {"from_year": 2013, "to_year": 2014}),
        ("ambient_effect_at_constant_load", {}),
        ("query_turbine_data", {"sql": "SELECT year, AVG(TEY) AS tey FROM turbine GROUP BY year ORDER BY year"}),
    ]
    for name, kwargs in checks:
        res = TOOL_REGISTRY[name]["fn"](**kwargs)
        print(f"\n[{name}] {kwargs if kwargs else ''}")
        print(f"  -> {res.headline}")
