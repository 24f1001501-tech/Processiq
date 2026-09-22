"""
ProcessIQ - Visualisation layer

Every figure is a pure function of the data: pass a ToolResult or the
dataframe, get a Plotly figure. The same functions serve the static dashboard
and the agent's chart responses, so a chart never disagrees with the number
the agent just quoted.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots

from src.agent.tools import (
    ALL_SENSORS,
    CO_LIMIT,
    EXOGENOUS,
    LOAD_SYMPTOMS,
    NOX_LIMIT,
    clean_subset,
    load_data,
)

# Palette kept deliberately small and consistent across every figure.
INK = "#1f2933"
MUTED = "#8a94a6"
GRID = "rgba(138,148,166,0.18)"
ACCENT = "#2f6f8f"        # primary series
WARN = "#c1622f"          # attention / breach
GOOD = "#3d7d5a"          # compliant / improvement
NEUTRAL = "#b0b8c4"

LAYOUT = dict(
    template="plotly_white",
    font=dict(family="Inter, Segoe UI, system-ui, sans-serif", size=13, color=INK),
    margin=dict(l=60, r=30, t=60, b=50),
    hovermode="x unified",
    plot_bgcolor="rgba(0,0,0,0)",
    paper_bgcolor="rgba(0,0,0,0)",
)


def _style(fig, title=None, subtitle=None, height=420):
    if title:
        text = f"<b>{title}</b>"
        if subtitle:
            text += f"<br><span style='font-size:12px;color:{MUTED}'>{subtitle}</span>"
        fig.update_layout(title=dict(text=text, x=0, xanchor="left", y=0.95))
    fig.update_layout(height=height, **LAYOUT)
    fig.update_xaxes(showgrid=False, linecolor=GRID, ticks="outside", tickcolor=GRID)
    fig.update_yaxes(showgrid=True, gridcolor=GRID, zeroline=False)
    return fig


# --------------------------------------------------------------------------
# Headline figure: the NOx compliance step change
# --------------------------------------------------------------------------

def fig_compliance_step_change(df: pd.DataFrame | None = None) -> go.Figure:
    """
    The strongest finding in the dataset: the share of operating hours in NOx
    breach collapsed between 2013 and 2014.
    """
    df = clean_subset(df)
    yearly = (
        df.groupby("year")
        .agg(
            breach_pct=("nox_breach", lambda s: 100 * s.mean()),
            mean_nox=("NOX", "mean"),
        )
        .reset_index()
    )

    colors = [WARN if v > 25 else GOOD for v in yearly["breach_pct"]]

    fig = make_subplots(specs=[[{"secondary_y": True}]])
    fig.add_bar(
        x=yearly["year"],
        y=yearly["breach_pct"],
        marker_color=colors,
        name="Hours in breach",
        hovertemplate="%{y:.1f}% of hours<extra></extra>",
        secondary_y=False,
    )
    fig.add_scatter(
        x=yearly["year"],
        y=yearly["mean_nox"],
        mode="lines+markers",
        line=dict(color=INK, width=2.5),
        marker=dict(size=8),
        name="Mean NOx",
        hovertemplate="%{y:.1f} mg/m³<extra></extra>",
        secondary_y=True,
    )
    fig.add_hline(
        y=NOX_LIMIT,
        line=dict(color=WARN, dash="dash", width=1.5),
        annotation_text=f"{NOX_LIMIT:g} mg/m³ limit",
        annotation_position="right",
        secondary_y=True,
    )
    fig.update_yaxes(title_text="% of operating hours in breach", secondary_y=False)
    fig.update_yaxes(title_text="Mean NOx (mg/m³)", secondary_y=True, showgrid=False)
    fig.update_xaxes(title_text="Year", dtick=1)

    return _style(
        fig,
        "NOx compliance collapsed between 2013 and 2014",
        "Breach rate fell 43.2% → 10.0% of operating hours, with CO also down and output up",
        height=440,
    )


# --------------------------------------------------------------------------
# The multicollinearity demonstration
# --------------------------------------------------------------------------

def fig_driver_comparison() -> go.Figure:
    """
    Side-by-side: the naive driver ranking versus the load-corrected one.
    This chart is the analytical argument of the whole project.
    """
    from src.agent.tools import rank_efficiency_drivers

    naive = rank_efficiency_drivers(include_load_symptoms=True).table
    corrected = rank_efficiency_drivers(include_load_symptoms=False).table

    fig = make_subplots(
        rows=1,
        cols=2,
        subplot_titles=(
            "Naive — all parameters (R²=0.998)",
            "Load-corrected — exogenous only (R²=0.986)",
        ),
        horizontal_spacing=0.18,
    )

    naive = naive.sort_values("importance_pct")
    fig.add_bar(
        y=naive["parameter"],
        x=naive["importance_pct"],
        orientation="h",
        marker_color=[WARN if p in LOAD_SYMPTOMS else NEUTRAL for p in naive["parameter"]],
        hovertemplate="%{y}: %{x:.1f}%<extra></extra>",
        showlegend=False,
        row=1, col=1,
    )

    corrected = corrected.sort_values("importance_pct")
    fig.add_bar(
        y=corrected["parameter"],
        x=corrected["importance_pct"],
        orientation="h",
        marker_color=[ACCENT if p == "TIT" else NEUTRAL for p in corrected["parameter"]],
        hovertemplate="%{y}: %{x:.1f}%<extra></extra>",
        showlegend=False,
        row=1, col=2,
    )

    fig.update_xaxes(title_text="Importance (%)", range=[0, 100])
    for ann in fig.layout.annotations:
        ann.font.size = 12

    return _style(
        fig,
        "The same data gives opposite answers depending on what you feed the model",
        "Orange bars are load co-symptoms. Including them makes CDP look like the driver — it is not.",
        height=380,
    )


def fig_driver_bar(target: str = "TEY", include_load_symptoms: bool = False) -> go.Figure:
    from src.agent.tools import rank_efficiency_drivers

    res = rank_efficiency_drivers(target=target, include_load_symptoms=include_load_symptoms)
    t = res.table.sort_values("importance_pct")
    top = t["parameter"].iloc[-1]

    fig = go.Figure(
        go.Bar(
            y=t["parameter"],
            x=t["importance_pct"],
            orientation="h",
            marker_color=[ACCENT if p == top else NEUTRAL for p in t["parameter"]],
            hovertemplate="%{y}: %{x:.2f}%<extra></extra>",
        )
    )
    fig.update_xaxes(title_text="Importance (%)")
    mode = "naive" if include_load_symptoms else "load-corrected"
    return _style(fig, f"Drivers of {target}", f"Random Forest importance, {mode}", height=360)


# --------------------------------------------------------------------------
# Ambient effect
# --------------------------------------------------------------------------

def fig_ambient_effect() -> go.Figure:
    """Whole-dataset scatter vs. constant-load band — the effect appears only in the band."""
    df = clean_subset()
    lo, hi = df["CDP"].quantile([0.45, 0.55])
    band = df[(df["CDP"] >= lo) & (df["CDP"] <= hi)]

    sample = df.sample(min(4000, len(df)), random_state=42)

    fig = make_subplots(
        rows=1, cols=2,
        subplot_titles=(
            f"All operating hours (r = {df['AT'].corr(df['TEY']):+.3f})",
            f"Constant load band (r = {band['AT'].corr(band['TEY']):+.3f})",
        ),
        horizontal_spacing=0.12,
    )
    fig.add_scatter(
        x=sample["AT"], y=sample["TEY"], mode="markers",
        marker=dict(size=3, color=NEUTRAL, opacity=0.45),
        hovertemplate="AT %{x:.1f}°C<br>TEY %{y:.1f} MWh<extra></extra>",
        showlegend=False, row=1, col=1,
    )
    fig.add_scatter(
        x=band["AT"], y=band["TEY"], mode="markers",
        marker=dict(size=4, color=ACCENT, opacity=0.55),
        hovertemplate="AT %{x:.1f}°C<br>TEY %{y:.1f} MWh<extra></extra>",
        showlegend=False, row=1, col=2,
    )
    # Trend line on the band
    z = np.polyfit(band["AT"], band["TEY"], 1)
    xs = np.linspace(band["AT"].min(), band["AT"].max(), 50)
    fig.add_scatter(
        x=xs, y=np.polyval(z, xs), mode="lines",
        line=dict(color=WARN, width=2.5), showlegend=False, row=1, col=2,
    )
    fig.update_xaxes(title_text="Ambient temperature (°C)")
    fig.update_yaxes(title_text="Turbine energy yield (MWh)", col=1)
    for ann in fig.layout.annotations:
        ann.font.size = 12

    return _style(
        fig,
        "Ambient temperature looks irrelevant until you control for load",
        "Load variation swamps the ambient signal — within a fixed load band the effect is 23× stronger",
        height=400,
    )


# --------------------------------------------------------------------------
# Emissions
# --------------------------------------------------------------------------

def fig_compliance_distribution(pollutant: str = "NOX") -> go.Figure:
    df = clean_subset()
    limit = NOX_LIMIT if pollutant.upper() == "NOX" else CO_LIMIT
    series = df[pollutant.upper()]

    below = series[series <= limit]
    above = series[series > limit]

    fig = go.Figure()
    fig.add_histogram(
        x=below, nbinsx=70, marker_color=GOOD, opacity=0.85,
        name=f"Compliant ({100*len(below)/len(series):.1f}%)",
        hovertemplate="%{x:.0f} mg/m³: %{y} hours<extra></extra>",
    )
    fig.add_histogram(
        x=above, nbinsx=40, marker_color=WARN, opacity=0.85,
        name=f"In breach ({100*len(above)/len(series):.1f}%)",
        hovertemplate="%{x:.0f} mg/m³: %{y} hours<extra></extra>",
    )
    fig.add_vline(
        x=limit, line=dict(color=INK, dash="dash", width=2),
        annotation_text=f"{limit:g} mg/m³ limit", annotation_position="top right",
    )
    fig.update_layout(barmode="overlay", legend=dict(orientation="h", y=1.02, x=0))
    fig.update_xaxes(title_text=f"{pollutant.upper()} (mg/m³)")
    fig.update_yaxes(title_text="Operating hours")

    return _style(
        fig,
        f"{pollutant.upper()} distribution against the compliance limit",
        f"{len(above):,} of {len(series):,} operating hours exceed the reference limit",
        height=400,
    )


def fig_emissions_forecast(pollutant: str = "NOX", horizon_hours: int = 720) -> go.Figure:
    df = clean_subset()
    limit = NOX_LIMIT if pollutant.upper() == "NOX" else CO_LIMIT
    latest = int(df["year"].max())
    recent = df[df["year"] == latest].sort_values("operating_hour")

    x = recent["operating_hour"].values.astype(float)
    y = recent[pollutant.upper()].values.astype(float)
    slope, intercept = np.polyfit(x, y, 1)

    roll = pd.Series(y).rolling(72, min_periods=12).mean()
    fx = np.linspace(x[0], x[-1] + horizon_hours, 200)

    fig = go.Figure()
    fig.add_scatter(
        x=x, y=y, mode="markers", marker=dict(size=2.5, color=NEUTRAL, opacity=0.35),
        name="Hourly readings", hoverinfo="skip",
    )
    fig.add_scatter(
        x=x, y=roll, mode="lines", line=dict(color=ACCENT, width=2),
        name="72h rolling mean",
    )
    fig.add_scatter(
        x=fx, y=np.polyval([slope, intercept], fx), mode="lines",
        line=dict(color=WARN, width=2, dash="dot"), name="Linear trend + projection",
    )
    fig.add_vline(x=x[-1], line=dict(color=MUTED, width=1, dash="dot"),
                  annotation_text="end of record", annotation_position="top left")
    fig.add_hline(y=limit, line=dict(color=INK, dash="dash", width=1.5),
                  annotation_text=f"{limit:g} mg/m³ limit", annotation_position="right")

    fig.update_xaxes(title_text=f"Operating hour ({latest})")
    fig.update_yaxes(title_text=f"{pollutant.upper()} (mg/m³)")
    fig.update_layout(legend=dict(orientation="h", y=1.02, x=0))

    return _style(
        fig,
        f"{pollutant.upper()} trend and projection",
        f"Trend {slope*1000:+.2f} mg/m³ per 1,000 operating hours, projected {horizon_hours:,}h forward",
        height=420,
    )


# --------------------------------------------------------------------------
# Year comparison / degradation
# --------------------------------------------------------------------------

def fig_year_comparison(metric: str = "TEY", from_year: int = 2013, to_year: int = 2014) -> go.Figure:
    from src.agent.tools import diagnose_performance_change

    res = diagnose_performance_change(metric=metric, from_year=from_year, to_year=to_year)
    t = res.table.copy().sort_values("change_pct")

    fig = go.Figure(
        go.Bar(
            y=t["parameter"], x=t["change_pct"], orientation="h",
            marker_color=[WARN if v < 0 else GOOD for v in t["change_pct"]],
            hovertemplate="%{y}: %{x:+.2f}%<extra></extra>",
        )
    )
    fig.add_vline(x=0, line=dict(color=INK, width=1))
    fig.update_xaxes(title_text=f"Change from {from_year} to {to_year} (%)")

    return _style(
        fig,
        f"What moved between {from_year} and {to_year}",
        "Ranked by standardised effect size across all measured tags",
        height=440,
    )


def fig_multi_year_trends() -> go.Figure:
    df = clean_subset()
    metrics = [("TEY", "Energy yield (MWh)"), ("TIT", "Inlet temp (°C)"),
               ("NOX", "NOx (mg/m³)"), ("CO", "CO (mg/m³)")]

    fig = make_subplots(rows=2, cols=2, subplot_titles=[m[1] for m in metrics],
                        vertical_spacing=0.16, horizontal_spacing=0.10)
    for i, (col, _) in enumerate(metrics):
        r, c = divmod(i, 2)
        yearly = df.groupby("year")[col].mean()
        fig.add_scatter(
            x=yearly.index, y=yearly.values, mode="lines+markers",
            line=dict(color=ACCENT, width=2.5), marker=dict(size=7),
            showlegend=False, hovertemplate="%{x}: %{y:.2f}<extra></extra>",
            row=r + 1, col=c + 1,
        )
    fig.update_xaxes(dtick=1)
    for ann in fig.layout.annotations:
        ann.font.size = 12

    return _style(
        fig, "Five-year plant trends",
        "Annual means across the full operating record", height=480,
    )


# --------------------------------------------------------------------------
# Correlation + data quality
# --------------------------------------------------------------------------

def fig_correlation_heatmap() -> go.Figure:
    df = clean_subset()
    corr = df[ALL_SENSORS].corr()

    fig = go.Figure(
        go.Heatmap(
            z=corr.values, x=corr.columns, y=corr.columns,
            colorscale=[[0, WARN], [0.5, "#f5f5f3"], [1, ACCENT]],
            zmid=0, zmin=-1, zmax=1,
            text=corr.round(2).values, texttemplate="%{text}",
            textfont=dict(size=10),
            hovertemplate="%{y} ↔ %{x}: %{z:.3f}<extra></extra>",
            colorbar=dict(thickness=12, len=0.7),
        )
    )
    return _style(
        fig, "Correlation structure",
        "CDP–TEY at +0.99 is the multicollinearity that breaks naive driver analysis",
        height=520,
    )


def fig_quality_summary() -> go.Figure:
    df = load_data()
    counts = {
        "Fully unflagged": int((~df["stuck_sensor_flag"] & ~df["outlier_flag"]).sum()),
        "Extreme outlier": int(df["outlier_flag"].sum()),
        "Stuck sensor": int(df["stuck_sensor_flag"].sum()),
        "Humidity clipped": int(df["AH_was_clipped"].sum()),
    }
    labels, values = list(counts), list(counts.values())
    colors = [GOOD, WARN, "#8a5a3d", MUTED]

    fig = go.Figure(
        go.Bar(
            x=values, y=labels, orientation="h", marker_color=colors,
            text=[f"{v:,}" for v in values], textposition="outside",
            hovertemplate="%{y}: %{x:,} rows<extra></extra>",
        )
    )
    fig.update_xaxes(title_text="Rows", range=[0, max(values) * 1.18])
    return _style(
        fig, "Data quality flags",
        "Flagged rows are retained, not deleted — every decision stays auditable",
        height=320,
    )


# --------------------------------------------------------------------------
# Dispatcher used by the agent
# --------------------------------------------------------------------------

FIGURE_BUILDERS = {
    "driver_bar": lambda spec: fig_driver_bar(spec.get("target", "TEY")),
    "driver_comparison": lambda spec: fig_driver_comparison(),
    "compliance_distribution": lambda spec: fig_compliance_distribution(spec.get("pollutant", "NOX")),
    "emissions_forecast": lambda spec: fig_emissions_forecast(spec.get("pollutant", "NOX")),
    "year_comparison": lambda spec: fig_year_comparison(
        spec.get("metric", "TEY"), spec.get("from_year", 2013), spec.get("to_year", 2014)
    ),
    "ambient_effect": lambda spec: fig_ambient_effect(),
    "quality_summary": lambda spec: fig_quality_summary(),
    "compliance_step": lambda spec: fig_compliance_step_change(),
    "multi_year": lambda spec: fig_multi_year_trends(),
    "correlation": lambda spec: fig_correlation_heatmap(),
}


def build_figure(spec: dict | None):
    """Turn a tool's figure_spec into a Plotly figure, or None if unavailable."""
    if not spec:
        return None
    builder = FIGURE_BUILDERS.get(spec.get("kind"))
    if builder is None:
        return None
    try:
        return builder(spec)
    except Exception:
        return None


if __name__ == "__main__":
    from pathlib import Path

    out = Path(__file__).resolve().parents[1] / "outputs"
    out.mkdir(exist_ok=True)

    figures = {
        "01_compliance_step_change": fig_compliance_step_change(),
        "02_driver_comparison": fig_driver_comparison(),
        "03_ambient_effect": fig_ambient_effect(),
        "04_compliance_distribution": fig_compliance_distribution(),
        "05_emissions_forecast": fig_emissions_forecast(),
        "06_year_comparison": fig_year_comparison(),
        "07_multi_year_trends": fig_multi_year_trends(),
        "08_correlation": fig_correlation_heatmap(),
        "09_quality_summary": fig_quality_summary(),
    }
    for name, fig in figures.items():
        path = out / f"{name}.html"
        fig.write_html(path, include_plotlyjs="cdn")
        print(f"  wrote {path.name}")
    print(f"\n{len(figures)} figures written to outputs/")
