"""
ProcessIQ - Streamlit application

Run:  streamlit run src/app.py
"""

from __future__ import annotations

import sys
from pathlib import Path

import streamlit as st

# Allow `streamlit run src/app.py` to resolve `src.*` imports
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src import viz  # noqa: E402
from src.agent.agent import SAMPLE_QUESTIONS, ProcessIQAgent  # noqa: E402
from src.agent.tools import clean_subset, load_data  # noqa: E402
from src.llm import get_client  # noqa: E402

st.set_page_config(
    page_title="ProcessIQ",
    page_icon="◆",
    layout="wide",
    initial_sidebar_state="expanded",
)

CSS = """
<style>
  .block-container { padding-top: 2.2rem; max-width: 1280px; }
  h1, h2, h3 { letter-spacing: -0.02em; }
  .subtle { color: #8a94a6; font-size: 0.9rem; }
  .metric-card {
      background: rgba(47,111,143,0.06);
      border-left: 3px solid #2f6f8f;
      padding: 0.9rem 1.1rem; border-radius: 4px; margin-bottom: 0.6rem;
  }
  .metric-value { font-size: 1.55rem; font-weight: 650; color: #1f2933; line-height:1.1; }
  .metric-label { font-size: 0.78rem; color: #8a94a6; text-transform: uppercase;
                  letter-spacing: 0.06em; margin-top: 0.15rem; }
  .answer-box {
      background: rgba(47,111,143,0.05); border-radius: 6px;
      padding: 1.1rem 1.3rem; line-height: 1.65;
  }
  .badge { display:inline-block; padding:0.12rem 0.55rem; border-radius:10px;
           font-size:0.72rem; font-weight:600; margin-right:0.3rem; }
  .badge-ok   { background:#e3f0e8; color:#3d7d5a; }
  .badge-warn { background:#f7e9e0; color:#c1622f; }
  .caveat { color:#8a94a6; font-size:0.82rem; border-left:2px solid #d8dde5;
            padding-left:0.7rem; margin-top:0.5rem; }
</style>
"""
st.markdown(CSS, unsafe_allow_html=True)


@st.cache_resource
def get_agent():
    return ProcessIQAgent()


@st.cache_data
def get_data():
    return load_data()


def metric_card(value, label):
    st.markdown(
        f'<div class="metric-card"><div class="metric-value">{value}</div>'
        f'<div class="metric-label">{label}</div></div>',
        unsafe_allow_html=True,
    )


def llm_status():
    """
    Return (online, detail). Detail explains WHY when offline.

    Every attribute access here is defensive. Streamlit reruns this script
    without reimporting its modules, so during a redeploy this function can
    briefly run against an older version of src.llm than it was written for.
    Reading an attribute that version lacks would replace a helpful message
    with an AttributeError, which is precisely the failure this function
    exists to prevent.
    """
    try:
        client = get_client()
    except Exception as e:
        return False, f"Could not initialise the client — {type(e).__name__}: {e}"

    if getattr(client, "configured", False):
        provider = getattr(client.provider, "name", "provider")
        models = getattr(client.provider, "models", ())
        return True, f"{provider} · {models[0] if models else 'unknown model'}"

    diagnostic = getattr(client, "diagnostic", None)
    if diagnostic:
        return False, diagnostic

    env_key = getattr(getattr(client, "provider", None), "env_key", "API key")
    return False, f"{env_key} is not set."


# --------------------------------------------------------------------------
# Sidebar
# --------------------------------------------------------------------------

with st.sidebar:
    st.markdown("### ◆ ProcessIQ")
    st.markdown(
        '<div class="subtle">Agentic analytics for combined-cycle gas turbine '
        "operations</div>",
        unsafe_allow_html=True,
    )
    st.divider()

    online, detail = llm_status()
    if online:
        st.markdown(
            f'<span class="badge badge-ok">NARRATIVE ONLINE</span>'
            f'<div class="subtle" style="margin-top:0.4rem">{detail}</div>',
            unsafe_allow_html=True,
        )
    else:
        st.markdown(
            f'<span class="badge badge-warn">NARRATIVE OFFLINE</span>'
            f'<div class="subtle" style="margin-top:0.4rem">{detail}<br>'
            "Analysis still runs — all tools are deterministic.</div>",
            unsafe_allow_html=True,
        )

    st.divider()
    df = get_data()
    clean = clean_subset(df)
    st.markdown("**Dataset**")
    st.markdown(
        f'<div class="subtle">'
        f"{len(df):,} hourly readings<br>"
        f"11 sensor tags · 2011–2015<br>"
        f"{100*len(clean)/len(df):.1f}% unflagged<br><br>"
        f"UCI Gas Turbine CO &amp; NOx<br>Emission Data Set"
        "</div>",
        unsafe_allow_html=True,
    )

    st.divider()
    st.markdown(
        '<div class="subtle">'
        "<b>Data provenance</b><br>"
        "This application runs on the public UCI <i>Gas Turbine CO and NOx "
        "Emission Data Set</i> — a combined-cycle plant in north-western "
        "Turkey, 2011–2015. <b>It contains no proprietary or client data.</b>"
        "<br><br>"
        "The project originated from diagnosing a gas-turbine heat-rate "
        "deviation by manually screening 120 process parameters during an "
        "analytics internship at MRPL Refinery. That work informed the "
        "approach; none of its data appears here."
        "</div>",
        unsafe_allow_html=True,
    )


# --------------------------------------------------------------------------
# Header
# --------------------------------------------------------------------------

st.title("ProcessIQ")
st.markdown(
    '<div class="subtle">Ask a question about the plant. The agent selects an '
    "analysis, computes the result, and explains what it means.</div>",
    unsafe_allow_html=True,
)
st.write("")

tab_ask, tab_findings, tab_quality, tab_method = st.tabs(
    ["Ask", "Findings", "Data quality", "Method"]
)


# --------------------------------------------------------------------------
# Tab: Ask
# --------------------------------------------------------------------------

with tab_ask:
    if "history" not in st.session_state:
        st.session_state.history = []

    st.markdown("**Try one of these**")
    cols = st.columns(4)
    clicked = None
    for i, q in enumerate(SAMPLE_QUESTIONS[:4]):
        if cols[i].button(q, key=f"s{i}", width='stretch'):
            clicked = q

    cols2 = st.columns(3)
    for i, q in enumerate(SAMPLE_QUESTIONS[4:7]):
        if cols2[i].button(q, key=f"s{i+4}", width='stretch'):
            clicked = q

    st.write("")
    typed = st.chat_input("Ask about turbine performance, emissions or data quality…")
    question = clicked or typed

    if question:
        with st.spinner("Selecting analysis, computing, interpreting…"):
            response = get_agent().ask(question)
        st.session_state.history.insert(0, response)

    for idx, r in enumerate(st.session_state.history):
        st.divider()
        st.markdown(f"#### {r.question}")

        badge_route = "badge-ok" if r.routed_by == "llm" else "badge-warn"
        badge_narr = "badge-ok" if r.narrated_by == "llm" else "badge-warn"
        st.markdown(
            f'<span class="badge {badge_route}">routed: {r.routed_by}</span>'
            f'<span class="badge {badge_narr}">narrated: {r.narrated_by}</span>'
            f'<span class="badge" style="background:#eef1f5;color:#5a6472">'
            f"tool: {r.tool}</span>",
            unsafe_allow_html=True,
        )
        st.write("")

        st.markdown(
            f'<div class="answer-box">{r.answer.replace(chr(10)+chr(10), "<br><br>")}</div>',
            unsafe_allow_html=True,
        )

        fig = viz.build_figure(r.result.figure_spec)
        if fig is not None:
            st.plotly_chart(fig, width='stretch', key=f"fig{idx}")

        if r.result.table is not None and not r.result.table.empty:
            with st.expander(f"Computed data ({len(r.result.table)} rows)"):
                st.dataframe(r.result.table, width='stretch', hide_index=True)

        if r.result.caveats:
            for c in r.result.caveats:
                st.markdown(f'<div class="caveat">{c}</div>', unsafe_allow_html=True)

        with st.expander("Execution trace"):
            for step in r.trace:
                st.markdown(f'<div class="subtle">• {step}</div>', unsafe_allow_html=True)
            st.markdown(
                f'<div class="subtle">• arguments: <code>{r.tool_args}</code></div>',
                unsafe_allow_html=True,
            )


# --------------------------------------------------------------------------
# Tab: Findings
# --------------------------------------------------------------------------

with tab_findings:
    clean = clean_subset(get_data())

    c1, c2, c3, c4 = st.columns(4)
    with c1:
        metric_card(f"{100*clean['nox_breach'].mean():.1f}%", "hours over NOx limit")
    with c2:
        metric_card("43.2% → 10.0%", "breach rate, 2013 → 2014")
    with c3:
        metric_card("TIT", "true output driver (93.7%)")
    with c4:
        metric_card("−2.57%", "hot-hour output penalty")

    st.write("")
    st.markdown("#### The compliance step change")
    st.markdown(
        '<div class="subtle">The strongest signal in five years of operation: '
        "NOx breaches collapsed between 2013 and 2014, while CO also fell and output "
        "rose — an across-the-board improvement rather than a tuning tradeoff.</div>",
        unsafe_allow_html=True,
    )
    st.plotly_chart(viz.fig_compliance_step_change(), width='stretch')

    st.divider()
    st.markdown("#### The multicollinearity trap")
    st.markdown(
        '<div class="subtle">The same dataset gives opposite answers depending on '
        "which parameters you feed the model. Including load co-symptoms makes CDP "
        "look like the dominant driver at 97.6% importance — a 99.8% accurate model "
        "that reports load predicting load.</div>",
        unsafe_allow_html=True,
    )
    st.plotly_chart(viz.fig_driver_comparison(), width='stretch')

    st.divider()
    st.markdown("#### Ambient effect, hidden by load")
    st.plotly_chart(viz.fig_ambient_effect(), width='stretch')

    st.divider()
    left, right = st.columns(2)
    with left:
        st.plotly_chart(viz.fig_compliance_distribution("NOX"), width='stretch')
    with right:
        st.plotly_chart(viz.fig_emissions_forecast("NOX"), width='stretch')

    st.divider()
    st.plotly_chart(viz.fig_multi_year_trends(), width='stretch')
    st.plotly_chart(viz.fig_correlation_heatmap(), width='stretch')


# --------------------------------------------------------------------------
# Tab: Data quality
# --------------------------------------------------------------------------

with tab_quality:
    from src.agent.tools import data_quality_report

    res = data_quality_report()
    st.markdown(f"#### {res.headline}")
    st.markdown(
        '<div class="subtle">Cleaning was conservative by design: physically '
        "impossible values were corrected, exact duplicates removed, and everything "
        "else suspicious was flagged and retained. Conclusions are drawn only from "
        "unflagged rows, but nothing was silently discarded.</div>",
        unsafe_allow_html=True,
    )
    st.write("")

    c1, c2, c3, c4 = st.columns(4)
    f = res.facts
    with c1:
        metric_card(f"{f['total_rows']:,}", "total readings")
    with c2:
        metric_card(f"{f['fully_unflagged']:,}", "unflagged")
    with c3:
        metric_card(f"{f['stuck_sensor_rows']:,}", "stuck-sensor rows")
    with c4:
        metric_card(f"{f['outlier_rows']:,}", "extreme outliers")

    st.plotly_chart(viz.fig_quality_summary(), width='stretch')

    st.markdown("**Issues detected**")
    st.dataframe(res.table, width='stretch', hide_index=True)

    st.markdown(
        """
**What was found**

- **478 humidity readings above 100% RH** — physically impossible, indicating
  calibration drift. Clipped to 100% and flagged.
- **A 102-hour window with NOx frozen at a constant value** — every other tag
  varies within 5–12 hours, so this is a stuck sensor tag, not steady-state
  operation. Retained but excluded from stated conclusions.
- **7 exact duplicate rows** — removed.
- **2,374 extreme outliers** (beyond a 3× IQR fence) — mostly in TAT and CO,
  consistent with startup and shutdown transients rather than sensor error, so
  they were flagged rather than deleted.
"""
    )


# --------------------------------------------------------------------------
# Tab: Method
# --------------------------------------------------------------------------

with tab_method:
    st.markdown(
        """
#### How ProcessIQ answers a question

Three stages, each able to degrade independently.

**1 · Route** — the question is mapped to one of seven analysis tools. A language
model does this using the semantic layer, which describes every sensor tag, its
units, its physical meaning and its role in the process. If the model is
unavailable or returns an invalid tool, a deterministic keyword router takes
over.

**2 · Execute** — the selected tool runs. **No tool calls a language model.**
Given the same arguments they return the same numbers.

**3 · Narrate** — the computed facts are handed back to the model, which writes
the explanation and a recommendation. It is instructed to use only the supplied
numbers. If it is unavailable, a template renders the results directly.

The consequence: **the model never produces a statistic.** It chooses what to
compute and explains what came back, but every number on screen was calculated
deterministically. If the LLM is down the answers get blunter, not wrong.

---

#### Why the semantic layer matters

Raw column names like `AFDP` or `TIT` mean nothing to a language model. The
semantic layer maps each tag to its physical meaning, operating envelope and
role — and records which parameters must be excluded from driver analysis and
why. It doubles as the governance artifact: every assumption the system makes
is written down in one reviewable file rather than buried in prompts.

---

#### The analytical core

The headline finding is a negative result. A random forest predicting energy
yield from all eight operating parameters achieves **R² = 0.998** and reports
**CDP at 97.6% importance**. That model is accurate and useless: CDP correlates
**+0.987** with output because both track unit load. It reports that load
predicts load.

Restricting the model to genuinely upstream variables — ambient conditions,
filter fouling, firing setpoint — drops R² to **0.986** and puts **TIT at
93.7%**, which matches both turbine thermodynamics and the operator's actual
control lever. The lower R² is the correct outcome; the leakage was removed.

The same trap hides the ambient-temperature effect. Across the whole dataset the
AT–TEY correlation is **−0.033**, which would lead an analyst to dismiss weather
entirely. Within a fixed load band it is **−0.773**, and hot hours cost
**2.57%** of output.

---

#### Data

UCI Machine Learning Repository — *Gas Turbine CO and NOx Emission Data Set*.
36,733 hourly readings from a combined-cycle power plant in north-western
Turkey, 2011–2015, across 11 sensor tags.

Emissions limits used here (70 mg/m³ NOx, 20 mg/m³ CO) are indicative EU
industrial-emissions reference values, not this plant's actual permit
conditions. Timestamps are synthesised from sequential operating hours — the
raw data carries none — so they are valid for ordering and trend analysis but
not for calendar-date claims.
"""
    )
