# ProcessIQ

**An agentic analytics copilot for combined-cycle gas turbine operations.**

Ask a plant question in plain English. The agent selects an analysis, computes
the result deterministically, and explains what it means for the plant — with a
recommendation attached.

---

## The problem this solves

During an analytics internship at an operating refinery I diagnosed a
gas-turbine heat-rate deviation by screening 120 process parameters by hand.
The analysis was sound. It also took weeks, had to be repeated from scratch for
the next question, and lived in a spreadsheet nobody else could interrogate.

ProcessIQ is that workflow as a system: the parameter screen, the compliance
check and the degradation attribution run in seconds, and anyone can ask for
them in plain English.

---

## The finding that matters

The headline result of this project is a **negative** one, and it is the reason
the agent is built the way it is.

A random forest predicting turbine energy yield from all eight operating
parameters achieves **R² = 0.998** and reports **compressor discharge pressure
(CDP) as 97.6% of importance.**

That model is accurate and useless. CDP correlates **+0.987** with output
because both track unit load. The model has discovered that load predicts load.

Restricting the inputs to variables genuinely upstream of output — ambient
conditions, air-filter fouling, firing setpoint — and excluding the load
co-symptoms gives:

| | Naive model | Load-corrected |
|---|---|---|
| R² | 0.998 | 0.986 |
| Top driver | **CDP — 97.6%** ❌ | **TIT — 93.7%** ✅ |

The lower R² is the correct outcome: the leakage was removed. TIT is the
operator's actual control lever, and the result now agrees with turbine
thermodynamics rather than contradicting it.

**The same trap hides the ambient-temperature effect.** Across the whole
dataset the AT–TEY correlation is **−0.033**, which would lead an analyst to
dismiss weather entirely. Within a fixed load band it is **−0.773** — hot hours
cost **2.57%** of output.

ProcessIQ's driver tool excludes load co-symptoms by default, so the agent
cannot reproduce this mistake.

---

## What else the data showed

**NOx compliance collapsed between 2013 and 2014.** The share of operating
hours exceeding the 70 mg/m³ reference limit fell from **43.2% to 10.0%** in a
single year. CO fell 21% over the same step and output rose — so this is not a
NOx–CO tuning tradeoff but an across-the-board improvement, consistent with a
combustor or emissions-control upgrade.

**Over the full record, 27.5% of operating hours exceed the NOx limit**, with
average headroom of only 8%.

**Two data-quality defects were found and are surfaced rather than hidden:**
478 humidity readings above 100% RH (physically impossible — calibration
drift), and a **102-hour window with the NOx sensor frozen** at a constant
value while every other tag varied normally.

---

## Architecture

Three stages, each able to degrade independently.

```
question
   │
   ├─ 1. ROUTE      LLM picks one of 7 tools using the semantic layer.
   │                Falls back to a deterministic keyword router.
   │
   ├─ 2. EXECUTE    The tool runs. No tool calls an LLM.
   │                Same arguments always produce the same numbers.
   │
   └─ 3. NARRATE    Computed facts go back to the LLM, which writes the
                    explanation and recommendation using only those numbers.
                    Falls back to a template.
```

**The LLM never produces a statistic.** It decides what to compute and explains
what came back. Every number on screen was calculated deterministically, so the
model cannot fabricate one. If the LLM is unavailable the answers get blunter,
not wrong — the dashboard stays fully functional.

### The semantic layer

`src/semantic_layer.yaml` maps every sensor tag to its physical meaning, units,
operating envelope and role in the process. Raw column names like `AFDP` mean
nothing to a language model; this is what makes generated SQL and generated
narrative grounded rather than plausible.

It doubles as the governance artifact — every assumption the system makes about
this plant, including which parameters must be excluded from driver analysis
and why, is written down in one reviewable file instead of buried in prompts.

---

## Evaluation

`eval/golden_questions.json` holds 20 questions with known-correct tool
selections and expected computed values, derived from the verified analysis.

```
                      Routing    Facts
Full agent (LLM)        100%      100%
Rule-based fallback      95%      100%
```

Routing and fact accuracy are measured separately because they fail for
different reasons. Routing is probabilistic — the LLM can pick the wrong
analysis. Fact accuracy is deterministic and must be 100%; anything less is a
bug, not model variance.

```bash
python eval/run_eval.py                # full agent
python eval/run_eval.py --rules-only   # no-LLM fallback
```

---

## Data quality approach

Cleaning is **conservative: nothing is silently discarded.**

| Issue | Rows | Action |
|---|---|---|
| Humidity > 100% RH | 478 | Clipped to 100%, flagged |
| Exact duplicate readings | 7 | Removed |
| Stuck sensor windows (NOx, AH) | 126 | **Flagged, retained** |
| Extreme outliers (3× IQR) | 2,374 | **Flagged, retained** |
| Fully unflagged | 34,251 (93.3%) | — |

Conclusions are drawn only from unflagged rows, but every flagged row stays in
the dataset and every transformation is logged to
`data/processed/cleaning_log.csv`. A reviewer can reproduce or reverse any
decision. The extreme outliers in TAT and CO are concentrated in startup and
shutdown transients — genuine plant behaviour, not sensor error, which is why
they are flagged rather than deleted.

---

## Running it

```bash
pip install -r requirements.txt
```

Download the dataset from the
[UCI Machine Learning Repository](https://archive.ics.uci.edu/dataset/551/gas+turbine+co+and+nox+emission+data+set)
and place `gt_2011.csv` … `gt_2015.csv` in `data/raw/`.

```bash
cp .env.example .env        # add a Groq API key (free tier)
python src/data_audit.py    # data quality audit
python src/clean_data.py    # cleaning + feature engineering
python src/analysis.py      # hypothesis testing
streamlit run src/app.py    # the application
```

The app runs without an API key — the chat layer shows an offline badge and
renders computed results directly.

### LLM provider

Provider-agnostic via `LLM_PROVIDER` in `.env`: `groq`, `huggingface`,
`openrouter` or a local `ollama`. All speak the OpenAI chat-completions shape,
so switching is configuration rather than a rewrite.

> This project was originally wired to Hugging Face and hit `HTTP 402` mid-build
> when the free tier ran out. That is why the provider is swappable and why the
> analytical layer never depends on the model being up.

---

## Project structure

```
processiq/
├── data/
│   ├── raw/                     UCI CSVs, 2011-2015
│   └── processed/               cleaned parquet + cleaning audit log
├── src/
│   ├── data_audit.py            8-check data quality audit
│   ├── clean_data.py            conservative cleaning + feature engineering
│   ├── analysis.py              hypothesis testing (H1/H2/H3)
│   ├── semantic_layer.yaml      tag definitions + governance assumptions
│   ├── llm.py                   provider-agnostic client with fallback chain
│   ├── viz.py                   Plotly figures
│   ├── app.py                   Streamlit application
│   └── agent/
│       ├── tools.py             7 deterministic analysis tools
│       └── agent.py             routing, execution, narration
├── eval/
│   ├── golden_questions.json    20 cases with known-correct answers
│   └── run_eval.py              evaluation harness
└── outputs/                     exported figures
```

---

## Data

UCI Machine Learning Repository — *Gas Turbine CO and NOx Emission Data Set*.
36,733 hourly readings from a combined-cycle power plant in north-western
Turkey, 2011–2015, across 11 sensor tags.

Kaya, H., Tüfekci, P., Uzun, E. (2019).

**Two stated assumptions:**

- Emissions limits (70 mg/m³ NOx, 20 mg/m³ CO) are indicative EU
  industrial-emissions reference values, **not this plant's actual permit
  conditions.**
- The raw data carries no timestamp column and contains fewer rows than hours
  in a year, so wall-clock time is unrecoverable. Timestamps are synthesised
  from sequential operating hours — valid for ordering and trend analysis,
  **not for calendar-date claims.**
