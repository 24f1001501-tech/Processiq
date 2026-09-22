"""
ProcessIQ - Agent

Three stages, each independently degradable:

  1. ROUTE    question -> (tool, arguments)
              LLM first; a keyword router takes over if the LLM is unavailable
              or returns something invalid.
  2. EXECUTE  run the chosen tool. Always deterministic, never uses the LLM.
  3. NARRATE  turn the returned numbers into an answer with a recommendation.
              LLM first; a template takes over if unavailable.

The important property: stage 2 is the only stage that produces numbers, and
it never calls a model. The LLM cannot invent a statistic, because it is only
ever handed facts that have already been computed. If the model is down the
answer gets blunter, not wrong.
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from pathlib import Path

import yaml

from src.agent.tools import TOOL_REGISTRY, ToolResult
from src.llm import LLMResult, get_client

SEMANTIC_PATH = Path(__file__).resolve().parents[1] / "semantic_layer.yaml"


def load_semantic_layer() -> dict:
    with open(SEMANTIC_PATH, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


@dataclass
class AgentResponse:
    question: str
    tool: str
    tool_args: dict
    result: ToolResult
    answer: str
    routed_by: str          # "llm" or "rules"
    narrated_by: str        # "llm" or "template"
    model: str | None = None
    trace: list[str] = field(default_factory=list)


# --------------------------------------------------------------------------
# Stage 1 - routing
# --------------------------------------------------------------------------

# Keyword fallback. Ordered: the first pattern that matches wins, so more
# specific intents are listed before general ones.
# Patterns are deliberately PREFIX matches (\b at the start only). A trailing
# \b would stop "clean" matching "cleaning" and "season" matching "seasonal",
# which is exactly how an earlier version of this router lost a third of its
# accuracy.
RULE_PATTERNS: list[tuple[str, str, dict]] = [
    (r"\b(data quality|trust|reliab|clean|missing value|stuck|sensor fail|"
     r"sensor issue|governance|audit|cleaning decision)",
     "data_quality_report", {}),
    (r"\b(forecast|predict|future|will we|going to|on track|project|"
     r"looking forward|next year|expect)",
     "forecast_emissions", {"pollutant": "NOX"}),
    (r"\b(complian|breach|exceed|limit|regulat|permit|emission|nox|"
     r"carbon monoxide|pollut)",
     "analyse_emissions_compliance", {"pollutant": "NOX"}),
    (r"\b(ambient|weather|season|hot day|cold day|outside temperature|"
     r"summer|winter|air temperature)",
     "ambient_effect_at_constant_load", {}),
    (r"\b(chang|degrad|compare|worse|better|happened|decline|drop|"
     r"between \d{4}|over the (five|5) year|deteriorat)",
     "diagnose_performance_change", {}),
    (r"\b(driver|drive|affect|influenc|caus|important|matter|impact|"
     r"determin|responsible for)",
     "rank_efficiency_drivers", {}),
]


def route_with_rules(question: str) -> tuple[str, dict]:
    """Deterministic keyword routing. Always returns something runnable."""
    q = question.lower()

    for pattern, tool, args in RULE_PATTERNS:
        if re.search(pattern, q):
            args = dict(args)
            # Pull explicit years out of the question where the tool takes them
            years = [int(y) for y in re.findall(r"\b(20[01]\d)\b", q)]
            if tool == "diagnose_performance_change" and len(years) >= 2:
                args["from_year"], args["to_year"] = min(years), max(years)
            elif tool in ("analyse_emissions_compliance",) and len(years) == 1:
                args["year"] = years[0]
            if "co " in q or " co," in q or "carbon monoxide" in q:
                if tool in ("analyse_emissions_compliance", "forecast_emissions"):
                    args["pollutant"] = "CO"
            return tool, args

    # Nothing matched: the driver ranking is the most generally informative answer.
    return "rank_efficiency_drivers", {}


def _routing_prompt(question: str, semantic: dict) -> list[dict]:
    tools_desc = []
    for name, spec in TOOL_REGISTRY.items():
        args = ", ".join(f"{k} ({v})" for k, v in spec["args"].items()) or "no arguments"
        tools_desc.append(f"- {name}: {spec['description']}\n    args: {args}")

    columns = "\n".join(
        f"  {col}: {meta['name']} ({meta.get('unit','-')}) - {meta['role']}"
        for col, meta in semantic["columns"].items()
    )

    system = f"""You route questions about a gas turbine power plant to exactly one analysis tool.

AVAILABLE TOOLS:
{chr(10).join(tools_desc)}

DATA COLUMNS:
{columns}

The table is called `turbine`. Years available: 2011-2015.

Respond with ONLY a JSON object, no prose:
{{"tool": "<tool_name>", "args": {{...}}, "reason": "<one short sentence>"}}

Rules:
- Use query_turbine_data ONLY when no other tool fits; write valid DuckDB SQL.
- For "what drives/affects X" use rank_efficiency_drivers.
- For regulatory/limit/breach questions use analyse_emissions_compliance.
- For "why did X change between years" use diagnose_performance_change.
- Never invent a tool name or an argument name."""

    return [
        {"role": "system", "content": system},
        {"role": "user", "content": question},
    ]


def route(question: str, semantic: dict) -> tuple[str, dict, str, LLMResult | None]:
    """Return (tool_name, args, routed_by, llm_result)."""
    client = None
    try:
        client = get_client()
    except Exception:
        pass

    if client and client.configured:
        obj, res = client.json_chat(_routing_prompt(question, semantic), max_tokens=1200)
        if obj and obj.get("tool") in TOOL_REGISTRY:
            args = obj.get("args") or {}
            if isinstance(args, dict):
                # Drop any argument the tool does not accept - small models
                # occasionally invent plausible-looking parameter names.
                valid = set(TOOL_REGISTRY[obj["tool"]]["args"])
                args = {k: v for k, v in args.items() if k in valid}
                return obj["tool"], args, "llm", res
        # LLM answered but unusably: fall through to rules.
        tool, args = route_with_rules(question)
        return tool, args, "rules", res

    tool, args = route_with_rules(question)
    return tool, args, "rules", None


# --------------------------------------------------------------------------
# Stage 3 - narration
# --------------------------------------------------------------------------

def _narration_prompt(question: str, result: ToolResult, semantic: dict) -> list[dict]:
    findings = "\n".join(
        f"- {f['finding'].strip()}" for f in semantic.get("verified_findings", [])
    )

    system = f"""You are a process-analytics consultant reporting to a plant manager.

You will be given a question and the OUTPUT OF A COMPUTATION. Write the answer.

ABSOLUTE RULES:
- Use ONLY the numbers in the FACTS block. Never state a figure that is not there.
- Never estimate, extrapolate or recall numbers from memory.
- If the facts don't answer the question, say so plainly.

PREVIOUSLY VERIFIED CONTEXT (established facts about this plant, safe to reference):
{findings}

FORMAT (no headings, no bullets, no markdown):
Paragraph 1 - the direct answer with the key numbers.
Paragraph 2 - what it means for the plant, mechanistically.
Paragraph 3 - one sentence starting "Recommendation:" with a concrete action.

Keep it under 160 words. Write plainly. No filler."""

    user = f"QUESTION: {question}\n\n{result.summary_for_llm()}"
    return [{"role": "system", "content": system}, {"role": "user", "content": user}]


def narrate_with_template(question: str, result: ToolResult) -> str:
    """Deterministic narration used when the LLM is unavailable."""
    lines = [result.headline, ""]
    if result.facts:
        lines.append("Key figures:")
        for k, v in list(result.facts.items())[:8]:
            label = k.replace("_", " ")
            if isinstance(v, float):
                lines.append(f"  - {label}: {v:,.4g}")
            elif isinstance(v, (list, dict)):
                continue
            else:
                lines.append(f"  - {label}: {v}")
    if result.caveats:
        lines.append("")
        lines.append("Caveats:")
        lines.extend(f"  - {c}" for c in result.caveats)
    lines.append("")
    lines.append("(Narrative layer offline - showing computed results directly.)")
    return "\n".join(lines)


def narrate(question: str, result: ToolResult, semantic: dict) -> tuple[str, str, str | None]:
    """Return (answer_text, narrated_by, model_name)."""
    try:
        client = get_client()
    except Exception:
        return narrate_with_template(question, result), "template", None

    if not client.configured:
        return narrate_with_template(question, result), "template", None

    res = client.chat(_narration_prompt(question, result, semantic), max_tokens=1800)
    if res.ok and res.text.strip():
        return res.text.strip(), "llm", res.model
    return narrate_with_template(question, result), "template", None


# --------------------------------------------------------------------------
# Orchestration
# --------------------------------------------------------------------------

class ProcessIQAgent:
    def __init__(self):
        self.semantic = load_semantic_layer()

    def ask(self, question: str) -> AgentResponse:
        trace = []

        tool_name, args, routed_by, route_res = route(question, self.semantic)
        trace.append(f"routed to `{tool_name}` by {routed_by}")
        if route_res and route_res.attempts:
            trace.append(f"routing attempts: {', '.join(route_res.attempts)}")

        try:
            result = TOOL_REGISTRY[tool_name]["fn"](**args)
        except TypeError as e:
            # Bad arguments from the router: retry the tool with its defaults.
            trace.append(f"argument error ({e}); retrying with defaults")
            args = {}
            result = TOOL_REGISTRY[tool_name]["fn"]()
        trace.append(f"executed with args {args or '{}'}")

        answer, narrated_by, model = narrate(question, result, self.semantic)
        trace.append(f"narrated by {narrated_by}")

        return AgentResponse(
            question=question,
            tool=tool_name,
            tool_args=args,
            result=result,
            answer=answer,
            routed_by=routed_by,
            narrated_by=narrated_by,
            model=model,
            trace=trace,
        )


SAMPLE_QUESTIONS = [
    "What actually drives turbine energy yield at this plant?",
    "How exposed are we on NOx compliance?",
    "Why did performance change between 2013 and 2014?",
    "Does ambient temperature really affect our output?",
    "Are we on track to breach the NOx limit?",
    "Can I trust this data?",
    "What was the average energy yield each year?",
]


if __name__ == "__main__":
    import sys

    # Windows consoles default to cp1252 and crash on characters the model
    # emits (narrow no-break space, en dashes). Force UTF-8 for this CLI.
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")

    agent = ProcessIQAgent()
    for q in SAMPLE_QUESTIONS[:4]:
        print("\n" + "=" * 78)
        print(f"Q: {q}")
        print("=" * 78)
        r = agent.ask(q)
        print(f"[tool: {r.tool} | routed by {r.routed_by} | narrated by {r.narrated_by}"
              f"{' via ' + r.model if r.model else ''}]\n")
        print(r.answer)
