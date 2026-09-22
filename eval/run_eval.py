"""
ProcessIQ - Evaluation harness

Measures two things separately, because they fail for different reasons:

  ROUTING ACCURACY  - did the agent pick the right analysis for the question?
                      This is the part the LLM can get wrong.
  FACT ACCURACY     - were the computed numbers correct?
                      This is deterministic and should be 100%; anything less
                      is a genuine bug, not model variance.

Run:  python eval/run_eval.py
      python eval/run_eval.py --rules-only    (evaluate the no-LLM fallback)
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.agent.agent import ProcessIQAgent, route_with_rules  # noqa: E402
from src.agent.tools import TOOL_REGISTRY  # noqa: E402

GOLDEN = ROOT / "eval" / "golden_questions.json"


def approx_equal(a, b, tol_pct: float) -> bool:
    if isinstance(a, str) or isinstance(b, str):
        return str(a).upper() == str(b).upper()
    if isinstance(a, bool) or isinstance(b, bool):
        return a == b
    try:
        a, b = float(a), float(b)
    except (TypeError, ValueError):
        return a == b
    if b == 0:
        return abs(a) < 1e-9
    return abs(a - b) / abs(b) * 100 <= tol_pct


def evaluate(rules_only: bool = False):
    spec = json.loads(GOLDEN.read_text(encoding="utf-8"))
    tol = spec.get("tolerance_pct", 2.0)
    cases = spec["cases"]

    agent = None if rules_only else ProcessIQAgent()

    route_hits = 0
    fact_checks = 0
    fact_hits = 0
    rows = []

    for case in cases:
        q = case["question"]

        if rules_only:
            tool, args = route_with_rules(q)
            result = TOOL_REGISTRY[tool]["fn"](**args)
            routed_by = "rules"
        else:
            resp = agent.ask(q)
            tool, result, routed_by = resp.tool, resp.result, resp.routed_by

        route_ok = tool == case["expect_tool"]
        route_hits += route_ok

        fact_results = []
        for key, expected in (case.get("expect_facts") or {}).items():
            fact_checks += 1
            actual = result.facts.get(key)
            ok = actual is not None and approx_equal(actual, expected, tol)
            fact_hits += ok
            fact_results.append((key, expected, actual, ok))

        rows.append(
            {
                "id": case["id"],
                "question": q,
                "expected_tool": case["expect_tool"],
                "actual_tool": tool,
                "route_ok": route_ok,
                "routed_by": routed_by,
                "facts": fact_results,
            }
        )

    return rows, route_hits, fact_hits, fact_checks, tol


def report(rows, route_hits, fact_hits, fact_checks, tol, rules_only):
    mode = "RULE-BASED FALLBACK (no LLM)" if rules_only else "FULL AGENT (LLM routing)"
    print("=" * 78)
    print(f"ProcessIQ evaluation - {mode}")
    print("=" * 78)

    for r in rows:
        mark = "PASS" if r["route_ok"] else "FAIL"
        print(f"\n[{mark}] {r['id']}")
        print(f"       {r['question']}")
        if not r["route_ok"]:
            print(f"       expected tool: {r['expected_tool']}")
            print(f"       actual tool  : {r['actual_tool']}  (routed by {r['routed_by']})")
        for key, expected, actual, ok in r["facts"]:
            fmark = "ok  " if ok else "MISS"
            print(f"       [{fmark}] {key}: expected {expected}, got {actual}")

    n = len(rows)
    route_pct = 100 * route_hits / n
    fact_pct = 100 * fact_hits / fact_checks if fact_checks else 100.0

    print("\n" + "=" * 78)
    print("SUMMARY")
    print("=" * 78)
    print(f"  Routing accuracy : {route_hits}/{n}  ({route_pct:.1f}%)")
    print(f"  Fact accuracy    : {fact_hits}/{fact_checks}  ({fact_pct:.1f}%)  "
          f"[tolerance {tol}%]")
    print()
    if fact_pct < 100:
        print("  NOTE: fact accuracy below 100% indicates a computation bug, not")
        print("  model variance - the tools are deterministic.")
    return route_pct, fact_pct


if __name__ == "__main__":
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")

    ap = argparse.ArgumentParser()
    ap.add_argument("--rules-only", action="store_true",
                    help="Evaluate the deterministic fallback router instead of the LLM.")
    args = ap.parse_args()

    rows, rh, fh, fc, tol = evaluate(rules_only=args.rules_only)
    route_pct, fact_pct = report(rows, rh, fh, fc, tol, args.rules_only)

    # CI gate: facts must be perfect; routing is allowed some slack because the
    # LLM is probabilistic and several questions are genuinely ambiguous.
    sys.exit(0 if fact_pct == 100.0 and route_pct >= 75.0 else 1)
