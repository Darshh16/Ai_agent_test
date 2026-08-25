"""
Evaluation harness: runs every case in visible-cases.json plus
custom-cases.json through the real agent (needs a live LLM call -- uses
whichever provider is configured in .env), applies the checks from
checks.py, and reports results grouped by category.

Usage (from the repo root):
    python evaluation/run_eval.py                       # run everything, print results
    python evaluation/run_eval.py --save-as baseline      # also save to evaluation/results/baseline.json
    python evaluation/run_eval.py --save-as final          # save to evaluation/results/final.json
    python evaluation/run_eval.py --case standard-return-window   # run just one case, for debugging
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT / "src"))
sys.path.insert(0, str(REPO_ROOT / "evaluation"))

from dotenv import load_dotenv  # noqa: E402
load_dotenv(REPO_ROOT / ".env")

import checks  # noqa: E402
from agent import Agent, AgentResponse  # noqa: E402
from llm_client import GeminiClient, GroqClient, LLMClient  # noqa: E402
from memory import Session, rewrite_query  # noqa: E402
from retrieval import Retriever  # noqa: E402
from tools.order_lookup import OrderLookupTool  # noqa: E402


def load_cases(case_id_filter: str | None = None) -> list[dict]:
    visible = json.loads((REPO_ROOT / "evaluation" / "visible-cases.json").read_text())["cases"]
    custom_path = REPO_ROOT / "evaluation" / "custom-cases.json"
    custom = json.loads(custom_path.read_text())["cases"] if custom_path.exists() else []
    all_cases = visible + custom
    if case_id_filter:
        return [c for c in all_cases if c["id"] == case_id_filter]
    return all_cases


def build_backend() -> tuple[Agent, LLMClient]:
    provider = os.environ.get("LLM_PROVIDER", "groq").lower()
    llm = GroqClient() if provider == "groq" else GeminiClient()
    retriever = Retriever.from_kb_dir(str(REPO_ROOT / "knowledge-base"))
    order_tool = OrderLookupTool(str(REPO_ROOT / "data" / "orders.json"))
    return Agent(retriever, order_tool, llm), llm


def apply_checks(llm: LLMClient, expect: dict, result: AgentResponse) -> dict:
    """Pure function of (expect, result) -> per-check results. Deliberately
    separated from run_case() so it can be unit tested against synthetic
    AgentResponse objects without needing to run a real agent turn --
    see tests/test_run_eval_harness.py."""
    check_results: dict = {}

    if "must_include" in expect:
        check_results["must_include"] = checks.check_must_include(result.answer, expect["must_include"], llm=llm)
    if "must_not_include" in expect:
        check_results["must_not_include"] = checks.check_must_not_include(result.answer, expect["must_not_include"])
    if "required_sources" in expect:
        check_results["required_sources"] = checks.check_required_sources(result.sources, expect["required_sources"])
    if "forbidden_sources_as_authority" in expect:
        check_results["forbidden_sources_as_authority"] = checks.check_forbidden_sources_as_authority(
            result.sources, expect["forbidden_sources_as_authority"]
        )
    if "tool" in expect:
        check_results["tool"] = checks.check_tool(result.tool_called, expect["tool"])
    if "tool_arguments" in expect:
        check_results["tool_arguments"] = checks.check_tool_arguments(result.tool_arguments, expect["tool_arguments"])
    if "handoff" in expect:
        check_results["handoff"] = checks.check_handoff(result.handoff, expect["handoff"])
    if "must_not_silently_choose_one" in expect:
        check_results["must_not_silently_choose_one"] = checks.check_must_not_silently_choose_one(
            result.handoff, expect["must_not_silently_choose_one"]
        )
    if "must_include_concepts" in expect:
        check_results["must_include_concepts"] = checks.check_must_include_concepts(
            llm, result.answer, expect["must_include_concepts"]
        )
    if "must_ask_for" in expect:
        check_results["must_ask_for"] = checks.check_must_ask_for(llm, result.answer, expect["must_ask_for"])
    if "must_refuse_to_disclose" in expect:
        check_results["must_refuse_to_disclose"] = checks.check_must_refuse_to_disclose(
            llm, result.answer, expect["must_refuse_to_disclose"]
        )
    if "must_not_invent" in expect:
        check_results["must_not_invent"] = checks.check_must_not_invent(llm, result.answer, expect["must_not_invent"])
    if "must_not_follow" in expect:
        check_results["must_not_follow"] = checks.check_must_not_follow(llm, result.answer, expect["must_not_follow"])

    return check_results


def run_case(agent: Agent, llm: LLMClient, case: dict) -> dict:
    session = Session(session_id=f"eval-{case['id']}")
    result: AgentResponse | None = None

    for msg in case["messages"]:
        if msg["role"] != "user":
            continue
        query = rewrite_query(llm, session, msg["content"])
        result = agent.handle_message(msg["content"], history=session.to_gemini_contents(), retrieval_query=query)
        session.add_turn("user", msg["content"])
        session.add_turn("agent", result.answer)

    check_results = apply_checks(llm, case["expect"], result)
    passed = all(c.passed for c in check_results.values())

    return {
        "id": case["id"],
        "category": case["category"],
        "passed": passed,
        "checks": {k: {"passed": v.passed, "detail": v.detail} for k, v in check_results.items()},
        "answer": result.answer,
        "sources": result.sources,
        "handoff": result.handoff,
        "tool_called": result.tool_called,
        "tool_arguments": result.tool_arguments,
    }


def print_report(results: list[dict]) -> None:
    print("\n" + "=" * 70)
    print("DETAILED RESULTS")
    print("=" * 70)
    for r in results:
        status = "PASS" if r["passed"] else "FAIL"
        print(f"\n[{status}] {r['id']} ({r['category']})")
        if r.get("error"):
            print(f"  ERROR: {r['error']}")
            continue
        for name, c in r["checks"].items():
            print(f"  - {name}: {'ok' if c['passed'] else 'FAILED — ' + c['detail']}")

    print("\n" + "=" * 70)
    print("SUMMARY BY CATEGORY")
    print("=" * 70)
    by_category: dict[str, list[bool]] = {}
    for r in results:
        by_category.setdefault(r["category"], []).append(r["passed"])
    for cat, passes in sorted(by_category.items()):
        print(f"  {cat:28} {sum(passes)}/{len(passes)}")

    total_passed = sum(r["passed"] for r in results)
    print(f"\nTOTAL: {total_passed}/{len(results)} passed")


def run_case_with_retry(agent: Agent, llm: LLMClient, case: dict, max_retries: int = 3) -> dict:
    last_error = None
    for attempt in range(max_retries + 1):
        try:
            return run_case(agent, llm, case)
        except Exception as e:
            last_error = e
            is_rate_limit = "429" in str(e) or "rate_limit" in str(e).lower()
            if is_rate_limit and attempt < max_retries:
                wait = 8 * (attempt + 1)  # 8s, 16s, 24s -- free-tier TPM limits reset per minute
                print(f"\n  Rate limited, waiting {wait}s before retry {attempt + 1}/{max_retries}...", end=" ", flush=True)
                time.sleep(wait)
                continue
            break
    return {"id": case["id"], "category": case["category"], "passed": False, "checks": {}, "error": str(last_error)}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--save-as", default=None, help="save to evaluation/results/<name>.json, e.g. baseline or final")
    parser.add_argument("--case", default=None, help="run only the case with this id")
    args = parser.parse_args()

    cases = load_cases(args.case)
    if not cases:
        print(f"No matching case(s) found for --case={args.case!r}")
        return

    agent, llm = build_backend()

    results = []
    for case in cases:
        print(f"Running: {case['id']} ({case['category']})...", end=" ", flush=True)
        r = run_case_with_retry(agent, llm, case)
        results.append(r)
        print("PASS" if r["passed"] else "FAIL")

    print_report(results)

    if args.save_as:
        out_dir = REPO_ROOT / "evaluation" / "results"
        out_dir.mkdir(parents=True, exist_ok=True)
        out_path = out_dir / f"{args.save_as}.json"
        by_category: dict[str, list[bool]] = {}
        for r in results:
            by_category.setdefault(r["category"], []).append(r["passed"])
        out_path.write_text(json.dumps({
            "run_at": datetime.now(timezone.utc).isoformat(),
            "total": len(results),
            "passed": sum(r["passed"] for r in results),
            "by_category": {cat: {"passed": sum(p), "total": len(p)} for cat, p in by_category.items()},
            "results": results,
        }, indent=2))
        print(f"\nSaved to {out_path}")


if __name__ == "__main__":
    main()