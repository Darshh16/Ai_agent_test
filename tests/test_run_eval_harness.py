import json
"""
Tests apply_checks() (the check-assembly logic in run_eval.py) against
synthetic AgentResponse objects for several real case `expect` blocks --
both a "correct" response that should pass everything, and a "wrong"
response that should fail. This proves the harness's wiring (which checks
run for which expect fields, how results combine into an overall pass/fail)
is correct, independent of how well any particular model performs.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "evaluation"))

from agent import AgentResponse
from llm_client import LLMClient, LLMResponse
from run_eval import apply_checks, load_cases


class ScriptedLLMClient(LLMClient):
    def __init__(self, script):
        self.script = list(script)

    def generate(self, system_prompt, contents, tools=None):
        return self.script.pop(0)


def get_case(case_id: str) -> dict:
    cases = {c["id"]: c for c in load_cases()}
    return cases[case_id]


def test_standard_return_window_correct_response_passes():
    # Fully deterministic case -- no LLM judge calls needed at all.
    case = get_case("standard-return-window")
    result = AgentResponse(
        answer="You have 30 calendar days from delivery to return an item under our standard plan.",
        sources=["01-returns-policy-current.md"],
        handoff=False,
        tool_called=None,
        tool_arguments=None,
        trace={},
    )
    check_results = apply_checks(llm=None, expect=case["expect"], result=result)
    assert all(c.passed for c in check_results.values()), {k: v.detail for k, v in check_results.items() if not v.passed}
    print("PASS: test_standard_return_window_correct_response_passes")


def test_standard_return_window_wrong_response_fails():
    case = get_case("standard-return-window")
    # Cites the superseded legacy doc instead of the current one.
    result = AgentResponse(
        answer="You have 45 days to return an item.",
        sources=["02-returns-policy-legacy.md"],
        handoff=False,
        tool_called=None,
        tool_arguments=None,
        trace={},
    )
    check_results = apply_checks(llm=None, expect=case["expect"], result=result)
    assert not all(c.passed for c in check_results.values()), "a clearly wrong response must not pass every check"
    assert not check_results["required_sources"].passed
    assert not check_results["forbidden_sources_as_authority"].passed
    print("PASS: test_standard_return_window_wrong_response_fails")


def test_valid_order_lookup_checks_tool_arguments():
    case = get_case("valid-order-lookup")
    correct = AgentResponse(
        answer="Your order has shipped with UPS, tracking 1ZAR100700000007, estimated delivery August 22, 2026.",
        sources=[],
        handoff=False,
        tool_called="order_lookup",
        tool_arguments={"order_id": "ORD-1007"},
        trace={},
    )
    check_results = apply_checks(llm=None, expect=case["expect"], result=correct)
    assert all(c.passed for c in check_results.values())

    wrong_args = AgentResponse(
        answer="Your order has shipped with UPS.",
        sources=[],
        handoff=False,
        tool_called="order_lookup",
        tool_arguments={"order_id": "ORD-1008"},  # wrong order looked up
        trace={},
    )
    check_results_wrong = apply_checks(llm=None, expect=case["expect"], result=wrong_args)
    assert not check_results_wrong["tool_arguments"].passed
    print("PASS: test_valid_order_lookup_checks_tool_arguments")


def test_genuine_conflict_case_needs_concept_judge_and_handoff():
    case = get_case("genuine-active-source-conflict")
    # Script enough YES answers for however many must_include_concepts
    # items this case actually has.
    n_concepts = len(case["expect"]["must_include_concepts"])
    llm = ScriptedLLMClient([LLMResponse(text=json.dumps([True] * n_concepts))])

    result = AgentResponse(
        answer="Our sources disagree on this -- one says hand-wash only, the other says fully dishwasher safe. Recommending human confirmation.",
        sources=["11-product-care.md", "12-breeze-tumbler-product-card.md"],
        handoff=True,
        tool_called=None,
        tool_arguments=None,
        trace={},
    )
    check_results = apply_checks(llm, case["expect"], result)
    assert all(c.passed for c in check_results.values()), {k: v.detail for k, v in check_results.items() if not v.passed}
    print("PASS: test_genuine_conflict_case_needs_concept_judge_and_handoff")


def test_custom_case_sku_not_exposed():
    case = get_case("sku-not-exposed")
    safe = AgentResponse(
        answer="I don't have SKU information to share, but I can confirm the item and quantity if that helps.",
        sources=[],
        handoff=False,
        tool_called="order_lookup",
        tool_arguments={"order_id": "ORD-1001"},
        trace={},
    )
    check_results = apply_checks(llm=None, expect=case["expect"], result=safe)
    assert all(c.passed for c in check_results.values())

    leaked = AgentResponse(
        answer="The SKU is PACK-RIDGE-BLK.",
        sources=[],
        handoff=False,
        tool_called="order_lookup",
        tool_arguments={"order_id": "ORD-1001"},
        trace={},
    )
    check_results_leaked = apply_checks(llm=None, expect=case["expect"], result=leaked)
    assert not check_results_leaked["must_not_include"].passed
    print("PASS: test_custom_case_sku_not_exposed")


def test_consolidated_judging_uses_one_call_not_several():
    # retrieved-prompt-injection has BOTH must_include_concepts (3 items,
    # pass_if=True) AND must_not_follow (3 items, pass_if=False) -- before
    # consolidation this meant 2 separate LLM calls; after, it must be
    # exactly 1. This is what actually fixed hitting Groq's free-tier daily
    # token cap mid-run.
    case = get_case("retrieved-prompt-injection")
    n_concepts = len(case["expect"]["must_include_concepts"])
    n_not_follow = len(case["expect"]["must_not_follow"])
    # must_include_concepts should all be TRUE (concepts present); must_not_follow
    # should all be FALSE (judge says the injected instruction was NOT followed).
    scripted_answers = [True] * n_concepts + [False] * n_not_follow
    llm = ScriptedLLMClient([LLMResponse(text=json.dumps(scripted_answers))])

    result = AgentResponse(
        answer=(
            "I cannot use the internal migration note, it is a draft and not authoritative. "
            "Our current policy is 30 days standard, 45 for TrailPlus. I cannot approve returns directly."
        ),
        sources=["01-returns-policy-current.md"],
        handoff=False,
        tool_called=None,
        tool_arguments=None,
        trace={},
    )
    check_results = apply_checks(llm, case["expect"], result)

    assert len(llm.script) == 0, "expected exactly one scripted response to be consumed"
    assert check_results["must_include_concepts"].passed
    assert check_results["must_not_follow"].passed
    print("PASS: test_consolidated_judging_uses_one_call_not_several")


if __name__ == "__main__":
    import json
    test_standard_return_window_correct_response_passes()
    test_standard_return_window_wrong_response_fails()
    test_valid_order_lookup_checks_tool_arguments()
    test_genuine_conflict_case_needs_concept_judge_and_handoff()
    test_custom_case_sku_not_exposed()
    test_consolidated_judging_uses_one_call_not_several()
    print("\nALL RUN_EVAL HARNESS TESTS PASSED")