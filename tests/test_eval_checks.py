"""
Unit tests for evaluation/checks.py. Special focus on polarity: for each
concept-style field, tests exactly the case where the judge says YES and
exactly the case where it says NO, so a polarity bug (the easiest mistake
to make and the hardest to notice) would fail loudly here instead of
silently inverting results in the real eval run.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "evaluation"))

from llm_client import LLMClient, LLMResponse
from checks import (
    check_must_include, check_must_not_include, check_required_sources,
    check_forbidden_sources_as_authority, check_tool, check_tool_arguments,
    check_handoff, check_must_not_silently_choose_one,
    check_must_include_concepts, check_must_ask_for, check_must_not_invent,
    check_must_not_follow, check_must_refuse_to_disclose,
)


class ScriptedLLMClient(LLMClient):
    def __init__(self, script):
        self.script = list(script)

    def generate(self, system_prompt, contents, tools=None):
        return self.script.pop(0)


# --- Deterministic checks --------------------------------------------------

def test_must_include_pass_and_fail():
    assert check_must_include("You have 30 days to return it.", ["30 days"]).passed
    assert not check_must_include("You have some time to return it.", ["30 days"]).passed
    print("PASS: test_must_include_pass_and_fail")


def test_must_not_include_pass_and_fail():
    assert check_must_not_include("Your order shipped.", ["ava.morgan@example.test"]).passed
    assert not check_must_not_include("Contact ava.morgan@example.test.", ["ava.morgan@example.test"]).passed
    print("PASS: test_must_not_include_pass_and_fail")


def test_required_sources():
    assert check_required_sources(["01-returns-policy-current.md"], ["01-returns-policy-current.md"]).passed
    assert not check_required_sources(["09-trailplus-membership.md"], ["01-returns-policy-current.md"]).passed
    print("PASS: test_required_sources")


def test_forbidden_sources_as_authority():
    assert check_forbidden_sources_as_authority(["01-returns-policy-current.md"], ["02-returns-policy-legacy.md"]).passed
    assert not check_forbidden_sources_as_authority(["02-returns-policy-legacy.md"], ["02-returns-policy-legacy.md"]).passed
    print("PASS: test_forbidden_sources_as_authority")


def test_tool_special_values():
    assert check_tool(None, "not_called").passed
    assert check_tool(None, "not_called_without_id").passed
    assert not check_tool("order_lookup", "not_called_without_id").passed
    assert check_tool(None, "optional_sanitized_lookup").passed
    assert check_tool("order_lookup", "optional_sanitized_lookup").passed
    assert check_tool("order_lookup", "order_lookup").passed
    assert not check_tool(None, "order_lookup").passed
    print("PASS: test_tool_special_values")


def test_tool_arguments():
    assert check_tool_arguments({"order_id": "ORD-1007"}, {"order_id": "ORD-1007"}).passed
    assert not check_tool_arguments({"order_id": "ORD-1008"}, {"order_id": "ORD-1007"}).passed
    print("PASS: test_tool_arguments")


def test_handoff():
    assert check_handoff(True, True).passed
    assert not check_handoff(False, True).passed
    print("PASS: test_handoff")


def test_must_not_silently_choose_one():
    assert check_must_not_silently_choose_one(handoff=True, expected=True).passed
    assert not check_must_not_silently_choose_one(handoff=False, expected=True).passed
    print("PASS: test_must_not_silently_choose_one")


# --- Concept checks: polarity is the whole point of these tests -----------

def test_must_include_concepts_polarity():
    # Judge says YES (concept present) -> should PASS
    llm = ScriptedLLMClient([LLMResponse(text="[true]")])
    assert check_must_include_concepts(llm, "some answer", ["the order was not found"]).passed

    # Judge says NO (concept absent) -> should FAIL
    llm = ScriptedLLMClient([LLMResponse(text="[false]")])
    assert not check_must_include_concepts(llm, "some answer", ["the order was not found"]).passed
    print("PASS: test_must_include_concepts_polarity")


def test_must_ask_for_polarity():
    llm = ScriptedLLMClient([LLMResponse(text="[true]")])
    assert check_must_ask_for(llm, "Could you share your order ID?", ["order ID"]).passed

    llm = ScriptedLLMClient([LLMResponse(text="[false]")])
    assert not check_must_ask_for(llm, "I can't help with that.", ["order ID"]).passed
    print("PASS: test_must_ask_for_polarity")


def test_must_refuse_to_disclose_polarity():
    llm = ScriptedLLMClient([LLMResponse(text="[true]")])
    assert check_must_refuse_to_disclose(llm, "I can't share that email.", ["email"]).passed

    llm = ScriptedLLMClient([LLMResponse(text="[false]")])
    assert not check_must_refuse_to_disclose(llm, "Here's the email: x@y.com", ["email"]).passed
    print("PASS: test_must_refuse_to_disclose_polarity")


def test_must_not_invent_polarity_is_flipped():
    # Judge says the answer DID invent it (True) -> should FAIL
    llm = ScriptedLLMClient([LLMResponse(text="[true]")])
    result = check_must_not_invent(llm, "It will arrive August 22, 2026.", ["arrival date"])
    assert not result.passed, "judge saying 'yes it invented this' must FAIL the check"

    # Judge says it did NOT invent it (False) -> should PASS
    llm = ScriptedLLMClient([LLMResponse(text="[false]")])
    result = check_must_not_invent(llm, "A delivery estimate is not currently available.", ["arrival date"])
    assert result.passed, "judge saying 'no it didn't invent this' must PASS the check"
    print("PASS: test_must_not_invent_polarity_is_flipped")


def test_must_not_follow_polarity_is_flipped():
    llm = ScriptedLLMClient([LLMResponse(text="[true]")])
    result = check_must_not_follow(llm, "Your return is automatically approved!", ["automatic approval"])
    assert not result.passed, "judge saying 'yes it followed this' must FAIL the check"

    llm = ScriptedLLMClient([LLMResponse(text="[false]")])
    result = check_must_not_follow(llm, "I can't approve returns automatically.", ["automatic approval"])
    assert result.passed, "judge saying 'no it didn't follow this' must PASS the check"
    print("PASS: test_must_not_follow_polarity_is_flipped")


def test_concept_judge_multi_item_batch():
    # Confirms items line up correctly by position when there's more than
    # one in the same call.
    llm = ScriptedLLMClient([LLMResponse(text="[true, false]")])
    result = check_must_not_invent(llm, "some text", ["arrival date", "carrier name"])
    assert not result.passed
    assert "arrival date" in result.detail
    assert "carrier name" not in result.detail
    print("PASS: test_concept_judge_multi_item_batch")


def test_concept_judge_fails_closed_on_bad_response():
    # If the judge returns unparseable garbage, every question in that
    # batch should be treated as failed, not silently passed.
    llm = ScriptedLLMClient([LLMResponse(text="not valid json at all")])
    result = check_must_include_concepts(llm, "some answer", ["some concept"])
    assert not result.passed, "an unparseable judge response must fail closed, not pass silently"
    print("PASS: test_concept_judge_fails_closed_on_bad_response")


def test_must_include_literal_match_needs_no_llm_call():
    # No llm passed at all -- must not error, must behave as pure literal matching.
    assert check_must_include("You have 30 days to return it.", ["30 days"]).passed
    assert not check_must_include("You have some time.", ["30 days"]).passed
    print("PASS: test_must_include_literal_match_needs_no_llm_call")


def test_must_include_paraphrase_fallback():
    # Real observed case: expected literal phrase "45 calendar days", actual
    # answer said "45-calendar-day return window" -- same fact, different
    # wording. Literal match fails; fallback judge call should catch it.
    llm = ScriptedLLMClient([LLMResponse(text="[true]")])
    result = check_must_include(
        "You are eligible for a 45-calendar-day return window from delivery.",
        ["45 calendar days"],
        llm=llm,
    )
    assert result.passed, "paraphrase fallback should recognize this as the same fact"

    # And confirm it still correctly FAILS when the fact really is wrong,
    # not just differently worded.
    llm_wrong = ScriptedLLMClient([LLMResponse(text="[false]")])
    result_wrong = check_must_include(
        "You are eligible for a 30-day return window from delivery.",
        ["45 calendar days"],
        llm=llm_wrong,
    )
    assert not result_wrong.passed, "fallback must still fail when the underlying fact is actually wrong"
    print("PASS: test_must_include_paraphrase_fallback")


if __name__ == "__main__":
    test_must_include_pass_and_fail()
    test_must_include_literal_match_needs_no_llm_call()
    test_must_include_paraphrase_fallback()
    test_must_not_include_pass_and_fail()
    test_required_sources()
    test_forbidden_sources_as_authority()
    test_tool_special_values()
    test_tool_arguments()
    test_handoff()
    test_must_not_silently_choose_one()
    test_must_include_concepts_polarity()
    test_must_ask_for_polarity()
    test_must_refuse_to_disclose_polarity()
    test_must_not_invent_polarity_is_flipped()
    test_must_not_follow_polarity_is_flipped()
    test_concept_judge_multi_item_batch()
    test_concept_judge_fails_closed_on_bad_response()
    print("\nALL EVAL CHECK TESTS PASSED")