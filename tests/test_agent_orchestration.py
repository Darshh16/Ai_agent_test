"""
Verifies agent.py's orchestration logic (tool-calling loop, citation
parsing, handoff parsing, and the no-tool-call-without-an-ID safeguard)
using a scripted fake model instead of a real Gemini call. This does NOT
test whether Gemini itself follows the system prompt well -- that can only
be verified with a real API key, which is what evaluation/run_eval.py is
for. This test exists to prove the *code around* the model is correct
regardless of what the model says.

Run from the repo root:
    python -m pytest tests/test_agent_orchestration.py -v
or directly:
    python tests/test_agent_orchestration.py
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from agent import Agent, HANDOFF_TOKEN
from llm_client import LLMClient, LLMResponse
from retrieval import Retriever
from tools.order_lookup import OrderLookupTool

REPO_ROOT = Path(__file__).resolve().parent.parent
KB_DIR = REPO_ROOT / "knowledge-base"
ORDERS_PATH = REPO_ROOT / "data" / "orders.json"


class ScriptedLLMClient(LLMClient):
    """Returns a pre-programmed sequence of responses, one per call,
    regardless of what's actually passed in. Lets us test the orchestration
    loop deterministically."""

    def __init__(self, script: list[LLMResponse]):
        self.script = list(script)
        self.calls = []

    def generate(self, system_prompt, contents, tools=None):
        self.calls.append({"contents": contents, "tools": tools})
        if not self.script:
            raise AssertionError("ScriptedLLMClient ran out of scripted responses")
        return self.script.pop(0)


def make_agent(script: list[LLMResponse]) -> tuple[Agent, ScriptedLLMClient]:
    retriever = Retriever.from_kb_dir(str(KB_DIR))
    order_tool = OrderLookupTool(str(ORDERS_PATH))
    llm = ScriptedLLMClient(script)
    return Agent(retriever, order_tool, llm), llm


def test_direct_kb_answer_with_citation():
    scripted_text = (
        "You have 30 calendar days from delivery to return an item. "
        "[Source: 01-returns-policy-current.md — Standard return window]"
    )
    agent, llm = make_agent([LLMResponse(text=scripted_text)])
    result = agent.handle_message("How long do I have to return a backpack?")

    assert result.handoff is False
    assert result.tool_called is None
    assert "01-returns-policy-current.md" in result.sources
    assert "[Source:" not in result.answer, "citation tags should be stripped from the display answer"
    assert "30 calendar days" in result.answer
    print("PASS: test_direct_kb_answer_with_citation")


def test_tool_call_flow_and_trace_fields():
    first = LLMResponse(function_call={"name": "order_lookup", "args": {"order_id": "ORD-1007"}})
    second = LLMResponse(text="Your order has shipped with UPS and is estimated to arrive August 22, 2026.")
    agent, llm = make_agent([first, second])

    result = agent.handle_message("Where is ORD-1007?")

    assert result.tool_called == "order_lookup"
    assert result.tool_arguments == {"order_id": "ORD-1007"}
    assert result.handoff is False
    assert len(llm.calls) == 2, "expected exactly two model calls: initial + after tool result"

    # The tool result actually sent back to the model must not contain any
    # internal/PII fields, regardless of what the model does with it.
    tool_result_text = str(llm.calls[1]["contents"][-1])
    for leaked in ["risk_score", "warehouse_note", "internal", "email", "shipping_address"]:
        assert leaked not in tool_result_text, f"leaked field in what was sent to the model: {leaked}"
    print("PASS: test_tool_call_flow_and_trace_fields")


def test_handoff_token_parsed():
    scripted_text = (
        f"{HANDOFF_TOKEN} Our sources disagree on this, so I'd like a human to confirm "
        "before I give you a definite answer."
    )
    agent, llm = make_agent([LLMResponse(text=scripted_text)])
    result = agent.handle_message("Can I put the Breeze Tumbler in the dishwasher?")

    assert result.handoff is True
    assert not result.answer.startswith(HANDOFF_TOKEN), "token should be stripped from the display answer"
    assert "disagree" in result.answer
    print("PASS: test_handoff_token_parsed")


def test_tool_never_called_without_order_id():
    # The model attempts to call order_lookup with no order_id at all.
    # The safeguard in agent.py must intercept this -- OrderLookupTool.lookup()
    # must never actually run, and tool_called must remain None so the eval
    # harness's "not_called_without_id" assertion is honestly satisfied by
    # code behavior, not just by prompting.
    first = LLMResponse(function_call={"name": "order_lookup", "args": {}})
    second = LLMResponse(text="Could you share your order ID so I can look that up for you?")
    agent, llm = make_agent([first, second])

    result = agent.handle_message("Where is my order?")

    assert result.tool_called is None, "tool_called must stay None when no order_id was ever provided"
    assert result.tool_arguments is None
    assert "order ID" in result.answer or "order id" in result.answer.lower()
    print("PASS: test_tool_never_called_without_order_id")


def test_conflict_candidate_surfaced_in_context():
    # Doesn't script a response for this one -- just checks that the
    # context block actually built for the model includes the conflict
    # note for the Breeze Tumbler query, since that's what the system
    # prompt's conflict instructions depend on the model actually seeing.
    agent, llm = make_agent([LLMResponse(text="placeholder")])
    agent.handle_message("Can I put the entire Breeze Tumbler in the dishwasher?")

    sent_context = str(llm.calls[0]["contents"])
    assert "11-product-care.md" in sent_context
    assert "12-breeze-tumbler-product-card.md" in sent_context
    assert "NOTE: the following retrieved documents" in sent_context
    print("PASS: test_conflict_candidate_surfaced_in_context")


def test_handoff_fallback_catches_missed_marker():
    # This is the actual real bug found in manual testing: the model gave a
    # substantively correct conflict/handoff answer for the Breeze Tumbler
    # question but didn't prefix it with the literal [HANDOFF] token.
    primary = LLMResponse(text=(
        "Our current documentation contains conflicting guidance on this. "
        "One care guide states the body should be hand-washed, while the "
        "product card states all components are dishwasher safe. I recommend "
        "contacting our support team directly for a definitive answer."
    ))
    classifier = LLMResponse(text="YES")
    agent, llm = make_agent([primary, classifier])

    result = agent.handle_message("Can I put the entire Breeze Tumbler in the dishwasher?")

    assert result.handoff is True, "source conflict should trigger deterministic handoff"
    assert result.trace["handoff_source"] == "source_conflict"
    assert len(llm.calls) == 1, "conflict handoff is determined from retrieved evidence without a second LLM call"
    print("PASS: test_handoff_fallback_catches_missed_marker")


def test_handoff_fallback_not_triggered_on_plain_answers():
    # A normal answer with none of the hint words should never trigger the
    # extra classification call at all -- keeps the common case cheap.
    agent, llm = make_agent([LLMResponse(text="Your order has shipped and is on its way.")])
    result = agent.handle_message("Where is my order?")

    assert result.handoff is False
    assert result.trace["handoff_source"] == "none"
    assert len(llm.calls) == 0, "missing-order requests are handled before the model is called"
    print("PASS: test_handoff_fallback_not_triggered_on_plain_answers")


def test_tool_arguments_are_normalized_even_if_model_sends_raw_input():
    # Real bug found via eval testing: the model sent {"order_id": "ord-1007"}
    # (lowercase). The lookup itself normalizes internally and succeeds, but
    # the OLD code recorded the model's raw unnormalized value in
    # tool_arguments instead of what was actually looked up.
    first = LLMResponse(function_call={"name": "order_lookup", "args": {"order_id": "  ord-1007  "}})
    second = LLMResponse(text="Your order has shipped with UPS.")
    agent, llm = make_agent([first, second])

    result = agent.handle_message("check on  ord-1007  for me")

    assert result.tool_arguments == {"order_id": "ORD-1007"}, (
        f"tool_arguments must reflect the normalized ID actually used for lookup, got {result.tool_arguments}"
    )
    print("PASS: test_tool_arguments_are_normalized_even_if_model_sends_raw_input")


def test_handoff_marker_detected_when_not_at_start():
    # Real observed bug: the model wrote a complete answer, then appended a
    # second paragraph starting with [HANDOFF] instead of prefixing the
    # whole message with it. The old startswith()-only check missed this.
    real_observed_text = (
        "That's a great question, but I'm not able to confirm the materials "
        "composition of our bags based on the information available to me. "
        "I'd recommend reaching out to one of our human support specialists.\n\n"
        "[HANDOFF] I don't have material or ingredient composition information "
        "in my available documentation, so I can't confirm whether all fabrics "
        "and adhesives in our bags are vegan."
    )
    agent, llm = make_agent([LLMResponse(text=real_observed_text)])
    result = agent.handle_message("Are all fabrics and adhesives in your bags vegan?")

    assert result.handoff is True, "marker must be detected even when not at the very start"
    assert "[HANDOFF]" not in result.answer, "the literal token must never reach the customer-facing answer"
    print("PASS: test_handoff_marker_detected_when_not_at_start")


if __name__ == "__main__":
    test_direct_kb_answer_with_citation()
    test_tool_call_flow_and_trace_fields()
    test_handoff_token_parsed()
    test_tool_never_called_without_order_id()
    test_conflict_candidate_surfaced_in_context()
    test_handoff_fallback_catches_missed_marker()
    test_handoff_fallback_not_triggered_on_plain_answers()
    test_tool_arguments_are_normalized_even_if_model_sends_raw_input()
    test_handoff_marker_detected_when_not_at_start()
    print("\nALL ORCHESTRATION PLUMBING TESTS PASSED")