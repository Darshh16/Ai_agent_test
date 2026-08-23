"""
Regression test for a real bug found during manual testing: a reasoning
model (qwen/qwen3.6-27b on Groq) returned its internal <think>...</think>
trace as part of response.text instead of a clean answer. Repro: run the
Canada follow-up flow against that model and inspect the "rewritten query"
-- it came back as the model's full chain of thought, not a question.

Root cause: _strip_reasoning_trace didn't exist -- raw model text was
passed straight through unmodified.

Fix: strip <think>...</think> blocks at the provider-client boundary in
llm_client.py, so every caller gets clean text automatically.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from llm_client import _strip_reasoning_trace


# This is the actual (trimmed) shape of what qwen/qwen3.6-27b returned via
# Groq during manual testing of the Canada follow-up scenario.
REAL_OBSERVED_OUTPUT = """<think>
Here's a thinking process:
1. Analyze User Input...
2. Identify Context & Entities...
I'll go with: "Do you ship internationally to Canada?"
</think>

Do you ship internationally to Canada?"""


def test_strips_real_observed_think_block():
    result = _strip_reasoning_trace(REAL_OBSERVED_OUTPUT)
    assert result == "Do you ship internationally to Canada?"
    assert "<think>" not in result
    assert "thinking process" not in result
    print("PASS: test_strips_real_observed_think_block")


def test_leaves_normal_text_unchanged():
    result = _strip_reasoning_trace("You have 30 days to return an item.")
    assert result == "You have 30 days to return an item."
    print("PASS: test_leaves_normal_text_unchanged")


def test_handles_none():
    assert _strip_reasoning_trace(None) is None
    print("PASS: test_handles_none")


def test_does_not_return_empty_if_entire_text_was_a_think_block():
    # Defensive: if a model somehow returns ONLY a think block with nothing
    # after it, we'd rather show the raw (odd) text than silently return
    # an empty answer.
    only_think = "<think>just thinking, no final answer</think>"
    result = _strip_reasoning_trace(only_think)
    assert result == only_think
    print("PASS: test_does_not_return_empty_if_entire_text_was_a_think_block")


if __name__ == "__main__":
    test_strips_real_observed_think_block()
    test_leaves_normal_text_unchanged()
    test_handles_none()
    test_does_not_return_empty_if_entire_text_was_a_think_block()
    print("\nALL REASONING-TRACE REGRESSION TESTS PASSED")