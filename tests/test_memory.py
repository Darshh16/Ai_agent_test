"""
Verifies memory.py's session handling and query rewriting using a scripted
fake model -- proves the plumbing (session isolation, history capping,
rewrite fallback behavior) is correct independent of how well a real model
actually performs the rewrite.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from llm_client import LLMClient, LLMResponse
from memory import SessionStore, rewrite_query, MAX_TURNS_KEPT


class ScriptedLLMClient(LLMClient):
    def __init__(self, script: list[LLMResponse]):
        self.script = list(script)
        self.calls = []

    def generate(self, system_prompt, contents, tools=None):
        self.calls.append(contents)
        if not self.script:
            raise AssertionError("ScriptedLLMClient ran out of scripted responses")
        return self.script.pop(0)


def test_sessions_are_isolated():
    store = SessionStore()
    a = store.get_or_create("session-a")
    b = store.get_or_create("session-b")
    a.add_turn("user", "Do you ship internationally?")

    assert len(a.turns) == 1
    assert len(b.turns) == 0, "session-b must not see session-a's turns"
    print("PASS: test_sessions_are_isolated")


def test_history_is_capped():
    store = SessionStore()
    s = store.get_or_create("session-cap")
    for i in range(10):
        s.add_turn("user", f"message {i}")
    assert len(s.turns) == MAX_TURNS_KEPT
    assert s.turns[-1]["text"] == "message 9", "must keep the most recent turns, not the oldest"
    print("PASS: test_history_is_capped")


def test_first_turn_skips_rewrite_entirely():
    # No history yet -- rewrite_query should return the message unchanged
    # WITHOUT even calling the model, since there's nothing to rewrite against.
    store = SessionStore()
    s = store.get_or_create("session-first")
    llm = ScriptedLLMClient([])  # deliberately empty -- must not be called

    result = rewrite_query(llm, s, "Do you ship internationally?")

    assert result == "Do you ship internationally?"
    assert len(llm.calls) == 0, "should not call the model on the first turn of a session"
    print("PASS: test_first_turn_skips_rewrite_entirely")


def test_follow_up_gets_rewritten():
    store = SessionStore()
    s = store.get_or_create("session-followup")
    s.add_turn("user", "Do you ship internationally?")
    s.add_turn("agent", "Yes, we ship to several countries including Canada and the UK.")

    llm = ScriptedLLMClient([LLMResponse(text="Does Aster & Row ship to Canada?")])
    result = rewrite_query(llm, s, "what about Canada?")

    assert result == "Does Aster & Row ship to Canada?"
    assert len(llm.calls) == 1
    print("PASS: test_follow_up_gets_rewritten")


def test_rewrite_failure_falls_back_to_original():
    # If the rewrite call raises for any reason, the user must still get an
    # answer -- a broken rewrite step must never block the conversation.
    store = SessionStore()
    s = store.get_or_create("session-fail")
    s.add_turn("user", "Do you ship internationally?")

    class BrokenLLMClient(LLMClient):
        def generate(self, *a, **kw):
            raise RuntimeError("simulated API failure")

    result = rewrite_query(BrokenLLMClient(), s, "what about Canada?")

    assert result == "what about Canada?"
    print("PASS: test_rewrite_failure_falls_back_to_original")


if __name__ == "__main__":
    test_sessions_are_isolated()
    test_history_is_capped()
    test_first_turn_skips_rewrite_entirely()
    test_follow_up_gets_rewritten()
    test_rewrite_failure_falls_back_to_original()
    print("\nALL MEMORY/REWRITE PLUMBING TESTS PASSED")