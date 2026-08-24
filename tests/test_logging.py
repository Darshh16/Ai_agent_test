import json
import sys
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from agent import AgentResponse
from logging_utils import log_turn
from tools.order_lookup import FORBIDDEN_LEAK_MARKERS


def make_fake_response(**overrides) -> AgentResponse:
    defaults = dict(
        answer="Your order has shipped with UPS.",
        sources=[],
        handoff=False,
        tool_called="order_lookup",
        tool_arguments={"order_id": "ORD-1007"},
        trace={
            "user_message": "Where is ORD-1007?",
            "retrieval_query": "Where is ORD-1007?",
            "retrieved": [],
            "conflict_candidate": None,
            "tool_calls": [{
                "name": "order_lookup",
                "args": {"order_id": "ORD-1007"},
                "result": {"found": True, "status": "shipped", "carrier": "UPS"},
            }],
            "raw_final_response": "Your order has shipped with UPS. [no citation needed]",
            "handoff_source": "none",
            "decision": "answered",
        },
    )
    defaults.update(overrides)
    return AgentResponse(**defaults)


def test_log_entry_has_expected_fields():
    with tempfile.TemporaryDirectory() as tmp:
        log_path = Path(tmp) / "trace.jsonl"
        entry = log_turn("session-abc", make_fake_response(), log_path=log_path)

        assert entry["session_id"] == "session-abc"
        assert entry["tool_called"] == "order_lookup"
        assert entry["decision"] == "answered"
        assert "timestamp" in entry
        print("PASS: test_log_entry_has_expected_fields")


def test_log_file_is_valid_jsonl_and_appends():
    with tempfile.TemporaryDirectory() as tmp:
        log_path = Path(tmp) / "trace.jsonl"
        log_turn("session-1", make_fake_response())
        log_turn("session-1", make_fake_response(), log_path=log_path)
        log_turn("session-1", make_fake_response(answer="A different answer."), log_path=log_path)

        lines = log_path.read_text(encoding="utf-8").strip().split("\n")
        assert len(lines) == 2, "second and third calls should both append to the same file"
        parsed = [json.loads(line) for line in lines]
        assert parsed[1]["answer"] == "A different answer."
        print("PASS: test_log_file_is_valid_jsonl_and_appends")


def test_logged_entry_never_contains_forbidden_fields():
    # Independent re-check of the same leak markers order_lookup.py's own
    # test uses -- defense in depth. Even if the tool boundary were ever
    # bypassed upstream, this confirms the log itself doesn't become a
    # second place PII could leak from.
    with tempfile.TemporaryDirectory() as tmp:
        log_path = Path(tmp) / "trace.jsonl"
        entry = log_turn("session-privacy-check", make_fake_response(), log_path=log_path)
        serialized = json.dumps(entry)

        for marker in FORBIDDEN_LEAK_MARKERS:
            assert marker not in serialized, f"LEAK in log entry: {marker!r}"
        print("PASS: test_logged_entry_never_contains_forbidden_fields")


if __name__ == "__main__":
    test_log_entry_has_expected_fields()
    test_log_file_is_valid_jsonl_and_appends()
    test_logged_entry_never_contains_forbidden_fields()
    print("\nALL LOGGING TESTS PASSED")