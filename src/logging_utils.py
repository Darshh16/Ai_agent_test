"""
Structured JSON-lines logging for each conversation turn.

Design:
- One JSON object per line in logs/trace.jsonl (JSONL: easy to tail, grep,
  or stream-parse without loading the whole file as one document).
- Built entirely from data agent.py's AgentResponse already collected --
  this module's only job is adding a timestamp/session_id and writing it
  out, not gathering new data itself.
- Nothing new to leak here: the tool payload was already allowlist-filtered
  in order_lookup.py before it ever reached AgentResponse.trace, so what
  gets logged is exactly what a reviewer is already allowed to see. The
  same FORBIDDEN_LEAK_MARKERS check used in order_lookup.py's own test is
  re-applied here as a second, independent safety net -- see
  tests/test_logging.py.
"""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path

DEFAULT_LOG_PATH = Path("logs") / "trace.jsonl"


def build_log_entry(session_id: str, agent_response) -> dict:
    trace = agent_response.trace
    return {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "session_id": session_id,
        "user_message": trace.get("user_message"),
        "retrieval_query": trace.get("retrieval_query"),
        "retrieved": trace.get("retrieved"),
        "conflict_candidate": trace.get("conflict_candidate"),
        "tool_calls": trace.get("tool_calls"),
        "tool_called": agent_response.tool_called,
        "tool_arguments": agent_response.tool_arguments,
        "answer": agent_response.answer,
        "sources": agent_response.sources,
        "handoff": agent_response.handoff,
        "handoff_source": trace.get("handoff_source"),
        "decision": trace.get("decision"),
    }


def log_turn(session_id: str, agent_response, log_path: Path | str = DEFAULT_LOG_PATH) -> dict:
    """Writes one JSON line and also returns the entry, so callers (and
    tests) can inspect exactly what was logged without re-reading the
    file."""
    log_path = Path(log_path)
    log_path.parent.mkdir(parents=True, exist_ok=True)

    entry = build_log_entry(session_id, agent_response)
    with open(log_path, "a", encoding="utf-8") as f:
        f.write(json.dumps(entry, default=str) + "\n")
    return entry