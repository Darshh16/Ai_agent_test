"""
Session memory + query rewriting for multi-turn conversations.

Design:
- Each session keeps its own turn history, keyed by session_id, so one
  customer's conversation never bleeds into another's.
- Before retrieval, a short follow-up message ("what about Canada?") is
  rewritten into a standalone question using the recent conversation, so
  the retriever isn't searching for the literal 4-word fragment. This is a
  separate, small, cheap LLM call -- not the same call that answers the
  question -- so it can be tested and reasoned about independently of the
  main agent response.
- History is capped at MAX_TURNS_KEPT turns so a long-running session
  doesn't grow the prompt unboundedly or drag in stale context from much
  earlier in the conversation.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from llm_client import LLMClient, LLMResponse

MAX_TURNS_KEPT = 6  # user+agent turns, i.e. 3 exchanges

REWRITE_SYSTEM_PROMPT = """You rewrite a customer's follow-up message into a fully \
standalone question, using the recent conversation for context. Output ONLY the \
rewritten question, nothing else -- no preamble, no quotes, no explanation.

Rules:
- If the message is already a standalone question unrelated to what came before, \
return it unchanged.
- If it's a short follow-up ("what about Canada?", "when will it arrive?", "does that \
apply to final-sale items too?"), rewrite it into a full question that stands on its \
own, using the topic/entity from the prior turn.
- Do not answer the question. Only rewrite it.
"""


@dataclass
class Session:
    session_id: str
    turns: list[dict] = field(default_factory=list)  # [{"role": "user"/"agent", "text": ...}]

    def add_turn(self, role: str, text: str) -> None:
        self.turns.append({"role": role, "text": text})
        if len(self.turns) > MAX_TURNS_KEPT:
            self.turns = self.turns[-MAX_TURNS_KEPT:]

    def recent_context_text(self) -> str:
        return "\n".join(f"{t['role']}: {t['text']}" for t in self.turns)

    def to_gemini_contents(self) -> list[dict]:
        """The conversation history in the same {"role", "parts"} shape
        agent.py builds new turns in, so it can be passed straight through
        as the `history` argument to Agent.handle_message()."""
        contents = []
        for t in self.turns:
            role = "user" if t["role"] == "user" else "model"
            contents.append({"role": role, "parts": [{"text": t["text"]}]})
        return contents


class SessionStore:
    """In-memory session storage, keyed by session_id. Fine for this
    assignment's scope (no persistence needed); if this needed to survive a
    process restart, this is the one place that would change."""

    def __init__(self):
        self._sessions: dict[str, Session] = {}

    def get_or_create(self, session_id: str) -> Session:
        if session_id not in self._sessions:
            self._sessions[session_id] = Session(session_id=session_id)
        return self._sessions[session_id]

    def reset(self, session_id: str) -> None:
        self._sessions.pop(session_id, None)


def rewrite_query(llm: LLMClient, session: Session, latest_message: str) -> str:
    """Returns a standalone version of latest_message using session history.
    Falls back to the original message unchanged if there's no history yet
    (first turn) or if the rewrite call fails for any reason -- a broken
    rewrite step should never block the user from getting an answer."""
    if not session.turns:
        return latest_message

    prompt = (
        f"Recent conversation:\n{session.recent_context_text()}\n\n"
        f"Follow-up message: {latest_message}\n\n"
        f"Rewritten standalone question:"
    )
    try:
        response: LLMResponse = llm.generate(REWRITE_SYSTEM_PROMPT, [
            {"role": "user", "parts": [{"text": prompt}]}
        ])
        rewritten = (response.text or "").strip()
        return rewritten if rewritten else latest_message
    except Exception:
        return latest_message