"""
Minimal Streamlit UI for the Aster & Row support agent.

Lives at the repo root (not inside src/) per the project layout -- so it
adds src/ to the import path itself rather than assuming a particular
working directory when `streamlit run app.py` is invoked.

This file is intentionally thin: all real logic (retrieval, tool safety,
orchestration, memory) lives in already-tested modules under src/. This
file's only jobs are wiring them together and rendering the result --
answer, sources, and a handoff banner, per the README's requirement that
the final response make those three things easy to see.
"""

import os
import sys
import uuid
from pathlib import Path

import streamlit as st
from dotenv import load_dotenv

BASE_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(BASE_DIR / "src"))
load_dotenv(BASE_DIR / ".env")

from retrieval import Retriever          # noqa: E402
from tools.order_lookup import OrderLookupTool  # noqa: E402
from llm_client import GeminiClient, GroqClient  # noqa: E402
from agent import Agent                   # noqa: E402
from memory import SessionStore, rewrite_query  # noqa: E402
from logging_utils import log_turn                # noqa: E402


PROVIDER = os.environ.get("LLM_PROVIDER", "groq").lower()

st.set_page_config(page_title="Aster & Row Support", page_icon="🎒")


@st.cache_resource
def load_backend():
    retriever = Retriever.from_kb_dir(str(BASE_DIR / "knowledge-base"))
    order_tool = OrderLookupTool(str(BASE_DIR / "data" / "orders.json"))
    llm = GroqClient() if PROVIDER == "groq" else GeminiClient()
    agent = Agent(retriever, order_tool, llm)
    store = SessionStore()
    return agent, llm, store


agent, llm, store = load_backend()

if "session_id" not in st.session_state:
    st.session_state.session_id = str(uuid.uuid4())
if "display_history" not in st.session_state:
    st.session_state.display_history = []

session = store.get_or_create(st.session_state.session_id)

title_col, button_col = st.columns([5, 1])
with title_col:
    st.title("🎒 Aster & Row Support")
    st.caption(f"Answers are grounded in company documents and real order data — powered by {PROVIDER.title()}.")
with button_col:
    st.write("")
    if st.button("New session"):
        store.reset(st.session_state.session_id)
        st.session_state.session_id = str(uuid.uuid4())
        st.session_state.display_history = []
        st.rerun()

for turn in st.session_state.display_history:
    with st.chat_message(turn["role"]):
        st.write(turn["content"])
        if turn.get("handoff"):
            st.warning("🚩 Recommending human support for this question.")
        if turn.get("sources"):
            st.caption("Sources: " + ", ".join(turn["sources"]))
        if turn.get("trace") is not None:
            with st.expander("🔍 Trace"):
                st.json(turn["trace"])

user_input = st.chat_input("Ask about returns, shipping, warranty, or an order status...")

if user_input:
    st.session_state.display_history.append({"role": "user", "content": user_input})
    with st.chat_message("user"):
        st.write(user_input)

    with st.chat_message("assistant"):
        with st.spinner("Looking that up..."):
            search_query = rewrite_query(llm, session, user_input)
            result = agent.handle_message(
                user_input,
                history=session.to_gemini_contents(),
                retrieval_query=search_query,
            )
        st.write(result.answer)
        if result.handoff:
            st.warning("🚩 Recommending human support for this question.")
        if result.sources:
            st.caption("Sources: " + ", ".join(result.sources))
        with st.expander("🔍 Trace"):
            st.json(result.trace)

    session.add_turn("user", user_input)
    session.add_turn("agent", result.answer)
    log_turn(st.session_state.session_id, result, log_path=BASE_DIR / "logs" / "trace.jsonl")

    st.session_state.display_history.append({
        "role": "assistant",
        "content": result.answer,
        "handoff": result.handoff,
        "sources": result.sources,
        "trace": result.trace,
    })