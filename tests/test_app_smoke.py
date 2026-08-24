"""
Headless smoke test for app.py, using Streamlit's official AppTest API.
This actually runs app.py and constructs the real Retriever and
OrderLookupTool against your real knowledge-base/ and data/ files -- it
catches import-path bugs, wiring bugs, and cold-load exceptions before you
ever open a browser.

It does NOT send a real chat message (that would need a live LLM API call,
which this test suite deliberately doesn't depend on -- same reasoning as
tests/test_agent_orchestration.py). Actually sending a message and reading
the response is what you verify manually in the browser, which also
doubles as your demo recording.

Run from the repo root:
    python tests/test_app_smoke.py
"""

import os
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

# Client construction (GroqClient()/GeminiClient()) doesn't make a network
# call -- only actually sending a message does. A dummy key is enough to
# get past the "is a key configured" check and exercise the rest of
# app.py's real wiring (Retriever, OrderLookupTool, session state) without
# needing a real credential or hitting the network in this test.
os.environ.setdefault("GROQ_API_KEY", "dummy-key-for-smoke-test-only")
os.environ.setdefault("LLM_PROVIDER", "groq")

from streamlit.testing.v1 import AppTest


def test_app_loads_without_exceptions():
    at = AppTest.from_file(str(REPO_ROOT / "app.py"))
    at.run(timeout=30)

    assert not at.exception, f"app.py raised on cold load: {at.exception}"
    print("PASS: app.py loads and constructs the real backend without exceptions")


def test_title_and_new_session_button_render():
    at = AppTest.from_file(str(REPO_ROOT / "app.py"))
    at.run(timeout=30)

    titles = [t.value for t in at.title]
    assert any("Aster & Row" in t for t in titles), f"expected title not found, got: {titles}"

    button_labels = [b.label for b in at.button]
    assert "New session" in button_labels, f"expected 'New session' button, got: {button_labels}"
    print("PASS: title and New session button render correctly")


def test_chat_input_present():
    at = AppTest.from_file(str(REPO_ROOT / "app.py"))
    at.run(timeout=30)

    assert len(at.chat_input) >= 1, "expected a chat input widget on the page"
    print("PASS: chat input widget is present")


if __name__ == "__main__":
    test_app_loads_without_exceptions()
    test_title_and_new_session_button_render()
    test_chat_input_present()
    print("\nALL APP SMOKE TESTS PASSED")