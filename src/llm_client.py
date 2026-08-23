"""
Thin wrapper around the Gemini API. Deliberately isolated to this one file
so a provider swap -- or a Gemini SDK version difference -- only ever
requires editing here. Nothing else in the codebase imports the Gemini SDK
directly.

Uses the `google-genai` package (`pip install google-genai`). If your
installed SDK version has a slightly different call shape than what's
here, this is the file to adjust -- check https://ai.google.dev for the
current quickstart if `GeminiClient.generate()` errors on import or call.
"""

from __future__ import annotations

import os
from dataclasses import dataclass


@dataclass
class LLMResponse:
    text: str | None = None
    function_call: dict | None = None  # {"name": str, "args": dict}


class LLMClient:
    """Interface both the real Gemini client and the test fake implement,
    so agent.py never depends on which one it's talking to."""

    def generate(self, system_prompt: str, contents: list[dict], tools: list[dict] | None = None) -> LLMResponse:
        raise NotImplementedError


