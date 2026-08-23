"""
Thin wrappers around LLM provider APIs. Deliberately isolated to this one
file so a provider swap only ever requires editing here -- nothing else in
the codebase (agent.py, tests, eval harness) imports a provider SDK
directly, or needs to change when you switch providers.

Two options are provided, both free tier, no credit card:
- GeminiClient: Google Gemini API (`pip install google-genai`)
- GroqClient: Groq API, OpenAI-compatible (`pip install groq`)

To switch which one your app uses, change ONE line wherever you construct
the agent (see app.py / run_eval.py):
    from llm_client import GeminiClient as ActiveClient   # or GroqClient
    llm = ActiveClient()
"""

from __future__ import annotations

import json
import os
from dataclasses import dataclass


@dataclass
class LLMResponse:
    text: str | None = None
    function_call: dict | None = None  # {"name": str, "args": dict}


class LLMClient:
    """Interface every provider client and the test fake implement, so
    agent.py never depends on which one it's talking to."""

    def generate(self, system_prompt: str, contents: list[dict], tools: list[dict] | None = None) -> LLMResponse:
        raise NotImplementedError


class GeminiClient(LLMClient):
    def __init__(self, model: str = "gemini-2.5-flash", api_key: str | None = None):
        from google import genai
        from google.genai import types

        self._types = types
        api_key = api_key or os.environ.get("GEMINI_API_KEY")
        if not api_key:
            raise RuntimeError("GEMINI_API_KEY is not set -- check your .env file")
        self._client = genai.Client(api_key=api_key)
        self._model = model

    def generate(self, system_prompt: str, contents: list[dict], tools: list[dict] | None = None) -> LLMResponse:
        types = self._types
        gemini_tools = None
        if tools:
            gemini_tools = [
                types.Tool(
                    function_declarations=[
                        types.FunctionDeclaration(
                            name=t["name"],
                            description=t["description"],
                            parameters=t["parameters"],
                        )
                        for t in tools
                    ]
                )
            ]

        config = types.GenerateContentConfig(
            system_instruction=system_prompt,
            tools=gemini_tools,
        )

        response = self._client.models.generate_content(
            model=self._model,
            contents=contents,
            config=config,
        )

        candidate = response.candidates[0]
        for part in candidate.content.parts:
            if getattr(part, "function_call", None):
                fc = part.function_call
                return LLMResponse(function_call={"name": fc.name, "args": dict(fc.args)})

        return LLMResponse(text=response.text)


class GroqClient(LLMClient):
    """Groq's API is OpenAI-compatible. agent.py always builds `contents` in
    the same shape (a list of {"role": ..., "parts": [...]} dicts, with tool
    calls/results following a fixed pattern -- see agent.py's handle_message
    loop), so this class's job is purely to translate that shape into
    OpenAI-style chat messages and translate the response back."""

    def __init__(self, model: str = "qwen/qwen3.6-27b", api_key: str | None = None):
        from groq import Groq

        api_key = api_key or os.environ.get("GROQ_API_KEY")
        if not api_key:
            raise RuntimeError("GROQ_API_KEY is not set -- check your .env file")
        self._client = Groq(api_key=api_key)
        self._model = model

    def _contents_to_messages(self, system_prompt: str, contents: list[dict]) -> list[dict]:
        messages = [{"role": "system", "content": system_prompt}]
        last_tool_call_id = None

        for item in contents:
            role = item.get("role")
            for part in item.get("parts", []):
                if "function_call" in part:
                    fc = part["function_call"]
                    call_id = f"call_{len(messages)}"
                    messages.append({
                        "role": "assistant",
                        "content": None,
                        "tool_calls": [{
                            "id": call_id,
                            "type": "function",
                            "function": {
                                "name": fc["name"],
                                "arguments": json.dumps(fc.get("args", {})),
                            },
                        }],
                    })
                    last_tool_call_id = call_id
                elif "text" in part:
                    if role == "user" and last_tool_call_id is not None:
                        # This text turn immediately follows a function_call
                        # turn -- per agent.py's loop, that means it's the
                        # tool's result being reported back, not a new user
                        # message.
                        messages.append({
                            "role": "tool",
                            "tool_call_id": last_tool_call_id,
                            "content": part["text"],
                        })
                        last_tool_call_id = None
                    else:
                        messages.append({
                            "role": "user" if role == "user" else "assistant",
                            "content": part["text"],
                        })
        return messages

    def _tools_to_openai_format(self, tools: list[dict]) -> list[dict]:
        return [{"type": "function", "function": t} for t in tools]

    def generate(self, system_prompt: str, contents: list[dict], tools: list[dict] | None = None) -> LLMResponse:
        messages = self._contents_to_messages(system_prompt, contents)
        groq_tools = self._tools_to_openai_format(tools) if tools else None

        response = self._client.chat.completions.create(
            model=self._model,
            messages=messages,
            tools=groq_tools,
        )

        message = response.choices[0].message
        if message.tool_calls:
            tc = message.tool_calls[0]
            return LLMResponse(function_call={
                "name": tc.function.name,
                "args": json.loads(tc.function.arguments),
            })

        return LLMResponse(text=message.content)