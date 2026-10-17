"""Mesh API client: the single gateway for every LLM call in FinPilot.

Mesh is OpenAI-compatible, so we point the OpenAI SDK at its base URL. Each
agent calls `chat_json` with its own system prompt; when no key is configured
(or FINPILOT_OFFLINE=1) the call returns None and the agent uses its
deterministic fallback, so the pipeline always runs end to end.
"""
from __future__ import annotations

import json
import logging
import re
import time
from dataclasses import dataclass, field
from typing import Any

from openai import OpenAI

from app import config

log = logging.getLogger("finpilot.llm")


@dataclass
class LLMStats:
    calls: int = 0
    failures: int = 0
    latency_s: float = 0.0
    by_agent: dict[str, int] = field(default_factory=dict)


class MeshLLM:
    def __init__(self, api_key: str | None = None, offline: bool | None = None):
        key = config.MESH_API_KEY if api_key is None else api_key
        self.offline = config.OFFLINE if offline is None else offline
        self.client = OpenAI(base_url=config.MESH_BASE_URL, api_key=key) if key else None
        self.stats = LLMStats()

    @property
    def available(self) -> bool:
        return self.client is not None and not self.offline

    def chat_json(
        self,
        agent: str,
        system: str,
        user: str | dict | list,
        model: str | None = None,
        temperature: float = 0.0,
    ) -> dict | None:
        """Ask for a JSON object. Returns None when unavailable or on failure."""
        if not self.available:
            return None
        if not isinstance(user, str):
            user = json.dumps(user, default=str, ensure_ascii=False)
        messages = [
            {"role": "system", "content": system + "\nRespond with a single JSON object only."},
            {"role": "user", "content": user},
        ]
        model = model or config.MODEL_FAST
        self.stats.calls += 1
        self.stats.by_agent[agent] = self.stats.by_agent.get(agent, 0) + 1
        started = time.perf_counter()
        try:
            try:
                resp = self.client.chat.completions.create(
                    model=model,
                    messages=messages,
                    temperature=temperature,
                    response_format={"type": "json_object"},
                )
            except Exception as exc:  # some routed models reject response_format
                if "response_format" not in str(exc):
                    raise
                resp = self.client.chat.completions.create(
                    model=model, messages=messages, temperature=temperature
                )
            return _extract_json(resp.choices[0].message.content or "")
        except Exception as exc:
            self.stats.failures += 1
            log.warning("Mesh call by %s failed: %s", agent, exc)
            return None
        finally:
            self.stats.latency_s += time.perf_counter() - started


def _extract_json(text: str) -> dict | None:
    text = text.strip()
    try:
        out = json.loads(text)
        return out if isinstance(out, dict) else {"items": out}
    except json.JSONDecodeError:
        match = re.search(r"\{.*\}", text, re.S)
        if match:
            try:
                return json.loads(match.group(0))
            except json.JSONDecodeError:
                return None
    return None


_default: MeshLLM | None = None


def get_llm() -> MeshLLM:
    global _default
    if _default is None:
        _default = MeshLLM()
    return _default


def set_llm(llm: MeshLLM) -> None:
    """Swap the process-wide client (tests, eval)."""
    global _default
    _default = llm


def as_json(obj: Any) -> str:
    return json.dumps(obj, default=str, ensure_ascii=False)
