"""Mesh API client: the single gateway for every LLM call in FinPilot.

Mesh is OpenAI-compatible, so we point the OpenAI SDK at its base URL. Each
agent calls `chat_json` with its own system prompt; when no key is configured
(or FINPILOT_OFFLINE=1) the call returns None and the agent uses its
deterministic fallback, so the pipeline always runs end to end.

Every request has a timeout and a bounded retry budget, so a slow model can
never hang the dashboard or the weekly digest: the agent simply falls back.
Account-level errors (bad key, zero balance, no access) cannot fix themselves
mid-run, so the first one switches the client off for the rest of the process
instead of failing every later call the same way.

    python -m app.llm          # check the key and the configured model ids (no credits spent)
    python -m app.llm --ping   # ...and make one tiny chat call to confirm the balance covers it
"""
from __future__ import annotations

import json
import logging
import re
import threading
import time
from dataclasses import dataclass, field
from typing import Any

import httpx
from openai import APIStatusError, OpenAI

from app import config

log = logging.getLogger("finpilot.llm")

# 401 bad key, 402 balance exhausted, 403 no access: retrying the same account will not help.
ACCOUNT_ERRORS = {401: "invalid Mesh API key", 402: "Mesh balance exhausted (top up to use paid models)",
                  403: "this key has no access to the model"}


@dataclass
class LLMStats:
    calls: int = 0
    failures: int = 0
    latency_s: float = 0.0
    by_agent: dict[str, int] = field(default_factory=dict)


class MeshLLM:
    def __init__(self, api_key: str | None = None, offline: bool | None = None):
        key = config.MESH_API_KEY if api_key is None else api_key
        self._key = key
        self.offline = config.OFFLINE if offline is None else offline
        self.blocked_reason: str | None = None  # set by the first account-level error
        self.client = (
            OpenAI(base_url=config.MESH_BASE_URL, api_key=key, timeout=config.LLM_TIMEOUT_S,
                   max_retries=config.LLM_MAX_RETRIES)
            if key else None
        )
        self.stats = LLMStats()
        self._lock = threading.Lock()  # agents may call from worker threads (categorizer batches)

    @property
    def available(self) -> bool:
        return self.client is not None and not self.offline and self.blocked_reason is None

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
        with self._lock:
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
        except APIStatusError as exc:
            with self._lock:
                self.stats.failures += 1
                reason = ACCOUNT_ERRORS.get(exc.status_code)
                first = reason is not None and self.blocked_reason is None
                if first:
                    self.blocked_reason = f"{reason} (HTTP {exc.status_code})"
            if first:
                log.error("Mesh disabled for this process: %s. Agents use their deterministic fallbacks.",
                          self.blocked_reason)
            elif reason is None:
                log.warning("Mesh call by %s failed: %s", agent, exc)
            return None
        except Exception as exc:
            with self._lock:
                self.stats.failures += 1
            log.warning("Mesh call by %s failed: %s", agent, exc)
            return None
        finally:
            with self._lock:
                self.stats.latency_s += time.perf_counter() - started

    def check(self) -> dict:
        """Verify the key and the configured model ids with a metadata call (spends no credits).

        Mesh returns GET /v1/models as a bare JSON array, which the OpenAI SDK's page type cannot
        parse, so this uses plain HTTP. It cannot see the account balance: a valid key with a zero
        balance passes here and then fails chat calls with HTTP 402.
        """
        wanted = sorted({config.MODEL_FAST, config.MODEL_SMART})
        if self.client is None or self.offline:
            return {"ok": False, "mode": "offline", "reason": "no MESH_API_KEY set or FINPILOT_OFFLINE=1"}
        try:
            resp = httpx.get(f"{config.MESH_BASE_URL.rstrip('/')}/models", timeout=config.LLM_TIMEOUT_S,
                             headers={"Authorization": f"Bearer {self._key}"})
        except httpx.HTTPError as exc:
            return {"ok": False, "mode": "mesh", "reason": f"could not reach Mesh: {exc}"}
        if resp.status_code != 200:
            return {"ok": False, "mode": "mesh",
                    "reason": ACCOUNT_ERRORS.get(resp.status_code, f"HTTP {resp.status_code}: {resp.text[:200]}")}
        body = resp.json()
        models = body if isinstance(body, list) else body.get("data", [])
        ids = {m["id"] if isinstance(m, dict) else str(m) for m in models}
        missing = [m for m in wanted if m not in ids]
        return {"ok": not missing, "mode": "mesh", "key_valid": True, "models_available": len(ids),
                "configured": wanted, "missing": missing, "blocked": self.blocked_reason}

    def ping(self) -> dict:
        """One tiny chat call (a fraction of a cent) to confirm the account can actually run the models."""
        reply = self.chat_json("ping", 'Reply with {"ok": true}.', "ping")
        if reply is not None:
            return {"ok": True}
        return {"ok": False, "reason": self.blocked_reason or "chat call failed; see the log above"}


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


if __name__ == "__main__":
    import sys

    llm = MeshLLM()
    result = llm.check()
    if result["ok"] and "--ping" in sys.argv:
        result["chat"] = llm.ping()
    print(json.dumps(result, indent=2))
