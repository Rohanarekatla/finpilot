"""Coordinator routing and the LLM paths, exercised with a scripted fake LLM (no network)."""
from __future__ import annotations

import pandas as pd
import pytest

from app.agents import categorizer
from app.audit import AuditTrail
from app.graph import run_pipeline
from app.llm import MeshLLM


class FakeLLM(MeshLLM):
    """Behaves like an available Mesh client; each agent's reply comes from a script."""

    def __init__(self, script: dict):
        super().__init__(api_key="", offline=True)
        self.script = script
        self.calls: list[tuple[str, object]] = []

    @property
    def available(self) -> bool:
        return True

    def chat_json(self, agent, system, user, model=None, temperature=0.0):
        self.calls.append((agent, user))
        reply = self.script.get(agent)
        return reply(user) if callable(reply) else reply


def _advisor_script(bad_then_fixed: bool):
    """First draft: one grounded rec + one with an invented figure. Revision: fix it (or repeat the mistake)."""
    def reply(payload):
        cands = payload["candidates"]
        good, target = cands[0], cands[1]
        bad = {**target, "action": target["action"] + " That is ₹98,765 a year."}
        revising = "critic_feedback" in payload
        second = target if (revising and bad_then_fixed) else bad
        recs = [good, second]
        return {"recommendations": [{k: r[k] for k in ("title", "action", "why", "finding_ids", "monthly_impact",
                                                        "one_off_impact")} for r in recs]}
    return reply


@pytest.fixture
def run(statement):
    def _run(llm):
        df, bal = statement("professional", 0)
        audit = AuditTrail("t")
        return run_pipeline(df, bal, audit, llm, mode="heuristic"), audit
    return _run


def test_critic_feedback_makes_the_advisor_revise(run):
    st, audit = run(FakeLLM({"advisor": _advisor_script(bad_then_fixed=True)}))
    assert st["attempts"] == 2, "one rejection should trigger exactly one revision"
    assert st["first_pass_rate"] == 0.5
    revision = [c for c in audit.records if c.agent == "coordinator" and c.decision == "advise (revision)"]
    assert revision, "the revision route is recorded in the audit trail"
    llm_written = [r for r in st["final"] if r["source"] == "llm"]
    assert len(llm_written) == 2 and all("98,765" not in r["action"] for r in st["final"])


def test_recommendation_that_keeps_failing_is_dropped_and_backfilled(run):
    st, audit = run(FakeLLM({"advisor": _advisor_script(bad_then_fixed=False)}))
    assert st["attempts"] == 3, "first draft + MAX_ADVISOR_REVISIONS (2) revisions"
    assert any(d.step == "drop" for d in audit.records)
    assert any(d.step == "backfill" for d in audit.records)
    assert all("98,765" not in r["action"] for r in st["final"])
    assert len(st["final"]) >= 3


def test_short_history_skips_the_forecast_but_still_advises(statement):
    df, bal = statement("professional", 0)
    recent = df[df["date"] >= df["date"].max() - pd.Timedelta(days=20)].copy()
    audit = AuditTrail("t")
    st = run_pipeline(recent, bal, audit, MeshLLM(api_key="", offline=True), mode="heuristic")
    assert st["forecast"] is None
    assert any(f.id == "insufficient_history:data" for f in st["forecast_findings"])
    routes = [d.decision for d in audit.records if d.agent == "coordinator" and d.step == "route"]
    assert "skip_forecast" in routes


def test_merchant_memory_avoids_asking_the_llm_twice(statement):
    df, _ = statement("professional", 0)

    def classify(payload):
        return {"items": [{"key": m["key"], "category": "shopping", "confidence": 0.9, "reason": "test"}
                          for m in payload["merchants"]] + [{"key": "INVENTED|debit", "category": "rent"}]}

    llm = FakeLLM({"categorizer": classify})
    memory: dict = {}
    first = categorizer.categorize(df, mode="auto", llm=llm, memory=memory)
    calls = len(llm.calls)
    assert calls >= 1 and memory and "INVENTED|debit" not in memory
    assert (first["category_source"] == "llm").any()
    second = categorizer.categorize(df, mode="auto", llm=llm, memory=memory)
    assert len(llm.calls) == calls, "every unknown merchant was answered from memory"
    assert second["category"].equals(first["category"])


def test_invalid_llm_categories_fall_back_to_heuristics(statement):
    df, _ = statement("student", 0)
    llm = FakeLLM({"categorizer": {"items": [{"key": "X|debit", "category": "not_a_category"}]}})
    out = categorizer.categorize(df, mode="auto", llm=llm, memory={})
    assert out["category"].notna().all()
    assert not (out["category_source"] == "llm").any()
    assert (out["category_source"] == "heuristic").any()


def test_an_account_error_switches_mesh_off_once_instead_of_failing_every_call():
    import httpx
    from openai import APIStatusError

    class Completions:
        calls = 0

        def create(self, **_):
            Completions.calls += 1
            req = httpx.Request("POST", "https://api.meshapi.ai/v1/chat/completions")
            raise APIStatusError("Insufficient balance", response=httpx.Response(402, request=req), body=None)

    from types import SimpleNamespace

    llm = MeshLLM(api_key="rsk_test", offline=False)
    llm.client = SimpleNamespace(chat=SimpleNamespace(completions=Completions()))
    assert llm.chat_json("advisor", "s", "u") is None
    assert llm.blocked_reason and "402" in llm.blocked_reason
    assert not llm.available
    assert llm.chat_json("advisor", "s", "u") is None and Completions.calls == 1, "no further calls after a 402"
