"""Offline tests: deterministic, no Mesh key needed."""
from __future__ import annotations

import os

os.environ["FINPILOT_OFFLINE"] = "1"
os.environ["FINPILOT_SCHEDULER"] = "0"

import pandas as pd  # noqa: E402
import pytest  # noqa: E402

from app.agents import analyzer, categorizer, critic, forecaster  # noqa: E402
from app.agents.coordinator import parse_amount, parse_when  # noqa: E402
from app.audit import AuditTrail  # noqa: E402
from app.graph import run_ask, run_pipeline  # noqa: E402
from app.grounding import inr, ungrounded  # noqa: E402
from app.ingest import parse_csv  # noqa: E402
from app.llm import MeshLLM  # noqa: E402
from app.synthetic import generate  # noqa: E402
from app.taxonomy import merchant_key  # noqa: E402

LLM = MeshLLM(api_key="", offline=True)


@pytest.fixture(scope="module")
def professional():
    s = generate("professional", seed=0)
    df = s.transactions.drop(columns=["true_category"]).copy()
    df["merchant"] = df["description"].map(merchant_key)
    c = categorizer.categorize(df, mode="heuristic", llm=LLM)
    return c, analyzer.analyze(c, narrate=False, llm=LLM), float(df["balance"].iloc[-1])


def test_merchant_key_strips_refs_and_months():
    assert merchant_key("UPI-SWIGGY-swiggy8@ybl-412345678901-PAYMENT") == "SWIGGY"
    assert merchant_key("POS 4821 CROMA BENGALURU") == "CROMA"
    a = merchant_key("NEFT CR-ICIC0000101-GLOBEX INDIA PVT LTD-SALARY SEP")
    assert a == merchant_key("NEFT CR-ICIC0000101-GLOBEX INDIA PVT LTD-SALARY OCT")


def test_ingest_handles_debit_credit_columns_and_drops_labels():
    csv = "Txn Date,Narration,Debit,Credit,Balance,true_category\n2026-09-01,SALARY,,50000,50000,income\n2026-09-02,UPI-SWIGGY-x@ybl-1-P,300,,49700,food_delivery\n"
    df = parse_csv(csv)
    assert list(df["amount"]) == [50000, -300]
    assert "true_category" not in df.columns


def test_categories_depend_on_the_transaction(professional):
    c, _, _ = professional
    assert c.loc[c["description"].str.contains("SWIGGY"), "category"].eq("food_delivery").all()
    assert c.loc[c["description"].str.contains("SALARY"), "category"].eq("income").all()
    assert c["category"].nunique() > 10


def test_planted_patterns_are_found(professional):
    _, a, _ = professional
    ids = {f.id for f in a.findings}
    assert "price_hike:NETFLIX" in ids
    assert "large_transaction:CROMA" in ids
    assert "overlapping_subscriptions:video_streaming" in ids
    salary = [r for r in a.recurring if r.amount > 0]
    assert salary and salary[0].period_months == 1


def test_purchase_shifts_forecast_exactly(professional):
    c, a, bal = professional
    r = forecaster.simulate(c, a, bal, forecaster.Scenario(purchases=[{"date": "2026-10-20", "amount": 25000}]))
    assert r["delta_end_balance_p50"] == -25000


def test_cancel_subscription_saves_its_cost(professional):
    c, a, bal = professional
    r = forecaster.simulate(c, a, bal, forecaster.Scenario(cancel=["NETFLIX"]))
    assert r["delta_end_balance_p50"] > 0
    assert r["delta_end_balance_p50"] % 649 == 0


def test_critic_rejects_ungrounded_and_generic_advice(professional):
    _, a, _ = professional
    findings = {f.id: f for f in a.findings}
    bad = [
        {"title": "Save more", "action": "Try to save more money every month.", "why": "Saving is good.",
         "finding_ids": [], "monthly_impact": 0, "one_off_impact": 0},
        {"title": "Cancel Hotstar", "action": "Cancel DISNEY HOTSTAR to save ₹4,321/month.", "why": "Overlap.",
         "finding_ids": ["overlapping_subscriptions:video_streaming"], "monthly_impact": 4321, "one_off_impact": 0},
        {"title": "Fix it", "action": "Reduce spending.", "why": "Too high.",
         "finding_ids": ["price_hike:NETFLIX"], "monthly_impact": 0, "one_off_impact": 0},
    ]
    verdicts = critic.critique(bad, findings, [], a.stats)
    assert not any(v["passed"] for v in verdicts)
    assert any("do not appear" in i for i in verdicts[1]["issues"])
    assert any("exceeds" in i for i in verdicts[1]["issues"])
    assert any("generic" in i for i in verdicts[2]["issues"])


def test_pipeline_delivers_only_critic_approved_recommendations():
    s = generate("student", seed=1)
    df = s.transactions.drop(columns=["true_category"]).copy()
    df["merchant"] = df["description"].map(merchant_key)
    audit = AuditTrail("t")
    st = run_pipeline(df, float(df["balance"].iloc[-1]), audit, LLM, mode="heuristic")
    assert st["final"], "expected at least one recommendation"
    findings = {f.id: f for f in st["analysis"].findings + st["forecast_findings"]}
    assert all(v["passed"] for v in critic.critique(st["final"], findings, st["candidates"], st["analysis"].stats))
    agents = {r.agent for r in audit.records}
    assert {"coordinator", "categorizer", "analyzer", "forecaster", "advisor", "critic"} <= agents


def test_ask_routes_affordability_to_simulation(professional):
    c, a, bal = professional
    st = run_ask("Can I afford a 60k trip next month?", c, a, bal, AuditTrail("t"), LLM)
    assert st["intent"] == "afford"
    assert st["simulation"]["scenario"]["purchases"][0]["amount"] == 60000
    assert st["simulation"]["verdict"] in {"affordable", "tight", "not_affordable", "affordable_but_delays_goal"}


def test_parsers():
    assert parse_amount("a 1.5 lakh bike") == 150000
    assert parse_amount("₹60,000 in December 2026") == 60000
    assert parse_when("next month", pd.Timestamp("2026-09-30")) == pd.Timestamp("2026-10-15")
    assert inr(1234567) == "₹12,34,567"
    assert ungrounded("costs ₹1,499 a month, 12% more", {1499.0, 12.0}) == []


def test_api_end_to_end(tmp_path, monkeypatch):
    from app import config, db
    monkeypatch.setattr(config, "DB_PATH", tmp_path / "t.db")
    from fastapi.testclient import TestClient

    from app.main import app
    with TestClient(app) as client:
        assert client.post("/api/users/u1/sample/family").json()["imported"] > 300
        res = client.post("/api/users/u1/analyze").json()
        assert res["recommendations"]
        rec = res["recommendations"][0]
        why = client.get(f"/api/recommendations/{rec['rec_id']}/why").json()
        assert why["evidence"]
        assert client.get(f"/api/runs/{res['run_id']}/audit").json()["decisions"]
        ans = client.post("/api/users/u1/ask", json={"question": "What if I cancel Netflix?"}).json()
        assert ans["intent"] == "what_if"
        assert client.post("/api/users/u1/digest").json()["digest"]
    assert db.list_digests("u1")
