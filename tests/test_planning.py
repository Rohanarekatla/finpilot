"""Forward-looking features: declared bills, goal-adaptive advice, calibrated spread, and their API."""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
from fastapi.testclient import TestClient

from app.agents import advisor, analyzer, categorizer, forecaster
from app.llm import MeshLLM

LLM = MeshLLM(api_key="", offline=True)


@pytest.fixture
def prepared(statement):
    df, bal = statement("professional", 0)
    c = categorizer.categorize(df, mode="heuristic", llm=LLM)
    return c, analyzer.analyze(c, narrate=False, llm=LLM), bal


def test_declared_bill_is_in_the_forecast_and_gets_a_set_aside_action(prepared):
    c, a, bal = prepared
    due = (a.as_of + pd.Timedelta(days=40)).date().isoformat()
    bill = {"name": "Car insurance", "amount": 24000, "next_date": due, "every_months": 12}
    plain = forecaster.forecast(c, a, bal)
    with_bill = forecaster.forecast(c, a, bal, bills=[bill])
    assert round(float(np.median(with_bill.paths[:, -1]) - np.median(plain.paths[:, -1]))) == -24000
    status = with_bill.bills[0]
    assert status["in_forecast"] and status["due_date"] == due and status["months_until"] == 2
    assert status["set_aside_monthly"] == 12000
    findings = forecaster.forecast_findings(with_bill, a)
    bill_finding = next(f for f in findings if f.kind == "upcoming_bill")
    assert bill_finding.severity == "warn"
    cand = next(x for x in advisor.build_candidates(a, findings) if x["finding_ids"] == [bill_finding.id])
    assert "₹12,000" in cand["action"] and "Car insurance" in cand["action"]


def test_bill_dates_roll_forward_and_repeat():
    start, end = pd.Timestamp("2026-10-01"), pd.Timestamp("2027-12-31")
    dates = forecaster.bill_dates({"next_date": "2025-11-15", "every_months": 6}, start, end)
    assert [d.date().isoformat() for d in dates] == ["2026-11-15", "2027-05-15", "2027-11-15"]


def _goal_kinds(a, c, bal, goal):
    fc = forecaster.forecast(c, a, bal, goals=[goal])
    return fc.goals[0], {f.kind for f in forecaster.forecast_findings(fc, a)}


def test_advice_adapts_to_goal_progress(prepared):
    c, a, bal = prepared
    when = (a.as_of + pd.DateOffset(months=6)).date().isoformat()
    easy, easy_kinds = _goal_kinds(a, c, bal, {"name": "Laptop", "target_amount": 10000, "target_date": when})
    hard, hard_kinds = _goal_kinds(a, c, bal, {"name": "Laptop", "target_amount": 5_000_000, "target_date": when})
    assert easy["status"] == "ahead" and "goal_ahead" in easy_kinds and "goal_at_risk" not in easy_kinds
    assert hard["status"] == "behind" and "goal_at_risk" in hard_kinds
    assert hard["required_extra_monthly"] > 0 and easy["surplus"] > 0


def test_goal_trend_since_last_run_is_reported(prepared):
    c, a, bal = prepared
    when = (a.as_of + pd.DateOffset(months=6)).date().isoformat()
    goal = {"name": "Trip", "target_amount": 5_000_000, "target_date": when, "previous_probability_pct": 55.0}
    status, _ = _goal_kinds(a, c, bal, goal)
    fc = forecaster.forecast(c, a, bal, goals=[goal])
    finding = next(f for f in forecaster.forecast_findings(fc, a) if f.kind == "goal_at_risk")
    assert status["change_pct_points"] <= -10
    assert "at the last check" in finding.detail and finding.severity == "alert"


def test_spread_inflation_widens_the_band_but_keeps_the_median(prepared):
    c, a, bal = prepared
    raw = forecaster.forecast(c, a, bal, spread=1.0, n_sims=2000)
    wide = forecaster.forecast(c, a, bal, spread=2.0, n_sims=2000)
    width = lambda fc: np.subtract(*np.percentile(fc.paths[:, -1], [90, 10]))  # noqa: E731
    assert width(wide) > 1.5 * width(raw)
    assert abs(np.median(wide.paths[:, -1]) - np.median(raw.paths[:, -1])) < 0.1 * width(raw)


def test_planning_api_round_trip():
    from app.main import app
    with TestClient(app) as client:
        assert client.get("/api/health").json()["scheduler"] is False
        assert client.get("/api/health?deep=true").json()["mesh_check"]["mode"] == "offline"
        assert "skip_forecast" in client.get("/api/graph").text
        client.post("/api/users/u2/sample/professional")
        bill = client.post("/api/users/u2/bills", json={"name": "Health insurance", "amount": 18000,
                                                        "next_date": "2026-11-20"}).json()
        goal = client.post("/api/users/u2/goals", json={"name": "Emergency fund", "target_amount": 300000,
                                                        "target_date": "2027-06-30"}).json()
        first = client.post("/api/users/u2/analyze").json()
        assert any(f["kind"] == "upcoming_bill" for f in first["findings"])
        assert first["forecast"]["bills"][0]["name"] == "Health insurance"
        client.post("/api/users/u2/analyze")
        overview = client.get("/api/users/u2/goals").json()[0]
        assert overview["status"] in {"behind", "on_track", "ahead"} and len(overview["history"]) == 2
        sim = client.post("/api/users/u2/simulate", json={"purchases": [{"date": "2026-10-25", "amount": 15000}]})
        assert sim.status_code == 200 and sim.json()["delta_end_balance_p50"] == -15000
        bad = client.post("/api/users/u2/simulate", json={"category_cuts": {"not_a_category": 0.5}})
        assert bad.status_code == 400
        assert client.delete(f"/api/users/u2/bills/{bill['id']}").json()["deleted"] == bill["id"]
        assert client.delete(f"/api/users/u2/goals/{goal['id']}").status_code == 200
        assert client.get("/api/users/u2/bills").json() == []


def test_too_little_history_gives_clear_errors_not_crashes():
    from app.main import app
    csv = "date,description,amount\n2026-09-01,SALARY,50000\n2026-09-03,UPI-SWIGGY-x@ybl-1-P,-450\n"
    with TestClient(app) as client:
        client.post("/api/users/u3/upload", files={"file": ("s.csv", csv, "text/csv")})
        res = client.post("/api/users/u3/analyze").json()
        assert res["forecast"] is None
        assert any(f["kind"] == "insufficient_history" for f in res["findings"])
        assert client.post("/api/users/u3/simulate", json={"cancel": []}).status_code == 400
        assert client.post("/api/users/u3/ask", json={"question": "Can I afford a 20k phone?"}).status_code == 200


def test_small_purchase_is_not_blamed_for_a_risk_that_already_exists(statement):
    # Over a horizon to March 2027 this user's balance is already at risk before any purchase.
    df, bal = statement("student", 4)
    c = categorizer.categorize(df, mode="heuristic", llm=LLM)
    a = analyzer.analyze(c, narrate=False, llm=LLM)
    goals = [{"name": "Emergency fund", "target_amount": 200000, "target_date": "2027-03-31"}]
    small = forecaster.simulate(c, a, bal, forecaster.Scenario(purchases=[{"date": "2026-10-20", "amount": 500}]),
                                goals=goals, seed=4)
    assert small["baseline"]["prob_negative_pct"] > 10
    assert small["verdict"] == "tight" and "already at risk" in small["reasons"][0]
    big = forecaster.simulate(c, a, bal, forecaster.Scenario(purchases=[{"date": "2026-10-20", "amount": bal * 3}]))
    assert big["verdict"] == "not_affordable"
