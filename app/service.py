"""Application service: connects the pure LangGraph pipelines to the database.

This is the only layer that loads and persists. It also keeps two small,
explicit caches:

* merchant memory (SQLite-backed): long-tail merchants the LLM has already
  categorised are never sent to Mesh again, across users and restarts;
* the latest analysis per user (in memory): Ask and Simulate reuse it until
  the user's transactions or balance change, instead of re-running the analyzer
  on every question.
"""
from __future__ import annotations

import threading

import pandas as pd

from app import db, ingest, synthetic
from app.agents import analyzer, forecaster
from app.audit import AuditTrail
from app.graph import run_ask, run_pipeline
from app.llm import get_llm

_analysis_cache: dict[str, tuple[tuple, analyzer.Analysis]] = {}
_cache_lock = threading.Lock()


# --- ingestion ------------------------------------------------------------------------------------

def import_csv(user_id: str, data: bytes | str, opening_balance: float | None = None) -> int:
    df = ingest.parse_csv(data)
    db.ensure_user(user_id, opening_balance)
    return db.insert_transactions(user_id, df)


def load_sample(user_id: str, persona: str, seed: int = 0) -> int:
    s = synthetic.generate(persona, seed=seed)
    csv = s.transactions.drop(columns=["true_category"]).to_csv(index=False)
    return import_csv(user_id, csv, opening_balance=s.opening_balance)


def _frame(user_id: str) -> pd.DataFrame:
    df = db.load_transactions(user_id)
    if df.empty:
        raise LookupError(f"No transactions for user '{user_id}'. Upload a CSV or load a sample first.")
    return df


def _goals_with_history(user_id: str) -> list[dict]:
    """Goals plus the probability from the previous run, so the forecaster can report the trend."""
    previous = db.latest_goal_snapshots(user_id)
    goals = db.list_goals(user_id)
    for g in goals:
        if g["id"] in previous:
            g["previous_probability_pct"] = previous[g["id"]]["probability_pct"]
    return goals


# --- the multi-agent pipeline ---------------------------------------------------------------------

def analyze_user(user_id: str) -> dict:
    """Run the full multi-agent pipeline and persist everything it decided."""
    df = _frame(user_id)
    audit = AuditTrail(user_id)
    llm = get_llm()
    calls_before = dict(llm.stats.by_agent)
    goals = _goals_with_history(user_id)
    memory = db.load_merchant_memory()
    known = set(memory)
    state = run_pipeline(df, db.current_balance(user_id, df), audit, llm, goals=goals,
                         bills=db.list_bills(user_id), memory=memory)
    db.save_merchant_memory({k: v for k, v in memory.items() if k not in known})
    db.save_categories(state["txns"])
    a, fc = state["analysis"], state.get("forecast")
    findings = a.findings + state["forecast_findings"]
    db.save_findings(audit.run_id, user_id, findings)
    db.save_recommendations(audit.run_id, user_id, state["final"])
    if fc is not None:
        db.save_goal_snapshots(audit.run_id, user_id, goals, fc.goals)
    summary = {
        "narrative": a.narrative,
        "stats": a.stats,
        "forecast": fc.summary() if fc is not None else None,
        "series": fc.series() if fc is not None else None,
        "recurring": [r.brief() for r in a.recurring],
        "monthly": {str(m): {k: round(v) for k, v in row.items() if v} for m, row in a.monthly.iterrows()},
        "advisor_passes": state.get("attempts", 0),
        "first_pass_rate": state.get("first_pass_rate"),
        "llm_calls": {k: v - calls_before.get(k, 0) for k, v in llm.stats.by_agent.items()
                      if v - calls_before.get(k, 0)},
    }
    db.save_run(audit.run_id, user_id, "pipeline", summary)
    audit.flush()
    _invalidate(user_id)
    return {"run_id": audit.run_id, **summary, "findings": [f.brief() | {"severity": f.severity} for f in findings],
            "recommendations": state["final"]}


def _invalidate(user_id: str) -> None:
    with _cache_lock:
        _analysis_cache.pop(user_id, None)


def _analysis(user_id: str) -> tuple[pd.DataFrame, analyzer.Analysis, float]:
    """Categorised transactions, their analysis and the current balance (cached per user)."""
    df = _frame(user_id)
    if df["category"].isna().any():
        analyze_user(user_id)
        df = _frame(user_id)
    balance = db.current_balance(user_id, df)
    fingerprint = (len(df), int(df["id"].max()), round(float(df["amount"].sum()), 2), round(balance, 2))
    with _cache_lock:
        hit = _analysis_cache.get(user_id)
    if hit and hit[0] == fingerprint:
        return df, hit[1], balance
    a = analyzer.analyze(df, narrate=False)
    with _cache_lock:
        _analysis_cache[user_id] = (fingerprint, a)
    return df, a, balance


def ask(user_id: str, question: str) -> dict:
    df, a, balance = _analysis(user_id)
    audit = AuditTrail(user_id)
    state = run_ask(question, df, a, balance, audit, get_llm(), goals=db.list_goals(user_id),
                    bills=db.list_bills(user_id))
    db.save_run(audit.run_id, user_id, "ask", {"question": question, "intent": state["intent"],
                                                "answer": state["answer"]})
    audit.flush()
    return {"run_id": audit.run_id, "intent": state["intent"], "answer": state["answer"],
            "scenario": state["scenario"].describe(), "simulation": state.get("simulation")}


def simulate(user_id: str, scenario: forecaster.Scenario) -> dict:
    df, a, balance = _analysis(user_id)
    if not forecaster.has_enough_history(df):
        raise ValueError(f"At least {forecaster.MIN_HISTORY_DAYS} days of transactions are needed to simulate.")
    audit = AuditTrail(user_id)
    result = forecaster.simulate(df, a, balance, scenario, goals=db.list_goals(user_id), audit=audit,
                                 bills=db.list_bills(user_id))
    db.save_run(audit.run_id, user_id, "simulate", {"verdict": result["verdict"], "scenario": scenario.describe()})
    audit.flush()
    return {"run_id": audit.run_id, **result}


# --- goals, bills and explanations ----------------------------------------------------------------

def goals_overview(user_id: str) -> list[dict]:
    """Each goal with its latest status and how its probability has moved across runs."""
    latest = db.latest_goal_snapshots(user_id)
    out = []
    for g in db.list_goals(user_id):
        snap = latest.get(g["id"])
        out.append({**g, "status": snap["status"] if snap else None,
                    "probability_pct": snap["probability_pct"] if snap else None,
                    "history": db.goal_history(user_id, g["id"])})
    return out


def explain(rec_id: str) -> dict | None:
    """Why was this recommendation made? Evidence + the decisions around it."""
    rec = db.get_recommendation(rec_id)
    if not rec:
        return None
    findings = {f["finding_id"]: f for f in db.load_findings(rec["run_id"])}
    decisions = db.load_decisions(rec["run_id"])
    related = [d for d in decisions if d["decision"] in rec["finding_ids"]
               or rec["title"] in d["decision"] or d["agent"] == "coordinator"]
    return {"recommendation": rec, "evidence": [findings[f] for f in rec["finding_ids"] if f in findings],
            "decisions": related}
