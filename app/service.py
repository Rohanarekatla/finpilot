"""Application service: connects the pure LangGraph pipelines to the database."""
from __future__ import annotations

import pandas as pd

from app import db, ingest, synthetic
from app.agents import analyzer, forecaster
from app.audit import AuditTrail
from app.graph import run_ask, run_pipeline
from app.llm import get_llm


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


def analyze_user(user_id: str) -> dict:
    """Run the full multi-agent pipeline and persist everything it decided."""
    df = _frame(user_id)
    audit = AuditTrail(user_id)
    llm = get_llm()
    state = run_pipeline(df, db.current_balance(user_id, df), audit, llm, goals=db.list_goals(user_id))
    db.save_categories(state["txns"])
    a, fc = state["analysis"], state["forecast"]
    findings = a.findings + state["forecast_findings"]
    db.save_findings(audit.run_id, user_id, findings)
    db.save_recommendations(audit.run_id, user_id, state["final"])
    summary = {
        "narrative": a.narrative,
        "stats": a.stats,
        "forecast": fc.summary(),
        "series": fc.series(),
        "recurring": [r.brief() for r in a.recurring],
        "monthly": {str(m): {k: round(v) for k, v in row.items() if v} for m, row in a.monthly.iterrows()},
        "advisor_passes": state["attempts"],
        "first_pass_rate": state.get("first_pass_rate"),
        "llm_calls": dict(llm.stats.by_agent),
    }
    db.save_run(audit.run_id, user_id, "pipeline", summary)
    audit.flush()
    return {"run_id": audit.run_id, **summary, "findings": [f.brief() | {"severity": f.severity} for f in findings],
            "recommendations": state["final"]}


def _analysis(user_id: str) -> tuple[pd.DataFrame, analyzer.Analysis, float]:
    df = _frame(user_id)
    if df["category"].isna().any():
        analyze_user(user_id)
        df = _frame(user_id)
    return df, analyzer.analyze(df, narrate=False), db.current_balance(user_id, df)


def ask(user_id: str, question: str) -> dict:
    df, a, balance = _analysis(user_id)
    audit = AuditTrail(user_id)
    state = run_ask(question, df, a, balance, audit, get_llm(), goals=db.list_goals(user_id))
    db.save_run(audit.run_id, user_id, "ask", {"question": question, "intent": state["intent"],
                                                "answer": state["answer"]})
    audit.flush()
    sim = state.get("simulation")
    return {"run_id": audit.run_id, "intent": state["intent"], "answer": state["answer"],
            "scenario": state["scenario"].describe(), "simulation": sim}


def simulate(user_id: str, scenario: forecaster.Scenario) -> dict:
    df, a, balance = _analysis(user_id)
    audit = AuditTrail(user_id)
    result = forecaster.simulate(df, a, balance, scenario, goals=db.list_goals(user_id), audit=audit)
    db.save_run(audit.run_id, user_id, "simulate", {"verdict": result["verdict"], "scenario": scenario.describe()})
    audit.flush()
    return {"run_id": audit.run_id, **result}


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
