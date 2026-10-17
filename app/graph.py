"""LangGraph orchestration.

Pipeline (runs on upload, on demand and from the scheduler):

    categorize -> analyze -> forecast -> advise -> critique --(issues & attempts left)--> advise
                                                          \\--(all pass / out of attempts)--> finalize

Ask (free-form questions):

    classify --(afford / what_if)--> simulate --> respond
             \\--(question)--------> answer ----/

These graphs are pure: they take a DataFrame and return results. Persistence
lives in service.py, so the eval harness runs the exact same graph.
"""
from __future__ import annotations

from typing import Any, TypedDict

import pandas as pd
from langgraph.graph import END, START, StateGraph

from app import config
from app.agents import advisor, analyzer, categorizer, coordinator, critic, forecaster
from app.audit import AuditTrail
from app.llm import MeshLLM

COORD = "coordinator"


class PipelineState(TypedDict, total=False):
    audit: AuditTrail
    llm: MeshLLM
    txns: pd.DataFrame
    start_balance: float
    goals: list[dict]
    mode: str
    analysis: analyzer.Analysis
    forecast: forecaster.Forecast
    forecast_findings: list[analyzer.Finding]
    candidates: list[dict]
    recommendations: list[dict]
    verdicts: list[dict]
    attempts: int
    final: list[dict]
    first_pass_rate: float


def _categorize(s: PipelineState) -> dict:
    df = s["txns"]
    todo = df["category"].isna() if "category" in df else pd.Series(True, index=df.index)
    s["audit"].log(COORD, "route", "categorize", f"{int(todo.sum())} uncategorised transactions")
    if not todo.any():
        return {}
    done = categorizer.categorize(df[todo], mode=s.get("mode", "auto"), audit=s["audit"], llm=s["llm"])
    out = df.copy()
    for col in ("category", "category_source", "category_confidence"):
        if col not in out:
            out[col] = None
        out.loc[todo, col] = done[col]
    return {"txns": out}


def _analyze(s: PipelineState) -> dict:
    return {"analysis": analyzer.analyze(s["txns"], audit=s["audit"], llm=s["llm"])}


def _forecast(s: PipelineState) -> dict:
    fc = forecaster.forecast(s["txns"], s["analysis"], s["start_balance"], goals=s.get("goals"))
    found = forecaster.forecast_findings(fc, s["analysis"])
    summary = fc.summary()
    s["audit"].log(forecaster.AGENT, "forecast", f"P50 balance {summary['end_balance_p50']:.0f} on {summary['to']}",
                   f"{len(s['analysis'].recurring)} scheduled flows + 13-week block bootstrap of everyday spending",
                   {k: summary[k] for k in ("start_balance", "end_balance_p10", "end_balance_p50", "end_balance_p90",
                                            "prob_negative_pct")})
    for f in found:
        s["audit"].log(forecaster.AGENT, "finding", f.id, f.detail, f.data)
    cands = advisor.build_candidates(s["analysis"], found)
    return {"forecast": fc, "forecast_findings": found, "candidates": cands, "attempts": 0}


def _advise(s: PipelineState) -> dict:
    attempts = s.get("attempts", 0)
    feedback = previous = None
    if attempts:
        feedback = [v for v in s["verdicts"] if not v["passed"]]
        previous = s["recommendations"]
        s["audit"].log(COORD, "route", "advise (revision)",
                       f"Critic rejected {len(feedback)} recommendation(s); sending feedback for revision {attempts}")
    recs = advisor.advise(s["analysis"], s["forecast_findings"], s["forecast"].summary(), s["candidates"],
                          feedback=feedback, previous=previous, audit=s["audit"], llm=s["llm"])
    return {"recommendations": recs, "attempts": attempts + 1}


def _all_findings(s: PipelineState) -> dict[str, analyzer.Finding]:
    return {f.id: f for f in s["analysis"].findings + s["forecast_findings"]}


def _critique(s: PipelineState) -> dict:
    verdicts = critic.critique(s["recommendations"], _all_findings(s), s["candidates"], s["analysis"].stats, s["audit"])
    out: dict[str, Any] = {"verdicts": verdicts}
    if s["attempts"] == 1:
        out["first_pass_rate"] = sum(v["passed"] for v in verdicts) / len(verdicts) if verdicts else 1.0
    return out


def _after_critique(s: PipelineState) -> str:
    failed = [v for v in s["verdicts"] if not v["passed"]]
    from_llm = any(r.get("source") == "llm" for r in s["recommendations"])
    if failed and from_llm and s["attempts"] <= config.MAX_ADVISOR_REVISIONS:
        return "advise"
    return "finalize"


def _finalize(s: PipelineState) -> dict:
    audit = s["audit"]
    kept = [r for r, v in zip(s["recommendations"], s["verdicts"]) if v["passed"]]
    for r, v in zip(s["recommendations"], s["verdicts"]):
        if not v["passed"]:
            audit.log(COORD, "drop", r["title"], "Still failing the critic after revisions: " + "; ".join(v["issues"]))
    covered = {fid for r in kept for fid in r["finding_ids"]}
    want = min(3, len(s["candidates"]))
    backfill = [c for c in s["candidates"] if c["finding_ids"][0] not in covered][: max(want - len(kept), 0)]
    if backfill:
        extra = advisor.templates(backfill)
        verdicts = critic.critique(extra, _all_findings(s), s["candidates"], s["analysis"].stats, audit)
        extra = [r for r, v in zip(extra, verdicts) if v["passed"]]
        kept += extra
        audit.log(COORD, "backfill", f"{len(extra)} computed recommendation(s) added",
                  "Fewer than 3 recommendations survived the critic; added the highest-impact remaining candidates")
    for i, r in enumerate(kept, 1):
        r["rec_id"] = f"{audit.run_id}-R{i}"
    audit.log(COORD, "finalize", f"{len(kept)} recommendations delivered",
              f"{s['attempts']} advisor pass(es); every delivered recommendation passed the critic")
    return {"final": kept}


def build_pipeline():
    g = StateGraph(PipelineState)
    g.add_node("categorize", _categorize)
    g.add_node("analyze", _analyze)
    g.add_node("forecast", _forecast)
    g.add_node("advise", _advise)
    g.add_node("critique", _critique)
    g.add_node("finalize", _finalize)
    g.add_edge(START, "categorize")
    g.add_edge("categorize", "analyze")
    g.add_edge("analyze", "forecast")
    g.add_edge("forecast", "advise")
    g.add_edge("advise", "critique")
    g.add_conditional_edges("critique", _after_critique, {"advise": "advise", "finalize": "finalize"})
    g.add_edge("finalize", END)
    return g.compile()


class AskState(TypedDict, total=False):
    question: str
    audit: AuditTrail
    llm: MeshLLM
    txns: pd.DataFrame
    analysis: analyzer.Analysis
    start_balance: float
    goals: list[dict]
    intent: str
    scenario: forecaster.Scenario
    simulation: dict
    answer: str


def _classify(s: AskState) -> dict:
    intent, sc = coordinator.classify(s["question"], s["analysis"], s["llm"], s["audit"])
    return {"intent": intent, "scenario": sc}


def _route_ask(s: AskState) -> str:
    route = "simulate" if s["intent"] in ("afford", "what_if") else "answer"
    s["audit"].log(COORD, "route", route, f"intent={s['intent']}")
    return route


def _simulate(s: AskState) -> dict:
    sim = forecaster.simulate(s["txns"], s["analysis"], s["start_balance"], s["scenario"], goals=s.get("goals"),
                              audit=s["audit"])
    return {"simulation": sim}


def _respond(s: AskState) -> dict:
    base = coordinator.verdict_text(s["simulation"])
    return {"answer": coordinator.rewrite(s["simulation"], base, s["llm"], s["audit"])}


def _answer(s: AskState) -> dict:
    fc = forecaster.forecast(s["txns"], s["analysis"], s["start_balance"], goals=s.get("goals"))
    return {"answer": coordinator.answer_question(s["question"], s["analysis"], fc.summary(), s["llm"], s["audit"])}


def build_ask():
    g = StateGraph(AskState)
    g.add_node("classify", _classify)
    g.add_node("simulate", _simulate)
    g.add_node("respond", _respond)
    g.add_node("answer", _answer)
    g.add_edge(START, "classify")
    g.add_conditional_edges("classify", _route_ask, {"simulate": "simulate", "answer": "answer"})
    g.add_edge("simulate", "respond")
    g.add_edge("respond", END)
    g.add_edge("answer", END)
    return g.compile()


PIPELINE = build_pipeline()
ASK = build_ask()


def run_pipeline(txns: pd.DataFrame, start_balance: float, audit: AuditTrail, llm: MeshLLM,
                 goals: list[dict] | None = None, mode: str = "auto") -> PipelineState:
    return PIPELINE.invoke({"txns": txns, "start_balance": start_balance, "goals": goals or [], "mode": mode,
                            "audit": audit, "llm": llm})


def run_ask(question: str, txns: pd.DataFrame, analysis: analyzer.Analysis, start_balance: float,
            audit: AuditTrail, llm: MeshLLM, goals: list[dict] | None = None) -> AskState:
    return ASK.invoke({"question": question, "txns": txns, "analysis": analysis, "start_balance": start_balance,
                       "goals": goals or [], "audit": audit, "llm": llm})
