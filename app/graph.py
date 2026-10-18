"""LangGraph orchestration: the coordinator's routing between the specialist agents.

Pipeline (runs on upload, on demand and from the scheduler):

    categorize -> analyze --(enough history?)--> forecast ------> plan --(any actionable finding?)--> advise
                          \\--(no)--------------> skip_forecast -/      \\--(no)--> finalize
    advise -> critique --(rejections, LLM-written, attempts left)--> advise   (revision with critic feedback)
                       \\--(all pass / out of attempts)------------> finalize (drop failures, backfill, ids)

Ask (free-form questions):

    classify --(afford / what_if, enough history)--> simulate --> respond
             \\--(question, or not enough history)--> answer

Every routing decision is written to the audit trail by the coordinator, so a
run shows not only what each agent decided but which path the run took and why.

These graphs are pure: they take a DataFrame and return results. Persistence
lives in service.py, so the eval harness runs the exact same graph.

    python -m app.graph    # print both graphs as Mermaid diagrams
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
    # inputs
    audit: AuditTrail
    llm: MeshLLM
    txns: pd.DataFrame
    start_balance: float
    goals: list[dict]
    bills: list[dict]
    mode: str
    memory: categorizer.MerchantMemory | None
    # produced by the agents
    analysis: analyzer.Analysis
    forecast: forecaster.Forecast | None
    forecast_findings: list[analyzer.Finding]
    candidates: list[dict]
    recommendations: list[dict]
    verdicts: list[dict]
    attempts: int
    first_pass_rate: float
    final: list[dict]


# --- pipeline nodes -------------------------------------------------------------------------------

def _categorize(s: PipelineState) -> dict:
    df = s["txns"]
    todo = df["category"].isna() if "category" in df else pd.Series(True, index=df.index)
    s["audit"].log(COORD, "route", "categorize", f"{int(todo.sum())} uncategorised transactions")
    if not todo.any():
        return {}
    done = categorizer.categorize(df[todo], mode=s.get("mode", "auto"), audit=s["audit"], llm=s["llm"],
                                  memory=s.get("memory"))
    out = df.copy()
    for col in ("category", "category_source", "category_confidence"):
        if col not in out:
            out[col] = None
        out.loc[todo, col] = done[col]
    return {"txns": out}


def _analyze(s: PipelineState) -> dict:
    return {"analysis": analyzer.analyze(s["txns"], audit=s["audit"], llm=s["llm"])}


def _after_analyze(s: PipelineState) -> str:
    df = s["txns"]
    span = (df["date"].max() - df["date"].min()).days + 1
    if forecaster.has_enough_history(df):
        s["audit"].log(COORD, "route", "forecast", f"{span} days of history (minimum {forecaster.MIN_HISTORY_DAYS})")
        return "forecast"
    s["audit"].log(COORD, "route", "skip_forecast",
                   f"Only {span} days of history; a forecast needs at least {forecaster.MIN_HISTORY_DAYS}. "
                   "Advising on the detected patterns only.")
    return "skip_forecast"


def _forecast(s: PipelineState) -> dict:
    fc = forecaster.forecast(s["txns"], s["analysis"], s["start_balance"], goals=s.get("goals"), bills=s.get("bills"))
    found = forecaster.forecast_findings(fc, s["analysis"])
    summary = fc.summary()
    s["audit"].log(forecaster.AGENT, "forecast", f"P50 balance {summary['end_balance_p50']:.0f} on {summary['to']}",
                   f"{len(s['analysis'].recurring)} scheduled flows, {len(fc.bills)} declared bill(s) + 13-week block "
                   f"bootstrap of everyday spending (spread x{forecaster.SPREAD_INFLATION})",
                   {k: summary[k] for k in ("start_balance", "end_balance_p10", "end_balance_p50", "end_balance_p90",
                                            "prob_negative_pct")})
    for f in found:
        s["audit"].log(forecaster.AGENT, "finding", f.id, f.detail, f.data)
    return {"forecast": fc, "forecast_findings": found}


def _skip_forecast(s: PipelineState) -> dict:
    df = s["txns"]
    span = (df["date"].max() - df["date"].min()).days + 1
    note = analyzer.Finding(
        "insufficient_history:data", "insufficient_history", "Not enough history to forecast yet",
        f"Your data covers {span} days; FinPilot needs at least {forecaster.MIN_HISTORY_DAYS} days to project your "
        "balance. Upload a longer statement to unlock forecasts and what-if simulations.",
        "info", {"days": span, "needed_days": forecaster.MIN_HISTORY_DAYS})
    return {"forecast": None, "forecast_findings": [note]}


def _plan(s: PipelineState) -> dict:
    """Coordinator: turn every actionable finding into a quantified candidate action for the advisor."""
    cands = advisor.build_candidates(s["analysis"], s["forecast_findings"])
    s["audit"].log(COORD, "plan", f"{len(cands)} candidate action(s)",
                   "One computed action per actionable finding, ranked by yearly impact and severity",
                   [{"finding": c["finding_ids"][0], "priority": round(c["priority"])} for c in cands])
    return {"candidates": cands, "attempts": 0, "recommendations": [], "verdicts": []}


def _after_plan(s: PipelineState) -> str:
    if s["candidates"]:
        s["audit"].log(COORD, "route", "advise", f"{len(s['candidates'])} actionable finding(s)")
        return "advise"
    s["audit"].log(COORD, "route", "finalize", "Nothing actionable was found, so the advisor is not called")
    return "finalize"


def _advise(s: PipelineState) -> dict:
    attempts = s.get("attempts", 0)
    feedback = previous = None
    if attempts:
        feedback = [v for v in s["verdicts"] if not v["passed"]]
        previous = s["recommendations"]
        s["audit"].log(COORD, "route", "advise (revision)",
                       f"Critic rejected {len(feedback)} recommendation(s); sending feedback for revision {attempts}")
    summary = s["forecast"].summary() if s.get("forecast") else {}
    recs = advisor.advise(s["analysis"], s["forecast_findings"], summary, s["candidates"],
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
    recs, verdicts = s.get("recommendations", []), s.get("verdicts", [])
    kept = [r for r, v in zip(recs, verdicts) if v["passed"]]
    for r, v in zip(recs, verdicts):
        if not v["passed"]:
            audit.log(COORD, "drop", r["title"], "Still failing the critic after revisions: " + "; ".join(v["issues"]))
    covered = {fid for r in kept for fid in r["finding_ids"]}
    want = min(3, len(s["candidates"]))
    backfill = [c for c in s["candidates"] if c["finding_ids"][0] not in covered][: max(want - len(kept), 0)]
    if backfill:
        extra = advisor.templates(backfill)
        checks = critic.critique(extra, _all_findings(s), s["candidates"], s["analysis"].stats, audit)
        extra = [r for r, v in zip(extra, checks) if v["passed"]]
        kept += extra
        audit.log(COORD, "backfill", f"{len(extra)} computed recommendation(s) added",
                  "Fewer than 3 recommendations survived the critic; added the highest-impact remaining candidates")
    for i, r in enumerate(kept, 1):
        r["rec_id"] = f"{audit.run_id}-R{i}"
    audit.log(COORD, "finalize", f"{len(kept)} recommendations delivered",
              f"{s.get('attempts', 0)} advisor pass(es); every delivered recommendation passed the critic")
    return {"final": kept}


def build_pipeline():
    g = StateGraph(PipelineState)
    for name, fn in [("categorize", _categorize), ("analyze", _analyze), ("forecast", _forecast),
                     ("skip_forecast", _skip_forecast), ("plan", _plan), ("advise", _advise),
                     ("critique", _critique), ("finalize", _finalize)]:
        g.add_node(name, fn)
    g.add_edge(START, "categorize")
    g.add_edge("categorize", "analyze")
    g.add_conditional_edges("analyze", _after_analyze, {"forecast": "forecast", "skip_forecast": "skip_forecast"})
    g.add_edge("forecast", "plan")
    g.add_edge("skip_forecast", "plan")
    g.add_conditional_edges("plan", _after_plan, {"advise": "advise", "finalize": "finalize"})
    g.add_edge("advise", "critique")
    g.add_conditional_edges("critique", _after_critique, {"advise": "advise", "finalize": "finalize"})
    g.add_edge("finalize", END)
    return g.compile()


# --- ask graph ------------------------------------------------------------------------------------

class AskState(TypedDict, total=False):
    question: str
    audit: AuditTrail
    llm: MeshLLM
    txns: pd.DataFrame
    analysis: analyzer.Analysis
    start_balance: float
    goals: list[dict]
    bills: list[dict]
    intent: str
    scenario: forecaster.Scenario
    simulation: dict
    answer: str


def _classify(s: AskState) -> dict:
    intent, sc = coordinator.classify(s["question"], s["analysis"], s["llm"], s["audit"])
    return {"intent": intent, "scenario": sc}


def _route_ask(s: AskState) -> str:
    if s["intent"] in ("afford", "what_if"):
        if forecaster.has_enough_history(s["txns"]):
            s["audit"].log(COORD, "route", "simulate", f"intent={s['intent']}")
            return "simulate"
        s["audit"].log(COORD, "route", "answer",
                       f"intent={s['intent']}, but there is too little history to simulate; answering from the data")
        return "answer"
    s["audit"].log(COORD, "route", "answer", f"intent={s['intent']}")
    return "answer"


def _simulate(s: AskState) -> dict:
    sim = forecaster.simulate(s["txns"], s["analysis"], s["start_balance"], s["scenario"], goals=s.get("goals"),
                              bills=s.get("bills"), audit=s["audit"])
    return {"simulation": sim}


def _respond(s: AskState) -> dict:
    base = coordinator.verdict_text(s["simulation"])
    return {"answer": coordinator.rewrite(s["simulation"], base, s["llm"], s["audit"])}


def _answer(s: AskState) -> dict:
    summary = {}
    if forecaster.has_enough_history(s["txns"]):
        summary = forecaster.forecast(s["txns"], s["analysis"], s["start_balance"], goals=s.get("goals"),
                                      bills=s.get("bills")).summary()
    return {"answer": coordinator.answer_question(s["question"], s["analysis"], summary, s["llm"], s["audit"])}


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
                 goals: list[dict] | None = None, mode: str = "auto", bills: list[dict] | None = None,
                 memory: categorizer.MerchantMemory | None = None) -> PipelineState:
    return PIPELINE.invoke({"txns": txns, "start_balance": start_balance, "goals": goals or [], "bills": bills or [],
                            "mode": mode, "memory": memory, "audit": audit, "llm": llm})


def run_ask(question: str, txns: pd.DataFrame, analysis: analyzer.Analysis, start_balance: float,
            audit: AuditTrail, llm: MeshLLM, goals: list[dict] | None = None,
            bills: list[dict] | None = None) -> AskState:
    return ASK.invoke({"question": question, "txns": txns, "analysis": analysis, "start_balance": start_balance,
                       "goals": goals or [], "bills": bills or [], "audit": audit, "llm": llm})


def mermaid() -> str:
    """Both graphs as Mermaid, generated from the compiled graphs (so the diagram cannot drift)."""
    return ("%% pipeline\n" + PIPELINE.get_graph().draw_mermaid() + "\n%% ask\n" + ASK.get_graph().draw_mermaid())


if __name__ == "__main__":
    print(mermaid())
