"""Coordinator agent for free-form questions.

It classifies the question (LLM, regex fallback) and routes it: affordability
and what-if questions become a Scenario for the forecaster; everything else
goes to a question-answering step over the user's computed data. Answers are
built from computed numbers; an LLM rewrite is accepted only if every figure
in it is grounded.
"""
from __future__ import annotations

import re

import pandas as pd

from app.agents.analyzer import Analysis
from app.agents.forecaster import Scenario
from app.audit import AuditTrail
from app.grounding import allowed_numbers, inr, ungrounded
from app.llm import MeshLLM
from app.taxonomy import CATEGORIES, LABELS

AGENT = "coordinator"

CLASSIFY_SYSTEM = """You are the Coordinator agent of FinPilot. Classify the user's money question and extract
a what-if scenario when there is one. Today is {as_of}. Known recurring payments: {recurring}.
Spending categories: {categories}.
intent is one of:
- "afford": can I afford / should I buy a one-off purchase -> fill purchases
- "what_if": a change to regular behaviour (new monthly cost, cutting a category, cancelling a subscription)
- "question": anything else about their money
Dates are ISO (YYYY-MM-DD); "next month" means the 15th of next month. Amounts in INR as numbers
(1.5 lakh = 150000, 60k = 60000). category_cuts are fractions (cut by 30% -> 0.3). monthly_changes amounts
are negative for new costs and positive for new income. cancel lists recurring merchant names exactly as given.
Return JSON: {{"intent": str, "label": str, "purchases": [{{"date": str, "amount": number, "label": str}}],
"monthly_changes": [{{"amount": number, "label": str, "start": str|null}}], "category_cuts": {{category: fraction}},
"cancel": [str]}}"""

QA_SYSTEM = """You are FinPilot's assistant. Answer the user's question about their own money in 2-4 sentences
using ONLY the numbers in the provided data (round to the rupee, write amounts like ₹1,499). If the data
cannot answer it, say what you can tell them instead. Return JSON: {"answer": str}"""

REWRITE_SYSTEM = """Rewrite this affordability verdict for the user in 2-3 friendly, direct sentences. Keep every
number exactly as given, do not add new numbers, and end with one specific suggestion drawn from the
reasons. Return JSON: {"answer": str}"""

_AMOUNT = re.compile(r"(?:₹|rs\.?|inr)?\s*(\d+(?:[.,]\d+)*)\s*(k|thousand|lakhs?|lacs?|l)?\b", re.I)
_MONTHS = {m.lower(): i for i, m in enumerate(["jan", "feb", "mar", "apr", "may", "jun", "jul", "aug", "sep", "oct", "nov", "dec"], 1)}
_CATEGORY_WORDS = {
    "food_delivery": ["food delivery", "swiggy", "zomato", "ordering food", "delivery"],
    "dining": ["dining", "eating out", "restaurant", "cafe"],
    "groceries": ["grocer"], "shopping": ["shopping", "amazon", "flipkart"],
    "transport": ["transport", "cab", "uber", "ola", "auto"], "fuel": ["fuel", "petrol"],
    "entertainment": ["entertainment", "movies"], "travel": ["travel", "trip", "flight"],
    "subscriptions": ["subscription", "netflix", "spotify", "ott"],
}


def parse_amount(text: str) -> float | None:
    best = None
    for m in _AMOUNT.finditer(text):
        raw, unit = m.group(1).replace(",", ""), (m.group(2) or "").lower()
        try:
            v = float(raw)
        except ValueError:
            continue
        if not unit and not re.match(r"\s*(₹|rs|inr)", m.group(0), re.I) and 1990 <= v <= 2100:
            continue  # a year, not an amount
        if unit in ("k", "thousand"):
            v *= 1_000
        elif unit.startswith(("lakh", "lac")) or unit == "l":
            v *= 100_000
        if v >= 100 and (best is None or v > best):
            best = v
    return best


def parse_when(text: str, as_of: pd.Timestamp) -> pd.Timestamp:
    t = text.lower()
    if "tomorrow" in t:
        return as_of + pd.Timedelta(days=1)
    if "next week" in t:
        return as_of + pd.Timedelta(days=7)
    if m := re.search(r"in (\d+) months?", t):
        return as_of + pd.DateOffset(months=int(m.group(1)))
    if m := re.search(r"in (\d+) weeks?", t):
        return as_of + pd.Timedelta(weeks=int(m.group(1)))
    if "next month" in t:
        nxt = (as_of + pd.offsets.MonthBegin(1)).normalize()
        return nxt + pd.Timedelta(days=14)
    for name, num in _MONTHS.items():
        if re.search(rf"\b{name}[a-z]*\b", t):
            year = as_of.year + (1 if num <= as_of.month else 0)
            return pd.Timestamp(year=year, month=num, day=15)
    return as_of + pd.Timedelta(days=14)


def _mentioned_categories(text: str) -> list[str]:
    t = text.lower()
    return [c for c, words in _CATEGORY_WORDS.items() if any(w in t for w in words)]


def classify(question: str, analysis: Analysis, llm: MeshLLM, audit: AuditTrail | None = None) -> tuple[str, Scenario]:
    system = CLASSIFY_SYSTEM.format(as_of=analysis.as_of.date().isoformat(),
                                    recurring=[r.merchant for r in analysis.recurring if r.amount < 0],
                                    categories=list(CATEGORIES))
    resp = llm.chat_json(AGENT, system, question)
    if resp and resp.get("intent") in ("afford", "what_if", "question"):
        known = {r.merchant.upper() for r in analysis.recurring}
        try:
            sc = Scenario(
                label=str(resp.get("label") or question[:60]),
                purchases=[{"date": str(pd.Timestamp(p["date"]).date()), "amount": abs(float(p["amount"])),
                            "label": str(p.get("label", "purchase"))} for p in resp.get("purchases") or []],
                monthly_changes=[{"amount": float(c["amount"]), "label": str(c.get("label", "change")),
                                  "start": c.get("start")} for c in resp.get("monthly_changes") or []],
                category_cuts={k: float(v) for k, v in (resp.get("category_cuts") or {}).items() if k in CATEGORIES},
                cancel=[c for c in resp.get("cancel") or [] if str(c).upper() in known],
            )
            intent = resp["intent"] if (resp["intent"] == "question" or not sc.is_baseline) else "question"
            if audit:
                audit.log(AGENT, "classify", intent, "LLM intent classification", sc.describe())
            return intent, sc
        except (KeyError, TypeError, ValueError):
            pass
    intent, sc = _rule_classify(question, analysis)
    if audit:
        audit.log(AGENT, "classify", intent, "Rule-based intent classification (LLM unavailable or invalid)", sc.describe())
    return intent, sc


def _rule_classify(question: str, analysis: Analysis) -> tuple[str, Scenario]:
    q = question.lower()
    amount = parse_amount(question)
    cancel = [r.merchant for r in analysis.recurring if r.amount < 0 and r.merchant.split()[0].lower() in q]
    if cancel and re.search(r"cancel|drop|stop|unsubscribe", q):
        return "what_if", Scenario(label=f"Cancel {', '.join(cancel)}", cancel=cancel)
    if m := re.search(r"(cut|reduce|lower)\D+?(\d+)\s*%", q):
        cats = _mentioned_categories(q)
        if cats:
            return "what_if", Scenario(label=question[:60], category_cuts={c: int(m.group(2)) / 100 for c in cats})
    if amount and re.search(r"every month|per month|monthly|a month", q):
        sign = 1 if re.search(r"raise|hike|increment|earn|extra income", q) else -1
        return "what_if", Scenario(label=question[:60], monthly_changes=[
            {"amount": sign * amount, "label": question[:40], "start": None}])
    if amount and re.search(r"afford|buy|purchase|trip|spend|book|get a|vacation|holiday", q):
        when = parse_when(question, analysis.as_of)
        return "afford", Scenario(label=question[:60], purchases=[
            {"date": str(when.date()), "amount": amount, "label": question[:40]}])
    return "question", Scenario(label=question[:60])


def verdict_text(sim: dict) -> str:
    sc = sim["scenario"]
    head = {
        "affordable": "Yes, you can afford this.",
        "affordable_but_delays_goal": "You can afford this, but it sets back a goal.",
        "tight": "You can technically afford it, but it would be tight.",
        "not_affordable": "Not right now.",
    }[sim["verdict"]]
    if not sc["purchases"] and sim["delta_end_balance_p50"] >= 0:
        head = "That would improve your position."
    
    what = "; ".join(
        [f"{inr(p['amount'])} on {p['date']}" for p in sc["purchases"]]
        + [f"{inr(abs(c['amount']))}/month {'more' if c['amount'] < 0 else 'less'} spending" for c in sc["monthly_changes"]]
        + [f"cutting {LABELS.get(k, k)} by {v:.0%}" for k, v in sc["category_cuts"].items()]
        + [f"cancelling {m}" for m in sc["cancel"]])
    b, a = sim["baseline"], sim["with_decision"]
    delta = sim["delta_end_balance_p50"]
    return (f"{head} Scenario: {what}. By {a['to']} your balance would be about {inr(a['end_balance_p50'])} "
            f"instead of {inr(b['end_balance_p50'])} ({'+' if delta >= 0 else '-'}{inr(abs(delta))}). "
            + " ".join(sim["reasons"]))


def rewrite(sim: dict, base_text: str, llm: MeshLLM, audit: AuditTrail | None) -> str:
    resp = llm.chat_json(AGENT, REWRITE_SYSTEM, {"verdict_text": base_text, "verdict": sim["verdict"]})
    text = (resp or {}).get("answer", "")
    if not text:
        return base_text
    bad = ungrounded(text, allowed_numbers([base_text, sim["baseline"], sim["with_decision"], sim["scenario"]]))
    if bad:
        if audit:
            audit.log(AGENT, "rewrite_rejected", "kept computed answer", f"Ungrounded numbers in rewrite: {bad}")
        return base_text
    return text


def qa_context(analysis: Analysis, forecast_summary: dict) -> dict:
    monthly = analysis.monthly.tail(6).round()
    return {
        "stats": analysis.stats,
        "monthly_spend_by_category": {str(m): {k: v for k, v in row.items() if v} for m, row in monthly.iterrows()},
        "recurring": [r.brief() for r in analysis.recurring],
        "findings": [f.brief() for f in analysis.findings],
        "forecast": {k: v for k, v in forecast_summary.items() if k != "month_ends"},
    }


def answer_question(question: str, analysis: Analysis, forecast_summary: dict, llm: MeshLLM,
                    audit: AuditTrail | None = None) -> str:
    ctx = qa_context(analysis, forecast_summary)
    resp = llm.chat_json(AGENT, QA_SYSTEM, {"question": question, "data": ctx})
    text = (resp or {}).get("answer", "")
    if text:
        bad = ungrounded(text, allowed_numbers(ctx, derive=False))
        if not bad:
            return text
        if audit:
            audit.log(AGENT, "qa_rejected", "LLM answer discarded", f"Ungrounded numbers: {bad}", {"text": text})
    return _rule_answer(question, analysis, forecast_summary)


def _rule_answer(question: str, analysis: Analysis, fs: dict) -> str:
    cats = _mentioned_categories(question)
    m = analysis.monthly
    if cats and len(m):
        parts = []
        for c in cats:
            if c in m.columns:
                parts.append(f"{LABELS[c].capitalize()}: {inr(m[c].iloc[-1])} in {m.index[-1].strftime('%b %Y')}, "
                             f"{inr(m[c].tail(6).mean())}/month on average over the last {min(len(m), 6)} months.")
        if parts:
            return " ".join(parts)
    s = analysis.stats
    return (f"You earn about {inr(s['avg_monthly_income'])} and spend about {inr(s['avg_monthly_spend'])} a month "
            f"(savings rate {s['savings_rate_pct']}%). Your balance is projected at {inr(fs['end_balance_p50'])} "
            f"by {fs['to']}.")
