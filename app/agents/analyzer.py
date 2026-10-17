"""Analyzer agent: behaviour over time, recurring payments and anomalies.

Detection is deterministic statistics over the user's own data, so every
finding is reproducible and carries the numbers behind it. The LLM only
writes the plain-language behaviour summary, and that summary is rejected if
it cites a figure that is not in the computed stats.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from app.audit import AuditTrail
from app.grounding import allowed_numbers, inr, ungrounded
from app.llm import MeshLLM, get_llm
from app.taxonomy import BEHAVIOURAL, LABELS, NON_CONSUMPTION, subscription_group

AGENT = "analyzer"

SPIKE_MIN_PCT = 0.35
SPIKE_MIN_DELTA = 1000.0
SPIKE_MIN_Z = 2.0
SPIKE_MIN_MONTHLY_TXNS = 2.0  # sparse categories are noise; big one-offs are caught as outliers
# Thresholds above were tuned on dev seeds 100-114 (eval reports test seeds 0-9).
OUTLIER_INCOME_SHARE = 0.20
OUTLIER_ROBUST_Z = 8.0
RECENT_DAYS = 45
LOW_SAVINGS_RATE = 0.10
NOT_DISCRETIONARY = {"rent", "emi_loan", "investments", "insurance", "education", "income"}


@dataclass
class Finding:
    id: str
    kind: str
    title: str
    detail: str
    severity: str  # info | warn | alert
    data: dict = field(default_factory=dict)
    monthly_impact: float = 0.0
    one_off_impact: float = 0.0

    def __post_init__(self) -> None:
        self.data = _plain(self.data)
        self.monthly_impact = float(self.monthly_impact)
        self.one_off_impact = float(self.one_off_impact)

    def brief(self) -> dict:
        return {"id": self.id, "kind": self.kind, "title": self.title, "detail": self.detail, "data": self.data}


def _plain(obj):
    """numpy scalars -> Python numbers so findings serialise cleanly."""
    if isinstance(obj, dict):
        return {k: _plain(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_plain(v) for v in obj]
    if isinstance(obj, np.integer):
        return int(obj)
    if isinstance(obj, np.floating):
        return float(obj)
    return obj


@dataclass
class Recurring:
    merchant: str
    category: str
    amount: float  # signed, latest charge
    period_months: int
    last_date: pd.Timestamp
    next_date: pd.Timestamp
    count: int
    txn_ids: list
    is_subscription: bool
    amounts: list = field(default_factory=list)  # chronological, absolute

    @property
    def monthly(self) -> float:
        return self.amount / self.period_months

    def brief(self) -> dict:
        return {"merchant": self.merchant, "category": self.category, "amount": self.amount,
                "every_months": self.period_months, "next_date": self.next_date.date().isoformat()}


@dataclass
class Analysis:
    as_of: pd.Timestamp
    latest_month: pd.Period
    monthly: pd.DataFrame  # complete months x category, consumption (positive)
    recurring: list[Recurring]
    findings: list[Finding]
    outlier_ids: set
    stats: dict
    narrative: str = ""

    @property
    def recurring_ids(self) -> set:
        return {i for r in self.recurring for i in r.txn_ids}

    def finding(self, fid: str) -> Finding | None:
        return next((f for f in self.findings if f.id == fid), None)


def analyze(df: pd.DataFrame, audit: AuditTrail | None = None, llm: MeshLLM | None = None,
            narrate: bool = True) -> Analysis:
    df = df.copy()
    df["month"] = df["date"].dt.to_period("M")
    as_of = df["date"].max()
    latest = as_of.to_period("M")
    if as_of.day < as_of.days_in_month - 2:  # current month still in progress
        latest -= 1
    findings: list[Finding] = []

    recurring = detect_recurring(df, as_of)
    rec_ids = {i for r in recurring for i in r.txn_ids}
    income = df[(df["amount"] > 0) & (df["category"] == "income")].groupby("month")["amount"].sum()
    complete = [m for m in sorted(df["month"].unique()) if m <= latest]
    avg_income = float(income.reindex(complete).fillna(0).tail(3).mean()) if complete else 0.0

    outliers = detect_outliers(df, rec_ids, avg_income)
    consumption = df[(df["amount"] < 0) & ~df["category"].isin(NON_CONSUMPTION)].assign(spend=lambda d: -d["amount"])
    monthly = consumption.pivot_table(index="month", columns="category", values="spend", aggfunc="sum").fillna(0.0)
    monthly = monthly.reindex(complete).fillna(0.0)
    regular = consumption[~consumption.index.isin(outliers)]
    monthly_ex = (regular.pivot_table(index="month", columns="category", values="spend", aggfunc="sum")
                  .reindex(complete).fillna(0.0))

    findings += _spikes(monthly_ex, regular, latest)
    findings += _large_transactions(df, outliers, as_of, avg_income)
    findings += _duplicates(df, rec_ids, as_of)
    findings += _subscriptions(recurring)

    invested = df[(df["amount"] < 0) & (df["category"] == "investments")].groupby("month")["amount"].sum().abs()
    last3 = complete[-3:]
    inc3 = float(income.reindex(last3).fillna(0).sum())
    # Underlying savings rate: one-off outliers are reported on their own, not as a habit.
    cons3 = float(monthly_ex.loc[last3].sum().sum()) if last3 else 0.0
    savings_rate = (inc3 - cons3) / inc3 if inc3 else 0.0
    if inc3 and savings_rate < LOW_SAVINGS_RATE:
        findings.append(Finding(
            "low_savings_rate:overall", "low_savings_rate", "Savings rate is low",
            f"Over the last {len(last3)} months you kept {savings_rate:.0%} of income after regular spending "
            f"(one-off purchases excluded).",
            "warn", {"savings_rate_pct": round(savings_rate * 100, 1), "avg_income": round(inc3 / len(last3)),
                     "avg_spend": round(cons3 / len(last3))}))

    latest_row = monthly.loc[latest] if latest in monthly.index else pd.Series(dtype=float)
    stats = {
        "as_of": as_of.date().isoformat(),
        "latest_month": str(latest),
        "avg_monthly_income": round(avg_income),
        "avg_monthly_spend": round(float(monthly.tail(3).sum(axis=1).mean())) if len(monthly) else 0,
        "avg_monthly_invested": round(float(invested.reindex(last3).fillna(0).mean())) if last3 else 0,
        "savings_rate_pct": round(savings_rate * 100, 1),
        "latest_month_spend": round(float(latest_row.sum())),
        "top_categories_latest": {k: round(v) for k, v in latest_row.sort_values(ascending=False).head(5).items()},
        "fixed_monthly_costs": round(-sum(r.monthly for r in recurring if r.amount < 0)),
        "subscription_monthly_total": round(-sum(r.monthly for r in recurring if r.is_subscription)),
    }
    analysis = Analysis(as_of, latest, monthly, recurring, findings, outliers, stats)

    if audit:
        audit.log(AGENT, "recurring", f"{len(recurring)} recurring payments detected",
                  "Same merchant, regular interval (monthly or quarterly), stable amount",
                  [r.brief() for r in recurring])
        for f in findings:
            audit.log(AGENT, "finding", f.id, f.detail, f.data)
    if narrate:
        analysis.narrative = _narrative(analysis, audit, llm or get_llm())
    return analysis


def detect_recurring(df: pd.DataFrame, as_of: pd.Timestamp) -> list[Recurring]:
    out = []
    for (merchant, positive), g in df.groupby([df["merchant"], df["amount"] > 0], sort=False):
        if len(g) < 2:
            continue
        g = g.sort_values("date")
        gaps = g["date"].diff().dt.days.dropna().to_numpy()
        med = float(np.median(gaps))
        if 25 <= med <= 35 and len(g) >= 3 and gaps.std() <= 6:
            period = 1
        elif 80 <= med <= 100 and gaps.std() <= 12:
            period = 3
        else:
            continue
        amts = g["amount"].abs().to_numpy()
        base = amts[:-1] if len(amts) > 2 else amts  # the latest charge may be a price change
        if base.std() / base.mean() > 0.35:
            continue
        last = g["date"].iloc[-1]
        if (as_of - last).days > period * 31 + 10:  # lapsed / cancelled
            continue
        category = g["category"].mode().iloc[0]
        out.append(Recurring(
            merchant=merchant, category=category, amount=float(g["amount"].iloc[-1]), period_months=period,
            last_date=last, next_date=last + pd.DateOffset(months=period), count=len(g),
            txn_ids=list(g.index), is_subscription=category in ("subscriptions", "fitness") and not positive,
            amounts=[float(a) for a in amts],
        ))
    return out


def detect_outliers(df: pd.DataFrame, rec_ids: set, avg_income: float) -> set:
    cand = df[(df["amount"] < 0) & ~df.index.isin(rec_ids) & ~df["category"].isin(NOT_DISCRETIONARY)]
    out = set()
    for _, g in cand.groupby("category"):
        amts = -g["amount"]
        med = amts.median()
        mad = (amts - med).abs().median()
        for idx, a in amts.items():
            big = avg_income > 0 and a >= OUTLIER_INCOME_SHARE * avg_income
            robust = len(g) >= 5 and mad > 0 and 0.6745 * (a - med) / mad >= OUTLIER_ROBUST_Z and a >= 5000
            if big or robust:
                out.add(idx)
    return out


def _spikes(monthly: pd.DataFrame, regular: pd.DataFrame, latest: pd.Period) -> list[Finding]:
    """Flag categories whose latest month is far above their own history.

    Monthly spend in a category is a compound Poisson sum (how many purchases x
    how much each), so its standard deviation is sqrt(rate * E[amount^2]),
    estimated from all historical transactions. That is far more stable than
    the sample SD of five or six monthly totals.
    """
    out = []
    if latest not in monthly.index:
        return out
    hist = monthly.loc[monthly.index < latest].tail(6)
    if len(hist) < 2:
        return out
    for cat in sorted(BEHAVIOURAL & set(monthly.columns)):
        tx = regular[(regular["category"] == cat) & regular["month"].isin(hist.index)]["spend"]
        rate = len(tx) / len(hist)
        if rate < SPIKE_MIN_MONTHLY_TXNS:
            continue
        cur, base = monthly.at[latest, cat], hist[cat].mean()
        sd = float(np.sqrt(rate * (tx ** 2).mean()))
        delta = cur - base
        if base <= 0 or delta < SPIKE_MIN_DELTA or delta / base < SPIKE_MIN_PCT or delta / sd < SPIKE_MIN_Z:
            continue
        pct = delta / base * 100
        now = regular[(regular["category"] == cat) & (regular["month"] == latest)]
        top = (now.groupby("merchant")["spend"].agg(["sum", "count"]).sort_values("sum", ascending=False).head(3))
        top_merchants = [{"merchant": m, "amount": round(r["sum"]), "count": int(r["count"])} for m, r in top.iterrows()]
        out.append(Finding(
            f"category_spike:{cat}", "category_spike", f"{LABELS[cat].capitalize()} up {pct:.0f}% this month",
            f"{LABELS[cat].capitalize()} was {inr(cur)} in {latest.strftime('%b %Y')} vs a "
            f"{len(hist)}-month average of {inr(base)} (+{inr(delta)}); mostly "
            + ", ".join(f"{t['merchant']} ({inr(t['amount'])})" for t in top_merchants) + ".",
            "warn", {"category": cat, "month": str(latest), "current": round(cur), "baseline": round(base),
                     "increase": round(delta), "pct_change": round(pct, 1), "z_score": round(delta / sd, 2),
                     "top_merchants": top_merchants},
            monthly_impact=round(delta)))
    return out


def _large_transactions(df: pd.DataFrame, outliers: set, as_of: pd.Timestamp, avg_income: float) -> list[Finding]:
    out = []
    recent = df.loc[sorted(outliers)]
    recent = recent[recent["date"] >= as_of - pd.Timedelta(days=RECENT_DAYS)]
    for idx, r in recent.iterrows():
        a = -r["amount"]
        share = a / avg_income * 100 if avg_income else 0
        out.append(Finding(
            f"large_transaction:{r['merchant']}", "large_transaction",
            f"Unusually large {LABELS[r['category']]} payment at {r['merchant']}",
            f"{inr(a)} at {r['merchant']} on {r['date']:%d %b} is {share:.0f}% of a month's income and far above "
            f"your usual {LABELS[r['category']]} spend.",
            "alert" if share >= 40 else "warn",
            {"merchant": r["merchant"], "amount": round(a), "date": r["date"].date().isoformat(),
             "category": r["category"], "pct_of_income": round(share, 1), "txn_id": int(idx)},
            one_off_impact=round(a)))
    return out


def _duplicates(df: pd.DataFrame, rec_ids: set, as_of: pd.Timestamp) -> list[Finding]:
    deb = df[(df["amount"] < 0) & ~df.index.isin(rec_ids) & (df["date"] >= as_of - pd.Timedelta(days=RECENT_DAYS))]
    out, seen = [], set()
    for (merchant, amount), g in deb.groupby(["merchant", "amount"]):
        if len(g) < 2 or merchant in seen:
            continue
        dates = g["date"].sort_values()
        close = dates.diff().dt.days.le(1)
        if close.any():
            seen.add(merchant)
            d = dates[close].iloc[0]
            out.append(Finding(
                f"duplicate_charge:{merchant}", "duplicate_charge", f"Possible duplicate charge from {merchant}",
                f"Two identical charges of {inr(-amount)} from {merchant} on {d:%d %b}.",
                "alert", {"merchant": merchant, "amount": round(-amount), "date": d.date().isoformat(),
                          "txn_ids": [int(i) for i in g.index]},
                one_off_impact=round(-amount)))
    return out


def _subscriptions(recurring: list[Recurring]) -> list[Finding]:
    subs = [r for r in recurring if r.is_subscription]
    out = []
    if subs:
        total = -sum(r.monthly for r in subs)
        out.append(Finding(
            "subscriptions:summary", "subscriptions", f"{len(subs)} active subscriptions",
            f"You pay {inr(total)}/month across {', '.join(r.merchant for r in subs)}.",
            "info", {"total_monthly": round(total), "items": [{"merchant": r.merchant, "monthly": round(-r.monthly)} for r in subs]}))
    groups: dict[str, list[Recurring]] = {}
    for r in subs:
        g = subscription_group(r.merchant)
        if g:
            groups.setdefault(g, []).append(r)
    for g, items in groups.items():
        if len(items) < 2:
            continue
        costs = sorted((-r.monthly for r in items), reverse=True)
        saving = sum(costs[1:])
        names = ", ".join(r.merchant for r in items)
        out.append(Finding(
            f"overlapping_subscriptions:{g}", "overlapping_subscriptions",
            f"{len(items)} overlapping {g.replace('_', ' ')} services",
            f"You pay for {names} ({inr(sum(costs))}/month). Keeping only one saves at least {inr(saving)}/month.",
            "warn", {"group": g, "services": [{"merchant": r.merchant, "monthly": round(-r.monthly)} for r in items],
                     "total_monthly": round(sum(costs)), "saving_if_keep_one": round(saving)},
            monthly_impact=round(saving)))
    return out + _price_hikes(recurring)


def _price_hikes(recurring: list[Recurring]) -> list[Finding]:
    out = []
    for r in recurring:
        if r.amount >= 0 or len(r.amounts) < 3:
            continue
        prev = np.array(r.amounts[:-1])
        old, new = float(np.median(prev)), r.amounts[-1]
        if prev.std() / prev.mean() > 0.05 or new < old * 1.05 or new - old < 20:
            continue
        inc = (new - old) / r.period_months
        out.append(Finding(
            f"price_hike:{r.merchant}", "price_hike", f"{r.merchant} price went up",
            f"{r.merchant} charged {inr(new)} this cycle vs {inr(old)} before (+{inr(new - old)}).",
            "warn", {"merchant": r.merchant, "old_amount": round(old), "new_amount": round(new),
                     "increase_monthly": round(inc), "pct_change": round((new - old) / old * 100, 1)},
            monthly_impact=round(inc)))
    return out


NARRATIVE_SYSTEM = """You are the Analyzer agent of FinPilot. Write a 2-3 sentence plain-English summary of
this person's money behaviour for them, using second person. Use ONLY numbers that appear in the
input (you may round to the nearest rupee). Mention the most important pattern first. No advice.
Return JSON: {"summary": "..."}"""


def _narrative(a: Analysis, audit: AuditTrail | None, llm: MeshLLM) -> str:
    s = a.stats
    fallback = (
        f"In {a.latest_month.strftime('%B %Y')} you spent {inr(s['latest_month_spend'])} against an average income of "
        f"{inr(s['avg_monthly_income'])}; over the last three months you kept {s['savings_rate_pct']}% of income. "
        + (f"Biggest signal: {a.findings[0].title.lower()}." if a.findings else "No unusual activity detected.")
    )
    payload = {"stats": s, "findings": [f.brief() for f in a.findings]}
    resp = llm.chat_json(AGENT, NARRATIVE_SYSTEM, payload)
    text = (resp or {}).get("summary", "")
    if text:
        bad = ungrounded(text, allowed_numbers(payload, derive=False))
        if not bad:
            return text
        if audit:
            audit.log(AGENT, "narrative_rejected", "LLM summary discarded", f"Ungrounded numbers: {bad}", {"text": text})
    return fallback
