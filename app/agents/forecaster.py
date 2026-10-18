"""Forecaster agent: where the money is heading, and what-if simulation.

The forecast has two parts:
  * scheduled flows: every recurring payment/income the analyzer detected is
    placed on its next due dates (latest amount, so price hikes carry forward);
  * everyday spending: non-recurring, non-outlier flows from the last 13 weeks
    are block-bootstrapped a week at a time into N simulated futures, which
    keeps weekday/weekend rhythm and gives honest P10/P50/P90 bands.

A scenario (planned purchase, new monthly cost, category cut, cancelled
subscription) is applied on top of the same random draws as the baseline
(common random numbers), so the difference between the two is exactly the
effect of the decision, not simulation noise.
"""
from __future__ import annotations

import math
import re
from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from app import config
from app.agents.analyzer import Analysis, Finding
from app.audit import AuditTrail
from app.grounding import inr

AGENT = "forecaster"
LOOKBACK_DAYS = 91
MAX_HORIZON_DAYS = 400
MIN_HISTORY_DAYS = 28  # below this there are too few weekly blocks to bootstrap from
# Variance inflation for the everyday-spending bootstrap. Thirteen weekly blocks under-represent
# tail weeks, so the raw P10-P90 band was too narrow (63% coverage vs. the nominal 80%). Each
# simulated week is widened around the average week by this factor; the median is unchanged.
# Tuned on dev seeds 100-114 with `python -m eval.tune_forecast`; reported on test seeds 0-9.
SPREAD_INFLATION = 1.3


@dataclass
class Scenario:
    label: str = "baseline"
    purchases: list[dict] = field(default_factory=list)  # {"date", "amount" (>0 = spend), "label"}
    monthly_changes: list[dict] = field(default_factory=list)  # {"amount" (signed), "label", "start"}
    category_cuts: dict[str, float] = field(default_factory=dict)  # {"food_delivery": 0.3}
    cancel: list[str] = field(default_factory=list)  # recurring merchant names

    @property
    def is_baseline(self) -> bool:
        return not (self.purchases or self.monthly_changes or self.category_cuts or self.cancel)

    def describe(self) -> dict:
        return {"label": self.label, "purchases": self.purchases, "monthly_changes": self.monthly_changes,
                "category_cuts": self.category_cuts, "cancel": self.cancel}


@dataclass
class Forecast:
    dates: pd.DatetimeIndex
    start_balance: float
    paths: np.ndarray  # (n_sims, H) balance
    savings: np.ndarray  # (n_sims, H) cumulative new savings (balance change + invested)
    outflows: np.ndarray  # (n_sims, H) money going out each day (positive)
    goals: list[dict]
    notes: list[str]
    bills: list[dict] = field(default_factory=list)

    def pct(self, q: float) -> np.ndarray:
        return np.percentile(self.paths, q, axis=0)

    @property
    def prob_negative(self) -> float:
        return float((self.paths.min(axis=1) < 0).mean())

    def balance_on(self, day: pd.Timestamp) -> np.ndarray:
        idx = int(np.clip((day - self.dates[0]).days, 0, len(self.dates) - 1))
        return self.paths[:, idx]

    def month_ends(self) -> list[dict]:
        out = []
        frame = pd.Series(range(len(self.dates)), index=self.dates)
        for month, idxs in frame.groupby(self.dates.to_period("M")):
            i = int(idxs.iloc[-1])
            if self.dates[i].day < self.dates[i].days_in_month and i != len(self.dates) - 1:
                continue
            col = self.paths[:, i]
            out.append({"month": str(month), "date": self.dates[i].date().isoformat(),
                        "p10": round(float(np.percentile(col, 10))), "p50": round(float(np.median(col))),
                        "p90": round(float(np.percentile(col, 90)))})
        return out

    def summary(self) -> dict:
        p10, p50, p90 = self.pct(10), self.pct(50), self.pct(90)
        return {
            "from": self.dates[0].date().isoformat(),
            "to": self.dates[-1].date().isoformat(),
            "start_balance": round(self.start_balance),
            "end_balance_p10": round(float(p10[-1])),
            "end_balance_p50": round(float(p50[-1])),
            "end_balance_p90": round(float(p90[-1])),
            "min_balance_p10": round(float(p10.min())),
            "min_balance_p50": round(float(p50.min())),
            "min_balance_p50_date": self.dates[int(p50.argmin())].date().isoformat(),
            "prob_negative_pct": round(self.prob_negative * 100, 1),
            "month_ends": self.month_ends(),
            "goals": self.goals,
            "bills": self.bills,
        }

    def series(self, step: int = 1) -> dict:
        return {"dates": [d.date().isoformat() for d in self.dates[::step]],
                "p10": [round(float(v)) for v in self.pct(10)[::step]],
                "p50": [round(float(v)) for v in self.pct(50)[::step]],
                "p90": [round(float(v)) for v in self.pct(90)[::step]]}


def has_enough_history(df: pd.DataFrame) -> bool:
    """True when the data spans enough days for a meaningful bootstrap forecast."""
    if df.empty:
        return False
    return (df["date"].max() - df["date"].min()).days + 1 >= MIN_HISTORY_DAYS


def forecast(df: pd.DataFrame, analysis: Analysis, start_balance: float, horizon_days: int | None = None,
             n_sims: int | None = None, seed: int = 7, scenario: Scenario | None = None,
             goals: list[dict] | None = None, bills: list[dict] | None = None,
             lookback_days: int = LOOKBACK_DAYS, spread: float | None = None) -> Forecast:
    scenario = scenario or Scenario()
    goals = goals or []
    n_sims = n_sims or config.FORECAST_SIMULATIONS
    start = analysis.as_of.normalize() + pd.Timedelta(days=1)
    horizon = horizon_days or config.FORECAST_HORIZON_DAYS
    for g in goals:
        horizon = max(horizon, (pd.Timestamp(g["target_date"]) - start).days + 1)
    for p in scenario.purchases:
        horizon = max(horizon, (pd.Timestamp(p["date"]) - start).days + 60)
    horizon = min(horizon, MAX_HORIZON_DAYS)
    dates = pd.date_range(start, periods=horizon, freq="D")
    end = dates[-1]
    notes: list[str] = []

    scheduled = np.zeros(horizon)
    invested = np.zeros(horizon)
    cancelled = {c.upper() for c in scenario.cancel}

    def place(day: pd.Timestamp, amount: float, is_investment: bool = False) -> None:
        i = max((day - start).days, 0)
        if i < horizon:
            scheduled[i] += amount
            if is_investment:
                invested[i] += -amount

    for r in analysis.recurring:
        if r.merchant.upper() in cancelled:
            notes.append(f"cancelled {r.merchant} ({inr(-r.amount)} every {r.period_months} month(s))")
            continue
        k = 1
        while (day := r.last_date + pd.DateOffset(months=r.period_months * k)) <= end:
            place(day, r.amount, r.category == "investments")
            k += 1

    for ch in scenario.monthly_changes:
        first = pd.Timestamp(ch.get("start") or start)
        k = 0
        while (day := first + pd.DateOffset(months=k)) <= end:
            place(day, float(ch["amount"]))
            k += 1
    for p in scenario.purchases:
        place(pd.Timestamp(p["date"]), -abs(float(p["amount"])))
    for bill in bills or []:
        for day in bill_dates(bill, start, end):
            place(day, -abs(float(bill["amount"])))

    blocks, inv_blocks, out_blocks = _weekly_blocks(df, analysis, scenario.category_cuts, lookback_days)
    blocks = _inflate(blocks, SPREAD_INFLATION if spread is None else spread)
    rng = np.random.default_rng(seed)
    weeks = math.ceil(horizon / 7)
    pick = rng.integers(0, len(blocks), size=(n_sims, weeks))
    daily = blocks[pick].reshape(n_sims, -1)[:, :horizon]
    daily_inv = inv_blocks[pick].reshape(n_sims, -1)[:, :horizon]
    daily_out = out_blocks[pick].reshape(n_sims, -1)[:, :horizon] + np.clip(-scheduled, 0, None)

    paths = start_balance + np.cumsum(scheduled + daily, axis=1)
    savings = (paths - start_balance) + np.cumsum(invested + daily_inv, axis=1)
    fc = Forecast(dates, float(start_balance), paths, savings, daily_out, [], notes)
    fc.goals = [_goal_status(fc, g) for g in goals]
    fc.bills = [_bill_status(b, start, end) for b in bills or []]
    return fc


def bill_dates(bill: dict, start: pd.Timestamp, end: pd.Timestamp) -> list[pd.Timestamp]:
    """Due dates of a user-declared bill (e.g. yearly insurance) between start and end, inclusive."""
    every = max(int(bill.get("every_months") or 12), 1)
    day = pd.Timestamp(bill["next_date"]).normalize()
    while day < start:
        day += pd.DateOffset(months=every)
    out = []
    while day <= end:
        out.append(day)
        day += pd.DateOffset(months=every)
    return out


def _inflate(blocks: np.ndarray, factor: float) -> np.ndarray:
    """Widen each week's deviation from the average week by `factor` (1.0 = raw bootstrap)."""
    if factor == 1.0 or len(blocks) < 2:
        return blocks
    mean_week = blocks.mean(axis=0, keepdims=True)
    return mean_week + factor * (blocks - mean_week)


def _weekly_blocks(df: pd.DataFrame, analysis: Analysis, cuts: dict[str, float],
                   lookback_days: int = LOOKBACK_DAYS) -> tuple[np.ndarray, ...]:
    lo = max(analysis.as_of.normalize() - pd.Timedelta(days=lookback_days - 1), df["date"].min().normalize())
    hist = df[(df["date"] >= lo) & ~df.index.isin(analysis.recurring_ids) & ~df.index.isin(analysis.outlier_ids)].copy()
    for cat, cut in cuts.items():
        mask = (hist["category"] == cat) & (hist["amount"] < 0)
        hist.loc[mask, "amount"] *= 1 - float(np.clip(cut, 0, 1))
    days = pd.date_range(lo, analysis.as_of.normalize(), freq="D")
    flow = hist.groupby(hist["date"].dt.normalize())["amount"].sum().reindex(days, fill_value=0.0)
    inv = (hist[hist["category"] == "investments"].groupby(hist["date"].dt.normalize())["amount"].sum()
           .reindex(days, fill_value=0.0) * -1)
    out = (hist[hist["amount"] < 0].groupby(hist["date"].dt.normalize())["amount"].sum()
           .reindex(days, fill_value=0.0) * -1)
    n = max(len(days) // 7, 1)
    tail = slice(len(days) - n * 7, len(days))
    return tuple(x.to_numpy()[tail].reshape(n, 7) for x in (flow, inv, out))


GOAL_BEHIND_BELOW_PCT = 60.0  # probability of hitting the goal below which it is "behind"
GOAL_AHEAD_FROM_PCT = 85.0  # ...and from which it is "ahead"
GOAL_TREND_POINTS = 10.0  # a change this large since the last run is worth telling the user about


def goal_status_label(probability_pct: float) -> str:
    if probability_pct < GOAL_BEHIND_BELOW_PCT:
        return "behind"
    return "ahead" if probability_pct >= GOAL_AHEAD_FROM_PCT else "on_track"


def _goal_status(fc: Forecast, goal: dict) -> dict:
    """Where a savings goal stands, and how that changed since the previous run (if known)."""
    target_date = pd.Timestamp(goal["target_date"])
    idx = int(np.clip((target_date - fc.dates[0]).days, 0, len(fc.dates) - 1))
    at = fc.savings[:, idx]
    p50 = float(np.median(at))
    target = float(goal["target_amount"])
    months_left = max((target_date - fc.dates[0]).days / 30.44, 0.5)
    median_path = np.median(fc.savings, axis=0)
    hit = np.nonzero(median_path >= target)[0]
    probability = round(float((at >= target).mean()) * 100, 1)
    out = {
        "name": goal["name"], "target_amount": round(target), "target_date": target_date.date().isoformat(),
        "projected_p50": round(p50), "probability_pct": probability,
        "status": goal_status_label(probability),
        "shortfall": round(max(target - p50, 0.0)),
        "surplus": round(max(p50 - target, 0.0)),
        "required_extra_monthly": round(max(target - p50, 0.0) / months_left),
        "on_track_date": fc.dates[int(hit[0])].date().isoformat() if len(hit) else None,
    }
    previous = goal.get("previous_probability_pct")
    if previous is not None:
        out["previous_probability_pct"] = round(float(previous), 1)
        out["change_pct_points"] = round(probability - float(previous), 1)
    return out


def _bill_status(bill: dict, start: pd.Timestamp, end: pd.Timestamp) -> dict:
    """Next due date of a declared bill and how much to set aside each month until then."""
    due = bill_dates(bill, start, start + pd.DateOffset(years=2))[0]
    days_until = (due - start).days
    months_until = max(math.ceil(days_until / 30.44), 1)
    amount = abs(float(bill["amount"]))
    return {
        "name": bill["name"], "amount": round(amount), "every_months": int(bill.get("every_months") or 12),
        "due_date": due.date().isoformat(), "days_until": days_until, "months_until": months_until,
        "set_aside_monthly": round(amount / months_until), "in_forecast": bool(due <= end),
    }


def _slug(text: str) -> str:
    return re.sub(r"[^a-z0-9]+", "_", text.lower()).strip("_")


def forecast_findings(fc: Forecast, analysis: Analysis) -> list[Finding]:
    s = fc.summary()
    out = [Finding(
        "trajectory:balance", "trajectory", "Where your balance is heading",
        f"If current behaviour continues your balance goes from {inr(s['start_balance'])} to about "
        f"{inr(s['end_balance_p50'])} by {s['to']} (likely range {inr(s['end_balance_p10'])} to {inr(s['end_balance_p90'])}).",
        "info", {k: s[k] for k in ("start_balance", "end_balance_p10", "end_balance_p50", "end_balance_p90", "to")},
    )]
    buffer = analysis.stats.get("fixed_monthly_costs", 0)
    if s["prob_negative_pct"] > 10 or s["min_balance_p50"] < buffer:
        out.append(Finding(
            "cash_shortfall:balance", "cash_shortfall", "Balance could run low",
            f"Your balance is projected to dip to {inr(s['min_balance_p50'])} around {s['min_balance_p50_date']}, "
            f"below one month of fixed costs ({inr(buffer)}); {s['prob_negative_pct']}% chance of going negative.",
            "alert" if s["prob_negative_pct"] > 10 else "warn",
            {"min_balance_p50": s["min_balance_p50"], "date": s["min_balance_p50_date"],
             "fixed_monthly_costs": buffer, "prob_negative_pct": s["prob_negative_pct"]}))
    for g in fc.goals:
        out.extend(_goal_findings(g))
    for b in fc.bills:
        if b["days_until"] <= 366:
            out.append(Finding(
                f"upcoming_bill:{_slug(b['name'])}", "upcoming_bill", f"{b['name']} is due on {b['due_date']}",
                f"{b['name']} ({inr(b['amount'])}, every {b['every_months']} months) is due on {b['due_date']}. "
                f"Setting aside {inr(b['set_aside_monthly'])} a month covers it"
                + (" — it is already in your forecast." if b["in_forecast"] else "."),
                "warn" if b["days_until"] <= 60 else "info", b))
    return out


def _goal_findings(g: dict) -> list[Finding]:
    """The goal finding changes with the user's progress: behind, ahead, or a notable trend."""
    slug = _slug(g["name"])
    change = g.get("change_pct_points")
    trend = ""
    if change is not None and abs(change) >= GOAL_TREND_POINTS:
        trend = (f" Your chance was {g['previous_probability_pct']}% at the last check "
                 f"({'up' if change > 0 else 'down'} {abs(change)} points).")
    if g["status"] == "behind":
        falling = change is not None and change <= -GOAL_TREND_POINTS
        return [Finding(
            f"goal_at_risk:{slug}", "goal_at_risk", f"Goal '{g['name']}' is at risk",
            f"At the current pace you will have saved about {inr(g['projected_p50'])} of {inr(g['target_amount'])} "
            f"by {g['target_date']} ({g['probability_pct']}% chance). Closing the gap needs about "
            f"{inr(g['required_extra_monthly'])} more per month." + trend,
            "alert" if g["probability_pct"] < 30 or falling else "warn", g,
            monthly_impact=g["required_extra_monthly"])]
    if g["status"] == "ahead":
        return [Finding(
            f"goal_ahead:{slug}", "goal_ahead", f"Goal '{g['name']}' is ahead of plan",
            f"You are on course for about {inr(g['projected_p50'])} against {inr(g['target_amount'])} by "
            f"{g['target_date']} ({g['probability_pct']}% chance)"
            + (f", reaching the target around {g['on_track_date']}." if g["on_track_date"] else ".") + trend,
            "info", g)]
    if trend:
        return [Finding(f"goal_progress:{slug}", "goal_progress", f"Goal '{g['name']}' is on track",
                        f"You have a {g['probability_pct']}% chance of reaching {inr(g['target_amount'])} by "
                        f"{g['target_date']}." + trend, "info", g)]
    return []


GOAL_HIT_POINTS = 15.0  # a decision that cuts a goal's probability by this much "delays the goal"
MARGINAL_RISK_POINTS = 2.0  # extra chance of going negative that still counts as "not this decision's fault"


def _verdict(summary: dict, buffer: float) -> tuple[str, list[str]]:
    """Affordability of one forecast: can the balance absorb it with a month of fixed costs to spare?"""
    if summary["min_balance_p10"] >= buffer:
        return "affordable", [f"Even in a weak month (P10) your balance stays above {inr(summary['min_balance_p10'])}, "
                              f"more than one month of fixed costs ({inr(buffer)})."]
    if summary["min_balance_p50"] >= 0 and summary["prob_negative_pct"] <= 10:
        return "tight", [f"Your balance would dip to about {inr(summary['min_balance_p50'])} around "
                         f"{summary['min_balance_p50_date']}, below your safety buffer of one month's fixed costs "
                         f"({inr(buffer)})."]
    return "not_affordable", [f"There is a {summary['prob_negative_pct']}% chance your balance goes negative; the typical "
                              f"low point is {inr(summary['min_balance_p50'])} around {summary['min_balance_p50_date']}."]


def simulate(df: pd.DataFrame, analysis: Analysis, start_balance: float, scenario: Scenario,
             goals: list[dict] | None = None, seed: int = 7, audit: AuditTrail | None = None,
             bills: list[dict] | None = None) -> dict:
    """Compare a what-if scenario against the baseline on identical random draws."""
    base = forecast(df, analysis, start_balance, seed=seed, goals=goals, bills=bills, scenario=Scenario(
        purchases=[{"date": p["date"], "amount": 0} for p in scenario.purchases]))  # same horizon
    alt = forecast(df, analysis, start_balance, seed=seed, goals=goals, bills=bills, scenario=scenario)
    b, a = base.summary(), alt.summary()
    buffer = float(analysis.stats.get("fixed_monthly_costs", 0))
    goal_hit = []
    for gb, ga in zip(b["goals"], a["goals"]):
        drop = gb["probability_pct"] - ga["probability_pct"]
        if drop >= GOAL_HIT_POINTS:
            goal_hit.append({"goal": ga["name"], "probability_before_pct": gb["probability_pct"],
                             "probability_after_pct": ga["probability_pct"]})
    verdict, reasons = _verdict(a, buffer)
    added_risk = a["prob_negative_pct"] - b["prob_negative_pct"]
    if verdict == "not_affordable" and _verdict(b, buffer)[0] == "not_affordable" and added_risk <= MARGINAL_RISK_POINTS \
            and a["min_balance_p50"] >= 0:
        # The balance is at risk with or without this decision; the decision itself is not what breaks it.
        verdict = "tight"
        reasons = [f"Your balance is already at risk before this decision ({b['prob_negative_pct']}% chance of going "
                   f"negative); this changes that by {max(added_risk, 0):.1f} points. The typical low point is "
                   f"{inr(a['min_balance_p50'])} around {a['min_balance_p50_date']}."]
    if goal_hit and verdict == "affordable":
        verdict = "affordable_but_delays_goal"
    for g in goal_hit:
        reasons.append(f"Chance of reaching '{g['goal']}' falls from {g['probability_before_pct']}% to "
                       f"{g['probability_after_pct']}%.")
    result = {
        "scenario": scenario.describe(),
        "verdict": verdict,
        "reasons": reasons,
        "buffer": round(buffer),
        "baseline": b,
        "with_decision": a,
        "delta_end_balance_p50": round(float(np.median(alt.paths[:, -1]) - np.median(base.paths[:, -1]))),
        "goal_impact": goal_hit,
        "series": {"baseline": base.series(), "with_decision": alt.series()},
        "notes": alt.notes,
    }
    if audit:
        audit.log(AGENT, "simulate", f"{scenario.label}: {verdict}", " ".join(reasons),
                  {"scenario": scenario.describe(), "delta_end_balance_p50": result["delta_end_balance_p50"]})
    return result
