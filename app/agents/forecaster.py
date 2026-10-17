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
        }

    def series(self, step: int = 1) -> dict:
        return {"dates": [d.date().isoformat() for d in self.dates[::step]],
                "p10": [round(float(v)) for v in self.pct(10)[::step]],
                "p50": [round(float(v)) for v in self.pct(50)[::step]],
                "p90": [round(float(v)) for v in self.pct(90)[::step]]}


def forecast(df: pd.DataFrame, analysis: Analysis, start_balance: float, horizon_days: int | None = None,
             n_sims: int | None = None, seed: int = 7, scenario: Scenario | None = None,
             goals: list[dict] | None = None) -> Forecast:
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

    blocks, inv_blocks, out_blocks = _weekly_blocks(df, analysis, scenario.category_cuts)
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
    return fc


def _weekly_blocks(df: pd.DataFrame, analysis: Analysis, cuts: dict[str, float]) -> tuple[np.ndarray, ...]:
    lo = max(analysis.as_of.normalize() - pd.Timedelta(days=LOOKBACK_DAYS - 1), df["date"].min().normalize())
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


def _goal_status(fc: Forecast, goal: dict) -> dict:
    target_date = pd.Timestamp(goal["target_date"])
    idx = int(np.clip((target_date - fc.dates[0]).days, 0, len(fc.dates) - 1))
    at = fc.savings[:, idx]
    p50 = float(np.median(at))
    target = float(goal["target_amount"])
    months_left = max((target_date - fc.dates[0]).days / 30.44, 0.5)
    median_path = np.median(fc.savings, axis=0)
    hit = np.nonzero(median_path >= target)[0]
    return {
        "name": goal["name"], "target_amount": round(target), "target_date": target_date.date().isoformat(),
        "projected_p50": round(p50), "probability_pct": round(float((at >= target).mean()) * 100, 1),
        "shortfall": round(max(target - p50, 0.0)),
        "required_extra_monthly": round(max(target - p50, 0.0) / months_left),
        "on_track_date": fc.dates[int(hit[0])].date().isoformat() if len(hit) else None,
    }


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
        if g["probability_pct"] < 60:
            slug = re.sub(r"[^a-z0-9]+", "_", g["name"].lower()).strip("_")
            out.append(Finding(
                f"goal_at_risk:{slug}", "goal_at_risk", f"Goal '{g['name']}' is at risk",
                f"At the current pace you will have saved about {inr(g['projected_p50'])} of {inr(g['target_amount'])} "
                f"by {g['target_date']} ({g['probability_pct']}% chance). Closing the gap needs about "
                f"{inr(g['required_extra_monthly'])} more per month.",
                "alert" if g["probability_pct"] < 30 else "warn", g,
                monthly_impact=g["required_extra_monthly"]))
    return out


def simulate(df: pd.DataFrame, analysis: Analysis, start_balance: float, scenario: Scenario,
             goals: list[dict] | None = None, seed: int = 7, audit: AuditTrail | None = None) -> dict:
    """Compare a what-if scenario against the baseline on identical random draws."""
    base = forecast(df, analysis, start_balance, seed=seed, goals=goals, scenario=Scenario(
        purchases=[{"date": p["date"], "amount": 0} for p in scenario.purchases]))  # same horizon
    alt = forecast(df, analysis, start_balance, seed=seed, goals=goals, scenario=scenario)
    b, a = base.summary(), alt.summary()
    buffer = float(analysis.stats.get("fixed_monthly_costs", 0))
    reasons = []
    goal_hit = []
    for gb, ga in zip(b["goals"], a["goals"]):
        drop = gb["probability_pct"] - ga["probability_pct"]
        if drop >= 15:
            goal_hit.append({"goal": ga["name"], "probability_before_pct": gb["probability_pct"],
                             "probability_after_pct": ga["probability_pct"]})
    if a["min_balance_p10"] >= buffer:
        verdict = "affordable"
        reasons.append(f"Even in a weak month (P10) your balance stays above {inr(a['min_balance_p10'])}, "
                       f"more than one month of fixed costs ({inr(buffer)}).")
    elif a["min_balance_p50"] >= 0 and a["prob_negative_pct"] <= 10:
        verdict = "tight"
        reasons.append(f"Your balance would dip to about {inr(a['min_balance_p50'])} around {a['min_balance_p50_date']}, "
                       f"below your safety buffer of one month's fixed costs ({inr(buffer)}).")
    else:
        verdict = "not_affordable"
        reasons.append(f"There is a {a['prob_negative_pct']}% chance your balance goes negative; the typical low point "
                       f"is {inr(a['min_balance_p50'])} around {a['min_balance_p50_date']}.")
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
