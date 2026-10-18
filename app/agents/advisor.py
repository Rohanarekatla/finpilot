"""Advisor agent: turns findings + forecast into specific, grounded actions.

The system first computes candidate actions with exact impacts from each
finding (so the numbers are right by construction). The LLM then ranks them,
merges related ones and writes them for this person, citing finding ids. The
critic checks every recommendation; failing ones come back here with the
critic's feedback for a revision. Without an LLM, the templated candidates
are used directly.
"""
from __future__ import annotations

from app import config
from app.agents.analyzer import Analysis, Finding
from app.audit import AuditTrail
from app.grounding import inr
from app.llm import MeshLLM, get_llm
from app.taxonomy import LABELS

AGENT = "advisor"
MAX_RECS = 5
SEVERITY_BONUS = {"alert": 6000, "warn": 1500, "info": 0}

SYSTEM = """You are the Advisor agent of FinPilot, a proactive personal-finance assistant for Indian users.
You receive: the user's stats, a forecast summary, findings (each with an id and the numbers behind it)
and candidate actions the system computed from those findings (with exact rupee impacts).
Write up to 5 recommendations, most valuable first. Rules:
- Each recommendation cites the finding ids it is based on in "finding_ids" (at least one).
- Name the specific merchant, category, subscription or goal involved.
- Use only rupee figures and percentages that appear in the input. You may add two figures or
  multiply a monthly figure by 12; nothing else. Write amounts like ₹1,499.
- "action" is something concrete the user can do this week. "why" is one sentence pointing at the
  pattern or forecast that triggered it.
- No generic tips ("save more", "make a budget", "track your expenses").
- monthly_impact / one_off_impact must not exceed the candidate impacts of the cited findings.
Return JSON: {"recommendations": [{"title": str, "action": str, "why": str, "finding_ids": [str],
"monthly_impact": number, "one_off_impact": number}]}"""

REVISION = """Your previous recommendations were checked by the Critic. Fix every listed issue (or drop that
recommendation), keep the ones that passed unchanged, and return the full corrected list in the same JSON format."""


def build_candidates(analysis: Analysis, extra: list[Finding]) -> list[dict]:
    """One concrete, correctly-quantified action per actionable finding."""
    findings = analysis.findings + extra
    stats = analysis.stats
    cands: list[dict] = []
    savings_monthly = 0.0

    for f in findings:
        d, c = f.data, None
        if f.kind == "overlapping_subscriptions":
            services = sorted(d["services"], key=lambda s: -s["monthly"])
            keep, drop = services[0], services[1:]
            c = dict(title=f"Cancel overlapping {d['group'].replace('_', ' ')} subscriptions",
                     action=f"Keep {keep['merchant']} and cancel {', '.join(s['merchant'] for s in drop)} — saves "
                            f"{inr(d['saving_if_keep_one'])}/month ({inr(d['saving_if_keep_one'] * 12)}/year).",
                     why=f"You pay {inr(d['total_monthly'])}/month for {len(services)} services that do the same job.",
                     monthly_impact=d["saving_if_keep_one"])
        elif f.kind == "price_hike":
            c = dict(title=f"{d['merchant']} raised its price",
                     action=f"Switch {d['merchant']} to a cheaper plan or cancel it — it went from {inr(d['old_amount'])} "
                            f"to {inr(d['new_amount'])}.",
                     why=f"The increase alone costs you {inr(d['increase_monthly'])} more every month.",
                     monthly_impact=d["increase_monthly"])
        elif f.kind == "category_spike":
            label = LABELS[d["category"]]
            top = d.get("top_merchants") or []
            where = f" Most of it went to {top[0]['merchant']} ({inr(top[0]['amount'])})." if top else ""
            c = dict(title=f"Bring {label} back to normal",
                     action=f"Cap {label} at your usual {inr(d['baseline'])} next month.{where}",
                     why=f"{label.capitalize()} hit {inr(d['current'])} this month, {d['pct_change']:.0f}% above your "
                         f"average — {inr(d['increase'])} extra.",
                     monthly_impact=d["increase"])
        elif f.kind == "duplicate_charge":
            c = dict(title=f"Dispute duplicate {d['merchant']} charge",
                     action=f"Raise a dispute with {d['merchant']} or your bank for the repeated {inr(d['amount'])} "
                            f"charge on {d['date']}.",
                     why="Two identical charges from the same merchant on the same day are usually a payment error.",
                     one_off_impact=d["amount"])
        elif f.kind == "large_transaction":
            c = dict(title=f"Review the {inr(d['amount'])} payment at {d['merchant']}",
                     action=f"Confirm the {inr(d['amount'])} {LABELS[d['category']]} payment at {d['merchant']} on "
                            f"{d['date']} was planned; if it was on a card, consider a no-cost EMI instead of paying it at once.",
                     why=f"It is {d['pct_of_income']:.0f}% of a month's income and is excluded from your forecast as a one-off.")
        elif f.kind == "low_savings_rate":
            target = round(0.2 * d["avg_income"] - (d["avg_income"] - d["avg_spend"]))
            d["transfer_to_reach_20pct"] = target
            c = dict(title="Automate savings on salary day",
                     action=f"Set up an automatic transfer of {inr(target)} to savings on salary day to reach a 20% savings rate.",
                     why=f"You kept {d['savings_rate_pct']}% of income over recent months "
                         f"(spending {inr(d['avg_spend'])} of {inr(d['avg_income'])}).",
                     monthly_impact=target)
        elif f.kind == "cash_shortfall":
            c = dict(title="Protect your balance this month",
                     action=f"Hold back non-essential purchases until after {d['date']} and keep at least "
                            f"{inr(d['fixed_monthly_costs'])} available for fixed costs.",
                     why=f"Your balance is projected to fall to {inr(d['min_balance_p50'])} around {d['date']}.")
        elif f.kind == "goal_at_risk":
            c = dict(title=f"Get '{d['name']}' back on track",
                     action=f"Put an extra {inr(d['required_extra_monthly'])}/month towards '{d['name']}' "
                            f"to reach {inr(d['target_amount'])} by {d['target_date']}.",
                     why=f"At the current pace you reach about {inr(d['projected_p50'])} by then "
                         f"({d['probability_pct']}% chance of hitting the goal).",
                     monthly_impact=d["required_extra_monthly"])
        elif f.kind == "goal_ahead":
            when = (f"around {d['on_track_date']}, before {d['target_date']}" if d.get("on_track_date")
                    else f"by {d['target_date']}")
            c = dict(title=f"'{d['name']}' is ahead of plan",
                     action=f"Keep your current pace — you are projected to reach '{d['name']}' {when} with about "
                            f"{inr(d['surplus'])} to spare. Raise the target or start your next goal with the extra.",
                     why=f"You have a {d['probability_pct']}% chance of reaching {inr(d['target_amount'])} by "
                         f"{d['target_date']}.")
        elif f.kind == "upcoming_bill":
            c = dict(title=f"Set aside money for {d['name']}",
                     action=f"Move {inr(d['set_aside_monthly'])} a month into savings for {d['name']} so the "
                            f"{inr(d['amount'])} due on {d['due_date']} does not hit your balance all at once.",
                     why=f"It comes every {d['every_months']} months, so your monthly pattern does not show it"
                         + ("; it is already in your forecast." if d["in_forecast"] else "."))
        if c:
            c.setdefault("monthly_impact", 0.0)
            c.setdefault("one_off_impact", 0.0)
            c["finding_ids"] = [f.id]
            c["priority"] = c["monthly_impact"] * 12 + c["one_off_impact"] + SEVERITY_BONUS[f.severity]
            if f.kind in ("overlapping_subscriptions", "price_hike", "category_spike"):
                savings_monthly += c["monthly_impact"]
            cands.append(c)

    # A goal gap is easier to act on when tied to the cuts that would close it.
    for c in cands:
        if c["finding_ids"][0].startswith("goal_at_risk") and savings_monthly > 0:
            c["why"] += f" The cuts suggested here free up {inr(savings_monthly)}/month towards it."
            c["freed_by_cuts_monthly"] = round(savings_monthly)
    return sorted(cands, key=lambda c: -c["priority"])


def advise(analysis: Analysis, extra_findings: list[Finding], forecast_summary: dict, candidates: list[dict],
           feedback: list[dict] | None = None, previous: list[dict] | None = None,
           audit: AuditTrail | None = None, llm: MeshLLM | None = None) -> list[dict]:
    llm = llm or get_llm()
    payload = {
        "stats": analysis.stats,
        "forecast": {k: v for k, v in forecast_summary.items() if k != "month_ends"},
        "findings": [f.brief() for f in analysis.findings + extra_findings],
        "candidates": [{k: v for k, v in c.items() if k != "priority"} for c in candidates],
    }
    system = SYSTEM
    if feedback:
        system = SYSTEM + "\n\n" + REVISION
        payload["previous_recommendations"] = previous
        payload["critic_feedback"] = feedback
    resp = llm.chat_json(AGENT, system, payload, model=config.MODEL_SMART)
    recs = [_clean(r) for r in (resp or {}).get("recommendations", []) if isinstance(r, dict)]
    recs = [r for r in recs if r][:MAX_RECS]
    if recs:
        for r in recs:
            r["source"] = "llm"
        if audit:
            audit.log(AGENT, "revise" if feedback else "draft", f"{len(recs)} recommendations drafted",
                      "LLM ranked and personalised the computed candidate actions",
                      [{"title": r["title"], "finding_ids": r["finding_ids"]} for r in recs])
        return recs
    recs = templates(candidates)
    if audit:
        audit.log(AGENT, "template", f"{len(recs)} recommendations from templates",
                  "LLM unavailable or returned nothing usable; using computed candidate actions verbatim")
    return recs


def templates(candidates: list[dict], limit: int = MAX_RECS) -> list[dict]:
    return [{"title": c["title"], "action": c["action"], "why": c["why"], "finding_ids": list(c["finding_ids"]),
             "monthly_impact": float(c["monthly_impact"]), "one_off_impact": float(c["one_off_impact"]),
             "source": "template"} for c in candidates[:limit]]


def _clean(r: dict) -> dict | None:
    try:
        return {"title": str(r["title"]).strip(), "action": str(r["action"]).strip(), "why": str(r.get("why", "")).strip(),
                "finding_ids": [str(x) for x in r.get("finding_ids") or []],
                "monthly_impact": float(r.get("monthly_impact") or 0), "one_off_impact": float(r.get("one_off_impact") or 0)}
    except (KeyError, TypeError, ValueError):
        return None

