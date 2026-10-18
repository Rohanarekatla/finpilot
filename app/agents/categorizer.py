"""Categorizer agent: rules -> LLM (via Mesh) -> keyword heuristics.

Rules resolve the narrations they are sure about. Everything else is grouped
by (merchant, direction) so the LLM sees each unknown merchant once, with a
sample narration, its typical amount and how often it appears; the answer is
applied to every transaction of that merchant. If the LLM is unavailable or
returns an invalid category, generic keyword heuristics take over.

Merchant memory: once the LLM has classified a merchant it is never asked
again. The caller passes the memory in (service.py loads it from SQLite and
saves new entries, so it survives restarts); without one, a process-level
dict is used. Batches of unknown merchants are sent to Mesh in parallel.
"""
from __future__ import annotations

from collections.abc import MutableMapping
from concurrent.futures import ThreadPoolExecutor

import pandas as pd

from app import config
from app.audit import AuditTrail
from app.llm import MeshLLM, get_llm
from app.taxonomy import CATEGORIES, heuristic_category, rule_category

AGENT = "categorizer"
BATCH = 40

# key "MERCHANT|debit" -> (category, confidence, reason)
MerchantMemory = MutableMapping[str, tuple[str, float, str]]
_PROCESS_MEMORY: dict[str, tuple[str, float, str]] = {}

SYSTEM = f"""You are the Categorizer agent of FinPilot, a personal-finance system for Indian users.
You receive merchants from bank statements that keyword rules could not classify. Each has a
sample narration (UPI / POS / ECOM / NEFT format), whether money went out (debit) or came in
(credit), a typical amount in INR and how many times it appears.
Pick exactly one category per merchant from this taxonomy:
{chr(10).join(f"- {k}: {v}" for k, v in CATEGORIES.items())}
Guidance: use your knowledge of Indian brands and shop names (e.g. a 'kirana' or 'provision store'
is groceries, a 'tiffins' place is dining). A UPI payment to a person's name with no business
words is 'transfers'. Credits from individuals are 'transfers'. If genuinely unsure use 'other'
with low confidence.
Return JSON: {{"items": [{{"key": <key>, "category": <category>, "confidence": <0-1>, "reason": <short>}}]}}"""


def categorize(df: pd.DataFrame, mode: str = "auto", audit: AuditTrail | None = None,
               llm: MeshLLM | None = None, memory: MerchantMemory | None = None) -> pd.DataFrame:
    """mode: 'rules' (rules only, rest 'other'), 'heuristic' (rules + keywords),
    'auto' (rules + LLM, heuristics as fallback)."""
    llm = llm or get_llm()
    memory = _PROCESS_MEMORY if memory is None else memory
    out = df.copy()
    out["category"] = out["description"].map(rule_category)
    out["category_source"] = out["category"].notna().map({True: "rule", False: None})
    out["category_confidence"] = out["category"].notna().map({True: 0.95, False: 0.0})
    pending = out["category"].isna()
    if audit:
        audit.log(AGENT, "rules", f"{int((~pending).sum())} of {len(out)} transactions matched rules",
                  "Deterministic narration patterns for known brands and bank conventions")

    if pending.any() and mode == "auto" and llm.available:
        _llm_tier(out, pending, audit, llm, memory)
        pending = out["category"].isna()

    if pending.any():
        if mode == "rules":
            out.loc[pending, ["category", "category_source", "category_confidence"]] = ["other", "none", 0.0]
        else:
            guesses = [heuristic_category(d, a) for d, a in zip(out.loc[pending, "description"], out.loc[pending, "amount"])]
            out.loc[pending, "category"] = [g[0] for g in guesses]
            out.loc[pending, "category_source"] = "heuristic"
            out.loc[pending, "category_confidence"] = [g[1] for g in guesses]
            if audit:
                audit.log(AGENT, "heuristic", f"{int(pending.sum())} transactions categorised by keyword fallback",
                          "LLM unavailable or returned no valid category for these merchants")
    out["category_confidence"] = out["category_confidence"].astype(float)
    return out


def _llm_tier(out: pd.DataFrame, pending: pd.Series, audit: AuditTrail | None, llm: MeshLLM,
              memory: MerchantMemory) -> None:
    sub = out[pending].assign(direction=lambda d: d["amount"].gt(0).map({True: "credit", False: "debit"}))
    groups = []
    for (merchant, direction), g in sub.groupby(["merchant", "direction"], sort=False):
        groups.append({
            "key": f"{merchant}|{direction}",
            "sample_narration": g["description"].iloc[0],
            "direction": direction,
            "typical_amount": round(float(g["amount"].abs().median())),
            "count": int(len(g)),
        })
    decided = {g["key"]: tuple(memory[g["key"]]) for g in groups if g["key"] in memory}
    ask = [g for g in groups if g["key"] not in memory]
    if audit and decided:
        audit.log(AGENT, "memory", f"{len(decided)} merchants resolved from merchant memory",
                  "Previously classified by the LLM; not sent to Mesh again")
    batches = [ask[i:i + BATCH] for i in range(0, len(ask), BATCH)]
    if batches:
        workers = max(1, min(config.LLM_MAX_WORKERS, len(batches)))
        with ThreadPoolExecutor(max_workers=workers) as pool:
            responses = list(pool.map(lambda b: llm.chat_json(AGENT, SYSTEM, {"merchants": b}), batches))
        asked = {g["key"] for g in ask}
        for resp in responses:
            for item in (resp or {}).get("items", []):
                key, cat = item.get("key"), str(item.get("category", "")).strip().lower()
                if cat in CATEGORIES and key in asked:  # ignore invented keys and categories
                    try:
                        conf = float(item.get("confidence", 0.7) or 0.7)
                    except (TypeError, ValueError):
                        conf = 0.7
                    decided[key] = (cat, min(max(conf, 0.0), 1.0), str(item.get("reason", ""))[:200])
                    memory[key] = decided[key]

    keys = out["merchant"] + "|" + out["amount"].gt(0).map({True: "credit", False: "debit"})
    for key, (cat, conf, reason) in decided.items():
        mask = pending & (keys == key)
        out.loc[mask, ["category", "category_source", "category_confidence"]] = [cat, "llm", conf]
        if audit:
            audit.log(AGENT, "llm", f"{key.split('|')[0]} -> {cat}", reason or "LLM classification",
                      {"transactions": int(mask.sum()), "confidence": conf})
    if audit:
        audit.log(AGENT, "llm_summary", f"LLM categorised {len(decided)} of {len(groups)} unknown merchants",
                  "Long-tail merchants resolved in batches via Mesh API")
