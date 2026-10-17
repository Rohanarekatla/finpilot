"""Critic agent: verifies every recommendation before the user sees it.

Checks are deterministic so they cannot be talked around:
  1. evidence  - cites at least one finding id, and every cited id exists;
  2. grounding - every rupee figure / percentage appears in the cited evidence
                 (or is a simple derivation of it);
  3. specific  - names the merchant, category, subscription or goal involved;
  4. impact    - claimed savings do not exceed what the cited findings support;
  5. distinct  - does not repeat an earlier recommendation's evidence.
Failures go back to the advisor with the reasons (see graph.py).
"""
from __future__ import annotations

from app.agents.analyzer import Finding
from app.audit import AuditTrail
from app.grounding import allowed_numbers, ungrounded
from app.taxonomy import LABELS

AGENT = "critic"

_KIND_WORDS = {
    "low_savings_rate": ["saving"],
    "cash_shortfall": ["balance"],
    "trajectory": ["balance"],
}


def _entities(f: Finding) -> set[str]:
    d = f.data
    out = set(_KIND_WORDS.get(f.kind, []))
    for key in ("merchant", "name"):
        if d.get(key):
            out.add(str(d[key]))
    if d.get("category"):
        out.update({d["category"], LABELS.get(d["category"], d["category"])})
    if d.get("group"):
        out.add(d["group"].replace("_", " "))
    for item in d.get("services", []) + d.get("top_merchants", []) + d.get("items", []):
        out.add(item["merchant"])
    # Merchant keys can be long ("ACT FIBERNET BROADBAND"); the brand words are enough.
    for e in [e for e in out if e.isupper()]:
        words = e.split()
        out.add(" ".join(words[:2]))
        if len(words[0]) > 3:
            out.add(words[0])
    return {e.lower().replace("_", " ") for e in out if e}


def critique(recs: list[dict], findings: dict[str, Finding], candidates: list[dict], stats: dict,
             audit: AuditTrail | None = None) -> list[dict]:
    by_finding = {c["finding_ids"][0]: c for c in candidates}
    verdicts, seen = [], []
    for i, r in enumerate(recs):
        issues = []
        fids = r.get("finding_ids") or []
        unknown = [fid for fid in fids if fid not in findings]
        cited = [findings[fid] for fid in fids if fid in findings]
        if not fids:
            issues.append("cites no finding; every recommendation must point to the evidence behind it")
        if unknown:
            issues.append(f"cites unknown finding ids {unknown}")
        if cited:
            # Cited evidence may be combined (sums, x12); general stats may only be quoted as-is.
            evidence = [[f.data, f.title, f.detail] for f in cited] + [by_finding.get(f.id, {}) for f in cited]
            allowed = allowed_numbers(evidence) | allowed_numbers(stats, derive=False)
            text = f"{r['title']} {r['action']} {r['why']}"
            bad = ungrounded(text, allowed)
            if bad:
                issues.append(f"figures {bad} do not appear in the cited evidence")
            entities = set().union(*(_entities(f) for f in cited))
            low = text.lower().replace("-", " ").replace("_", " ")
            if entities and not any(e in low for e in entities):
                issues.append(f"too generic: mentions none of {sorted(entities)[:6]}")
            cap_monthly = sum(float(by_finding.get(f.id, {}).get("monthly_impact", f.monthly_impact)) for f in cited)
            cap_once = sum(float(by_finding.get(f.id, {}).get("one_off_impact", f.one_off_impact)) for f in cited)
            if r.get("monthly_impact", 0) > cap_monthly * 1.05 + 1:
                issues.append(f"monthly_impact {r['monthly_impact']:.0f} exceeds the {cap_monthly:.0f} the evidence supports")
            if r.get("one_off_impact", 0) > cap_once * 1.05 + 1:
                issues.append(f"one_off_impact {r['one_off_impact']:.0f} exceeds the {cap_once:.0f} the evidence supports")
        key = tuple(sorted(fids))
        if key and key in seen:
            issues.append("repeats the evidence of an earlier recommendation")
        passed = not issues
        if passed:
            seen.append(key)
        verdicts.append({"index": i, "title": r["title"], "passed": passed, "issues": issues})
        if audit:
            audit.log(AGENT, "check", f"{'PASS' if passed else 'REJECT'}: {r['title']}",
                      "; ".join(issues) or "evidence cited, figures grounded, specific, impact within bounds",
                      {"finding_ids": fids, "source": r.get("source")})
    return verdicts
