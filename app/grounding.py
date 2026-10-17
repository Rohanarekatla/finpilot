"""Number grounding: every figure an LLM writes must come from computed data.

Used by the critic (recommendations), the analyzer (narrative) and the
coordinator (answers). Text is checked for money amounts and percentages; each
must match a number in the evidence (or a simple derivation of one: a sum of
two, x12 annualised, /12 monthly) within a small tolerance.
"""
from __future__ import annotations

import re
from itertools import combinations
from typing import Any

_NUM = re.compile(
    r"(?P<cur>₹|\brs\.?|\binr)?\s*(?P<num>\d[\d,]*(?:\.\d+)?)\s*(?P<unit>%|k\b|lakhs?\b|lacs?\b|l\b)?",
    re.I,
)


def numbers_in(text: str) -> list[tuple[float, bool, str]]:
    """(value, is_percent, raw) for each number that could be a claim."""
    out = []
    for m in _NUM.finditer(text):
        raw = m.group(0).strip()
        try:
            v = float(m.group("num").replace(",", ""))
        except ValueError:
            continue
        unit = (m.group("unit") or "").lower()
        has_cur = bool(m.group("cur"))
        if unit == "k":
            v *= 1_000
        elif unit.startswith(("lakh", "lac", "l")):
            v *= 100_000
        is_pct = unit == "%"
        # Days, counts, months, years and list markers are not money claims.
        if not is_pct and not has_cur and not unit and (v < 32 or 1990 <= v <= 2100):
            continue
        out.append((v, is_pct, raw))
    return out


def collect_numbers(obj: Any) -> list[float]:
    found: list[float] = []
    if isinstance(obj, bool):
        return found
    if isinstance(obj, (int, float)):
        found.append(float(obj))
    elif isinstance(obj, str):
        found.extend(v for v, _, _ in numbers_in(obj))
    elif isinstance(obj, dict):
        for v in obj.values():
            found.extend(collect_numbers(v))
    elif isinstance(obj, (list, tuple, set)):
        for v in obj:
            found.extend(collect_numbers(v))
    return found


def allowed_numbers(evidence: Any, derive: bool = True) -> set[float]:
    base = {round(abs(v), 2) for v in collect_numbers(evidence)}
    if not derive:
        return base
    allowed = set(base)
    for v in base:
        allowed.update({round(v * 12, 2), round(v / 12, 2), round(v * 100, 2)})
    money = sorted(v for v in base if v >= 50)[:60]
    for a, b in combinations(money, 2):
        allowed.update({round(a + b, 2), round(abs(a - b), 2)})
    return allowed


def ungrounded(text: str, allowed: set[float]) -> list[str]:
    bad = []
    for v, is_pct, raw in numbers_in(text):
        if is_pct:
            ok = any(abs(v - a) <= 1.5 for a in allowed)
        else:
            ok = any(abs(v - a) <= max(2.0, 0.02 * a) for a in allowed)
            if not ok and re.search(r"k\b|lakh|lac|l\b", raw, re.I):  # "12k" is a rounded figure
                ok = any(abs(v - a) <= 0.05 * a for a in allowed)
        if not ok:
            bad.append(raw)
    return bad


def inr(v: float) -> str:
    """Indian digit grouping: 1234567 -> ₹12,34,567."""
    neg, v = v < 0, abs(round(v))
    s = str(int(v))
    if len(s) > 3:
        head, tail = s[:-3], s[-3:]
        head = ",".join([head[max(i - 2, 0):i] for i in range(len(head), 0, -2)][::-1])
        s = f"{head},{tail}"
    return f"{'-' if neg else ''}₹{s}"
