"""Synthetic Indian bank statements with ground-truth labels.

Each persona is simulated day by day: income and fixed bills land on their
usual dates, subscriptions renew monthly, and discretionary spending is drawn
from Poisson (how often) x lognormal (how much) processes with a weekend
bump. Known patterns are planted in the final month so the analyzer can be
scored against them, and every row carries its true category so the
categorizer can be scored too. The pipeline never sees either label.

    python -m app.synthetic --out data/
"""
from __future__ import annotations

import argparse
import calendar
import json
from dataclasses import dataclass
from datetime import date, timedelta
from pathlib import Path

import numpy as np
import pandas as pd

END_DATE = date(2026, 9, 30)

# Discretionary merchant pools. Rules know the first names in most pools; the
# rest are the long tail (local shops, brands outside the rule list) that the
# LLM tier is meant to handle.
MERCHANTS: dict[str, list[str]] = {
    "food_delivery": ["SWIGGY", "ZOMATO"],
    "dining": ["STARBUCKS", "CHAI POINT", "MCDONALDS", "SRI SAI TIFFINS", "UDUPI GRAND CAFE",
               "HALDIRAMS", "BARBEQUE NATION", "BLUE TOKAI"],
    "groceries": ["BIGBASKET", "DMART", "ZEPTO", "BLINKIT", "SRI LAKSHMI KIRANA",
                  "NEW BALAJI SUPERMART", "NATURES BASKET", "LICIOUS"],
    "transport": ["UBER", "OLA", "RAPIDO", "NAMMA METRO"],
    "fuel": ["INDIAN OIL", "HPCL", "SHELL"],
    "shopping": ["AMAZON", "FLIPKART", "MYNTRA", "DECATHLON", "LENSKART", "NYKAA", "PEPPERFRY"],
    "entertainment": ["BOOKMYSHOW", "PVR INOX", "TIMEZONE"],
    "health": ["APOLLO PHARMACY", "PHARMEASY", "TATA 1MG", "PRACTO", "SRI VENKATESWARA MEDICALS"],
    "travel": ["MAKEMYTRIP", "IRCTC", "REDBUS", "YATRA"],
    "education": ["COURSERA", "UDEMY", "UNACADEMY"],
    "cash_withdrawal": ["ATM"],
    "transfers": ["RAHUL SHARMA", "PRIYA NAIR", "KIRAN RAO", "SNEHA IYER"],
}

HANDLES = ["ybl", "okaxis", "okicici", "oksbi", "paytm", "ibl"]


@dataclass
class Spend:
    category: str
    per_week: float
    median: float
    sigma: float = 0.4


@dataclass
class Fixed:
    narration: str  # may contain {ref} / {mon}
    category: str
    amount: float
    day: int
    period: int = 1  # months
    offset: int = 0  # first month index it occurs in
    jitter: float = 0.0  # relative std of amount


PERSONAS: dict[str, dict] = {
    "student": {
        "opening_balance": 22000,
        "fixed": [
            Fixed("NEFT CR-SBIN0001234-IIT MADRAS-RESEARCH STIPEND {mon}", "income", 31000, 1),
            Fixed("UPI-ANAND KUMAR-anandk@okaxis-{ref}-RENT", "rent", 9000, 5),
            Fixed("ACH D- ZERODHA COIN SIP {ref}", "investments", 1000, 7),
            Fixed("JIO RECHARGE PREPAID {ref}", "mobile_internet", 299, 12),
            Fixed("SPOTIFY INDIA MUMBAI", "subscriptions", 119, 14),
            Fixed("YOUTUBE PREMIUM GOOGLE", "subscriptions", 129, 20),
            Fixed("NETFLIX.COM MUMBAI", "subscriptions", 199, 23),
        ],
        "spend": [
            Spend("food_delivery", 3.5, 290), Spend("dining", 2.5, 160), Spend("groceries", 1.2, 550),
            Spend("transport", 4, 110), Spend("shopping", 0.5, 1100), Spend("entertainment", 0.35, 420),
            Spend("transfers", 0.8, 450), Spend("health", 0.15, 380), Spend("education", 0.1, 1500),
        ],
        "planted": [
            {"kind": "category_spike", "target": "food_delivery", "factor": 1.9},
            {"kind": "duplicate_charge", "target": "SWIGGY"},
            {"kind": "overlapping_subscriptions", "target": "music_streaming"},
        ],
    },
    "professional": {
        "opening_balance": 140000,
        "fixed": [
            Fixed("NEFT CR-HDFC0000240-ACME TECH PVT LTD-SALARY {mon}", "income", 125000, 1),
            Fixed("UPI-SURESH REDDY-sureshreddy@oksbi-{ref}-RENT", "rent", 26000, 5),
            Fixed("ACH D- ZERODHA COIN SIP {ref}", "investments", 10000, 7),
            Fixed("ACH D- HDFC BANK LTD-EMI {ref}", "emi_loan", 12500, 10),
            Fixed("BESCOM ELECTRICITY BILL {ref}", "utilities", 1400, 15, jitter=0.2),
            Fixed("AIRTEL POSTPAID {ref}", "mobile_internet", 599, 18),
            Fixed("CULT FIT MEMBERSHIP", "fitness", 1499, 3),
            Fixed("NETFLIX.COM MUMBAI", "subscriptions", 499, 9),
            Fixed("AMAZON PRIME VIDEO", "subscriptions", 299, 12),
            Fixed("DISNEY+ HOTSTAR", "subscriptions", 299, 21),
            Fixed("SPOTIFY INDIA MUMBAI", "subscriptions", 119, 25),
        ],
        "spend": [
            Spend("food_delivery", 3, 450), Spend("dining", 2, 650), Spend("groceries", 1.5, 1400),
            Spend("transport", 5, 220), Spend("fuel", 0.6, 1800), Spend("shopping", 1, 2200),
            Spend("entertainment", 0.5, 800), Spend("travel", 0.12, 7000), Spend("health", 0.2, 600),
            Spend("transfers", 0.6, 1500), Spend("cash_withdrawal", 0.25, 3000),
        ],
        "planted": [
            {"kind": "category_spike", "target": "food_delivery", "factor": 1.9},
            {"kind": "price_hike", "target": "NETFLIX", "new_amount": 649},
            {"kind": "large_transaction", "target": "CROMA", "amount": 42999, "category": "shopping"},
            {"kind": "overlapping_subscriptions", "target": "video_streaming"},
        ],
    },
    "family": {
        "opening_balance": 260000,
        "fixed": [
            Fixed("NEFT CR-ICIC0000101-GLOBEX INDIA PVT LTD-SALARY {mon}", "income", 210000, 1),
            Fixed("ACH D- SBI HOME LOAN EMI {ref}", "emi_loan", 48000, 5),
            Fixed("ACH D- BSE STAR MF SIP {ref}", "investments", 20000, 7),
            Fixed("BESCOM ELECTRICITY BILL {ref}", "utilities", 3200, 14, jitter=0.25),
            Fixed("BWSSB WATER BILL {ref}", "utilities", 600, 16, jitter=0.15),
            Fixed("ACT FIBERNET BROADBAND {ref}", "mobile_internet", 999, 8),
            Fixed("AIRTEL POSTPAID {ref}", "mobile_internet", 798, 18),
            Fixed("NETFLIX.COM MUMBAI", "subscriptions", 649, 11),
            Fixed("GOOGLE ONE STORAGE", "subscriptions", 130, 22),
            Fixed("DPS SCHOOL FEES {ref}", "education", 42000, 10, period=3),
            Fixed("LIC OF INDIA PREMIUM {ref}", "insurance", 18500, 20, period=3, offset=1),
        ],
        "spend": [
            Spend("groceries", 2.5, 2200), Spend("dining", 1, 1600), Spend("food_delivery", 1.5, 650),
            Spend("transport", 3, 300), Spend("fuel", 1.2, 2500), Spend("shopping", 1.2, 3000),
            Spend("health", 0.5, 900), Spend("entertainment", 0.4, 1500), Spend("travel", 0.15, 12000),
            Spend("transfers", 0.5, 3000), Spend("cash_withdrawal", 0.4, 4000),
        ],
        "planted": [
            {"kind": "category_spike", "target": "shopping", "factor": 2.2},
            {"kind": "price_hike", "target": "ACT FIBERNET", "new_amount": 1199},
            {"kind": "large_transaction", "target": "RELIANCE DIGITAL", "amount": 78000, "category": "shopping"},
            {"kind": "duplicate_charge", "target": "BIGBASKET"},
        ],
    },
}


@dataclass
class Synthetic:
    persona: str
    seed: int
    transactions: pd.DataFrame  # date, description, amount, balance, true_category
    planted: list[dict]
    opening_balance: float


def _month_starts(end: date, months: int) -> list[date]:
    starts, y, m = [], end.year, end.month
    for _ in range(months):
        starts.append(date(y, m, 1))
        y, m = (y, m - 1) if m > 1 else (y - 1, 12)
    return starts[::-1]


def _narration(rng: np.random.Generator, category: str, merchant: str) -> str:
    ref = rng.integers(10**11, 10**12)
    if category == "cash_withdrawal":
        return f"ATM WDL BENGALURU {rng.integers(1000, 9999)}"
    if category == "transfers":
        vpa = merchant.split()[0].lower() + str(rng.integers(10, 99))
        return f"UPI-{merchant}-{vpa}@{rng.choice(HANDLES)}-{ref}-UPI"
    channel = rng.choice(["upi", "pos", "online"], p=[0.6, 0.25, 0.15])
    if channel == "upi":
        vpa = merchant.replace(" ", "").lower()[:10] + str(rng.integers(1, 9))
        return f"UPI-{merchant}-{vpa}@{rng.choice(HANDLES)}-{ref}-PAYMENT"
    if channel == "pos":
        return f"POS {rng.integers(1000, 9999)} {merchant} BENGALURU"
    return f"ECOM PUR {merchant} {ref}"


def generate(persona: str, seed: int = 0, months: int = 7, end: date = END_DATE) -> Synthetic:
    cfg = PERSONAS[persona]
    rng = np.random.default_rng(seed)
    planted = {p["kind"]: p for p in cfg["planted"]}
    rows: list[tuple] = []
    starts = _month_starts(end, months)

    def add(d: date, desc: str, amount: float, cat: str) -> None:
        rows.append((d, desc, round(amount, 2), cat))

    for mi, ms in enumerate(starts):
        last = mi == months - 1
        dim = calendar.monthrange(ms.year, ms.month)[1]
        for f in cfg["fixed"]:
            if mi < f.offset or (mi - f.offset) % f.period:
                continue
            amt = f.amount * (1 + rng.normal(0, f.jitter)) if f.jitter else f.amount
            hike = planted.get("price_hike")
            if last and hike and hike["target"] in f.narration:
                amt = hike["new_amount"]
            desc = f.narration.format(ref=rng.integers(10**8, 10**9), mon=ms.strftime("%b").upper())
            sign = 1 if f.category == "income" else -1
            add(ms.replace(day=min(f.day, dim)), desc, sign * round(amt), f.category)

        for d in range(dim):
            day = ms + timedelta(days=d)
            weekend = 1.4 if day.weekday() >= 5 else 0.85
            for s in cfg["spend"]:
                rate = s.per_week / 7
                if s.category in ("food_delivery", "dining", "entertainment"):
                    rate *= weekend
                spike = planted.get("category_spike")
                if last and spike and spike["target"] == s.category:
                    rate *= spike["factor"]
                for _ in range(rng.poisson(rate)):
                    pool = MERCHANTS[s.category]
                    weights = np.linspace(2.0, 1.0, len(pool))
                    merchant = pool[rng.choice(len(pool), p=weights / weights.sum())]
                    amt = float(np.round(rng.lognormal(np.log(s.median), s.sigma)))
                    if s.category == "transfers" and rng.random() < 0.3:
                        add(day, _narration(rng, s.category, merchant), amt, "transfers")  # friend pays back
                        continue
                    add(day, _narration(rng, s.category, merchant), -amt, s.category)

    last_start = starts[-1]
    if "large_transaction" in planted:
        p = planted["large_transaction"]
        add(last_start.replace(day=18), f"POS {rng.integers(1000, 9999)} {p['target']} BENGALURU",
            -p["amount"], p["category"])
    if "duplicate_charge" in planted:
        p = planted["duplicate_charge"]
        target_cat = "food_delivery" if p["target"] == "SWIGGY" else "groceries"
        day = last_start.replace(day=12)
        amt = 2140.0 if target_cat == "groceries" else 612.0
        for _ in range(2):
            ref = rng.integers(10**11, 10**12)
            add(day, f"UPI-{p['target']}-{p['target'].lower()}@ybl-{ref}-PAYMENT", -amt, target_cat)

    df = pd.DataFrame(rows, columns=["date", "description", "amount", "true_category"])
    df["date"] = pd.to_datetime(df["date"])
    df = df.sort_values("date", kind="stable").reset_index(drop=True)
    df["balance"] = (cfg["opening_balance"] + df["amount"].cumsum()).round(2)
    df = df[["date", "description", "amount", "balance", "true_category"]]
    return Synthetic(persona, seed, df, cfg["planted"], cfg["opening_balance"])


def main() -> None:
    ap = argparse.ArgumentParser(description="Write synthetic statements + planted-pattern labels")
    ap.add_argument("--out", default="data")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    for persona in PERSONAS:
        s = generate(persona, seed=args.seed)
        s.transactions.assign(date=s.transactions["date"].dt.strftime("%Y-%m-%d")).to_csv(
            out / f"{persona}.csv", index=False)
        (out / f"{persona}_planted.json").write_text(json.dumps(s.planted, indent=2))
        print(f"{persona}: {len(s.transactions)} transactions -> {out / f'{persona}.csv'}")


if __name__ == "__main__":
    main()
