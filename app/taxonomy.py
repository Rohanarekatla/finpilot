"""Category taxonomy and the deterministic categorisation tiers.

Tier 1 (RULES) knows the big national brands and bank narration conventions.
Tier 2 (HEURISTICS) is generic keyword matching used only when the LLM is
unavailable. The LLM tier sits between them and handles the long tail
(local shops, brands the rules do not list, person-to-person UPI).
"""
from __future__ import annotations

import re

CATEGORIES: dict[str, str] = {
    "income": "Salary, stipend, business or freelance income",
    "rent": "House rent",
    "emi_loan": "Loan EMIs and repayments",
    "investments": "SIPs, mutual funds, stocks, deposits",
    "insurance": "Life/health/vehicle insurance premiums",
    "groceries": "Supermarkets, kirana, quick commerce, fresh produce",
    "food_delivery": "Food delivery apps",
    "dining": "Restaurants, cafes, eateries",
    "transport": "Cabs, autos, metro, bus",
    "fuel": "Petrol/diesel",
    "utilities": "Electricity, water, gas",
    "mobile_internet": "Mobile recharge, broadband, DTH",
    "subscriptions": "Streaming and digital subscriptions",
    "fitness": "Gym and fitness memberships",
    "shopping": "Online and offline retail, electronics, fashion",
    "entertainment": "Movies, events, gaming",
    "health": "Pharmacy, doctors, diagnostics",
    "education": "Courses, school/college fees",
    "travel": "Flights, trains, hotels, holidays",
    "transfers": "Payments to/from individuals",
    "cash_withdrawal": "ATM withdrawals",
    "fees_charges": "Bank charges and fees",
    "other": "Anything that does not fit",
}

LABELS = {k: k.replace("_", " ") for k in CATEGORIES}
LABELS.update({"emi_loan": "loan EMIs", "mobile_internet": "mobile & internet", "fees_charges": "bank charges"})

# Money that is saved or moved, not consumed.
NON_CONSUMPTION = {"income", "investments"}
ESSENTIAL = {"rent", "emi_loan", "insurance", "groceries", "utilities", "mobile_internet", "health", "education", "fuel", "transport"}
# Categories where a month-over-month jump is a behavioural signal worth flagging.
BEHAVIOURAL = {"food_delivery", "dining", "groceries", "shopping", "entertainment", "transport", "fuel", "travel", "transfers", "cash_withdrawal", "health"}

# Ordered: first match wins. Patterns run on the upper-cased narration.
RULES: list[tuple[str, str]] = [
    (r"SALARY|STIPEND|FELLOWSHIP", "income"),
    (r"\bRENT\b", "rent"),
    (r"\bEMI\b|\bLOAN\b", "emi_loan"),
    (r"\bSIP\b|MUTUAL FUND|ZERODHA|GROWW|BSE STAR MF", "investments"),
    (r"LIC OF INDIA|HDFC ERGO|STAR HEALTH|INSURANCE", "insurance"),
    (r"NETFLIX|SPOTIFY|AMAZON PRIME|HOTSTAR|YOUTUBE PREMIUM|SONYLIV|ZEE5|GOOGLE ONE|ICLOUD", "subscriptions"),
    (r"CULT ?FIT|GOLDS GYM|ANYTIME FITNESS", "fitness"),
    (r"SWIGGY|ZOMATO", "food_delivery"),
    (r"BIGBASKET|DMART|RELIANCE FRESH|ZEPTO|BLINKIT", "groceries"),
    (r"STARBUCKS|MCDONALD|DOMINOS|\bKFC\b|CHAI POINT|THIRD WAVE", "dining"),
    (r"\bUBER\b|\bOLA\b|RAPIDO|\bMETRO\b|\bBMTC\b", "transport"),
    (r"INDIAN OIL|\bIOCL\b|\bHPCL\b|BHARAT PETROLEUM|\bSHELL\b", "fuel"),
    (r"BESCOM|ELECTRICITY|BWSSB|MAHANAGAR GAS", "utilities"),
    (r"AIRTEL|\bJIO\b|ACT FIBERNET|\bBSNL\b", "mobile_internet"),
    (r"AMAZON|FLIPKART|MYNTRA|\bAJIO\b|CROMA|DECATHLON|RELIANCE DIGITAL", "shopping"),
    (r"BOOKMYSHOW|\bPVR\b", "entertainment"),
    (r"APOLLO PHARMACY|PHARMEASY|MEDPLUS", "health"),
    (r"MAKEMYTRIP|GOIBIBO|IRCTC|INDIGO", "travel"),
    (r"COURSERA|UDEMY|SCHOOL FEES", "education"),
    (r"ATM WDL", "cash_withdrawal"),
    (r"CHARGES|ANNUAL FEE", "fees_charges"),
]

HEURISTICS: list[tuple[str, str]] = [
    (r"KIRANA|PROVISION|SUPERMART|GENERAL STORE|VEGETABLE|DAIRY", "groceries"),
    (r"TIFFIN|CAFE|RESTAURANT|DHABA|BIRYANI|BAKERY|SWEETS|BHAVAN", "dining"),
    (r"MEDICAL|PHARMA|CLINIC|DIAGNOSTIC", "health"),
    (r"PETROL|FUEL|FILLING", "fuel"),
    (r"SCHOOL|COLLEGE|ACADEMY|CLASSES", "education"),
]

_RULES = [(re.compile(p), c) for p, c in RULES]
_HEUR = [(re.compile(p), c) for p, c in HEURISTICS]

# Subscription families used to spot overlapping services.
SUBSCRIPTION_GROUPS = {
    "video_streaming": ["NETFLIX", "AMAZON PRIME", "HOTSTAR", "SONYLIV", "ZEE5"],
    "music_streaming": ["SPOTIFY", "YOUTUBE PREMIUM", "GAANA", "WYNK"],
    "cloud_storage": ["GOOGLE ONE", "ICLOUD", "DROPBOX"],
}

_STOP = {
    "UPI", "POS", "ACH", "D", "NEFT", "CR", "IMPS", "ECOM", "PUR", "PAYMENT", "PAY", "TO", "MERCHANT",
    "COLLECT", "LTD", "PVT", "PRIVATE", "LIMITED", "IN", "COM", "INDIA", "WWW",
    "BENGALURU", "BANGALORE", "MUMBAI", "HYDERABAD", "CHENNAI", "PUNE", "DELHI",
    # month stamps ("SALARY SEP") must not split one payer into twelve merchants
    "JAN", "FEB", "MAR", "APR", "MAY", "JUN", "JUL", "AUG", "SEP", "SEPT", "OCT", "NOV", "DEC",
}


def rule_category(description: str) -> str | None:
    d = description.upper()
    for pat, cat in _RULES:
        if pat.search(d):
            return cat
    return None


def heuristic_category(description: str, amount: float) -> tuple[str, float]:
    d = description.upper()
    for pat, cat in _HEUR:
        if pat.search(d):
            return cat, 0.6
    if d.startswith("UPI-"):  # UPI to/from an individual with no business keywords
        return "transfers", 0.45
    return ("transfers", 0.3) if amount > 0 else ("other", 0.3)


def merchant_key(description: str) -> str:
    """Stable merchant identifier: strip channel prefixes, refs, VPAs and cities."""
    d = description.upper()
    tokens = []
    for tok in re.split(r"[^A-Z0-9@.+]+", d):
        tok = tok.strip(".+")
        if not tok or "@" in tok or any(ch.isdigit() for ch in tok) or tok in _STOP:
            continue
        tokens.extend(t for t in tok.split(".") if t and t not in _STOP)
    return " ".join(tokens[:3]) or d[:20]


def subscription_group(key: str) -> str | None:
    for group, names in SUBSCRIPTION_GROUPS.items():
        if any(n in key for n in names):
            return group
    return None
