"""CSV ingestion: normalise common bank-statement layouts into one schema."""
from __future__ import annotations

import io

import pandas as pd

from app.taxonomy import merchant_key

# Columns that must never reach the pipeline (ground truth in synthetic files).
LABEL_COLUMNS = {"true_category", "label", "category_label"}

_ALIASES = {
    "date": ["date", "txn date", "transaction date", "value date", "posting date"],
    "description": ["description", "narration", "details", "particulars", "remarks"],
    "amount": ["amount", "amount (inr)", "txn amount"],
    "debit": ["debit", "withdrawal", "withdrawal amt.", "dr"],
    "credit": ["credit", "deposit", "deposit amt.", "cr"],
    "balance": ["balance", "closing balance", "running balance"],
}


def _find(cols: dict[str, str], key: str) -> str | None:
    for alias in _ALIASES[key]:
        if alias in cols:
            return cols[alias]
    return None


def parse_csv(data: bytes | str) -> pd.DataFrame:
    raw = pd.read_csv(io.StringIO(data.decode() if isinstance(data, bytes) else data))
    raw = raw.drop(columns=[c for c in raw.columns if c.strip().lower() in LABEL_COLUMNS])
    cols = {c.strip().lower(): c for c in raw.columns}
    date_col, desc_col = _find(cols, "date"), _find(cols, "description")
    if not date_col or not desc_col:
        raise ValueError("CSV needs a date column and a description/narration column")

    amount_col = _find(cols, "amount")
    if amount_col:
        amount = pd.to_numeric(raw[amount_col].astype(str).str.replace(",", ""), errors="coerce")
    else:
        debit, credit = _find(cols, "debit"), _find(cols, "credit")
        if not debit and not credit:
            raise ValueError("CSV needs an amount column or debit/credit columns")
        num = lambda c: pd.to_numeric(raw[c].astype(str).str.replace(",", ""), errors="coerce").fillna(0) if c else 0
        amount = num(credit) - num(debit)

    bal_col = _find(cols, "balance")
    df = pd.DataFrame({
        "date": pd.to_datetime(raw[date_col], errors="coerce", format="mixed", dayfirst=False),
        "description": raw[desc_col].astype(str).str.strip(),
        "amount": amount,
        "balance": pd.to_numeric(raw[bal_col].astype(str).str.replace(",", ""), errors="coerce") if bal_col else None,
    })
    df = df.dropna(subset=["date", "amount"])
    df = df[df["amount"] != 0].sort_values("date", kind="stable").reset_index(drop=True)
    df["merchant"] = df["description"].map(merchant_key)
    return df
