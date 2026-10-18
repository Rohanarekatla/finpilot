"""Shared test setup: always offline, never start the scheduler, never touch the real database."""
from __future__ import annotations

import os

os.environ["FINPILOT_OFFLINE"] = "1"
os.environ["FINPILOT_SCHEDULER"] = "0"

import pandas as pd  # noqa: E402
import pytest  # noqa: E402

from app.synthetic import generate  # noqa: E402
from app.taxonomy import merchant_key  # noqa: E402


@pytest.fixture(autouse=True)
def isolated_db(tmp_path, monkeypatch):
    from app import config, db
    monkeypatch.setattr(config, "DB_PATH", tmp_path / "test.db")
    db.init_db()
    yield


def _statement(persona: str = "professional", seed: int = 0) -> tuple[pd.DataFrame, float]:
    s = generate(persona, seed=seed)
    df = s.transactions.drop(columns=["true_category"]).copy()
    df["merchant"] = df["description"].map(merchant_key)
    return df, float(df["balance"].iloc[-1])


@pytest.fixture
def statement():
    """Factory: statement(persona, seed) -> (transactions as the pipeline sees them, closing balance)."""
    return _statement
