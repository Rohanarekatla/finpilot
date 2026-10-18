"""SQLite persistence: transactions, goals, bills, runs, findings, recommendations, audit trail.

New tables are created with CREATE TABLE IF NOT EXISTS, so an existing
database picks them up on the next start without a migration step.
"""
from __future__ import annotations

import json
import sqlite3
from contextlib import contextmanager
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterator

import pandas as pd

from app import config

SCHEMA = """
CREATE TABLE IF NOT EXISTS users (
    user_id TEXT PRIMARY KEY,
    created_at TEXT NOT NULL,
    opening_balance REAL NOT NULL DEFAULT 0
);
CREATE TABLE IF NOT EXISTS transactions (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    user_id TEXT NOT NULL,
    date TEXT NOT NULL,
    description TEXT NOT NULL,
    amount REAL NOT NULL,
    balance REAL,
    merchant TEXT NOT NULL,
    category TEXT,
    category_source TEXT,
    category_confidence REAL,
    UNIQUE (user_id, date, description, amount)
);
CREATE INDEX IF NOT EXISTS ix_txn_user_date ON transactions (user_id, date);
CREATE TABLE IF NOT EXISTS goals (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    user_id TEXT NOT NULL,
    name TEXT NOT NULL,
    target_amount REAL NOT NULL,
    target_date TEXT NOT NULL,
    created_at TEXT NOT NULL
);
CREATE TABLE IF NOT EXISTS runs (
    run_id TEXT PRIMARY KEY,
    user_id TEXT NOT NULL,
    kind TEXT NOT NULL,
    created_at TEXT NOT NULL,
    summary TEXT
);
CREATE TABLE IF NOT EXISTS findings (
    run_id TEXT NOT NULL,
    finding_id TEXT NOT NULL,
    user_id TEXT NOT NULL,
    kind TEXT NOT NULL,
    title TEXT NOT NULL,
    detail TEXT NOT NULL,
    severity TEXT NOT NULL,
    data TEXT NOT NULL,
    PRIMARY KEY (run_id, finding_id)
);
CREATE TABLE IF NOT EXISTS recommendations (
    rec_id TEXT PRIMARY KEY,
    run_id TEXT NOT NULL,
    user_id TEXT NOT NULL,
    title TEXT NOT NULL,
    action TEXT NOT NULL,
    why TEXT NOT NULL,
    finding_ids TEXT NOT NULL,
    monthly_impact REAL NOT NULL DEFAULT 0,
    one_off_impact REAL NOT NULL DEFAULT 0,
    source TEXT NOT NULL,
    created_at TEXT NOT NULL
);
CREATE TABLE IF NOT EXISTS decisions (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    run_id TEXT NOT NULL,
    user_id TEXT NOT NULL,
    ts TEXT NOT NULL,
    agent TEXT NOT NULL,
    step TEXT NOT NULL,
    decision TEXT NOT NULL,
    reason TEXT NOT NULL,
    data TEXT
);
CREATE INDEX IF NOT EXISTS ix_decisions_run ON decisions (run_id);
CREATE TABLE IF NOT EXISTS bills (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    user_id TEXT NOT NULL,
    name TEXT NOT NULL,
    amount REAL NOT NULL,
    next_date TEXT NOT NULL,
    every_months INTEGER NOT NULL DEFAULT 12,
    created_at TEXT NOT NULL
);
CREATE TABLE IF NOT EXISTS goal_snapshots (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    goal_id INTEGER NOT NULL,
    user_id TEXT NOT NULL,
    run_id TEXT NOT NULL,
    created_at TEXT NOT NULL,
    probability_pct REAL NOT NULL,
    projected_p50 REAL NOT NULL,
    status TEXT NOT NULL
);
CREATE INDEX IF NOT EXISTS ix_goal_snapshots_goal ON goal_snapshots (goal_id, id);
CREATE TABLE IF NOT EXISTS merchant_memory (
    key TEXT PRIMARY KEY,
    category TEXT NOT NULL,
    confidence REAL NOT NULL,
    reason TEXT NOT NULL,
    updated_at TEXT NOT NULL
);
CREATE TABLE IF NOT EXISTS digests (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    user_id TEXT NOT NULL,
    run_id TEXT NOT NULL,
    created_at TEXT NOT NULL,
    channel TEXT NOT NULL,
    content TEXT NOT NULL
);
"""


def now_iso() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


@contextmanager
def connect(path: Path | None = None) -> Iterator[sqlite3.Connection]:
    conn = sqlite3.connect(path or config.DB_PATH)
    conn.row_factory = sqlite3.Row
    try:
        yield conn
        conn.commit()
    finally:
        conn.close()


def init_db(path: Path | None = None) -> None:
    with connect(path) as conn:
        conn.executescript(SCHEMA)


def ensure_user(user_id: str, opening_balance: float | None = None) -> None:
    with connect() as conn:
        conn.execute(
            "INSERT OR IGNORE INTO users (user_id, created_at, opening_balance) VALUES (?, ?, ?)",
            (user_id, now_iso(), opening_balance or 0.0),
        )
        if opening_balance is not None:
            conn.execute("UPDATE users SET opening_balance = ? WHERE user_id = ?", (opening_balance, user_id))


def list_users() -> list[str]:
    with connect() as conn:
        return [r["user_id"] for r in conn.execute("SELECT user_id FROM users ORDER BY user_id")]


def insert_transactions(user_id: str, df: pd.DataFrame) -> int:
    rows = [
        (user_id, r.date.strftime("%Y-%m-%d"), r.description, float(r.amount),
         None if pd.isna(r.balance) else float(r.balance), r.merchant)
        for r in df.itertuples()
    ]
    with connect() as conn:
        before = conn.total_changes
        conn.executemany(
            "INSERT OR IGNORE INTO transactions (user_id, date, description, amount, balance, merchant) "
            "VALUES (?, ?, ?, ?, ?, ?)",
            rows,
        )
        return conn.total_changes - before


def load_transactions(user_id: str) -> pd.DataFrame:
    with connect() as conn:
        df = pd.read_sql_query(
            "SELECT * FROM transactions WHERE user_id = ? ORDER BY date, id", conn, params=(user_id,)
        )
    df["date"] = pd.to_datetime(df["date"])
    return df


def save_categories(df: pd.DataFrame) -> None:
    with connect() as conn:
        conn.executemany(
            "UPDATE transactions SET category = ?, category_source = ?, category_confidence = ? WHERE id = ?",
            [(r.category, r.category_source, float(r.category_confidence), int(r.id)) for r in df.itertuples()],
        )


def current_balance(user_id: str, df: pd.DataFrame) -> float:
    if len(df) and df["balance"].notna().iloc[-1]:
        return float(df["balance"].iloc[-1])
    with connect() as conn:
        row = conn.execute("SELECT opening_balance FROM users WHERE user_id = ?", (user_id,)).fetchone()
    return float(row["opening_balance"] if row else 0.0) + float(df["amount"].sum())


def add_goal(user_id: str, name: str, target_amount: float, target_date: str) -> int:
    with connect() as conn:
        cur = conn.execute(
            "INSERT INTO goals (user_id, name, target_amount, target_date, created_at) VALUES (?, ?, ?, ?, ?)",
            (user_id, name, target_amount, target_date, now_iso()),
        )
        return int(cur.lastrowid)


def list_goals(user_id: str) -> list[dict]:
    with connect() as conn:
        return [dict(r) for r in conn.execute("SELECT * FROM goals WHERE user_id = ? ORDER BY id", (user_id,))]


def delete_goal(user_id: str, goal_id: int) -> bool:
    with connect() as conn:
        conn.execute("DELETE FROM goal_snapshots WHERE user_id = ? AND goal_id = ?", (user_id, goal_id))
        return conn.execute("DELETE FROM goals WHERE user_id = ? AND id = ?", (user_id, goal_id)).rowcount > 0


def latest_goal_snapshots(user_id: str) -> dict[int, dict]:
    """The most recent stored status of each goal, keyed by goal id."""
    with connect() as conn:
        rows = conn.execute(
            "SELECT s.* FROM goal_snapshots s JOIN (SELECT goal_id, MAX(id) AS id FROM goal_snapshots "
            "WHERE user_id = ? GROUP BY goal_id) last ON s.id = last.id", (user_id,)).fetchall()
    return {int(r["goal_id"]): dict(r) for r in rows}


def save_goal_snapshots(run_id: str, user_id: str, goals: list[dict], statuses: list[dict]) -> None:
    """`statuses` is the forecaster's goal output, in the same order as `goals`."""
    with connect() as conn:
        conn.executemany(
            "INSERT INTO goal_snapshots (goal_id, user_id, run_id, created_at, probability_pct, projected_p50, status) "
            "VALUES (?, ?, ?, ?, ?, ?, ?)",
            [(g["id"], user_id, run_id, now_iso(), s["probability_pct"], s["projected_p50"], s["status"])
             for g, s in zip(goals, statuses)],
        )


def goal_history(user_id: str, goal_id: int, limit: int = 20) -> list[dict]:
    with connect() as conn:
        rows = conn.execute(
            "SELECT created_at, probability_pct, projected_p50, status FROM goal_snapshots "
            "WHERE user_id = ? AND goal_id = ? ORDER BY id DESC LIMIT ?", (user_id, goal_id, limit)).fetchall()
    return [dict(r) for r in reversed(rows)]


def add_bill(user_id: str, name: str, amount: float, next_date: str, every_months: int = 12) -> int:
    with connect() as conn:
        cur = conn.execute(
            "INSERT INTO bills (user_id, name, amount, next_date, every_months, created_at) VALUES (?, ?, ?, ?, ?, ?)",
            (user_id, name, abs(amount), next_date, every_months, now_iso()),
        )
        return int(cur.lastrowid)


def list_bills(user_id: str) -> list[dict]:
    with connect() as conn:
        return [dict(r) for r in conn.execute("SELECT * FROM bills WHERE user_id = ? ORDER BY next_date", (user_id,))]


def delete_bill(user_id: str, bill_id: int) -> bool:
    with connect() as conn:
        return conn.execute("DELETE FROM bills WHERE user_id = ? AND id = ?", (user_id, bill_id)).rowcount > 0


def load_merchant_memory() -> dict[str, tuple[str, float, str]]:
    with connect() as conn:
        rows = conn.execute("SELECT key, category, confidence, reason FROM merchant_memory").fetchall()
    return {r["key"]: (r["category"], float(r["confidence"]), r["reason"]) for r in rows}


def save_merchant_memory(entries: dict[str, tuple[str, float, str]]) -> None:
    if not entries:
        return
    with connect() as conn:
        conn.executemany(
            "INSERT OR REPLACE INTO merchant_memory (key, category, confidence, reason, updated_at) "
            "VALUES (?, ?, ?, ?, ?)",
            [(k, cat, conf, reason, now_iso()) for k, (cat, conf, reason) in entries.items()],
        )


def save_run(run_id: str, user_id: str, kind: str, summary: dict) -> None:
    with connect() as conn:
        conn.execute(
            "INSERT OR REPLACE INTO runs (run_id, user_id, kind, created_at, summary) VALUES (?, ?, ?, ?, ?)",
            (run_id, user_id, kind, now_iso(), json.dumps(summary, default=str)),
        )


def latest_run(user_id: str, kind: str = "pipeline") -> dict | None:
    with connect() as conn:
        row = conn.execute(
            "SELECT * FROM runs WHERE user_id = ? AND kind = ? ORDER BY created_at DESC, rowid DESC LIMIT 1",
            (user_id, kind),
        ).fetchone()
    if not row:
        return None
    out = dict(row)
    out["summary"] = json.loads(out["summary"] or "{}")
    return out


def save_findings(run_id: str, user_id: str, findings: list) -> None:
    with connect() as conn:
        conn.executemany(
            "INSERT OR REPLACE INTO findings VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
            [(run_id, f.id, user_id, f.kind, f.title, f.detail, f.severity, json.dumps(f.data, default=str))
             for f in findings],
        )


def load_findings(run_id: str) -> list[dict]:
    with connect() as conn:
        rows = conn.execute("SELECT * FROM findings WHERE run_id = ?", (run_id,)).fetchall()
    return [{**dict(r), "data": json.loads(r["data"])} for r in rows]


def save_recommendations(run_id: str, user_id: str, recs: list[dict]) -> None:
    with connect() as conn:
        conn.executemany(
            "INSERT OR REPLACE INTO recommendations VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
            [(r["rec_id"], run_id, user_id, r["title"], r["action"], r["why"], json.dumps(r["finding_ids"]),
              r.get("monthly_impact", 0.0), r.get("one_off_impact", 0.0), r["source"], now_iso()) for r in recs],
        )


def load_recommendations(run_id: str) -> list[dict]:
    with connect() as conn:
        rows = conn.execute("SELECT * FROM recommendations WHERE run_id = ? ORDER BY rec_id", (run_id,)).fetchall()
    return [{**dict(r), "finding_ids": json.loads(r["finding_ids"])} for r in rows]


def get_recommendation(rec_id: str) -> dict | None:
    with connect() as conn:
        row = conn.execute("SELECT * FROM recommendations WHERE rec_id = ?", (rec_id,)).fetchone()
    return {**dict(row), "finding_ids": json.loads(row["finding_ids"])} if row else None


def save_decisions(rows: list[tuple]) -> None:
    with connect() as conn:
        conn.executemany(
            "INSERT INTO decisions (run_id, user_id, ts, agent, step, decision, reason, data) "
            "VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
            rows,
        )


def load_decisions(run_id: str) -> list[dict]:
    with connect() as conn:
        rows = conn.execute("SELECT * FROM decisions WHERE run_id = ? ORDER BY id", (run_id,)).fetchall()
    return [{**dict(r), "data": json.loads(r["data"]) if r["data"] else None} for r in rows]


def save_digest(user_id: str, run_id: str, channel: str, content: str) -> None:
    with connect() as conn:
        conn.execute(
            "INSERT INTO digests (user_id, run_id, created_at, channel, content) VALUES (?, ?, ?, ?, ?)",
            (user_id, run_id, now_iso(), channel, content),
        )


def list_digests(user_id: str) -> list[dict]:
    with connect() as conn:
        return [dict(r) for r in conn.execute(
            "SELECT * FROM digests WHERE user_id = ? ORDER BY id DESC", (user_id,))]
