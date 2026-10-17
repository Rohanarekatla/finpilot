"""Auditable decision trail.

Every agent records what it decided and why. Records are buffered in memory
during a run (so pure pipelines like the eval need no database) and flushed
to the `decisions` table at the end of a persisted run.
"""
from __future__ import annotations

import json
import uuid
from dataclasses import dataclass, field
from typing import Any

from app import db


@dataclass
class Decision:
    agent: str
    step: str
    decision: str
    reason: str
    data: Any = None
    ts: str = field(default_factory=db.now_iso)


class AuditTrail:
    def __init__(self, user_id: str, run_id: str | None = None):
        self.user_id = user_id
        self.run_id = run_id or f"run-{uuid.uuid4().hex[:10]}"
        self.records: list[Decision] = []

    def log(self, agent: str, step: str, decision: str, reason: str, data: Any = None) -> None:
        self.records.append(Decision(agent, step, decision, reason, data))

    def by_agent(self, agent: str) -> list[Decision]:
        return [r for r in self.records if r.agent == agent]

    def flush(self) -> None:
        db.save_decisions([
            (self.run_id, self.user_id, r.ts, r.agent, r.step, r.decision, r.reason,
             json.dumps(r.data, default=str) if r.data is not None else None)
            for r in self.records
        ])
        self.records.clear()
