"""FastAPI app: REST API + a single-page dashboard.

    uvicorn app.main:app --reload        (or: python main.py)
"""
from __future__ import annotations

import logging
from contextlib import asynccontextmanager
from datetime import date

from fastapi import FastAPI, File, Form, HTTPException, Request, UploadFile
from fastapi.responses import HTMLResponse, PlainTextResponse
from fastapi.templating import Jinja2Templates
from pydantic import BaseModel, Field

from app import config, db, digest, service
from app.agents.forecaster import Scenario
from app.graph import mermaid
from app.llm import get_llm
from app.synthetic import PERSONAS
from app.taxonomy import CATEGORIES

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(name)s %(levelname)s %(message)s")
templates = Jinja2Templates(directory=str(config.ROOT / "app" / "templates"))


@asynccontextmanager
async def lifespan(_: FastAPI):
    db.init_db()
    sched = digest.start_scheduler() if config.SCHEDULER_ENABLED else None
    yield
    if sched:
        sched.shutdown(wait=False)


app = FastAPI(title="FinPilot", description="Proactive multi-agent financial assistant", lifespan=lifespan)


# --- request bodies -------------------------------------------------------------------------------

class GoalIn(BaseModel):
    name: str = Field(min_length=1, max_length=80)
    target_amount: float = Field(gt=0)
    target_date: date


class BillIn(BaseModel):
    name: str = Field(min_length=1, max_length=80)
    amount: float = Field(gt=0)
    next_date: date
    every_months: int = Field(default=12, ge=1, le=60)


class Question(BaseModel):
    question: str = Field(min_length=3, max_length=500)


class PurchaseIn(BaseModel):
    date: date
    amount: float = Field(gt=0)
    label: str = "purchase"


class MonthlyChangeIn(BaseModel):
    amount: float = Field(description="Negative for a new monthly cost, positive for new monthly income")
    label: str = "change"
    start: date | None = None


class ScenarioIn(BaseModel):
    label: str = "what-if"
    purchases: list[PurchaseIn] = []
    monthly_changes: list[MonthlyChangeIn] = []
    category_cuts: dict[str, float] = Field(default={}, description="category -> fraction to cut, e.g. 0.3")
    cancel: list[str] = Field(default=[], description="recurring merchant names to cancel")

    def to_scenario(self) -> Scenario:
        unknown = sorted(set(self.category_cuts) - set(CATEGORIES))
        if unknown:
            raise HTTPException(400, f"Unknown categories {unknown}; choose from {list(CATEGORIES)}")
        return Scenario(
            label=self.label,
            purchases=[{"date": p.date.isoformat(), "amount": p.amount, "label": p.label} for p in self.purchases],
            monthly_changes=[{"amount": c.amount, "label": c.label, "start": c.start.isoformat() if c.start else None}
                             for c in self.monthly_changes],
            category_cuts={k: min(max(v, 0.0), 1.0) for k, v in self.category_cuts.items()},
            cancel=self.cancel,
        )


def _guard(fn, *args):
    """Map service errors to HTTP errors: missing data -> 404, invalid request -> 400."""
    try:
        return fn(*args)
    except LookupError as exc:
        raise HTTPException(404, str(exc)) from exc
    except ValueError as exc:
        raise HTTPException(400, str(exc)) from exc


# --- pages and meta -------------------------------------------------------------------------------

@app.get("/", response_class=HTMLResponse)
def index(request: Request):
    return templates.TemplateResponse(request, "index.html", {"personas": list(PERSONAS)})


@app.get("/api/health")
def health(deep: bool = False):
    """Liveness plus configuration. `?deep=true` also verifies the Mesh key and model ids (no credits spent)."""
    llm = get_llm()
    out = {"ok": True, "llm": "mesh" if llm.available else "offline-fallback", "llm_blocked": llm.blocked_reason,
           "model_fast": config.MODEL_FAST,
           "model_smart": config.MODEL_SMART, "scheduler": config.SCHEDULER_ENABLED,
           "digest_cron": config.DIGEST_CRON if config.SCHEDULER_ENABLED else None}
    if deep:
        out["mesh_check"] = llm.check()
    return out


@app.get("/api/graph", response_class=PlainTextResponse)
def graph():
    """The LangGraph pipeline and ask graph as Mermaid, generated from the compiled graphs."""
    return mermaid()


# --- data in --------------------------------------------------------------------------------------

@app.post("/api/users/{user_id}/upload")
async def upload(user_id: str, file: UploadFile = File(...), opening_balance: float | None = Form(None)):
    added = _guard(service.import_csv, user_id, await file.read(), opening_balance)
    return {"user_id": user_id, "imported": added}


@app.post("/api/users/{user_id}/sample/{persona}")
def sample(user_id: str, persona: str, seed: int = 0):
    if persona not in PERSONAS:
        raise HTTPException(404, f"Unknown persona; choose from {list(PERSONAS)}")
    return {"user_id": user_id, "imported": service.load_sample(user_id, persona, seed)}


@app.get("/api/users/{user_id}/transactions")
def transactions(user_id: str, limit: int = 200):
    df = db.load_transactions(user_id).tail(limit)
    df["date"] = df["date"].dt.strftime("%Y-%m-%d")
    return df.drop(columns=["user_id"]).to_dict(orient="records")


# --- agents ---------------------------------------------------------------------------------------

@app.post("/api/users/{user_id}/analyze")
def analyze(user_id: str):
    return _guard(service.analyze_user, user_id)


@app.get("/api/users/{user_id}/latest")
def latest(user_id: str):
    run = db.latest_run(user_id)
    if not run:
        raise HTTPException(404, "No analysis yet")
    return {"run_id": run["run_id"], "created_at": run["created_at"], **run["summary"],
            "findings": db.load_findings(run["run_id"]), "recommendations": db.load_recommendations(run["run_id"])}


@app.post("/api/users/{user_id}/ask")
def ask(user_id: str, q: Question):
    return _guard(service.ask, user_id, q.question)


@app.post("/api/users/{user_id}/simulate")
def simulate(user_id: str, sc: ScenarioIn):
    return _guard(service.simulate, user_id, sc.to_scenario())


# --- goals and bills ------------------------------------------------------------------------------

@app.post("/api/users/{user_id}/goals")
def add_goal(user_id: str, goal: GoalIn):
    db.ensure_user(user_id)
    return {"id": db.add_goal(user_id, goal.name, goal.target_amount, goal.target_date.isoformat())}


@app.get("/api/users/{user_id}/goals")
def goals(user_id: str):
    return service.goals_overview(user_id)


@app.delete("/api/users/{user_id}/goals/{goal_id}")
def delete_goal(user_id: str, goal_id: int):
    if not db.delete_goal(user_id, goal_id):
        raise HTTPException(404, "Unknown goal")
    return {"deleted": goal_id}


@app.post("/api/users/{user_id}/bills")
def add_bill(user_id: str, bill: BillIn):
    db.ensure_user(user_id)
    return {"id": db.add_bill(user_id, bill.name, bill.amount, bill.next_date.isoformat(), bill.every_months)}


@app.get("/api/users/{user_id}/bills")
def bills(user_id: str):
    return db.list_bills(user_id)


@app.delete("/api/users/{user_id}/bills/{bill_id}")
def delete_bill(user_id: str, bill_id: int):
    if not db.delete_bill(user_id, bill_id):
        raise HTTPException(404, "Unknown bill")
    return {"deleted": bill_id}


# --- audit and digest -----------------------------------------------------------------------------

@app.get("/api/runs/{run_id}/audit")
def audit(run_id: str):
    return {"run_id": run_id, "decisions": db.load_decisions(run_id), "findings": db.load_findings(run_id),
            "recommendations": db.load_recommendations(run_id)}


@app.get("/api/recommendations/{rec_id}/why")
def why(rec_id: str):
    out = service.explain(rec_id)
    if not out:
        raise HTTPException(404, "Unknown recommendation")
    return out


@app.post("/api/users/{user_id}/digest")
def send_digest(user_id: str):
    return _guard(digest.send_digest, user_id)


@app.get("/api/users/{user_id}/digests")
def digests(user_id: str):
    return db.list_digests(user_id)
