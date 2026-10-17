"""FastAPI app: REST API + a single-page dashboard.

    uvicorn app.main:app --reload
"""
from __future__ import annotations

import logging
from contextlib import asynccontextmanager
from datetime import date

from fastapi import FastAPI, File, Form, HTTPException, Request, UploadFile
from fastapi.responses import HTMLResponse
from fastapi.templating import Jinja2Templates
from pydantic import BaseModel, Field

from app import config, db, digest, service
from app.agents.forecaster import Scenario
from app.llm import get_llm
from app.synthetic import PERSONAS

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


class GoalIn(BaseModel):
    name: str
    target_amount: float = Field(gt=0)
    target_date: date


class Question(BaseModel):
    question: str = Field(min_length=3, max_length=500)


class ScenarioIn(BaseModel):
    label: str = "what-if"
    purchases: list[dict] = []
    monthly_changes: list[dict] = []
    category_cuts: dict[str, float] = {}
    cancel: list[str] = []


def _guard(fn, *args):
    try:
        return fn(*args)
    except LookupError as exc:
        raise HTTPException(404, str(exc)) from exc
    except ValueError as exc:
        raise HTTPException(400, str(exc)) from exc


@app.get("/", response_class=HTMLResponse)
def index(request: Request):
    return templates.TemplateResponse(request, "index.html", {"personas": list(PERSONAS)})


@app.get("/api/health")
def health():
    llm = get_llm()
    return {"ok": True, "llm": "mesh" if llm.available else "offline-fallback", "model_fast": config.MODEL_FAST,
            "model_smart": config.MODEL_SMART}


@app.post("/api/users/{user_id}/upload")
async def upload(user_id: str, file: UploadFile = File(...), opening_balance: float | None = Form(None)):
    added = _guard(service.import_csv, user_id, await file.read(), opening_balance)
    return {"user_id": user_id, "imported": added}


@app.post("/api/users/{user_id}/sample/{persona}")
def sample(user_id: str, persona: str, seed: int = 0):
    if persona not in PERSONAS:
        raise HTTPException(404, f"Unknown persona; choose from {list(PERSONAS)}")
    return {"user_id": user_id, "imported": service.load_sample(user_id, persona, seed)}


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


@app.get("/api/users/{user_id}/transactions")
def transactions(user_id: str, limit: int = 200):
    df = db.load_transactions(user_id).tail(limit)
    df["date"] = df["date"].dt.strftime("%Y-%m-%d")
    return df.drop(columns=["user_id"]).to_dict(orient="records")


@app.post("/api/users/{user_id}/goals")
def add_goal(user_id: str, goal: GoalIn):
    db.ensure_user(user_id)
    gid = db.add_goal(user_id, goal.name, goal.target_amount, goal.target_date.isoformat())
    return {"id": gid}


@app.get("/api/users/{user_id}/goals")
def goals(user_id: str):
    return db.list_goals(user_id)


@app.post("/api/users/{user_id}/ask")
def ask(user_id: str, q: Question):
    return _guard(service.ask, user_id, q.question)


@app.post("/api/users/{user_id}/simulate")
def simulate(user_id: str, sc: ScenarioIn):
    return _guard(service.simulate, user_id, Scenario(**sc.model_dump()))


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
