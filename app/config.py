"""Runtime configuration, read once from the environment (and a local .env)."""
from __future__ import annotations

import os
from pathlib import Path

from dotenv import load_dotenv

ROOT = Path(__file__).resolve().parent.parent
load_dotenv(ROOT / ".env")

# Every LLM call goes through Mesh API (OpenAI-compatible gateway).
MESH_BASE_URL = os.getenv("MESH_BASE_URL", "https://api.meshapi.ai/v1")
MESH_API_KEY = os.getenv("MESH_API_KEY", "")
MODEL_FAST = os.getenv("FINPILOT_MODEL_FAST", "openai/gpt-4o-mini")
MODEL_SMART = os.getenv("FINPILOT_MODEL_SMART", "openai/gpt-4o")

# FINPILOT_OFFLINE=1 forces the deterministic fallbacks (used by tests and CI).
OFFLINE = os.getenv("FINPILOT_OFFLINE", "0") == "1"
# A slow model must never hang a request: time out and fall back to the deterministic path.
LLM_TIMEOUT_S = float(os.getenv("FINPILOT_LLM_TIMEOUT", "30"))
LLM_MAX_RETRIES = int(os.getenv("FINPILOT_LLM_RETRIES", "1"))
LLM_MAX_WORKERS = int(os.getenv("FINPILOT_LLM_WORKERS", "4"))  # parallel categorizer batches

DB_PATH = Path(os.getenv("FINPILOT_DB", str(ROOT / "finpilot.db")))
CURRENCY = "₹"

FORECAST_HORIZON_DAYS = int(os.getenv("FINPILOT_HORIZON_DAYS", "90"))
FORECAST_SIMULATIONS = int(os.getenv("FINPILOT_SIMULATIONS", "500"))
MAX_ADVISOR_REVISIONS = int(os.getenv("FINPILOT_MAX_REVISIONS", "2"))

# Proactive digest delivery (all optional; without them digests are stored + logged).
# Opt-in, so `uvicorn --reload` and tests never start a second scheduler; deployments set it to 1.
SCHEDULER_ENABLED = os.getenv("FINPILOT_SCHEDULER", "0") == "1"
DIGEST_CRON = os.getenv("FINPILOT_DIGEST_CRON", "0 8 * * mon")  # Mondays 08:00
TIMEZONE = os.getenv("FINPILOT_TZ", "Asia/Kolkata")
TELEGRAM_BOT_TOKEN = os.getenv("TELEGRAM_BOT_TOKEN", "")
TELEGRAM_CHAT_ID = os.getenv("TELEGRAM_CHAT_ID", "")
