"""Proactive digest: a scheduled job re-runs the pipeline and sends one insight + one action."""
from __future__ import annotations

import json
import logging
import urllib.parse
import urllib.request

from apscheduler.schedulers.background import BackgroundScheduler
from apscheduler.triggers.cron import CronTrigger

from app import config, db, service
from app.grounding import allowed_numbers, ungrounded
from app.llm import get_llm

log = logging.getLogger("finpilot.digest")
AGENT = "digest_writer"
SEVERITY = {"alert": 0, "warn": 1, "info": 2}

SYSTEM = """You write FinPilot's weekly money digest. In at most 70 words: one key insight about the user's
money and exactly one action for this week, both taken from the input. Use only numbers from the input.
Plain text, no markdown headings. Return JSON: {"digest": str}"""


def compose(result: dict) -> str:
    findings = sorted((f for f in result["findings"] if f["kind"] not in ("subscriptions", "trajectory")),
                      key=lambda f: SEVERITY.get(f["severity"], 3))
    recs = result["recommendations"]
    insight = findings[0]["detail"] if findings else result["narrative"]
    action = f"This week: {recs[0]['action']}" if recs else "No action needed this week."
    base = f"FinPilot weekly: {insight} {action}"
    forecast = {k: v for k, v in (result.get("forecast") or {}).items() if k != "month_ends"}
    resp = get_llm().chat_json(AGENT, SYSTEM, {"insight": insight, "action": recs[0] if recs else None,
                                               "forecast": forecast})
    text = (resp or {}).get("digest", "")
    if text and not ungrounded(text, allowed_numbers([base, forecast, recs[:1]])):
        return text
    return base


def deliver(text: str) -> str:
    if config.TELEGRAM_BOT_TOKEN and config.TELEGRAM_CHAT_ID:
        url = f"https://api.telegram.org/bot{config.TELEGRAM_BOT_TOKEN}/sendMessage"
        body = urllib.parse.urlencode({"chat_id": config.TELEGRAM_CHAT_ID, "text": text}).encode()
        try:
            with urllib.request.urlopen(url, data=body, timeout=10) as resp:
                if json.load(resp).get("ok"):
                    return "telegram"
        except Exception as exc:
            log.warning("Telegram delivery failed: %s", exc)
    log.info("Digest (stored only): %s", text)
    return "stored"


def send_digest(user_id: str) -> dict:
    result = service.analyze_user(user_id)
    text = compose(result)
    channel = deliver(text)
    db.save_digest(user_id, result["run_id"], channel, text)
    return {"user_id": user_id, "run_id": result["run_id"], "channel": channel, "digest": text}


def send_all() -> list[dict]:
    out = []
    for user_id in db.list_users():
        try:
            out.append(send_digest(user_id))
        except LookupError:
            continue
        except Exception:
            log.exception("Digest failed for %s", user_id)
    return out


def start_scheduler() -> BackgroundScheduler:
    sched = BackgroundScheduler(timezone=config.TIMEZONE)
    sched.add_job(send_all, CronTrigger.from_crontab(config.DIGEST_CRON, timezone=config.TIMEZONE),
                  id="weekly_digest", replace_existing=True)
    sched.start()
    log.info("Digest scheduler started (%s %s)", config.DIGEST_CRON, config.TIMEZONE)
    return sched
