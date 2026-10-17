# FinPilot — plan to submission

**Deadline:** submissions close **20 Oct 2026, 11:00 pm IST** (registration closes 17 Oct, 11:00 pm IST).
**Track:** Agent Arena 2026 · Track 02 (FinPilot). **Repo:** https://github.com/Rohanarekatla/finpilot

## Where it stands (17 Oct)

Done and working offline (11 tests passing, eval in `eval/RESULTS_offline.md`):

- [x] Six LangGraph agents: coordinator, categorizer, analyzer, forecaster, advisor, critic (+ digest writer)
- [x] Mesh API client as the single LLM entry point (`app/llm.py`)
- [x] CSV ingestion + labelled synthetic personas (student, professional, family)
- [x] Forecast with P10/P50/P90, goals, what-if simulation on common random numbers
- [x] Critic → advisor revision loop (≤ 2 revisions), number grounding, audit trail + `/why` endpoint
- [x] FastAPI dashboard, weekly APScheduler digest (Telegram optional)
- [x] Offline self-evaluation + Agent Arena CI workflow

Not yet verified: **anything with a live Mesh key** (LLM categorization, LLM-written advice, critic first-pass rate).

## Day 0 — 17 Oct (today)

- [ ] Register on the hackathon dashboard before 11 pm IST and copy the `SUBMISSION_TOKEN`
- [ ] Add repo secrets `MESH_API_KEY` and `SUBMISSION_TOKEN`
- [ ] Put the key in a local `.env`, run `python -m eval.run_eval`, commit `eval/RESULTS.md` + `eval/results.json`
- [ ] Confirm all four critical checks pass in the Actions tab

## Day 1 — 18 Oct: harden the LLM paths

- [ ] Confirm which Mesh model ids work (`openai/gpt-4o-mini`, `openai/gpt-4o`) and that `response_format` is honoured; adjust defaults in `.env.example`
- [ ] Read the LLM run's audit trail: are critic rejections real problems or false rejections? Tune prompts, not the checks
- [ ] Report LLM vs. heuristic long-tail categorization accuracy, and how many large-transaction false alarms the LLM tier removes
- [ ] Add a test with a fake LLM client that returns a bad recommendation, to prove the revise → drop → backfill path

## Day 2 — 19 Oct: close the known gaps

- [ ] **Forecast calibration:** the P10–P90 band covers ~63% of regular months (target 80%). Try a longer, recency-weighted bootstrap or variance inflation; tune on dev seeds 100–114, report on test seeds 0–9
- [ ] **Spike recall** in low-frequency categories (family shopping): compare against a count-based test
- [ ] **Annual bills:** let the user mark a known yearly payment so the forecast includes it
- [ ] **Goal-adaptive advice:** show the advice changing as the user moves ahead of or behind a goal (judging bonus)

## Day 3 — 20 Oct: present and submit (stop coding by 6 pm IST)

- [ ] Deploy (Render / Railway / Fly) with `FINPILOT_SCHEDULER=1`; add the URL to the README
- [ ] Record a 2–3 minute demo: load persona → recommendations → "why?" → "Can I afford…?" → audit trail → digest
- [ ] Refresh README numbers from the final `eval/RESULTS*.md`; keep the "Known gaps" section honest
- [ ] Final push, check CI is green, submit the repo link

## Judging criteria → where to point the judges

| Criterion | Evidence in the repo |
|---|---|
| Multi-agent orchestration | `app/graph.py` (pipeline + ask graph), `app/agents/*` |
| Core loop: forecasting & simulation | `app/agents/forecaster.py`, eval §3–4 |
| Core domain: categorization & analysis | `app/agents/categorizer.py`, `app/agents/analyzer.py`, eval §1–2 |
| Supporting domain: proactive advice | `app/agents/advisor.py`, `app/agents/critic.py`, `app/digest.py` |
| Foundation, data model, ingestion | `app/db.py`, `app/ingest.py`, `app/synthetic.py` |
| Self-evaluation | `eval/run_eval.py`, `eval/RESULTS*.md` |
| Auditable decision trail | `app/audit.py`, `/api/runs/{id}/audit`, `/api/recommendations/{id}/why` |
| Code quality & reproducibility | `tests/`, offline mode, pinned CI workflow |

## Risks

- **Mesh model behaviour differs from expectations:** every agent has a deterministic fallback, so the demo still runs. Fix prompts on Day 1.
- **Running out of time on calibration:** ship with the honest "Known gaps" section rather than a rushed change.
- **CI secret mistakes:** check the Actions tab on Day 0, not Day 3.
