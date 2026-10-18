# FinPilot — plan 2: architecture changes and what is left

**Written 18 Oct 2026.** Follows [PLAN.md](PLAN.md), which describes the original build. This file records
what changed in the second pass, why, how each change was verified, and the short list of things that are
still left before the deadline (**20 Oct 2026, 11:00 pm IST**).

Contents: [Summary](#summary) · [Architecture changes](#architecture-changes) · [Feature gaps closed](#feature-gaps-closed) ·
[Repo hygiene](#repo-hygiene) · [Verification](#verification) · [What is left](#what-is-left-for-the-team) · [Changed files](#changed-files)

## Summary

| Area | Before | After |
|---|---|---|
| Orchestration | A straight line; the only decision was critic → advisor revision | The coordinator makes 3 routing decisions (history check, anything actionable?, revise or finalize) and logs each one |
| Robustness | A CSV with under a week of data crashed the forecaster | Short histories skip the forecast, still get advice, and the API returns clear 400s |
| Mesh calls | No timeout (SDK default 10 min × 2 retries) | 30 s timeout, 1 retry, then the deterministic fallback |
| Categorizer | LLM batches one after another; merchant memory lost on restart | Batches in parallel (4 workers); memory saved in SQLite |
| Ask / Simulate | Re-ran the analyzer on every question | Reuses the latest analysis until the data changes |
| Forecast band | P10–P90 covered 63% of regular months (target 80%) | Spread calibrated on dev seeds: **73%** on test seeds (81% on dev) |
| Yearly bills | Not forecast | Users declare them; they are in the forecast and get a "set aside ₹X/month" action |
| Goals | One "at risk" finding below 60% | Behind / on track / ahead, a trend since the last run, and advice that changes with it |
| Affordability | A ₹500 purchase was "not affordable" when the balance was already at risk | The verdict separates existing risk from the risk this decision adds |
| Tests | 11 | **24**, including a scripted fake LLM that drives the revise → drop → backfill path |

## Architecture changes

### 1. The coordinator routes instead of just chaining

```
categorize → analyze ─┬─ ≥ 28 days of history ─▶ forecast ──────┐
                      └─ less ──────────────────▶ skip_forecast ─┴▶ plan ─┬─ candidates ─▶ advise → critique ─┬─▶ finalize
                                                                          └─ nothing actionable ──────────────┘   ▲
                                                                                      critique ── revise ─▶ advise ┘
```

- **`analyze → forecast | skip_forecast`.** The bootstrap needs whole weeks of history. With less than
  `MIN_HISTORY_DAYS` (28), the coordinator skips the forecast and adds an `insufficient_history` finding.
  The user still gets advice on duplicates, large one-offs, subscriptions and so on. The Ask graph applies
  the same check: an affordability question on too little data is answered from the data, not simulated.
- **New `plan` node.** It turns findings into quantified candidate actions, so building them is now its own
  step in the graph and the audit trail.
- **`plan → advise | finalize`.** When nothing is actionable, the advisor (the most expensive LLM call) is
  not called at all.
- Every route is written to the audit trail with its reason. The dashboard shows the path a run took
  ("How the agents ran"), and `GET /api/graph` returns both graphs as Mermaid generated from the compiled
  LangGraph objects, so the diagram can never drift from the code.

### 2. Mesh client hardening (`app/llm.py`, `app/config.py`)

- `OpenAI(timeout=FINPILOT_LLM_TIMEOUT (30 s), max_retries=FINPILOT_LLM_RETRIES (1))`. A slow or failing model
  now degrades to the deterministic path instead of freezing the demo.
- Call counters are guarded by a lock, because the categorizer now calls from worker threads.
- `MeshLLM.check()` lists the models (a metadata call that spends no credits) and reports whether the configured
  model ids exist. Mesh returns that list as a bare JSON array, so this uses plain HTTP.
  - Run it with `python -m app.llm`, or `GET /api/health?deep=true`.
  - `--ping` adds one tiny chat call to confirm the balance covers it.
- **Account errors switch Mesh off once.** The first 401/402/403 (bad key, zero balance, no access) sets
  `blocked_reason`. Every agent then uses its fallback without calling Mesh again, the error is logged once, and
  `/api/health` reports it.
- The live eval pings Mesh first and refuses to write `RESULTS.md` if Mesh isn't usable or stops working
  mid-run, so a fallback-only run can never be labelled "mesh". Eval output is written as UTF-8, which fixes a
  crash on Windows.
- The scheduler is now opt-in (`FINPILOT_SCHEDULER=1`). `.env.example`, the Dockerfile and `render.yaml` turn
  it on; tests never start it.

### 3. Efficiency

- **Parallel categorizer batches.** The LLM requests for unknown merchants run on a thread pool
  (`FINPILOT_LLM_WORKERS`, default 4).
- **Persistent merchant memory.** The `merchant_memory` table holds every LLM categorization. `service.py` loads
  it, passes it into the graph, and saves only the new entries. The categorizer itself never touches the
  database, so agents stay pure. LLM replies with invented keys or categories are ignored.
- **Analysis cache.** `service._analysis` keys the latest analysis on (row count, last id, amount sum, balance),
  so repeated questions don't re-run the analyzer.
- `llm_calls` in a run summary is now per run, not a running total for the whole process.

### 4. Data model additions (`app/db.py`)

| Table | Purpose |
|---|---|
| `bills` | User-declared yearly/periodic payments: `name`, `amount`, `next_date`, `every_months` |
| `goal_snapshots` | One row per goal per pipeline run: probability, projected P50, status. Drives the trend and the dashboard history |
| `merchant_memory` | `key` (MERCHANT\|direction) → category, confidence, reason |

All are `CREATE TABLE IF NOT EXISTS`, so existing databases upgrade themselves on start.

### 5. API additions (`app/main.py`)

| Method | Path | Purpose |
|---|---|---|
| GET | `/api/health?deep=true` | Adds the Mesh key/model check |
| GET | `/api/graph` | Mermaid for both graphs |
| POST/GET | `/api/users/{id}/bills` | Declare / list bills |
| DELETE | `/api/users/{id}/bills/{bill_id}` | Remove a bill |
| GET | `/api/users/{id}/goals` | Now includes status, probability and per-run history |
| DELETE | `/api/users/{id}/goals/{goal_id}` | Remove a goal |

The `/simulate` body is now typed (`PurchaseIn`, `MonthlyChangeIn`), and unknown categories return 400.

## Feature gaps closed

### Forecast calibration (PLAN.md Day 2)

- **Method.** Each simulated week is widened around the average week by a factor k; the median is unchanged.
  `python -m eval.tune_forecast` sweeps k on **dev seeds 100–114 only** and picks the smallest k that reaches
  the nominal 80% coverage. Under-covering is the worse error for money, because it makes a risky month look safe.
- **Result.** k = 1.3 gives 81.1% on dev. On the held-out test seeds, regular-month coverage goes from
  **63.3% to 73.3%**, and balance MAPE is essentially unchanged (6.75% → 6.78%).
- The gap between 81% on dev and 73% on test is reported as it is, not tuned away. See
  [`eval/TUNING_forecast.md`](eval/TUNING_forecast.md).

### Yearly bills (PLAN.md Day 2)

- Declared bills are placed on their due dates in the forecast and in every what-if simulation.
- Each one produces an `upcoming_bill` finding. It is a warning when the bill is due within 60 days.
- The advisor turns that finding into "Move ₹X a month into savings for <bill> so the ₹Y due on <date> does
  not hit your balance all at once." The critic checks it like any other recommendation.
- Eval §4 adds an exact property: a declared ₹20,000 bill moves the P50 end balance by exactly −₹20,000 (15/15).

### Goal-adaptive advice (PLAN.md Day 2, judging bonus)

- **Status.** Each goal is `behind` (< 60% chance), `on_track`, or `ahead` (≥ 85%). The finding and the advice
  follow the status:
  - behind → `goal_at_risk`, with the exact extra amount per month needed, tied to the cuts that would free it up
  - ahead → `goal_ahead` ("you reach it around <date> with ₹X to spare; raise the target or start the next goal")
  - a change of ≥ 10 points since the last run → the finding says so, and a falling at-risk goal becomes an **alert**
- **Eval §6 (new).** Every dataset gets goals at 0.5×, 0.9×, 1.5× and 3× its own projected savings:
  - the chance of hitting the goal falls as the goal grows in 100% of datasets
  - the advice matches the status 100% of the time
  - for all 76 "behind" goals, saving the advised extra amount per month really reaches the target at P50

### The LLM revision path is now tested (PLAN.md Day 1)

`tests/test_orchestration.py` uses a scripted `FakeLLM`:

- First case: the advisor writes one recommendation with an invented figure. The critic rejects it, the
  coordinator routes back with feedback, the revised list passes, and `first_pass_rate == 0.5`.
- Second case: the advisor keeps the bad figure. After two revisions the recommendation is dropped and a
  computed one is backfilled. Both paths are visible in the audit trail.

### Affordability verdict against the baseline

- **Problem.** For a user whose balance is already at risk, even a ₹500 purchase was "not affordable".
- **Fix.** If the baseline is already not affordable and the decision adds ≤ 2 points of risk, the verdict is
  "tight" with an explicit reason: "your balance is already at risk before this decision".
- Eval §4 "tiny purchase not rejected" is back to 15/15.

## Repo hygiene

- **Secret.** A real Mesh key had been pasted into `.env.example`, which is committed. It was moved to `.env`
  (gitignored) before any commit. The key never reached git history.
- **`uv init` leftovers.**
  - `pyproject.toml` now lists the real dependencies and `requires-python >= 3.11`, matching the CI's Python
    3.11. Before, it was empty and required 3.13.
  - `main.py` is a real launcher (`python main.py` / `uv run main.py`) instead of "Hello from finpilot".
  - `requirements.txt` stays as the source the Agent Arena checks read.
- **Line endings.** `.gitattributes` (`* text=auto eol=lf`) stops Windows checkouts from showing every file as
  modified.
- **CI.** A second workflow, `.github/workflows/tests.yml`, runs the offline test suite on Python 3.11. The
  official `agent-arena-2026-checks.yml` is untouched.
- **Deploy.** `Dockerfile` (one worker, non-root, scheduler on, SQLite in `/data`), `.dockerignore`, and a
  Render blueprint `render.yaml`.
- `tests/conftest.py` makes every test offline, scheduler-free and on its own temporary database.

## Verification

- `pytest`: **25 passed** (11 original + 14 new), offline. Also run on Windows / Python 3.13: 24/24 before the last fix.
- `python -m eval.run_eval --offline` was re-run on the final code. See
  [`eval/RESULTS_offline.md`](eval/RESULTS_offline.md).
- `python -m eval.tune_forecast` was run: [`eval/TUNING_forecast.md`](eval/TUNING_forecast.md).
- Every `.py` file compiles on Python 3.10, so it compiles on the 3.11 CI too.
- The dashboard was exercised in headless Chromium: load persona → add goal → add bill → ask an affordability
  question. There were no JavaScript errors, and the page still works when the Chart.js CDN is unreachable.
- **Not verified:** anything with a live Mesh key. Neither machine available during this pass could reach
  `api.meshapi.ai`. That is the first item below.

## What is left for the team

**Today (18 Oct)**

1. [ ] **Top up the Mesh balance first.** The key is valid, but every chat call returns HTTP 402
       `spend_limit_exceeded`. Mesh's only free model is a moderation model, so the chat models need a positive
       balance. Then check the key, the model ids and one tiny chat call:
       ```
       uv run python -m app.llm --ping
       ```
       Look for `"ok": true`, `"missing": []` and `"chat": {"ok": true}`. To save credits, both
       `FINPILOT_MODEL_FAST` and `FINPILOT_MODEL_SMART` can be set to `openai/gpt-4o-mini` in `.env`.
2. [ ] Run the tests (expect 25 passed): `uv run pytest`
3. [ ] Run the live eval. It writes `eval/RESULTS.md` and `results.json`, which you commit. Then read the
       critic's REJECT lines in the audit trail: real problems or false alarms? Tune prompts, not the checks.
       ```
       uv run python -m eval.run_eval
       ```
4. [ ] Commit and push. Add the repo secrets `MESH_API_KEY` and `SUBMISSION_TOKEN` if they aren't there yet,
       and confirm both workflows are green in the Actions tab.

**19 Oct**

5. [ ] Report LLM vs. heuristic long-tail accuracy, plus the large-transaction false alarms the LLM tier removes
       (eval §1–2 with Mesh), in the README.
6. [ ] Optional: spike recall for low-frequency categories (family shopping). Try a count-based test.

**20 Oct (stop coding by 6 pm IST)**

7. [ ] Deploy: Render → New → Blueprint → this repo → paste `MESH_API_KEY`. Put the URL in the README.
8. [ ] Record a 2–3 minute demo:
       - load the professional persona → recommendations → "why?"
       - ask "Can I afford a ₹60,000 trip next month?"
       - add a goal and a yearly bill, and watch the advice change
       - show the "How the agents ran" path and the audit trail
       - send the digest
9. [ ] Final push, CI green, submit the repo link.

## Changed files

| File | Change |
|---|---|
| `app/graph.py` | Coordinator routing (history check, plan node, skip-advise), Mermaid export |
| `app/llm.py`, `app/config.py` | Timeouts/retries, thread-safe stats, `check()`, scheduler opt-in |
| `app/agents/forecaster.py` | Calibrated spread, bills, goal status/trend, baseline-aware verdict, history check |
| `app/agents/advisor.py` | Actions for `goal_ahead` and `upcoming_bill` |
| `app/agents/categorizer.py` | Parallel batches, injectable/persistent merchant memory, stricter reply validation |
| `app/agents/coordinator.py`, `app/digest.py` | Work when there is no forecast |
| `app/service.py`, `app/db.py`, `app/main.py` | Bills, goal snapshots, merchant memory, analysis cache, new endpoints |
| `app/templates/index.html` | Bills, goal status + history, run path, LLM mode badge, works without the chart CDN |
| `eval/run_eval.py`, `eval/tune_forecast.py` | Goal-adaptivity eval (§6), bill property, spread tuning on dev seeds |
| `tests/` | `conftest.py`, `test_orchestration.py`, `test_planning.py` |
| repo root | `pyproject.toml`, `main.py`, `.gitattributes`, `.gitignore`, `.env.example`, `Dockerfile`, `.dockerignore`, `render.yaml`, `.github/workflows/tests.yml` |
