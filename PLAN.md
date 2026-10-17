# FinPilot — plan to submission

**Deadline:** submissions close **20 Oct 2026, 11:00 pm IST** (registration closes 17 Oct, 11:00 pm IST).
**Track:** Agent Arena 2026 · Track 02 (FinPilot). **Repo:** https://github.com/Rohanarekatla/finpilot

Contents: [Status](#where-it-stands-17-oct) · [Day-by-day plan](#day-0--17-oct-today) · [Judging map](#judging-criteria--where-to-point-the-judges) · [Risks](#risks) · [Architecture](#architecture-as-built)

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

---

# Architecture (as built)

This section describes what is in the repo now: every component, how data flows through it, the data
model, the agent algorithms and their thresholds, the API, and how it is evaluated. Update it as the
Day 1–3 items above land.

## 1. System overview

```
                         ┌───────────────────────────────────────────────────────────────┐
  Browser dashboard ───▶ │ FastAPI (app/main.py)                                          │
  REST clients      ───▶ │   /upload /sample /analyze /ask /simulate /goals /why /audit   │
  APScheduler (weekly)─▶ │   /digest                                                      │
                         └──────────────┬────────────────────────────────────────────────┘
                                        │
                         ┌──────────────▼──────────────┐      ┌─────────────────────────┐
                         │ service.py                  │◀────▶│ SQLite (app/db.py)      │
                         │ load data → run graph →     │      │ users, transactions,    │
                         │ persist results + audit     │      │ goals, runs, findings,  │
                         └──────────────┬──────────────┘      │ recommendations,        │
                                        │                     │ decisions, digests      │
                         ┌──────────────▼──────────────┐      └─────────────────────────┘
                         │ graph.py (LangGraph)        │
                         │  PIPELINE graph + ASK graph │
                         └──────────────┬──────────────┘
          ┌──────────────┬──────────────┼──────────────┬──────────────┬──────────────┐
          ▼              ▼              ▼              ▼              ▼              ▼
     Categorizer     Analyzer      Forecaster       Advisor        Critic       Coordinator
     rules→LLM→      stats +       Monte Carlo +    candidates →   deterministic  intent routing,
     heuristics      LLM narrative what-if sims     LLM ranking    checks         grounded answers
          │              │                             │                             │
          └──────────────┴────────────┬────────────────┴─────────────────────────────┘
                                      ▼
                       llm.py: MeshLLM ── OpenAI SDK ──▶ https://api.meshapi.ai/v1
                       grounding.py: every LLM-written number is checked against data
```

Layering rules:
- **Agents are pure functions over DataFrames.** They never touch the database.
- **`graph.py` is pure too.** It takes a DataFrame and returns state. The eval harness runs the exact production graph.
- **`service.py` is the only layer that loads and persists.** `main.py` is a thin HTTP layer over it.
- **`llm.py` is the only place that talks to an LLM**, always through Mesh API.

## 2. Request flows

**Analyze** (`POST /api/users/{id}/analyze`, the scheduler, the dashboard's Analyze button):

1. `service.analyze_user` loads transactions, current balance and goals.
2. It runs `graph.run_pipeline` (§4.1).
3. It saves categories back to `transactions`, then saves `findings`, `recommendations`, a `runs` summary
   (narrative, stats, forecast, series, recurring, monthly table, LLM call counts) and the audit trail
   to `decisions`.

**Ask** (`POST /api/users/{id}/ask`):

1. Load data, recompute the analysis (no narrative, so no LLM call).
2. `graph.run_ask`: classify → simulate → respond, or classify → answer.
3. Persist a `runs` row of kind `ask` and its decisions.

**Simulate** (`POST /api/users/{id}/simulate`) runs an explicit `Scenario` through
`forecaster.simulate`. No LLM is involved.

**Why** (`GET /api/recommendations/{rec_id}/why`) returns the recommendation, the findings it cites
(with their data) and the run's related decisions.

**Digest** (APScheduler cron `0 8 * * mon`, Asia/Kolkata, or `POST /digest`):

1. Run the full analysis.
2. Compose one insight (the most severe finding) and one action (the top recommendation).
3. Optionally let the LLM rewrite it, keeping the rewrite only if its numbers are grounded.
4. Deliver to Telegram if configured; always store it in `digests`.

## 3. Data model (SQLite, `app/db.py`)

| Table | Key columns | Notes |
|---|---|---|
| `users` | `user_id`, `opening_balance` | Opening balance is used when the CSV has no balance column |
| `transactions` | `id`, `user_id`, `date`, `description`, `amount` (+credit / −debit), `balance`, `merchant`, `category`, `category_source` (rule/llm/heuristic/none), `category_confidence` | `UNIQUE(user_id, date, description, amount)` makes re-uploads idempotent |
| `goals` | `name`, `target_amount`, `target_date` | A goal means "new savings (balance gain + investments) by this date" |
| `runs` | `run_id`, `kind` (pipeline/ask/simulate), `summary` JSON | One row per execution |
| `findings` | `(run_id, finding_id)`, `kind`, `title`, `detail`, `severity`, `data` JSON | Ids are readable, e.g. `price_hike:NETFLIX` |
| `recommendations` | `rec_id` (`{run_id}-R{n}`), `action`, `why`, `finding_ids` JSON, `monthly_impact`, `one_off_impact`, `source` (llm/template) | Only critic-approved rows are stored |
| `decisions` | `run_id`, `agent`, `step`, `decision`, `reason`, `data` JSON, `ts` | The audit trail |
| `digests` | `user_id`, `run_id`, `channel`, `content` | Proactive messages sent |

**Ingestion (`app/ingest.py`):**
- Accepts `date` + `description`/`narration` + either `amount` or `debit`/`credit`, plus an optional `balance`.
- Column aliases cover common Indian bank exports.
- **Drops `true_category`/`label` columns** so ground truth can never leak into the pipeline.
- Computes `merchant` with `taxonomy.merchant_key`, which strips channel prefixes (UPI/POS/ACH/NEFT/ECOM),
  refs, VPAs, digits, cities, legal suffixes and month stamps, then keeps the first 3 tokens.

## 4. LangGraph orchestration (`app/graph.py`)

### 4.1 Pipeline graph

```
START → categorize → analyze → forecast → advise → critique ─┬─▶ finalize → END
                                             ▲               │
                                             └── revise ─────┘  if any rec failed AND recs came from the
                                                                LLM AND attempts ≤ MAX_ADVISOR_REVISIONS (2)
```

State (`PipelineState`): `txns`, `start_balance`, `goals`, `mode`, `audit`, `llm`, `analysis`,
`forecast`, `forecast_findings`, `candidates`, `recommendations`, `verdicts`, `attempts`,
`first_pass_rate`, `final`.

| Node | What it does |
|---|---|
| `categorize` | Categorizes only rows whose category is still null, so stored categories are reused |
| `analyze` | Runs the analyzer: recurring payments, findings, stats, narrative |
| `forecast` | Builds the baseline forecast, adds forecast findings (trajectory, cash shortfall, goal at risk), builds the advisor's candidate actions |
| `advise` | Drafts recommendations. On a revision pass it also sends the previous list and the critic's failures |
| `critique` | Runs the deterministic checks and records the first-pass rate |
| `finalize` | Keeps passed recs; logs dropped ones; if fewer than 3 survive, backfills the top uncovered candidates as templates (re-checked by the critic); assigns `rec_id`s |

### 4.2 Ask graph

```
START → classify ─┬─ afford / what_if ─▶ simulate → respond → END
                  └─ question ─────────▶ answer ───────────▶ END
```

## 5. Agents in detail

### 5.1 Categorizer (`app/agents/categorizer.py`, `app/taxonomy.py`)

- **Taxonomy:** 23 categories: income, rent, emi_loan, investments, insurance, groceries, food_delivery,
  dining, transport, fuel, utilities, mobile_internet, subscriptions, fitness, shopping, entertainment,
  health, education, travel, transfers, cash_withdrawal, fees_charges, other.
- **Tier 1, rules:** ordered regexes for national brands and bank conventions (SALARY, RENT, EMI, SIP, …).
  Confidence 0.95.
- **Tier 2, LLM via Mesh** (`mode="auto"`):
  - Unresolved rows are grouped by `(merchant, direction)`. Each group goes to the LLM once, with a sample
    narration, typical amount and count, in batches of 40.
  - The LLM returns a category, confidence and reason. Categories outside the taxonomy are rejected.
  - **Merchant memory**: a process-level cache, so the same merchant is never asked twice.
- **Tier 3, keyword heuristics** (fallback, or `mode="heuristic"`): kirana/tiffin/medical/petrol/school
  keywords. A UPI payment to a person becomes transfers. Confidence 0.3–0.6.
- **Audit:** rule coverage, each LLM merchant decision with its reason, and fallback counts.

### 5.2 Analyzer (`app/agents/analyzer.py`)

Detection is deterministic statistics; the LLM only writes the narrative.

- **Complete months:** the latest month counts as complete only if data reaches within 2 days of its end.
- **Recurring detection** per `(merchant, sign)`:
  - Monthly needs ≥3 occurrences, median gap 25–35 days and gap SD ≤ 6. Quarterly needs median gap 80–100
    days and gap SD ≤ 12.
  - Amount CV ≤ 0.35, measured excluding the latest charge so a price hike doesn't break detection.
  - A series is dropped as lapsed if the last charge is more than one period + 10 days old.
  - Subscriptions are recurring debits categorized as subscriptions or fitness.
- **Outliers (one-offs):** amount ≥ 20% of average monthly income, OR robust z = 0.6745·(x−median)/MAD ≥ 8
  with ≥5 transactions in the category and amount ≥ ₹5,000. Outliers are reported on their own and
  excluded from spikes, the savings rate and the forecast pool.

| Finding kind | Rule | Impact sized as |
|---|---|---|
| `category_spike:{cat}` | Behavioural category; ≥2 purchases/month; latest month vs. up to 6-month mean: +35%, +₹1,000 and z ≥ 2, with SD = √(rate·E[amount²]) (compound Poisson). Data includes the top 3 merchants. | monthly increase |
| `large_transaction:{merchant}` | An outlier in the last 45 days | one-off amount |
| `duplicate_charge:{merchant}` | Same merchant and amount within 1 day, not recurring, in the last 45 days | one-off amount |
| `subscriptions:summary` | All active subscriptions and their monthly total | info only |
| `overlapping_subscriptions:{group}` | ≥2 services in video, music or cloud storage | saving if you keep only the priciest |
| `price_hike:{merchant}` | Prior amounts stable (CV ≤ 5%) and latest ≥ +5% and ≥ +₹20 | monthly increase |
| `low_savings_rate:overall` | Underlying savings rate (excluding one-offs) over the last 3 months < 10% | transfer needed to reach 20% |

- **Stats:** average income, spend and investment; savings rate; latest month's spend and top categories;
  fixed monthly costs; subscription total.
- **Narrative:** the LLM summarizes stats and findings. It is rejected if any number isn't in the payload
  (no derived numbers allowed); the computed template is used instead.
- **Tuning provenance:** spike thresholds were chosen on dev seeds 100–114 and are reported on test seeds 0–9.

### 5.3 Forecaster (`app/agents/forecaster.py`)

- **Scheduled flows:** every active recurring item is placed on its future due dates using its latest
  amount. Investment outflows are tracked separately so they count as savings.
- **Everyday flows:**
  1. Take non-recurring, non-outlier transactions from the last 91 days, clipped to the start of the data.
  2. Aggregate them into daily totals and split them into 7-day blocks.
  3. Block-bootstrap N = 500 futures, which keeps the weekday/weekend rhythm.
- **Horizon:** 90 days by default, extended to cover goal dates and purchase date + 60 days, capped at 400.
- **Outputs:** P10/P50/P90 daily balance; month-end bands; minimum balance and its date; probability of
  going negative; daily outflows; savings paths.
- **Goals:** projected P50 savings by the target date, probability of hitting it, shortfall, extra amount
  needed per month, and the date the median path reaches the target.
- **Forecast findings:**
  - `trajectory:balance` (info).
  - `cash_shortfall:balance` if P(negative) > 10% or the P50 minimum is below one month of fixed costs.
  - `goal_at_risk:{slug}` if the probability of hitting a goal is < 60%.
- **`Scenario`:** `purchases`, `monthly_changes`, `category_cuts` (scales that category's debits),
  `cancel` (drops recurring merchants).
- **`simulate()`:**
  - Baseline and scenario use the **same seed** (common random numbers), so the delta is exactly the
    decision's effect: a ₹X purchase moves the end balance by exactly −₹X.
  - Verdicts:
    - `affordable`: P10 minimum ≥ buffer, where the buffer is one month of fixed costs.
    - `tight`: P50 minimum ≥ 0 and P(negative) ≤ 10%.
    - `not_affordable`: anything else.
    - `affordable_but_delays_goal`: affordable, but a goal's probability drops by ≥15 points.

### 5.4 Advisor (`app/agents/advisor.py`)

- **`build_candidates`** turns each actionable finding into one concrete action with an exact rupee impact
  (cancel overlap, revert price hike, cap a spiking category, dispute a duplicate, review a one-off,
  automate savings, protect the balance, close a goal gap).
- **Priority** = monthly impact × 12 + one-off impact + severity bonus (alert 6000, warn 1500).
- A goal-gap candidate also notes how much the suggested cuts would free up per month.
- **LLM step** (`MODEL_SMART`): ranks, merges and personalizes up to 5 recommendations; must cite
  `finding_ids` and may only use input numbers (sums or ×12 allowed).
- **Revision prompt:** "fix every issue or drop it, keep the ones that passed."
- **No LLM:** the candidates are used verbatim as templates.

### 5.5 Critic (`app/agents/critic.py`)

The critic is code, not a prompt. Every recommendation must pass all five checks:

1. **Evidence:** cites ≥1 finding id, and all cited ids exist.
2. **Grounding:** each ₹ figure or % is in the cited evidence, or derived from it (sum or difference of two,
   ×12, ÷12, ratio ×100). General stats may only be quoted verbatim. Tolerance is max(₹2, 2%) for money and
   ±1.5 points for percentages. Days, counts and years are ignored.
3. **Specific:** mentions a merchant, category label, subscription group, goal name, or "saving"/"balance"
   where that is the subject.
4. **Impact bound:** the claimed monthly or one-off impact is ≤ 105% of the cited candidates' impact.
5. **Distinct:** doesn't reuse an earlier passed recommendation's evidence set.

Every PASS or REJECT is logged with its reasons.

### 5.6 Coordinator (`app/agents/coordinator.py`)

- **Classify:** the LLM returns `intent` ∈ {afford, what_if, question} plus a Scenario JSON. The result is
  validated: dates parsed, categories in the taxonomy, cancel targets must be known recurring merchants.
- **Regex fallback:**
  - Amounts: ₹, k, lakh; years are ignored.
  - Dates: "next month" → the 15th, "in N months", month names.
  - Intent keywords: cancel / cut X% / per month / afford-buy-trip.
- **Respond:** a deterministic verdict sentence built from the simulation. The LLM may rewrite it only if
  every number survives grounding.
- **Answer:** LLM QA over a compact context (stats, 6-month category table, recurring payments, findings,
  forecast). It is rejected if any number isn't verbatim in the context; the rule-based answer is used instead.

### 5.7 Digest writer (`app/digest.py`)

The digest is one insight plus one action, at most about 70 words. An LLM rewrite is kept only if it is
grounded. Telegram delivery uses the Bot API through `urllib`. The digest is always saved to `digests`.

## 6. Cross-cutting

- **LLM client (`app/llm.py`):**
  - Uses the OpenAI SDK with `base_url=https://api.meshapi.ai/v1` and `MESH_API_KEY`.
  - `chat_json(agent, system, user, model)` asks for JSON mode and retries without `response_format` if a
    routed model rejects it. If the reply isn't clean JSON, it extracts the JSON object from it.
  - It counts calls and failures per agent and returns `None` when offline or on error, so every agent has
    a deterministic fallback.
- **Grounding (`app/grounding.py`):** `numbers_in` (understands ₹, k, lakh and %), `allowed_numbers` (with
  or without derivations), `ungrounded`, and `inr` (Indian digit grouping: ₹12,34,567).
- **Audit (`app/audit.py`):** `AuditTrail` buffers `Decision(agent, step, decision, reason, data)` in
  memory and flushes it to `decisions` at the end of a persisted run.
- **Configuration (`app/config.py`, `.env`):**

  | Variable | Purpose |
  |---|---|
  | `MESH_API_KEY` | Mesh API key |
  | `FINPILOT_MODEL_FAST` | Default `openai/gpt-4o-mini` |
  | `FINPILOT_MODEL_SMART` | Default `openai/gpt-4o`, used by the advisor |
  | `FINPILOT_OFFLINE` | `1` forces the deterministic fallbacks |
  | `FINPILOT_DB` | Database path |
  | `FINPILOT_HORIZON_DAYS` | Forecast horizon, default 90 |
  | `FINPILOT_SIMULATIONS` | Simulated futures, default 500 |
  | `FINPILOT_MAX_REVISIONS` | Advisor revisions, default 2 |
  | `FINPILOT_SCHEDULER`, `FINPILOT_DIGEST_CRON`, `FINPILOT_TZ` | Weekly digest schedule |
  | `TELEGRAM_BOT_TOKEN`, `TELEGRAM_CHAT_ID` | Optional digest delivery |

## 7. HTTP API (`app/main.py`)

| Method | Path | Body / params | Returns |
|---|---|---|---|
| GET | `/` | – | Dashboard (Jinja2 + Chart.js) |
| GET | `/api/health` | – | LLM mode, model ids |
| POST | `/api/users/{id}/upload` | multipart `file`, optional `opening_balance` | rows imported |
| POST | `/api/users/{id}/sample/{persona}` | `seed` | rows imported |
| POST | `/api/users/{id}/analyze` | – | run_id, narrative, stats, forecast, series, findings, recommendations |
| GET | `/api/users/{id}/latest` | – | latest stored run |
| GET | `/api/users/{id}/transactions` | `limit` | categorized transactions |
| POST/GET | `/api/users/{id}/goals` | `{name, target_amount, target_date}` | goal id / list |
| POST | `/api/users/{id}/ask` | `{question}` | intent, answer, scenario, simulation |
| POST | `/api/users/{id}/simulate` | Scenario JSON | verdict, reasons, baseline vs decision, series |
| GET | `/api/runs/{run_id}/audit` | – | decisions, findings, recommendations |
| GET | `/api/recommendations/{rec_id}/why` | – | recommendation + evidence + decisions |
| POST | `/api/users/{id}/digest` | – | channel, digest text |
| GET | `/api/users/{id}/digests` | – | digest history |

## 8. Synthetic data (`app/synthetic.py`)

**Personas.** Each covers 7 months, ending 30 Sep 2026:

| Persona | Monthly income | Fixed commitments |
|---|---|---|
| student | ₹31k stipend | rent, SIP |
| professional | ₹1.25L | rent, EMI, SIP, utilities, gym, 4 streaming subscriptions |
| family | ₹2.1L | home-loan EMI, SIP, utilities, quarterly school fees, quarterly LIC |

**How flows are generated:**
- Discretionary spend: daily Poisson(rate) × lognormal(median, σ = 0.4) per category, with a weekend boost
  for food, dining and entertainment.
- Merchants: brands the rules know, plus a long tail of local shops and brands they don't.
- Narrations follow real UPI/POS/ECOM/NEFT/ACH formats, with a running balance column.

**Planted ground truth** (final month), with a per-persona mix:
- a category spike (×1.9–2.2)
- a price hike (Netflix 499→649, broadband 999→1199)
- a large one-off (CROMA ₹42,999, RELIANCE DIGITAL ₹78,000)
- a duplicate charge
- overlapping subscriptions

Every row carries `true_category`. `python -m app.synthetic --out data` writes the CSVs and the
`*_planted.json` label files.

## 9. Evaluation (`eval/run_eval.py`)

1. **Categorization:** accuracy, macro-F1, rule coverage and long-tail accuracy for each tier: rules only,
   rules + heuristics, rules + LLM.
2. **Detection:** precision and recall per kind against the planted labels, over 30 datasets (3 personas ×
   seeds 0–9).
3. **Forecast:** a rolling-origin backtest, forecasting through month end.
   - Next-month spend MAPE against last-month and 3-month-mean baselines.
   - Month-end balance MAPE against repeating last month's net flow.
   - P10–P90 coverage, reported for all folds and for regular (non-anomaly) months.
4. **Simulation properties:** exact purchase delta, exact subscription saving, cuts never hurt, absurd
   purchases rejected, tiny purchases not rejected.
5. **Advice:** critic first-pass rate, rejections, revision loops, grounded / evidenced / quantified
   share of delivered advice, coverage of the top-3 issues, latency.

Outputs: `eval/RESULTS_offline.md` and `results_offline.json` (offline); `eval/RESULTS.md` and
`results.json` (with Mesh).

## 10. Testing and CI

- **`tests/test_finpilot.py`:** 11 offline tests covering the merchant key, ingestion, categorization,
  planted detection, exact simulation deltas, critic rejections, the critic-approved-only pipeline,
  ask routing, parsers and a full API run through `TestClient`.
- **`.github/workflows/agent-arena-2026-checks.yml`:** the official Agent Arena workflow. It checks that the
  code compiles on Python 3.11, that dependencies are present, that Mesh is used, and that the key is valid.

## 11. Repository layout

```
finpilot/
├── app/
│   ├── agents/  advisor.py analyzer.py categorizer.py coordinator.py critic.py forecaster.py
│   ├── templates/index.html      dashboard
│   ├── audit.py  config.py  db.py  digest.py  graph.py  grounding.py
│   ├── ingest.py  llm.py  main.py  service.py  synthetic.py  taxonomy.py
├── data/        student|professional|family .csv + *_planted.json
├── eval/        run_eval.py, RESULTS_offline.md, results_offline.json
├── tests/       test_finpilot.py
├── .github/workflows/agent-arena-2026-checks.yml
├── PLAN.md  README.md  requirements.txt  pytest.ini  .env.example  .gitignore
```
