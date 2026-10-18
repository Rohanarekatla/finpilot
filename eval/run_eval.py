"""FinPilot self-evaluation.

Runs the real pipeline on labelled synthetic statements and measures:
  1. categorisation accuracy / macro-F1 per tier (rules, +heuristics, +LLM);
  2. pattern & anomaly detection precision / recall against planted ground truth;
  3. forecast accuracy: rolling-origin backtest of next-month spend and month-end
     balance vs. naive baselines, plus P10-P90 interval coverage;
  4. simulation correctness: exact, checkable properties of what-if results;
  5. advice quality: critic pass rate, grounding and specificity of delivered advice;
  6. goal-adaptive advice: the goal status and advice track the user's progress, and
     following the advice ("save ₹X more a month") actually closes the gap.

Test seeds 0-9 are disjoint from the dev seeds (100-114) used to tune the analyzer.

    python -m eval.run_eval              # uses Mesh if MESH_API_KEY is set
    python -m eval.run_eval --offline    # deterministic, no LLM
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from collections import Counter
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from app.agents import analyzer, categorizer, critic, forecaster  # noqa: E402
from app.audit import AuditTrail  # noqa: E402
from app.graph import run_pipeline  # noqa: E402
from app.llm import MeshLLM, set_llm  # noqa: E402
from app.synthetic import PERSONAS, generate  # noqa: E402
from app.taxonomy import CATEGORIES, merchant_key  # noqa: E402

TEST_SEEDS = list(range(10))
ANOMALY_KINDS = {"category_spike", "large_transaction", "duplicate_charge", "price_hike", "overlapping_subscriptions"}
GOAL = {"name": "Emergency fund", "target_amount": 200000, "target_date": "2027-03-31"}


def prepare(persona: str, seed: int) -> tuple[pd.DataFrame, pd.Series, list[dict], float]:
    s = generate(persona, seed)
    df = s.transactions.drop(columns=["true_category"]).copy()
    df["merchant"] = df["description"].map(merchant_key)
    return df, s.transactions["true_category"], s.planted, float(df["balance"].iloc[-1])


def macro_f1(y_true: pd.Series, y_pred: pd.Series) -> float:
    f1s = []
    for c in sorted(set(y_true)):
        tp = int(((y_pred == c) & (y_true == c)).sum())
        fp = int(((y_pred == c) & (y_true != c)).sum())
        fn = int(((y_pred != c) & (y_true == c)).sum())
        p = tp / (tp + fp) if tp + fp else 0.0
        r = tp / (tp + fn) if tp + fn else 0.0
        f1s.append(2 * p * r / (p + r) if p + r else 0.0)
    return float(np.mean(f1s))


def eval_categorization(llm: MeshLLM, use_llm: bool) -> dict:
    tiers = ["rules", "heuristic"] + (["auto"] if use_llm else [])
    names = {"rules": "rules only", "heuristic": "rules + keyword heuristics", "auto": "rules + LLM (Mesh)"}
    out = {}
    for tier in tiers:
        trues, preds, srcs = [], [], []
        for persona in PERSONAS:
            for seed in TEST_SEEDS[:3] if tier == "auto" else TEST_SEEDS:  # LLM tier: fewer seeds to save credits
                df, truth, _, _ = prepare(persona, seed)
                c = categorizer.categorize(df, mode=tier, llm=llm)
                trues.append(truth)
                preds.append(c["category"])
                srcs.append(c["category_source"])
        y, p, s = pd.concat(trues), pd.concat(preds), pd.concat(srcs)
        long_tail = s != "rule"
        out[names[tier]] = {
            "transactions": int(len(y)),
            "accuracy": round(float((y == p).mean()), 4),
            "macro_f1": round(macro_f1(y, p), 4),
            "rule_coverage": round(float((s == "rule").mean()), 4),
            "long_tail_accuracy": round(float((y[long_tail] == p[long_tail]).mean()), 4) if long_tail.any() else None,
        }
    return out


def eval_detection(llm: MeshLLM, mode: str) -> dict:
    per_kind: dict[str, Counter] = {k: Counter() for k in ANOMALY_KINDS}
    misses, false_alarms = [], []
    for persona in PERSONAS:
        for seed in TEST_SEEDS:
            df, _, planted, _ = prepare(persona, seed)
            c = categorizer.categorize(df, mode=mode, llm=llm)
            a = analyzer.analyze(c, narrate=False)
            gold = {f"{p['kind']}:{p['target']}" for p in planted}
            got = {f.id for f in a.findings if f.kind in ANOMALY_KINDS}
            for g in gold:
                kind = g.split(":")[0]
                hit = any(x.startswith(kind + ":") and g.split(":", 1)[1] in x for x in got)
                per_kind[kind]["tp" if hit else "fn"] += 1
                if not hit:
                    misses.append(f"{persona}/{seed}: {g}")
            for x in got:
                kind = x.split(":")[0]
                if not any(g.startswith(kind + ":") and g.split(":", 1)[1] in x for g in gold):
                    per_kind[kind]["fp"] += 1
                    false_alarms.append(f"{persona}/{seed}: {x}")
    rows, tot = {}, Counter()
    for k, c in sorted(per_kind.items()):
        tot.update(c)
        rows[k] = {"tp": c["tp"], "fp": c["fp"], "fn": c["fn"],
                   "precision": round(c["tp"] / (c["tp"] + c["fp"]), 3) if c["tp"] + c["fp"] else None,
                   "recall": round(c["tp"] / (c["tp"] + c["fn"]), 3) if c["tp"] + c["fn"] else None}
    rows["overall"] = {"tp": tot["tp"], "fp": tot["fp"], "fn": tot["fn"],
                       "precision": round(tot["tp"] / (tot["tp"] + tot["fp"]), 3),
                       "recall": round(tot["tp"] / (tot["tp"] + tot["fn"]), 3)}
    return {"by_kind": rows, "datasets": len(PERSONAS) * len(TEST_SEEDS),
            "examples_missed": misses[:8], "examples_false_alarm": false_alarms[:8]}


def _mape(errs: list[float]) -> float:
    return round(float(np.mean(np.abs(errs)) * 100), 2)


def eval_forecast(llm: MeshLLM, mode: str) -> dict:
    """Rolling origin: history up to the end of month k, predict month k+1."""
    rec: dict[str, list[float]] = {k: [] for k in ("spend_fp", "spend_naive", "spend_mean3", "bal_fp", "bal_naive")}
    covered, n = 0, 0
    reg = {"bal": [], "cov": []}
    for persona in PERSONAS:
        for seed in TEST_SEEDS:
            df, _, _, _ = prepare(persona, seed)
            c = categorizer.categorize(df, mode=mode, llm=llm)
            months = sorted(c["date"].dt.to_period("M").unique())
            for cut in months[3:-1]:
                hist = c[c["date"].dt.to_period("M") <= cut]
                nxt = c[c["date"].dt.to_period("M") == cut + 1]
                a = analyzer.analyze(hist, narrate=False)
                start_bal = float(hist["balance"].iloc[-1])
                horizon = ((cut + 1).end_time.normalize() - a.as_of.normalize()).days  # through month end
                fc = forecaster.forecast(hist, a, start_bal, horizon_days=horizon, n_sims=300, seed=seed)
                actual_out = float(-nxt.loc[nxt["amount"] < 0, "amount"].sum())
                actual_bal = float(nxt["balance"].iloc[-1])
                pred_out = float(np.median(fc.outflows.sum(axis=1)))
                pred_bal = float(np.median(fc.paths[:, -1]))
                prev = hist[hist["date"].dt.to_period("M") == cut]
                outs = [float(-hist[(hist["date"].dt.to_period("M") == m) & (hist["amount"] < 0)]["amount"].sum())
                        for m in months if cut - 2 <= m <= cut]
                rec["spend_fp"].append((pred_out - actual_out) / actual_out)
                rec["spend_naive"].append((outs[-1] - actual_out) / actual_out)
                rec["spend_mean3"].append((np.mean(outs) - actual_out) / actual_out)
                rec["bal_fp"].append((pred_bal - actual_bal) / actual_bal)
                rec["bal_naive"].append((start_bal + float(prev["amount"].sum()) - actual_bal) / actual_bal)
                lo, hi = np.percentile(fc.paths[:, -1], [10, 90])
                covered += int(lo <= actual_bal <= hi)
                n += 1
                if cut + 1 != months[-1]:
                    reg["bal"].append((pred_bal - actual_bal) / actual_bal)
                    reg["cov"].append(lo <= actual_bal <= hi)
    return {
        "folds": n,
        "next_month_spend_mape_pct": {"finpilot": _mape(rec["spend_fp"]), "naive_last_month": _mape(rec["spend_naive"]),
                                      "mean_last_3_months": _mape(rec["spend_mean3"])},
        "month_end_balance_mape_pct": {"finpilot": _mape(rec["bal_fp"]), "naive_repeat_last_net_flow": _mape(rec["bal_naive"])},
        "p10_p90_coverage_pct": round(covered / n * 100, 1),
        "regular_months_only": {"folds": len(reg["bal"]), "month_end_balance_mape_pct": _mape(reg["bal"]),
                                "p10_p90_coverage_pct": round(float(np.mean(reg["cov"])) * 100, 1)},
        "note": "The final fold of each dataset contains the planted anomalies (spikes, one-off purchases), "
                "which no forecast should anticipate; 'regular months' excludes it.",
    }


def eval_simulation(llm: MeshLLM, mode: str) -> dict:
    checks = Counter()
    failures = []
    for persona in PERSONAS:
        for seed in TEST_SEEDS[:5]:
            df, _, _, bal = prepare(persona, seed)
            c = categorizer.categorize(df, mode=mode, llm=llm)
            a = analyzer.analyze(c, narrate=False)

            def run(sc):
                return forecaster.simulate(c, a, bal, sc, goals=[GOAL], seed=seed)

            amt = 50000.0
            r = run(forecaster.Scenario(purchases=[{"date": "2026-10-20", "amount": amt}]))
            ok = abs(r["delta_end_balance_p50"] + amt) <= 1
            checks["purchase_shifts_balance_exactly"] += ok
            if not ok:
                failures.append(f"{persona}/{seed} purchase delta {r['delta_end_balance_p50']}")

            subs = [x for x in a.recurring if x.is_subscription]
            if subs:
                s0 = subs[0]
                r = run(forecaster.Scenario(cancel=[s0.merchant]))
                cycles = sum(1 for k in range(1, 20)
                             if s0.last_date + pd.DateOffset(months=s0.period_months * k) <= pd.Timestamp(r["with_decision"]["to"]))
                ok = abs(r["delta_end_balance_p50"] - (-s0.amount) * cycles) <= 1
                checks["cancel_saves_exact_subscription_cost"] += ok
                if not ok:
                    failures.append(f"{persona}/{seed} cancel {s0.merchant} delta {r['delta_end_balance_p50']}")

            r = run(forecaster.Scenario(category_cuts={"shopping": 0.5}))
            checks["cutting_spend_never_hurts"] += r["delta_end_balance_p50"] >= 0

            bill = {"name": "Annual insurance", "amount": 20000, "next_date": "2026-11-10", "every_months": 12}
            with_bill = forecaster.simulate(c, a, bal, forecaster.Scenario(), goals=[GOAL], seed=seed, bills=[bill])
            plain = run(forecaster.Scenario())
            checks["declared_bill_shifts_balance_exactly"] += abs(
                with_bill["baseline"]["end_balance_p50"] - plain["baseline"]["end_balance_p50"] + 20000) <= 1

            big = run(forecaster.Scenario(purchases=[{"date": "2026-10-20", "amount": bal * 3}]))
            checks["unaffordable_purchase_rejected"] += big["verdict"] == "not_affordable"
            small = run(forecaster.Scenario(purchases=[{"date": "2026-10-20", "amount": 500}]))
            checks["tiny_purchase_not_rejected"] += small["verdict"] != "not_affordable"
            checks["_runs"] += 1
    runs = checks.pop("_runs")
    return {"runs": runs, "pass_rate": {k: f"{v}/{runs}" for k, v in checks.items()}, "failures": failures[:5]}


def eval_advice(llm: MeshLLM, use_llm: bool) -> dict:
    stats = Counter()
    passes, latency = [], []
    for persona in PERSONAS:
        for seed in TEST_SEEDS[:3] if use_llm else TEST_SEEDS:
            df, _, _, bal = prepare(persona, seed)
            audit = AuditTrail(f"eval-{persona}")
            t0 = time.perf_counter()
            st = run_pipeline(df, bal, audit, llm, goals=[GOAL], mode="auto" if use_llm else "heuristic")
            latency.append(time.perf_counter() - t0)
            findings = {f.id: f for f in st["analysis"].findings + st["forecast_findings"]}
            final = st["final"]
            stats["runs"] += 1
            stats["delivered"] += len(final)
            stats["llm_written"] += sum(r["source"] == "llm" for r in final)
            stats["rejections"] += sum(1 for d in audit.by_agent("critic") if d.decision.startswith("REJECT"))
            passes.append(st.get("first_pass_rate", 1.0))
            stats["revision_loops"] += max(st["attempts"] - 1, 0)
            recheck = critic.critique(final, findings, st["candidates"], st["analysis"].stats)
            stats["delivered_pass_recheck"] += sum(v["passed"] for v in recheck)
            stats["with_evidence"] += sum(bool(r["finding_ids"]) for r in final)
            stats["quantified"] += sum((r["monthly_impact"] or r["one_off_impact"]) > 0 for r in final)
            # Did the advice cover the highest-impact problems the system knew about?
            top = {c["finding_ids"][0] for c in st["candidates"][:3]}
            stats["top3_covered"] += len(top & {fid for r in final for fid in r["finding_ids"]})
            stats["top3_total"] += len(top)
    d = stats["delivered"] or 1
    return {
        "runs": stats["runs"],
        "recommendations_delivered": stats["delivered"],
        "llm_written_pct": round(stats["llm_written"] / d * 100, 1),
        "critic_first_pass_rate_pct": round(float(np.mean(passes)) * 100, 1),
        "critic_rejections": stats["rejections"],
        "revision_loops": stats["revision_loops"],
        "delivered_grounded_pct": round(stats["delivered_pass_recheck"] / d * 100, 1),
        "delivered_with_evidence_pct": round(stats["with_evidence"] / d * 100, 1),
        "delivered_quantified_pct": round(stats["quantified"] / d * 100, 1),
        "top3_issue_coverage_pct": round(stats["top3_covered"] / max(stats["top3_total"], 1) * 100, 1),
        "avg_pipeline_seconds": round(float(np.mean(latency)), 2),
    }


def eval_goals(llm: MeshLLM, mode: str) -> dict:
    """Goals set relative to each user's own trajectory: does the status (and so the advice) follow progress,
    and does saving the advised extra amount per month really reach the goal?"""
    scales = {"0.5x projected": 0.5, "0.9x projected": 0.9, "1.5x projected": 1.5, "3x projected": 3.0}
    status = {k: Counter() for k in scales}
    advice_matches, closes, behind, monotonic, runs = 0, 0, 0, 0, 0
    for persona in PERSONAS:
        for seed in TEST_SEEDS:
            df, _, _, bal = prepare(persona, seed)
            c = categorizer.categorize(df, mode=mode, llm=llm)
            a = analyzer.analyze(c, narrate=False)
            target_date = (a.as_of + pd.DateOffset(months=6)).date().isoformat()
            probe = forecaster.forecast(c, a, bal, goals=[{"name": "probe", "target_amount": 1, "target_date": target_date}])
            projected = max(probe.goals[0]["projected_p50"], 1000)
            probs = []
            for label, k in scales.items():
                goal = {"name": "Goal", "target_amount": round(projected * k), "target_date": target_date}
                fc = forecaster.forecast(c, a, bal, goals=[goal], seed=seed)
                g = fc.goals[0]
                status[label][g["status"]] += 1
                probs.append(g["probability_pct"])
                kinds = {f.kind for f in forecaster.forecast_findings(fc, a)}
                expected = {"behind": "goal_at_risk", "ahead": "goal_ahead"}.get(g["status"])
                advice_matches += expected is None or expected in kinds
                if g["status"] == "behind":
                    behind += 1
                    follow = forecaster.Scenario(monthly_changes=[{"amount": g["required_extra_monthly"],
                                                                   "label": "advised saving", "start": None}])
                    after = forecaster.forecast(c, a, bal, goals=[goal], seed=seed, scenario=follow).goals[0]
                    closes += after["projected_p50"] >= 0.97 * goal["target_amount"]
            monotonic += all(x >= y for x, y in zip(probs, probs[1:]))
            runs += 1
    return {
        "runs": runs,
        "status_by_goal_size": {k: dict(v) for k, v in status.items()},
        "probability_falls_as_goal_grows_pct": round(monotonic / runs * 100, 1),
        "advice_matches_status_pct": round(advice_matches / (runs * len(scales)) * 100, 1),
        "behind_goals": behind,
        "following_advice_reaches_goal_pct": round(closes / behind * 100, 1) if behind else None,
    }


def to_markdown(res: dict) -> str:
    L = ["# FinPilot evaluation results", "",
         f"Generated by `python -m eval.run_eval{' --offline' if res['mode'] == 'offline' else ''}` "
         f"(mode: **{res['mode']}**, test seeds {TEST_SEEDS[0]}-{TEST_SEEDS[-1]}, personas: {', '.join(PERSONAS)}).", ""]
    L += ["## 1. Categorisation", "", "| Tier | Txns | Accuracy | Macro-F1 | Rule coverage | Long-tail accuracy |", "|---|---|---|---|---|---|"]
    for k, v in res["categorization"].items():
        tail = "" if v["long_tail_accuracy"] is None else f"{v['long_tail_accuracy']:.1%}"
        L.append(f"| {k} | {v['transactions']} | {v['accuracy']:.1%} | {v['macro_f1']:.3f} | {v['rule_coverage']:.1%} | {tail} |")
    det = res["detection"]
    L += ["", f"## 2. Pattern & anomaly detection ({det['datasets']} labelled datasets)", "",
          "| Kind | TP | FP | FN | Precision | Recall |", "|---|---|---|---|---|---|"]
    for k, v in det["by_kind"].items():
        L.append(f"| {k} | {v['tp']} | {v['fp']} | {v['fn']} | {v['precision']} | {v['recall']} |")
    fc = res["forecast"]
    L += ["", f"## 3. Forecast backtest ({fc['folds']} rolling-origin folds)", "",
          "| Target | FinPilot | Baselines |", "|---|---|---|",
          f"| Next-month spend MAPE | {fc['next_month_spend_mape_pct']['finpilot']}% | last month {fc['next_month_spend_mape_pct']['naive_last_month']}%, "
          f"3-month mean {fc['next_month_spend_mape_pct']['mean_last_3_months']}% |",
          f"| Month-end balance MAPE | {fc['month_end_balance_mape_pct']['finpilot']}% | repeat last month's net flow "
          f"{fc['month_end_balance_mape_pct']['naive_repeat_last_net_flow']}% |",
          f"| Actual balance inside P10-P90 band | {fc['p10_p90_coverage_pct']}% | (ideal ≈ 80%) |",
          f"| Regular months only ({fc['regular_months_only']['folds']} folds): balance MAPE / band coverage | "
          f"{fc['regular_months_only']['month_end_balance_mape_pct']}% / {fc['regular_months_only']['p10_p90_coverage_pct']}% | |",
          "", f"_{fc['note']}_"]
    sim = res["simulation"]
    L += ["", f"## 4. What-if simulation properties ({sim['runs']} runs)", ""]
    L += [f"- {k.replace('_', ' ')}: **{v}**" for k, v in sim["pass_rate"].items()]
    adv = res["advice"]
    L += ["", f"## 5. Advice quality ({adv['runs']} pipeline runs)", "", "| Metric | Value |", "|---|---|"]
    L += [f"| {k.replace('_', ' ')} | {v} |" for k, v in adv.items()]
    gl = res["goals"]
    L += ["", f"## 6. Goal-adaptive advice ({gl['runs']} datasets x 4 goal sizes)", "",
          "Each user gets goals at 0.5x, 0.9x, 1.5x and 3x of their own projected savings in 6 months.", "",
          "| Goal size | ahead | on track | behind |", "|---|---|---|---|"]
    L += [f"| {k} | {v.get('ahead', 0)} | {v.get('on_track', 0)} | {v.get('behind', 0)} |"
          for k, v in gl["status_by_goal_size"].items()]
    L += ["", "| Metric | Value |", "|---|---|",
          f"| chance of hitting the goal falls as the goal grows | {gl['probability_falls_as_goal_grows_pct']}% |",
          f"| advice matches the goal status (at-risk vs. ahead) | {gl['advice_matches_status_pct']}% |",
          f"| behind goals where saving the advised extra/month reaches the target (P50) | "
          f"{gl['following_advice_reaches_goal_pct']}% of {gl['behind_goals']} |"]
    return "\n".join(L) + "\n"


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--offline", action="store_true", help="no LLM calls (deterministic)")
    ap.add_argument("--out", default=str(ROOT / "eval"))
    args = ap.parse_args()
    if args.offline:
        os.environ["FINPILOT_OFFLINE"] = "1"
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")  # ₹ and ≈ on a Windows console
    llm = MeshLLM(offline=args.offline or None)
    set_llm(llm)
    use_llm = llm.available
    if use_llm:
        # Fail fast instead of running the whole eval on fallbacks and labelling it "mesh".
        chat = llm.ping()
        if not chat["ok"]:
            sys.exit(f"Mesh is not usable: {chat['reason']}. Fix that, or run `python -m eval.run_eval --offline`.")
    print(f"Mode: {'mesh' if use_llm else 'offline'}")
    res = {"mode": "mesh" if use_llm else "offline"}
    mode = "auto" if use_llm else "heuristic"  # downstream evals use the real categoriser
    for name, fn in [("categorization", lambda: eval_categorization(llm, use_llm)), ("detection", lambda: eval_detection(llm, mode)),
                     ("forecast", lambda: eval_forecast(llm, mode)), ("simulation", lambda: eval_simulation(llm, mode)),
                     ("advice", lambda: eval_advice(llm, use_llm)), ("goals", lambda: eval_goals(llm, mode))]:
        t0 = time.perf_counter()
        res[name] = fn()
        print(f"  {name}: {time.perf_counter() - t0:.1f}s")
    res["llm_calls"] = {"total": llm.stats.calls, "failures": llm.stats.failures, "by_agent": llm.stats.by_agent}
    if use_llm and llm.blocked_reason:
        sys.exit(f"Mesh stopped working mid-run ({llm.blocked_reason}); results not written.")
    out = Path(args.out)
    suffix = "" if use_llm else "_offline"
    (out / f"results{suffix}.json").write_text(json.dumps(res, indent=2, default=str), encoding="utf-8")
    (out / f"RESULTS{suffix}.md").write_text(to_markdown(res), encoding="utf-8")
    print(to_markdown(res))


if __name__ == "__main__":
    main()
