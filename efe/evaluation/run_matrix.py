"""Scenario × filter × seed: every filter on every run, metrics tables and the main figures.

    python -m efe.evaluation.run_matrix --tier 0                      # Tier 0: simulate + evaluate
    python -m efe.evaluation.run_matrix --tier 1 --runs runs/tier1    # Tier 1: ingested Isaac/PX4 runs

Seed 0 is the tuning seed and is excluded from the reported numbers (--seeds defaults to 1-4).
Outputs: <out>/tables/metrics_all.csv (one row per run × filter), <out>/tables/summary.csv (mean ± std),
<out>/figures/*.pdf|png.
"""

from __future__ import annotations

import argparse
import os
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np
import pandas as pd

from ..estimation import models as M
from ..estimation.configs import load_filter, load_priors
from ..estimation.runner import run_filter, vehicle_params
from ..io.rundata import RunData
from ..scenario import REPO, load_scenario
from . import plots
from .metrics import errors, evaluate, nees, scenario_t0

FILTER_NAMES = ["KF", "EKF-A", "EKF-B", "EKF-B+bias", "EKF-C", "MO"]
SCENARIOS = ["S1_main", "S2_hover_only", "S3_sag_only", "S4_aggressive", "S5_payload"]


def parse_seeds(s):
    out = []
    for part in s.split(","):
        if "-" in part:
            a, b = part.split("-")
            out += list(range(int(a), int(b) + 1))
        else:
            out.append(int(part))
    return out


def tier0_run(scenario, seed, cache_dir=None, overrides=None):
    """Simulate (or load from cache) one Tier 0 run."""
    from ..pysim.sim import simulate

    path = Path(cache_dir) / scenario / f"seed{seed}" if cache_dir else None
    if path is not None and not overrides and (path / "imu.parquet").exists():
        return RunData.load(path)
    run = simulate(load_scenario(scenario, overrides), seed=seed)
    if path is not None and not overrides:
        run.save(path)
    return run


def get_run(tier, scenario, seed, runs_dir):
    if tier == 0:
        return tier0_run(scenario, seed, runs_dir)
    return RunData.load(Path(runs_dir) / scenario / f"seed{seed}")


def _job(args):
    tier, scenario, seed, runs_dir, filter_names, prm_overrides = args
    run = get_run(tier, scenario, seed, runs_dir)
    prm = vehicle_params(run, prm_overrides)
    priors = load_priors()
    rows, nees_t = [], {}
    for name in filter_names:
        cfg = load_filter(name, priors)
        res = run_filter(run, cfg, prm)
        rows.append(evaluate(res, run, cfg))
        if res.P is not None:
            e, _ = errors(res, run, getattr(cfg, "use_voltage", True))
            v, dof = nees(res, e, "F")
            nees_t[name] = (res.t - scenario_t0(run), v, dof, res.nis_acc)
    return rows, (scenario, seed, nees_t)


def summarise(df):
    num = [c for c in df.columns if c not in ("filter", "scenario", "seed") and pd.api.types.is_numeric_dtype(df[c])]
    long = df.melt(id_vars=["scenario", "filter", "seed"], value_vars=num, var_name="metric")
    g = long.groupby(["scenario", "filter", "metric"])["value"]
    return g.agg(["mean", "std", "count"]).reset_index()


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--tier", type=int, default=0, choices=[0, 1])
    ap.add_argument("--scenarios", nargs="+", default=SCENARIOS)
    ap.add_argument("--filters", nargs="+", default=FILTER_NAMES)
    ap.add_argument("--seeds", default="1-4")
    ap.add_argument("--runs", default=None, help="run directory (default runs/tier<N>)")
    ap.add_argument("--out", default=None, help="output directory (default results/tier<N>)")
    ap.add_argument("--jobs", type=int, default=max(1, (os.cpu_count() or 2) - 2))
    ap.add_argument("--figure-seed", type=int, default=1)
    args = ap.parse_args(argv)

    runs_dir = Path(args.runs or REPO / "runs" / f"tier{args.tier}")
    out = Path(args.out or REPO / "results" / f"tier{args.tier}")
    seeds = parse_seeds(args.seeds)
    jobs = [(args.tier, sc, s, str(runs_dir), args.filters, None) for sc in args.scenarios for s in seeds]
    print(f"[run_matrix] tier {args.tier}: {len(args.scenarios)} scenarios × {len(seeds)} seeds × "
          f"{len(args.filters)} filters on {args.jobs} processes")
    rows, nees_runs = [], []
    with ProcessPoolExecutor(args.jobs) as ex:
        for r, nt in ex.map(_job, jobs):
            rows += r
            nees_runs.append(nt)
    df = pd.DataFrame(rows)
    (out / "tables").mkdir(parents=True, exist_ok=True)
    df.to_csv(out / "tables" / "metrics_all.csv", index=False)
    summary = summarise(df)
    summary.to_csv(out / "tables" / "summary.csv", index=False)
    headline(summary, out / "tables" / "headline.csv")

    figures(args, runs_dir, out, summary, nees_runs)
    print(f"[run_matrix] wrote {out}/tables and {out}/figures")


HEADLINE = ["rmse_Fxy", "rmse_Fz", "bias_Fz_hover", "bias_Fz_end", "noise_F_hover", "rise_time", "detect_delay",
            "k_err_end_pct", "lam_err_end_pct", "anees_F", "nis_acc_in95", "step_us"]


def headline(summary, path):
    s = summary[summary["metric"].isin(HEADLINE)].copy()
    s["value"] = s.apply(lambda r: f"{r['mean']:.3g} ± {r['std']:.2g}" if np.isfinite(r["std"]) else f"{r['mean']:.3g}", axis=1)
    t = s.pivot_table(index=["scenario", "filter"], columns="metric", values="value", aggfunc="first")
    t = t[[c for c in HEADLINE if c in t.columns]]
    t.to_csv(path)
    with pd.option_context("display.width", 250, "display.max_columns", 30):
        print(t)


def figures(args, runs_dir, out, summary, nees_runs):
    fig = out / "figures"
    priors = load_priors()
    # per-scenario time series for the figure seed
    for sc in args.scenarios:
        run = get_run(args.tier, sc, args.figure_seed, runs_dir)
        prm = vehicle_params(run)
        results, by_name = [], {}
        for name in args.filters:
            cfg = load_filter(name, priors)
            res = run_filter(run, cfg, prm)
            results.append(res)
            by_name[name] = (res, run, getattr(cfg, "use_voltage", True))
        plots.force_vs_truth(results, run, fig / f"{sc}_force_vs_truth")
        plots.fz_bias_vs_battery(by_name, fig / f"{sc}_fz_error_vs_battery")
        if "EKF-B+bias" in by_name:
            res, _, _ = by_name["EKF-B+bias"]
            plots.params_convergence(res, run, fig / f"{sc}_params_convergence")
            plots.error_with_bounds(res, run, fig / f"{sc}_error_3sigma")
    # ANEES over the seeds, per scenario
    for sc in args.scenarios:
        per = [nt for (s, _, nt) in nees_runs if s == sc]
        if not per:
            continue
        series, nis, dof = {}, {}, 3
        for name in args.filters:
            vals = [p[name] for p in per if name in p]
            if not vals:
                continue
            n = min(len(v[1]) for v in vals)
            series[name] = (vals[0][0][:n], np.mean([v[1][:n] for v in vals], axis=0))
            nis[name] = np.concatenate([v[3][np.isfinite(v[3])] for v in vals])
            dof = vals[0][2]
        plots.nees_nis(series, fig / f"{sc}_nees_nis", dof, len(per), nis_acc=nis)
    plots.bars(summary, "rmse_Fz", fig / "bars_rmse_Fz", "$F_z$ RMSE [N]")
    plots.bars(summary, "rmse_Fxy", fig / "bars_rmse_Fxy", "lateral RMSE [N]")


if __name__ == "__main__":
    main()
