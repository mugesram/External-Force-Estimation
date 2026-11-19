"""Robustness, ablation and consistency experiments (plan §6.4-6.6).

    python -m efe.evaluation.experiments                 # all, Tier 0
    python -m efe.evaluation.experiments qf_sweep gnss   # some
    python -m efe.evaluation.experiments --tier 1 --base-seeds 1 ...

Every experiment is a list of variants; each variant runs on the held-out seeds and is summarised as mean ± std.
Outputs: results/tier<N>/experiments/<name>.csv, robustness.csv (all variants, the headline metrics), figures.
"""

from __future__ import annotations

import argparse
import os
from concurrent.futures import ProcessPoolExecutor
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd

from ..estimation import models as M
from ..estimation.configs import load_filter, load_priors
from ..estimation.runner import run_filter, vehicle_params
from ..scenario import REPO
from . import plots, replay
from .metrics import chi2_band, errors, evaluate, nees, scenario_t0
from .run_matrix import get_run, tier0_run

K_TRUE, TAU_TRUE = 10.3437818, 0.04          # truth (vehicle_iris.yaml); the prior-error sweep is relative to these


@dataclass
class Variant:
    experiment: str
    label: str
    scenario: str = "S1_main"
    filter: str = "EKF-B+bias"
    cfg: dict = field(default_factory=dict)        # filter config overrides
    prm: dict = field(default_factory=dict)        # vehicle parameter overrides (model error)
    sim: dict = field(default_factory=dict)        # truth overrides (Tier 0 only: re-simulates)
    transform: tuple = ()                          # data transform: (name, *args)
    replay_seed: int | None = None                 # resample sensor noise (replay MC)


def _apply(run, transform):
    if not transform:
        return run
    name, *a = transform
    if name == "gnss_rate":
        return replay.gnss_rate(run, *a)
    if name == "gnss_dropout":
        return replay.gnss_dropout(run, *a)
    if name == "battery_rate":
        return replay.battery_rate(run, *a)
    if name == "resample":
        return replay.resample_sensors(run, a[0], a[1] if len(a) > 1 else None)
    raise ValueError(name)


def _job(args):
    v, tier, seed, runs_dir = args
    if v.sim:
        run = tier0_run(v.scenario, seed, None, v.sim)
    else:
        run = get_run(tier, v.scenario, seed, runs_dir)
    run = _apply(run, v.transform)
    if v.replay_seed is not None:
        run = replay.resample_sensors(run, v.replay_seed)
    cfg = load_filter(v.filter, load_priors(), v.cfg)
    res = run_filter(run, cfg, vehicle_params(run, v.prm))
    row = evaluate(res, run, cfg)
    row.update({"experiment": v.experiment, "variant": v.label, "seed": seed})
    if v.replay_seed is not None:
        row["replay"] = v.replay_seed
    return row


def _run(variants, tier, seeds, runs_dir, jobs):
    work = [(v, tier, s, runs_dir) for v in variants for s in seeds]
    with ProcessPoolExecutor(jobs) as ex:
        return pd.DataFrame(list(ex.map(_job, work, chunksize=1)))


def summarise(df, metrics):
    g = df.groupby(["experiment", "variant", "filter"], sort=False)
    out = g[metrics].agg(["mean", "std"])
    out.columns = [f"{m}_{s}" for m, s in out.columns]
    return out.reset_index()


METRICS = ["rmse_Fxy", "rmse_Fz", "bias_Fz_hover", "bias_Fz_end", "noise_F_hover", "rise_time", "k_err_end_pct",
           "lam_err_end_pct", "k_conv_time", "anees_F", "anees_full", "nis_acc_in95"]


# --- the experiments -------------------------------------------------------------------------------------------
def exp_qf_sweep():
    return [Variant("qf_sweep", f"q_F={q:g}", cfg={"q_F": q}) for q in (0.02, 0.2, 2.0, 20.0, 200.0)]


def exp_prior_error():
    vs = []
    for f in (0.90, 0.95, 1.00, 1.05, 1.10):
        vs.append(Variant("prior_error", f"k0 x{f:.2f}", cfg={"k0": K_TRUE * f}))
    for f in (0.5, 1.0, 1.2, 1.5):
        vs.append(Variant("prior_error", f"tau0 x{f:.1f}", cfg={"lam0": 1.0 / (TAU_TRUE * f)}))
    return vs


def exp_model_error():
    return [
        Variant("model_error", "nominal"),
        Variant("model_error", "mass -10%", prm={"mass": 1.35}),
        Variant("model_error", "mass +10%", prm={"mass": 1.65}),
        Variant("model_error", "drag x0.5", prm={"drag": [0.25, 0.15, 0.0]}),
        Variant("model_error", "drag x1.5", prm={"drag": [0.75, 0.45, 0.0]}),
        Variant("model_error", "u0 = 0.05", prm={"u0": 0.05}),
        Variant("model_error", "u0 = 0.15", prm={"u0": 0.15}),
        Variant("model_error", "truth V^1.8", sim={"vehicle": {"voltage_exponent": 1.8}}),
        Variant("model_error", "truth motor spread 3%", sim={"vehicle": {"k_f_spread": 0.03}}),
    ]


def exp_accel_bias():
    vs = []
    for f in ("EKF-B", "EKF-B+bias"):
        vs.append(Variant("accel_bias", "IMU bias on", filter=f))
        vs.append(Variant("accel_bias", "IMU bias zeroed", filter=f, sim={"sensors": {"acc_bias_sigma": 1e-9}}))
    return vs


def exp_gnss():
    return [
        Variant("gnss", "10 Hz"),
        Variant("gnss", "5 Hz", transform=("gnss_rate", 5.0)),
        Variant("gnss", "1 Hz", transform=("gnss_rate", 1.0)),
        Variant("gnss", "5 s dropout in gust", transform=("gnss_dropout", 66.0, 5.0)),
    ]


def exp_voltage():
    return [
        Variant("voltage", "nominal (50 Hz, LP 2 Hz)"),
        Variant("voltage", "resampled, noise x1", transform=("resample", 0, {"bat_noise": 0.05})),
        Variant("voltage", "resampled, noise x5", transform=("resample", 0, {"bat_noise": 0.25})),
        Variant("voltage", "no low-pass", cfg={"voltage_lp_hz": 0.0}),
        Variant("voltage", "1 Hz (battery_status over DDS)", transform=("battery_rate", 1.0)),
    ]


def exp_attitude():
    return [Variant("attitude", m, cfg={"attitude_mode": m}) for m in ("gyro", "slerp", "truth")]


def exp_regularise():
    """The k_b pseudo-measurement (bench prior as a weak measurement) vs none."""
    return [Variant("regularise", "none"),
            Variant("regularise", "k pseudo sigma 2 N", cfg={"k_pseudo_sigma": 2.0}),
            Variant("regularise", "k pseudo sigma 5 N", cfg={"k_pseudo_sigma": 5.0})]


def exp_tilt():
    """Errors in the flight controller's attitude: does estimating a tilt error (EKF-C) help?"""
    vs = []
    for f in ("EKF-B+bias", "EKF-C"):
        vs.append(Variant("tilt", "estimated attitude", filter=f))
        vs.append(Variant("tilt", "true attitude", filter=f, cfg={"attitude_mode": "truth"}))
    return vs


EXPERIMENTS = {
    "qf_sweep": exp_qf_sweep, "prior_error": exp_prior_error, "model_error": exp_model_error,
    "accel_bias": exp_accel_bias, "gnss": exp_gnss, "voltage": exp_voltage, "attitude": exp_attitude,
    "regularise": exp_regularise, "tilt": exp_tilt,
}


# --- figure experiments ----------------------------------------------------------------------------------------
def observability(tier, runs_dir, out, seed):
    run1 = get_run(tier, "S1_main", seed, runs_dir)
    run2 = get_run(tier, "S2_hover_only", seed, runs_dir)
    cfg = load_filter("EKF-B+bias")
    r1 = run_filter(run1, cfg, vehicle_params(run1))
    r2 = run_filter(run2, cfg, vehicle_params(run2))
    plots.observability(r1, r2, run1, run2, out / "figures" / "observability")
    rows = []
    for name, r, run in [("S1 excited", r1, run1), ("S2 hover only", r2, run2)]:
        t = r.t - scenario_t0(run)
        for tq in (10, 35, 60, 119):
            i = min(np.searchsorted(t, tq), len(t) - 1)
            rows.append({"flight": name, "t": tq, "sd_k": r.sd[i, M.IK], "sd_lam": r.sd[i, M.IL], "sd_Fz": r.sd[i, 8]})
    pd.DataFrame(rows).to_csv(out / "experiments" / "observability.csv", index=False)


def _mc_job(args):
    tier, runs_dir, scenario, base_seed, k, filters = args
    run = replay.resample_sensors(get_run(tier, scenario, base_seed, runs_dir), 10_000 + k)
    out = {}
    for name in filters:
        cfg = load_filter(name)
        res = run_filter(run, cfg, vehicle_params(run))
        e, _ = errors(res, run, cfg.use_voltage)
        vF, dF = nees(res, e, "F")
        vA, dA = nees(res, e, "full")
        out[name] = (res.t - scenario_t0(run), vF, dF, vA, dA, res.nis_acc, res.nis_gnss)
    return out


def replay_mc(tier, runs_dir, out, base_seed, n, jobs, scenario="S1_main", filters=("EKF-B", "EKF-B+bias")):
    """N noise realisations on one truth trajectory: ANEES over time with the χ² band for N runs."""
    with ProcessPoolExecutor(jobs) as ex:
        res = list(ex.map(_mc_job, [(tier, runs_dir, scenario, base_seed, k, filters) for k in range(n)]))
    series, nis, rows = {}, {}, []
    for name in filters:
        m = min(len(r[name][1]) for r in res)
        t = res[0][name][0][:m]
        F = np.mean([r[name][1][:m] for r in res], axis=0)
        A = np.mean([r[name][3][:m] for r in res], axis=0)
        dF, dA = res[0][name][2], res[0][name][4]
        series[name] = (t, F)
        nis[name] = np.concatenate([r[name][5][np.isfinite(r[name][5])] for r in res])
        loF, hiF = chi2_band(dF, n)
        loA, hiA = chi2_band(dA, n)
        w = t >= 5.0
        g = np.concatenate([r[name][6] for r in res])
        lo3, hi3 = chi2_band(3, 1)
        lo6, hi6 = chi2_band(6, 1)
        rows.append({
            "filter": name, "runs": n,
            "ANEES_F_mean": float(np.mean(F[w])), "ANEES_F_dof": dF, "band_F": f"[{loF:.2f}, {hiF:.2f}]",
            "ANEES_F_frac_in_band": float(np.mean((F[w] >= loF) & (F[w] <= hiF))),
            "ANEES_full_mean": float(np.mean(A[w])), "ANEES_full_dof": dA, "band_full": f"[{loA:.2f}, {hiA:.2f}]",
            "ANEES_full_frac_in_band": float(np.mean((A[w] >= loA) & (A[w] <= hiA))),
            "NIS_acc_mean": float(np.mean(nis[name])), "NIS_acc_in95": float(np.mean((nis[name] >= lo3) & (nis[name] <= hi3))),
            "NIS_gnss_mean": float(np.mean(g)), "NIS_gnss_in95": float(np.mean((g >= lo6) & (g <= hi6))),
        })
    pd.DataFrame(rows).to_csv(out / "experiments" / "replay_mc_consistency.csv", index=False)
    plots.nees_nis(series, out / "figures" / "replay_mc_anees", res[0][filters[-1]][2], n, nis_acc=nis)
    print(pd.DataFrame(rows).to_string(index=False))


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("names", nargs="*", default=list(EXPERIMENTS) + ["observability", "replay_mc"])
    ap.add_argument("--tier", type=int, default=0, choices=[0, 1])
    ap.add_argument("--seeds", default="1-4")
    ap.add_argument("--runs", default=None)
    ap.add_argument("--out", default=None)
    ap.add_argument("--mc-runs", type=int, default=50)
    ap.add_argument("--jobs", type=int, default=max(1, (os.cpu_count() or 2) - 2))
    args = ap.parse_args(argv)
    from .run_matrix import parse_seeds

    seeds = parse_seeds(args.seeds)
    runs_dir = str(Path(args.runs or REPO / "runs" / f"tier{args.tier}"))
    out = Path(args.out or REPO / "results" / f"tier{args.tier}")
    (out / "experiments").mkdir(parents=True, exist_ok=True)

    tables = []
    for name in args.names:
        if name == "observability":
            observability(args.tier, runs_dir, out, seeds[0])
            continue
        if name == "replay_mc":
            replay_mc(args.tier, runs_dir, out, seeds[0], args.mc_runs, args.jobs)
            continue
        variants = EXPERIMENTS[name]()
        if args.tier == 1:
            variants = [v for v in variants if not v.sim]       # truth overrides need re-simulation
        df = _run(variants, args.tier, seeds, runs_dir, args.jobs)
        df.to_csv(out / "experiments" / f"{name}.csv", index=False)
        s = summarise(df, [m for m in METRICS if m in df.columns])
        tables.append(s)
        print(f"\n== {name}")
        cols = ["variant", "filter"] + [c for c in s.columns if c.endswith("_mean")]
        with pd.option_context("display.width", 250, "display.max_columns", 30, "display.precision", 3):
            print(s[cols].to_string(index=False))
        if name == "qf_sweep":
            q = s.assign(q_F=[float(v.split("=")[1]) for v in s["variant"]],
                         noise_F_hover=s["noise_F_hover_mean"], rise_time=s["rise_time_mean"])
            plots.tradeoff(q, out / "figures" / "qf_tradeoff", x="noise_F_hover", y="rise_time", label="q_F")
            plots.tradeoff(q.assign(bias=q["rmse_Fz_mean"]), out / "figures" / "qf_rmse_tradeoff",
                           x="noise_F_hover", y="bias", label="q_F", ylabel="$F_z$ RMSE [N]")
    if tables:
        path = out / "experiments" / "robustness.csv"
        new = pd.concat(tables)
        if path.exists():
            old = pd.read_csv(path)
            old = old[~old["experiment"].isin(new["experiment"].unique())]
            new = pd.concat([old, new])
        new.to_csv(path, index=False)


if __name__ == "__main__":
    main()
