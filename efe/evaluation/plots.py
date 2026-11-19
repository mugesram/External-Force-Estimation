"""Report figures (matplotlib, light background, PDF + PNG).

Colour follows the filter, never its rank: each filter has a fixed slot of the reference categorical palette, so a
figure that shows only some filters keeps their colours. Truth is primary ink, dashed. Every multi-series figure has a
legend (identity is never colour alone). Bands (±3σ, χ² bounds) are low-alpha fills of the series hue or neutral grey.
"""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

from ..estimation import models as M  # noqa: E402
from .metrics import chi2_band, errors, nees, scenario_t0  # noqa: E402

INK = "#0b0b0b"
INK2 = "#52514e"
GRID = "#e4e3df"
BAND = "#b9b8b2"
SURFACE = "#fcfcfb"
# reference categorical palette, fixed order; slot per filter (proposed filter first)
COLORS = {
    "EKF-B+bias": "#2a78d6",
    "EKF-B": "#eb6834",
    "EKF-A": "#1baf7a",
    "KF": "#eda100",
    "MO": "#e87ba4",
    "EKF-C": "#008300",
}
ORDER = ["KF", "MO", "EKF-A", "EKF-B", "EKF-C", "EKF-B+bias"]   # drawing order: proposed on top

plt.rcParams.update({
    "figure.facecolor": SURFACE, "axes.facecolor": SURFACE, "savefig.facecolor": SURFACE,
    "axes.edgecolor": INK2, "axes.labelcolor": INK, "xtick.color": INK2, "ytick.color": INK2,
    "axes.grid": True, "grid.color": GRID, "grid.linewidth": 0.6,
    "axes.spines.top": False, "axes.spines.right": False,
    "font.size": 9, "axes.titlesize": 10, "legend.fontsize": 8, "legend.frameon": False,
    "lines.linewidth": 1.4,
})


def color(name):
    return COLORS.get(name, INK2)


def save(fig, path):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path.with_suffix(".pdf"), bbox_inches="tight")
    fig.savefig(path.with_suffix(".png"), dpi=160, bbox_inches="tight")
    plt.close(fig)


def _sorted(results):
    return sorted(results, key=lambda r: ORDER.index(r.name) if r.name in ORDER else 99)


def _shade_phases(ax, run, t0):
    for name, (a, b) in (run.meta.get("phases") or {}).items():
        if name in ("z_step", "gust", "pickup", "figure8_gust"):
            ax.axvspan(a, b, color=GRID, alpha=0.5, lw=0)


def force_vs_truth(results, run, path, title=None):
    """Fig 1: F_hat vs truth, three axes, all filters."""
    t0 = scenario_t0(run)
    tt = run.truth["t"].to_numpy() - t0
    fig, axs = plt.subplots(3, 1, figsize=(7.5, 6.2), sharex=True)
    for j, (ax, lab) in enumerate(zip(axs, ["F_x", "F_y", "F_z"])):
        _shade_phases(ax, run, t0)
        for r in _sorted(results):
            ax.plot(r.t - t0, r.x[:, 6 + j], color=color(r.name), lw=1.0, label=r.name)
        ax.plot(tt, run.truth[["Fx", "Fy", "Fz"][j]], color=INK, lw=1.2, ls="--", label="truth")
        ax.set_ylabel(f"${lab}$ [N]")
    axs[0].legend(ncol=6, loc="lower left", bbox_to_anchor=(0, 1.02))
    axs[-1].set_xlabel("time [s]")
    if title:
        fig.suptitle(title, y=1.04, fontsize=10)
    save(fig, path)


def fz_bias_vs_battery(results_by_filter, path):
    """Fig 2 (headline): F_z error vs time as the battery drains. results_by_filter: {name: (res, run, use_v)}."""
    fig, (ax, axv) = plt.subplots(2, 1, figsize=(7.5, 4.8), sharex=True, gridspec_kw={"height_ratios": [3, 1]})
    run = None
    for name in [n for n in ORDER if n in results_by_filter]:
        res, run, use_v = results_by_filter[name]
        e, _ = errors(res, run, use_v)
        t0 = scenario_t0(run)
        k = max(1, int(round(0.5 / np.median(np.diff(res.t)))))           # 0.5 s moving average
        ez = np.convolve(e[:, M.IF][:, 2], np.ones(k) / k, mode="same")
        ax.plot(res.t - t0, ez, color=color(name), label=name)
    ax.axhline(0, color=INK2, lw=0.8)
    ax.set_ylabel("$F_z$ error [N] (0.5 s mean)")
    ax.legend(ncol=5, loc="upper left")
    t0 = scenario_t0(run)
    axv.plot(run.truth["t"] - t0, run.truth["V"], color=INK2, lw=1.2)
    axv.set_ylabel("battery [V]")
    axv.set_xlabel("time [s]")
    save(fig, path)


def params_convergence(res, run, path, use_voltage=True):
    """Fig 3: k (k_b s(V) shown as effective gain) and lambda vs truth with ±3σ, plus the accelerometer bias."""
    t0 = scenario_t0(run)
    t = res.t - t0
    e, Xt = errors(res, run, use_voltage)
    sd = res.sd
    c = color(res.name)
    fig, axs = plt.subplots(3, 1, figsize=(7.5, 6.0), sharex=True)
    for ax, i, lab in [(axs[0], M.IK, "$k_b$ [N]" if use_voltage else "$k$ [N]"), (axs[1], M.IL, r"$\lambda$ [1/s]")]:
        ax.fill_between(t, res.x[:, i] - 3 * sd[:, i], res.x[:, i] + 3 * sd[:, i], color=c, alpha=0.18, lw=0,
                        label=r"$\pm3\sigma$")
        ax.plot(t, res.x[:, i], color=c, label=res.name)
        ax.plot(t, Xt[:, i], color=INK, ls="--", label="truth")
        ax.set_ylabel(lab)
    axs[0].axhline(res.x[0, M.IK], color=INK2, lw=0.8, ls=":", label="bench prior")
    axs[0].legend(ncol=4, loc="upper right")
    for j, ls in zip(range(3), ["-", "--", ":"]):
        i = 12 + j
        axs[2].plot(t, res.x[:, i], color=c, ls=ls, label=f"$\\hat b_{'xyz'[j]}$")
        axs[2].plot(t, Xt[:, i], color=INK, ls=ls, lw=0.9, label=f"$b_{'xyz'[j]}$ truth")
    axs[2].set_ylabel("accel bias [m/s²]")
    axs[2].legend(ncol=6, loc="upper right")
    axs[2].set_xlabel("time [s]")
    save(fig, path)


def error_with_bounds(res, run, path, use_voltage=True):
    """Fig: error with ±3σ envelope for F_z, k, lambda, b_z."""
    t0 = scenario_t0(run)
    t = res.t - t0
    e, _ = errors(res, run, use_voltage)
    sd = res.sd
    c = color(res.name)
    items = [(8, "$F_z$ error [N]"), (M.IK, "$k$ error [N]"), (M.IL, r"$\lambda$ error [1/s]"), (14, "$b_z$ error [m/s²]")]
    items = [it for it in items if res.active[it[0]]]
    fig, axs = plt.subplots(len(items), 1, figsize=(7.5, 1.7 * len(items) + 0.6), sharex=True)
    axs = np.atleast_1d(axs)
    for ax, (i, lab) in zip(axs, items):
        ax.fill_between(t, -3 * sd[:, i], 3 * sd[:, i], color=c, alpha=0.18, lw=0, label=r"$\pm3\sigma$")
        ax.plot(t, e[:, i], color=c, lw=0.9, label="error")
        ax.axhline(0, color=INK2, lw=0.6)
        ax.set_ylabel(lab)
    axs[0].legend(ncol=2, loc="upper right")
    axs[-1].set_xlabel("time [s]")
    save(fig, path)


def nees_nis(anees_series, path, dof_nees, n_runs, nis_acc=None, t=None):
    """Fig 4: NEES averaged over runs with the χ² band; optional NIS (accelerometer) with its band.
    anees_series: {name: (t, anees)} averaged over n_runs runs."""
    ncol = 2 if nis_acc is not None else 1
    fig, axs = plt.subplots(1, ncol, figsize=(7.5 if ncol == 2 else 5.0, 3.0))
    axs = np.atleast_1d(axs)
    lo, hi = chi2_band(dof_nees, n_runs)
    ax = axs[0]
    for name in [n for n in ORDER if n in anees_series]:
        tt, v = anees_series[name]
        ax.plot(tt, v, color=color(name), lw=1.0, label=name)
    ax.axhspan(lo, hi, color=BAND, alpha=0.35, lw=0, label=f"95 % band (N={n_runs})")
    ax.set_yscale("log")
    ax.set_xlabel("time [s]")
    ax.set_ylabel(f"ANEES, force block (dof {dof_nees})")
    ax.legend(loc="upper right")
    if nis_acc is not None:
        lo1, hi1 = chi2_band(3, 1)
        ax = axs[1]
        for name in [n for n in ORDER if n in nis_acc]:
            ax.hist(np.clip(nis_acc[name], 0, 30), bins=60, density=True, histtype="step", color=color(name),
                    lw=1.2, label=name)
        x = np.linspace(0.01, 30, 300)
        from scipy.stats import chi2
        ax.plot(x, chi2.pdf(x, 3), color=INK, ls="--", lw=1.0, label=r"$\chi^2_3$")
        ax.axvspan(lo1, hi1, color=BAND, alpha=0.25, lw=0)
        ax.set_xlabel("NIS, accelerometer")
        ax.set_ylabel("density")
        ax.legend(loc="upper right")
    save(fig, path)


def observability(res_excited, res_hover, run_excited, run_hover, path):
    """Fig 5: σ_k and σ_λ over time, excited vs hover-only flight."""
    fig, axs = plt.subplots(1, 2, figsize=(7.5, 2.8))
    for res, run, lab, ls in [(res_excited, run_excited, "excited (S1)", "-"), (res_hover, run_hover, "hover only (S2)", "--")]:
        t = res.t - scenario_t0(run)
        axs[0].plot(t, res.sd[:, M.IK], color=color(res.name), ls=ls, label=lab)
        axs[1].plot(t, res.sd[:, M.IL], color=color(res.name), ls=ls, label=lab)
    for ax, lab in zip(axs, [r"$\sigma_{k_b}$ [N]", r"$\sigma_\lambda$ [1/s]"]):
        ax.axvspan(10, 35, color=GRID, alpha=0.6, lw=0)
        ax.set_xlabel("time [s]")
        ax.set_ylabel(lab)
    axs[0].legend(loc="upper right")
    save(fig, path)


def tradeoff(df, path, x="noise_F_hover", y="rise_time", label="q_F", ylabel="step rise time 10→90 % [s]"):
    """Fig 6: bandwidth vs noise: step rise time vs hover noise, one point per q_F."""
    fig, ax = plt.subplots(figsize=(5.0, 3.4))
    ax.plot(df[x], df[y], color=color("EKF-B+bias"), marker="o", ms=5)
    for _, r in df.iterrows():
        ax.annotate(f"{r[label]:g}", (r[x], r[y]), textcoords="offset points", xytext=(5, 4), fontsize=7, color=INK2)
    ax.set_xlabel("force noise in hover [N]")
    ax.set_ylabel(ylabel)
    ax.set_xscale("log")
    save(fig, path)


def bars(summary, metric, path, ylabel, scenarios=None):
    """Grouped bars: metric mean ± std per scenario, one bar per filter."""
    s = summary[summary["metric"] == metric]
    scen = scenarios or list(dict.fromkeys(s["scenario"]))
    names = [n for n in ORDER if n in set(s["filter"])]
    w = 0.8 / len(names)
    fig, ax = plt.subplots(figsize=(7.5, 3.0))
    for k, name in enumerate(names):
        rows = [s[(s["scenario"] == sc) & (s["filter"] == name)] for sc in scen]
        mean = [float(r["mean"].iloc[0]) if len(r) else np.nan for r in rows]
        std = [float(r["std"].iloc[0]) if len(r) else 0 for r in rows]
        xs = np.arange(len(scen)) + (k - (len(names) - 1) / 2) * w
        ax.bar(xs, mean, width=w * 0.92, yerr=std, color=color(name), label=name, error_kw={"lw": 0.8, "ecolor": INK2})
    ax.set_xticks(np.arange(len(scen)))
    ax.set_xticklabels(scen)
    ax.set_ylabel(ylabel)
    ax.set_yscale("log")
    ax.legend(ncol=5, loc="lower left", bbox_to_anchor=(0, 1.0))
    save(fig, path)
