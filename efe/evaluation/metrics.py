"""Evaluation metrics: accuracy against ground truth, and filter consistency (NEES / NIS).

Truth is interpolated onto the filter's output times. What 'true k' means depends on the filter: with the voltage
input the filter's k is k_b (the gain at V0); without it, the filter's k is the effective gain k_b * s(V) at the
current voltage.
"""

from __future__ import annotations

import numpy as np
from scipy.stats import chi2

from ..estimation import models as M


def scenario_t0(run):
    """Time at which the scenario starts: 0 in Tier 0; after boot, take-off and a warm-up in Tier 1."""
    t0 = run.meta.get("t_scenario_start")
    return float(t0) if t0 is not None else float(run.truth["t"].iloc[0])


def truth_states(run, t, use_voltage=True):
    """(N, NX) true state at times t, in the filter's state definition."""
    tr = run.truth
    tt = tr["t"].to_numpy()

    def col(c):
        return np.interp(t, tt, tr[c].to_numpy())

    X = np.zeros((len(t), M.NX))
    for j, c in enumerate(["px", "py", "pz", "vx", "vy", "vz", "Fx", "Fy", "Fz", "T"]):
        X[:, j] = col(c)
    X[:, M.IK] = col("k") if use_voltage else col("k_eff")
    X[:, M.IL] = col("lam")
    X[:, 12], X[:, 13], X[:, 14] = col("bax"), col("bay"), col("baz")
    X[:, M.IE] = tilt_error(run, t)
    return X


def tilt_error(run, t):
    """True tilt error of the attitude input: eps with R_true = (I + [eps]x) R_input, world x/y [rad]."""
    from ..estimation.frames import quat_to_rot

    ta = run.att["t"].to_numpy()
    tt = run.truth["t"].to_numpy()
    j = np.clip(np.searchsorted(ta, t, side="right") - 1, 0, len(ta) - 1)
    Ra = quat_to_rot(run.att[["qw", "qx", "qy", "qz"]].to_numpy()[j])
    k = np.clip(np.searchsorted(tt, ta[j]), 0, len(tt) - 1)
    Rt = quat_to_rot(run.truth[["qw", "qx", "qy", "qz"]].to_numpy()[k])
    E = np.einsum("nij,nkj->nik", Rt, Ra)            # R_true R_input^T = I + [eps]x
    return 0.5 * np.stack([E[:, 2, 1] - E[:, 1, 2], E[:, 0, 2] - E[:, 2, 0]], axis=1)


def _uses_voltage(res, cfgs):
    cfg = cfgs.get(res.name) if cfgs else None
    return bool(getattr(cfg, "use_voltage", True))


def errors(res, run, use_voltage=True):
    Xt = truth_states(run, res.t, use_voltage)
    return res.x - Xt, Xt


def _window(t, a, b):
    return (t >= a) & (t < b)


def rmse(e):
    return np.sqrt(np.nanmean(e ** 2, axis=0))


# --- event metrics ---------------------------------------------------------------------------------------------
def step_response(t, F_hat, t_step, axis, size, pre=3.0, post=8.0):
    """Rise time 10->90 % and time to 90 % of a force step (from the estimate)."""
    base = np.nanmean(F_hat[_window(t, t_step - pre, t_step - 0.2), axis])
    w = _window(t, t_step, t_step + post)
    y = (F_hat[w, axis] - base) / size
    tw = t[w]
    i10 = np.argmax(y >= 0.1) if np.any(y >= 0.1) else None
    i90 = np.argmax(y >= 0.9) if np.any(y >= 0.9) else None
    if i10 is None or i90 is None:
        return np.nan, np.nan
    return float(tw[i90] - tw[i10]), float(tw[i90] - t_step)


def detection_delay(t, F_hat, t_event, pre=3.0, window=1.0, min_threshold=0.5):
    """Time from the event until ||F_hat - baseline|| exceeds max(3 sigma of its baseline, min_threshold)."""
    wb = _window(t, t_event - pre, t_event - 0.2)
    mu = np.nanmean(F_hat[wb], axis=0)
    d_base = np.linalg.norm(F_hat[wb] - mu, axis=1)
    thr = max(float(np.mean(d_base) + 3 * np.std(d_base)), min_threshold)
    w = _window(t, t_event, t_event + window)
    d = np.linalg.norm(F_hat[w] - mu, axis=1)
    if not np.any(d > thr):
        return np.nan
    return float(t[w][np.argmax(d > thr)] - t_event)


def convergence_time(t, err_rel, tol=0.05, t0=0.0):
    """First time after which |err_rel| stays below tol until the end."""
    bad = np.abs(err_rel) >= tol
    bad &= t >= t0
    if not np.any(bad):
        return float(t[0] - t0) if len(t) else np.nan
    last_bad = np.where(bad)[0][-1]
    if last_bad == len(t) - 1:
        return np.nan
    return float(t[last_bad + 1] - t0)


# --- consistency -----------------------------------------------------------------------------------------------
BLOCKS = {"F": [6, 7, 8], "params": [9, 10, 11], "bias": [12, 13, 14], "tilt": [15, 16], "full": list(range(M.NX))}


def nees(res, e, block="full"):
    """Per-sample NEES over the estimated states of `block`; returns (values, dof)."""
    idx = [i for i in BLOCKS[block] if res.active[i]]
    if not idx or res.P is None:
        return None, 0
    Pb = res.P[:, idx][:, :, idx]
    eb = e[:, idx]
    sol = np.linalg.solve(Pb, eb[..., None])[..., 0]
    return np.einsum("ni,ni->n", eb, sol), len(idx)


def chi2_band(dof, n_runs=1, conf=0.95):
    """Two-sided band for the mean of n_runs chi^2(dof) samples."""
    a = (1 - conf) / 2
    return chi2.ppf(a, dof * n_runs) / n_runs, chi2.ppf(1 - a, dof * n_runs) / n_runs


def fraction_in_band(values, dof, conf=0.95):
    lo, hi = chi2_band(dof, 1, conf)
    v = values[np.isfinite(values)]
    return float(np.mean((v >= lo) & (v <= hi))) if len(v) else np.nan


# --- one run, one filter ---------------------------------------------------------------------------------------
def evaluate(res, run, cfg=None, settle=5.0, hover_phases=("hover", "hover2", "hover_sag", "hover_end")):
    use_v = bool(getattr(cfg, "use_voltage", True))
    e, Xt = errors(res, run, use_v)
    t = res.t
    rel = t - scenario_t0(run)
    out = {"filter": res.name, "scenario": run.meta.get("scenario"), "seed": run.meta.get("seed")}
    phases = run.meta.get("phases") or {}
    events = run.meta.get("events") or {}

    # exclude the settle time and 1 s after an impulse from the overall RMSE
    m = rel >= settle
    if "impulse" in events:
        ti = float(events["impulse"]["t"])
        m &= ~_window(rel, ti, ti + 1.0)
    eF = e[:, M.IF]
    r = rmse(eF[m])
    out.update({"rmse_Fx": r[0], "rmse_Fy": r[1], "rmse_Fz": r[2], "rmse_Fxy": float(np.sqrt(r[0] ** 2 + r[1] ** 2) / np.sqrt(2))})
    for name, (a, b) in phases.items():
        w = _window(rel, a, b)
        if np.any(w):
            out[f"rmse_Fz@{name}"] = float(rmse(eF[w, 2:3])[0])
            out[f"rmse_Fxy@{name}"] = float(np.sqrt(np.mean(eF[w, :2] ** 2)))
    hov = np.zeros_like(m)
    for name in hover_phases:
        if name in phases:
            hov |= _window(rel, *phases[name])
    if np.any(hov):
        out["bias_Fz_hover"] = float(np.mean(eF[hov, 2]))
        out["noise_F_hover"] = float(np.mean(np.std(eF[hov], axis=0)))
    # bias at the end of the flight, where the battery is lowest
    w_end = rel >= rel[-1] - 15.0
    out["bias_Fz_end"] = float(np.mean(eF[w_end, 2]))
    out["bias_Fxy_end"] = float(np.linalg.norm(np.mean(eF[w_end, :2], axis=0)))
    out["V_end"] = float(np.interp(t[-1], run.truth["t"], run.truth["V"]))

    F_hat = res.x[:, M.IF]
    if "step" in events:
        ev = events["step"]
        out["rise_time"], out["time_to_90"] = step_response(rel, F_hat, float(ev["t"]), int(ev["axis"]), float(ev["size"]))
    if "impulse" in events:
        out["detect_delay"] = detection_delay(rel, F_hat, float(events["impulse"]["t"]))

    if res.active[M.IK]:
        rel_k = e[:, M.IK] / Xt[:, M.IK]
        out["k_err_end_pct"] = float(100 * np.mean(rel_k[w_end]))
        out["k_conv_time"] = convergence_time(rel, rel_k, 0.05)
    if res.active[M.IL]:
        rel_l = e[:, M.IL] / Xt[:, M.IL]
        out["lam_err_end_pct"] = float(100 * np.mean(rel_l[w_end]))
    if res.active[12]:
        out["ba_err_end"] = float(np.linalg.norm(np.mean(e[w_end, 12:15], axis=0)))
    if res.active[15]:
        out["tilt_err_end_deg"] = float(np.degrees(np.linalg.norm(np.mean(e[w_end, 15:17], axis=0))))

    if res.P is not None:
        for block in ("F", "full"):
            v, dof = nees(res, e, block)
            if v is not None:
                mm = rel >= settle
                out[f"anees_{block}"] = float(np.mean(v[mm]))
                out[f"nees_{block}_dof"] = dof
                out[f"nees_{block}_in95"] = fraction_in_band(v[mm], dof)
        out["anis_acc"] = float(np.nanmean(res.nis_acc[rel >= settle]))
        out["nis_acc_in95"] = fraction_in_band(res.nis_acc[rel >= settle], 3)
        if len(res.nis_gnss):
            out["anis_gnss"] = float(np.mean(res.nis_gnss))
            out["nis_gnss_in95"] = fraction_in_band(res.nis_gnss, 6)
    out["step_us"] = res.step_us
    return out
