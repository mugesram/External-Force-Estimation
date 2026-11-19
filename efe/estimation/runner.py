"""Run a filter over a recorded flight (RunData) and collect its estimates, covariances and innovations."""

from __future__ import annotations

import time
from dataclasses import dataclass

import numpy as np

from . import models as M
from .ekf import FilterConfig, ForceEKF
from .inputs import inputs_at_imu
from .momentum_observer import MOConfig, MomentumObserver


@dataclass
class FilterResult:
    name: str
    t: np.ndarray            # (N,) output times (IMU)
    x: np.ndarray            # (N, 15)
    P: np.ndarray | None     # (N, 15, 15) or None (momentum observer)
    active: np.ndarray       # (15,) estimated states
    nis_acc: np.ndarray      # (N,)
    nis_gnss: np.ndarray     # (n_gnss,)
    t_gnss: np.ndarray
    step_us: float           # mean compute time per IMU step

    @property
    def sd(self):
        return np.sqrt(np.maximum(np.einsum("nii->ni", self.P), 0.0)) if self.P is not None else None


def vehicle_params(run, overrides=None):
    meta = run.meta
    prm = M.VehicleParams(
        mass=float(meta.get("mass", 1.5)),
        drag=np.array(meta.get("drag", [0.5, 0.3, 0.0]), dtype=float),
        V0=float(meta.get("V0", 16.8)),
        u0=float(meta.get("u0", 0.1)),
    )
    for k, v in (overrides or {}).items():
        setattr(prm, k, np.array(v, dtype=float) if k == "drag" else float(v))
    return prm


def motor_delay(run, cfg):
    d = getattr(cfg, "motor_delay", None)
    return float(d) if d is not None else float(run.meta.get("motor_delay_s", 0.0))


def run_filter(run, cfg, prm: M.VehicleParams, t_start=None):
    if isinstance(cfg, MOConfig):
        return _run_mo(run, cfg, prm, t_start)
    return _run_ekf(run, cfg, prm, t_start)


def _first_gnss(run, t_start):
    g = run.gnss
    tg = g["t"].to_numpy()
    j0 = int(np.searchsorted(tg, t_start if t_start is not None else tg[0]))
    return j0


def _run_ekf(run, cfg: FilterConfig, prm, t_start):
    ekf = ForceEKF(cfg, prm)
    t = run.t_imu
    acc = run.acc()
    u, V, R = inputs_at_imu(run, cfg.voltage_lp_hz, cfg.attitude_mode, motor_delay(run, cfg))
    g = run.gnss
    tg = g["t"].to_numpy()
    gp = g[["px", "py", "pz"]].to_numpy()
    gv = g[["vx", "vy", "vz"]].to_numpy()

    j = _first_gnss(run, t_start)
    ekf.init(gp[j], gv[j], T0=prm.mass * prm.g)
    t_now = tg[j]
    j += 1
    i0 = int(np.searchsorted(t, t_now, side="right"))

    n = len(t) - i0
    xs = np.empty((n, M.NX))
    Ps = np.empty((n, M.NX, M.NX))
    nis_acc = np.full(n, np.nan)
    nis_g, t_g = [], []
    k_idx = i0 - 1
    wall = time.perf_counter()
    for o, i in enumerate(range(i0, len(t))):
        # GNSS samples up to this IMU time: predict to each, then update
        while j < len(tg) and tg[j] <= t[i]:
            kk = max(i - 1, k_idx)
            ekf.predict(tg[j] - t_now, u[kk], V[kk], R[kk])
            t_now = tg[j]
            nis, _ = ekf.update_gnss(gp[j], gv[j])
            ekf.update_k_pseudo()
            nis_g.append(nis)
            t_g.append(tg[j])
            j += 1
        # inputs held since the previous IMU sample drive the step to this one
        kk = max(i - 1, 0)
        ekf.predict(t[i] - t_now, u[kk], V[kk], R[kk])
        t_now = t[i]
        nis_acc[o], _ = ekf.update_acc(acc[i], R[i])
        xs[o], Ps[o] = ekf.x, ekf.P
    step_us = (time.perf_counter() - wall) / max(n, 1) * 1e6
    return FilterResult(cfg.name, t[i0:], xs, Ps, ekf.active, nis_acc, np.array(nis_g), np.array(t_g), step_us)


def _run_mo(run, cfg: MOConfig, prm, t_start):
    t = run.t_imu
    u, V, R = inputs_at_imu(run, None, cfg.attitude_mode, motor_delay(run, cfg))
    g = run.gnss
    tg = g["t"].to_numpy()
    gv = g[["vx", "vy", "vz"]].to_numpy()
    j = _first_gnss(run, t_start)
    i0 = int(np.searchsorted(t, tg[j], side="right"))
    mo = MomentumObserver(cfg, prm)
    mo.init(gv[j])
    n = len(t) - i0
    xs = np.full((n, M.NX), np.nan)
    wall = time.perf_counter()
    t_now = tg[j]
    for o, i in enumerate(range(i0, len(t))):
        while j + 1 < len(tg) and tg[j + 1] <= t[i]:
            j += 1
        kk = max(i - 1, 0)
        mo.step(t[i] - t_now, u[kk], R[kk], gv[j])
        t_now = t[i]
        xs[o, M.IF] = mo.F
        xs[o, M.IT] = mo.T
        xs[o, M.IV] = mo.v_hat
    step_us = (time.perf_counter() - wall) / max(n, 1) * 1e6
    active = np.zeros(M.NX, dtype=bool)
    active[M.IF] = True
    return FilterResult(cfg.name, t[i0:], xs, None, active, np.full(n, np.nan), np.array([]), np.array([]), step_us)
