"""Replay Monte Carlo and data ablations.

resample_sensors: keep a run's true trajectory and its motor commands, redraw the IMU, attitude, GNSS and voltage
noise. This gives many statistically independent runs from one (slow) Isaac flight, enough for proper NEES
statistics. Assumption, stated in the report: the closed loop is not re-simulated, so the motor commands keep the
noise realisation of the original flight (a small effect at these noise levels).

The ablation helpers transform a run's streams: GNSS rate and dropout, voltage rate.
"""

from __future__ import annotations

import math

import numpy as np
import pandas as pd

from ..estimation.frames import quat_exp, quat_mul


def _interp(tt, df, cols, t):
    return np.column_stack([np.interp(t, tt, df[c].to_numpy()) for c in cols])


def resample_sensors(run, seed, sensors=None):
    """A new RunData with fresh sensor noise on the same truth. `sensors`: overrides of run.meta['sensors']."""
    sen = dict(run.meta.get("sensors") or {})
    sen.update(sensors or {})
    rng = np.random.default_rng(seed)
    tr = run.truth
    tt = tr["t"].to_numpy()

    # IMU at the run's IMU times
    t = run.t_imu
    dt = float(np.median(np.diff(t)))
    f = _interp(tt, tr, ["fx", "fy", "fz"], t)
    w = _interp(tt, tr, ["wx", "wy", "wz"], t)
    a_gm = math.exp(-dt / sen["acc_bias_tau"])
    b = np.empty((len(t), 3))
    b[0] = rng.normal(0, sen["acc_bias_sigma"], 3)
    drive = math.sqrt(1 - a_gm * a_gm) * sen["acc_bias_sigma"]
    noise = rng.normal(0, 1, (len(t), 3)) * drive
    for i in range(1, len(t)):
        b[i] = a_gm * b[i - 1] + noise[i]
    acc_sd = math.hypot(sen["acc_noise_density"] / math.sqrt(dt), sen.get("acc_vibration", 0.0))
    gyro_sd = sen["gyro_noise_density"] / math.sqrt(dt)
    acc = f + b + rng.normal(0, acc_sd, f.shape)
    gyro = w + rng.normal(0, sen["gyro_bias_sigma"], 3) + rng.normal(0, gyro_sd, w.shape)
    imu = pd.DataFrame(np.column_stack([t, acc, gyro]), columns=["t", "ax", "ay", "az", "gx", "gy", "gz"])

    # truth with the new bias, for NEES
    truth = tr.copy()
    for j, c in enumerate(["bax", "bay", "baz"]):
        truth[c] = np.interp(tt, t, b[:, j])

    # attitude estimate: truth with a Gauss-Markov error
    ta = run.att["t"].to_numpy()
    qa = _interp(tt, tr, ["qw", "qx", "qy", "qz"], ta)
    qa /= np.linalg.norm(qa, axis=1, keepdims=True)
    dta = float(np.median(np.diff(ta)))
    sig = math.radians(sen["att_error_deg"])
    a_att = math.exp(-dta / sen["att_error_tau"])
    err = rng.normal(0, sig, 3)
    rows = []
    for i in range(len(ta)):
        if i:
            err = a_att * err + math.sqrt(1 - a_att ** 2) * sig * rng.normal(0, 1, 3)
        q = quat_mul(quat_exp(err), qa[i])
        rows.append(q / np.linalg.norm(q))
    att = pd.DataFrame(np.column_stack([ta, np.array(rows)]), columns=["t", "qw", "qx", "qy", "qz"])

    # GNSS
    tg = run.gnss["t"].to_numpy()
    pg = _interp(tt, tr, ["px", "py", "pz"], tg) + rng.normal(0, 1, (len(tg), 3)) * np.asarray(sen["gnss_pos_sigma"])
    vg = _interp(tt, tr, ["vx", "vy", "vz"], tg) + rng.normal(0, sen["gnss_vel_sigma"], (len(tg), 3))
    gnss = pd.DataFrame(np.column_stack([tg, pg, vg]), columns=["t", "px", "py", "pz", "vx", "vy", "vz"])

    # battery voltage
    tb = run.battery["t"].to_numpy()
    Vb = np.interp(tb, tt, tr["V"].to_numpy()) + rng.normal(0, sen["bat_noise"], len(tb))
    battery = pd.DataFrame({"t": tb, "V": Vb})

    meta = dict(run.meta)
    meta["seed"] = int(seed)
    meta["replay_of"] = run.meta.get("seed")
    meta["sensors"] = sen
    return run.with_streams(imu=imu, att=att, gnss=gnss, battery=battery, truth=truth, meta=meta)


def gnss_rate(run, hz):
    g = run.gnss
    t = g["t"].to_numpy()
    keep = [0]
    for i in range(1, len(t)):
        if t[i] - t[keep[-1]] >= 1.0 / hz - 1e-6:
            keep.append(i)
    return run.with_streams(gnss=g.iloc[keep].reset_index(drop=True))


def gnss_dropout(run, t0, duration):
    """Remove GNSS for [t0, t0 + duration) of scenario time."""
    g = run.gnss
    ts = float(run.truth["t"].iloc[0])
    rel = g["t"].to_numpy() - ts
    return run.with_streams(gnss=g[(rel < t0) | (rel >= t0 + duration)].reset_index(drop=True))


def battery_rate(run, hz):
    b = run.battery
    t = b["t"].to_numpy()
    keep = [0]
    for i in range(1, len(t)):
        if t[i] - t[keep[-1]] >= 1.0 / hz - 1e-6:
            keep.append(i)
    return run.with_streams(battery=b.iloc[keep].reset_index(drop=True))
