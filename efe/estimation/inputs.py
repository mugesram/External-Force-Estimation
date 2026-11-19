"""The filter's inputs at the IMU timestamps: attitude R, motor commands u, filtered battery voltage V.

Everything here is causal (uses only samples at or before the IMU time), so the offline run behaves as an online
filter would.
"""

from __future__ import annotations

import math

import numpy as np

from .frames import quat_exp, quat_mul, quat_to_rot, slerp


def zoh_index(t_src, t_query):
    """Index of the latest source sample at or before each query time (-1 if none yet)."""
    return np.searchsorted(t_src, t_query, side="right") - 1


def attitude_at(t_imu, gyro, t_att, q_att, mode="gyro"):
    """Rotation matrices (N, 3, 3) at the IMU times.

    gyro   from the latest attitude sample, integrate the body rates to the IMU time (causal; what an onboard filter
           would do with 50 Hz vehicle_attitude and a 250 Hz IMU)
    slerp  interpolate between the attitude samples either side (non-causal; the 'ideal attitude' reference)
    """
    n = len(t_imu)
    out = np.empty((n, 3, 3))
    idx = zoh_index(t_att, t_imu)
    if mode == "slerp":
        for i in range(n):
            j = max(idx[i], 0)
            if j + 1 < len(t_att) and t_att[j + 1] > t_att[j] and idx[i] >= 0:
                fr = (t_imu[i] - t_att[j]) / (t_att[j + 1] - t_att[j])
                q = slerp(q_att[j], q_att[j + 1], min(max(fr, 0.0), 1.0))
            else:
                q = q_att[j]
            out[i] = quat_to_rot(q)
        return out
    if mode != "gyro":
        raise ValueError(f"unknown attitude mode {mode!r}")
    last_j, q, t_prev = -2, None, None
    for i in range(n):
        j = max(idx[i], 0)
        if j != last_j:
            q, t_prev, last_j = np.array(q_att[j], dtype=float), t_att[j], j
        dt = t_imu[i] - t_prev
        if dt > 0:
            q = quat_mul(q, quat_exp(gyro[i] * dt))
            q /= math.sqrt(q @ q)
            t_prev = t_imu[i]
        out[i] = quat_to_rot(q)
    return out


def lowpass(t, x, fc_hz):
    """Causal first-order low-pass, exact discretisation for irregular sampling."""
    x = np.asarray(x, dtype=float)
    if fc_hz is None or fc_hz <= 0:
        return x.copy()
    y = np.empty_like(x)
    y[0] = x[0]
    tau = 1.0 / (2 * math.pi * fc_hz)
    for i in range(1, len(x)):
        a = math.exp(-(t[i] - t[i - 1]) / tau)
        y[i] = a * y[i - 1] + (1 - a) * x[i]
    return y


def inputs_at_imu(run, voltage_lp_hz=2.0, attitude_mode="gyro", motor_delay=0.0):
    """u (N, n_motors), V (N,), R (N, 3, 3) at the IMU times of `run`. The motor commands are delayed by
    motor_delay [s]: the command logged at t acts on the rotors from t + motor_delay."""
    t = run.t_imu
    tm = run.motors["t"].to_numpy() + float(motor_delay or 0.0)
    u_all = run.motor_cmds()
    iu = np.clip(zoh_index(tm, t), 0, len(tm) - 1)
    u = u_all[iu]

    tb = run.battery["t"].to_numpy()
    Vf = lowpass(tb, run.battery["V"].to_numpy(), voltage_lp_hz)
    V = Vf[np.clip(zoh_index(tb, t), 0, len(tb) - 1)]

    if attitude_mode == "truth":
        q = run.truth[["qw", "qx", "qy", "qz"]].to_numpy()
        tt = run.truth["t"].to_numpy()
        R = quat_to_rot(q[np.clip(zoh_index(tt, t), 0, len(tt) - 1)])
    else:
        R = attitude_at(t, run.gyro(), run.att["t"].to_numpy(), run.quats(), attitude_mode)
    return u, V, R
