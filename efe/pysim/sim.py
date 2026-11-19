"""Tier 0: a pure-Python quadrotor flight with the same truth model as the Isaac/Pegasus extension.

Truth:   translational 6-DoF in NED, attitude following the controller's thrust direction with a first-order lag,
         per-motor lag + voltage-scaled open-loop ESC (efe.truth.motors), battery (efe.truth.battery), body-frame
         linear drag, external force schedule (efe.truth.forces).
Control: a PX4-like cascaded position PID that knows nothing about the battery: its integrator absorbs the sag, as
         PX4's does. Motor command from a fixed nominal thrust curve, plus small differential activity per motor.
Sensors: IMU (specific force + gyro, Pegasus' noise and bias model), attitude estimate at 50 Hz, RTK GNSS at 10 Hz,
         battery voltage, motor commands; all in PX4 conventions (NED / FRD / [w, x, y, z]).
Output:  RunData in the same format Tier 1 ingests.
"""

from __future__ import annotations

import math

import numpy as np
import pandas as pd

from ..estimation.frames import E3, quat_exp, quat_mul, quat_to_rot, rot_to_quat
from ..io.rundata import RunData
from ..scenario import Trajectory
from ..truth.battery import BatteryModel
from ..truth.forces import ForceSchedule

G = 9.80665


def _attitude_from_b3(b3, yaw):
    """R (body FRD -> NED) whose body z axis is b3 and whose heading is yaw."""
    b1c = np.array([math.cos(yaw), math.sin(yaw), 0.0])
    b2 = np.cross(b3, b1c)
    b2 /= np.linalg.norm(b2)
    b1 = np.cross(b2, b3)
    return np.column_stack([b1, b2, b3])


class GaussMarkov:
    """First-order Gauss-Markov process with stationary std `sigma` and time constant `tau` (vector)."""

    def __init__(self, sigma, tau, rng, n=3, start=None):
        self.sigma, self.tau, self.rng = np.asarray(sigma, dtype=float), float(tau), rng
        self.x = rng.normal(0.0, 1.0, n) * self.sigma if start is None else np.asarray(start, dtype=float)

    def step(self, dt):
        a = math.exp(-dt / self.tau)
        self.x = a * self.x + math.sqrt(1 - a * a) * self.sigma * self.rng.normal(0.0, 1.0, self.x.shape)
        return self.x


def simulate(sc, seed=0, sim_rate=500.0):
    """Fly scenario `sc` (from efe.scenario.load_scenario); returns RunData."""
    from ..truth.motors import MotorModel  # local: keeps module import light

    rng = np.random.default_rng(seed)
    veh, sen = sc["vehicle"], sc["sensors"]
    m = float(veh["mass"])
    D = np.diag(veh["drag"])
    motors = MotorModel.from_config(veh, rng=np.random.default_rng(seed + 1000))
    battery = BatteryModel.from_config(sc["battery"])
    forces = ForceSchedule(sc.get("forces"))
    traj = Trajectory(sc["trajectory"])
    duration = float(sc["duration"])

    dt = 1.0 / sim_rate
    imu_every = int(round(sim_rate / sen["imu_rate"]))
    gnss_every = int(round(sim_rate / sen["gnss_rate"]))
    att_every = int(round(sim_rate / sen["att_rate"]))
    bat_every = int(round(sim_rate / sen["bat_rate"]))
    ctrl_every = int(round(sim_rate / sen["motor_rate"]))

    # controller (PX4-like gains), with a nominal thrust curve fixed at the bench value
    kp, kd, ki = np.array([1.5, 1.5, 2.0]), np.array([2.5, 2.5, 3.0]), np.array([0.3, 0.3, 1.0])
    k_ctrl = motors.k_b()
    u0 = motors.u0
    tau_att = 0.05
    integ = np.zeros(3)

    # initial state: hovering at the first reference point
    p, _, _ = traj(0.0)
    p = p.copy()
    v = np.zeros(3)
    yaw = 0.0
    b3 = np.array([0.0, 0.0, 1.0])
    R = _attitude_from_b3(b3, yaw)
    V_now = battery.voltage
    hover_u = (1 + u0) * math.sqrt(m * G / 4 / motors.k_eff(V_now)) - u0
    u = np.full(4, hover_u)
    motors.reset(u, V_now)
    jitter = GaussMarkov(sen["motor_jitter"], 0.05, rng, n=4, start=np.zeros(4))

    # sensor error processes
    acc_bias = GaussMarkov(sen["acc_bias_sigma"], sen["acc_bias_tau"], rng)
    gyro_bias = rng.normal(0.0, sen["gyro_bias_sigma"], 3)
    att_err = GaussMarkov(math.radians(sen["att_error_deg"]), sen["att_error_tau"], rng)
    imu_dt = imu_every * dt
    acc_sd = math.hypot(sen["acc_noise_density"] / math.sqrt(imu_dt), sen.get("acc_vibration", 0.0))
    gyro_sd = sen["gyro_noise_density"] / math.sqrt(imu_dt)
    acc_rw_sd = sen["acc_bias_rw"] * math.sqrt(imu_dt)

    rows_imu, rows_att, rows_gnss, rows_mot, rows_bat, rows_truth = [], [], [], [], [], []
    w_body = np.zeros(3)
    b3_des = b3.copy()
    n_steps = int(round(duration * sim_rate))
    # Lockstep order, as PX4 SITL runs: the state at t is measured, the controller answers with u(t), and the physics
    # then integrates to t + dt with that u. So the IMU at t sees the thrust of the rotor speeds at t, which u(t) has
    # not yet changed: the same causality the filter's model assumes.
    for k in range(n_steps + 1):
        t = k * dt
        # ---- forces and true acceleration at t
        thrusts = motors.thrusts()
        T = float(np.sum(thrusts))
        F_ext = forces(t)
        drag_w = -R @ D @ R.T @ v
        acc = (-T * R @ E3 + F_ext + drag_w) / m + G * E3
        f_spec = R.T @ (acc - G * E3)                  # specific force, body FRD (true)

        # ---- sensors at t
        if k % imu_every == 0:
            ba = acc_bias.step(imu_dt) if acc_rw_sd > 0 else acc_bias.x
            z_acc = f_spec + ba + rng.normal(0.0, acc_sd, 3)
            z_gyro = w_body + gyro_bias + rng.normal(0.0, gyro_sd, 3)
            rows_imu.append((t, *z_acc, *z_gyro))
            rows_truth.append((t, *p, *v, *F_ext, T, motors.k_b(), 1.0 / motors.tau, *ba, V_now, battery.soc,
                               motors.k_eff(V_now), *rot_to_quat(R), *f_spec, *w_body))
        if k % att_every == 0:
            q_meas = quat_mul(quat_exp(att_err.step(att_every * dt)), rot_to_quat(R))
            rows_att.append((t, *(q_meas / np.linalg.norm(q_meas))))
        if k % gnss_every == 0:
            ps = np.asarray(sen["gnss_pos_sigma"], dtype=float)
            rows_gnss.append((t, *(p + rng.normal(0.0, 1.0, 3) * ps), *(v + rng.normal(0.0, sen["gnss_vel_sigma"], 3))))
        if k % bat_every == 0:
            rows_bat.append((t, V_now + rng.normal(0.0, sen["bat_noise"])))

        # ---- controller at the motor rate
        if k % ctrl_every == 0:
            p_ref, v_ref, a_ref = traj(t)
            e, ev = p_ref - p, v_ref - v
            integ = np.clip(integ + e * ctrl_every * dt, -10.0, 10.0)
            a_cmd = kp * e + kd * ev + ki * integ + a_ref
            a_cmd[:2] = np.clip(a_cmd[:2], -5.0, 5.0)
            a_cmd[2] = np.clip(a_cmd[2], -8.0, 6.0)
            f_des = m * (a_cmd - G * E3)              # desired -T R e3
            T_des = float(np.linalg.norm(f_des))
            b3_des = -f_des / T_des
            # tilt limit 35 deg
            if b3_des[2] < math.cos(math.radians(35)):
                h = b3_des[:2] / max(np.linalg.norm(b3_des[:2]), 1e-9) * math.sin(math.radians(35))
                b3_des = np.array([h[0], h[1], math.cos(math.radians(35))])
            # thrust along the actual body axis
            T_cmd = T_des / max(float(b3 @ b3_des), 0.5)
            u_coll = (1 + u0) * math.sqrt(max(T_cmd / 4.0 / k_ctrl, 0.0)) - u0
            u = np.clip(u_coll + jitter.step(ctrl_every * dt), 0.0, 1.0)
            rows_mot.append((t, *u))

        # ---- physics from t to t + dt
        v = v + acc * dt
        p = p + v * dt
        b3 = b3 + (b3_des - b3) * (dt / tau_att)
        b3 /= np.linalg.norm(b3)
        R_new = _attitude_from_b3(b3, yaw)
        w_body = _log_so3(R.T @ R_new) / dt
        R = R_new
        V_now = battery.step(motors.step(u, V_now, dt), dt)

    truth_cols = ["t", "px", "py", "pz", "vx", "vy", "vz", "Fx", "Fy", "Fz", "T", "k", "lam", "bax", "bay", "baz",
                  "V", "SoC", "k_eff", "qw", "qx", "qy", "qz", "fx", "fy", "fz", "wx", "wy", "wz"]
    meta = {
        "tier": 0, "scenario": sc.get("name"), "seed": int(seed), "mass": m, "drag": list(veh["drag"]),
        "V0": float(veh["V0"]), "u0": float(motors.u0), "duration": duration,
        "phases": sc.get("phases", {}), "events": sc.get("events", {}), "sensors": sen,
        "truth_params": {"k_b": motors.k_b(), "tau": motors.tau, "voltage_exponent": motors.voltage_exponent},
    }
    return RunData(
        imu=pd.DataFrame(rows_imu, columns=["t", "ax", "ay", "az", "gx", "gy", "gz"]),
        att=pd.DataFrame(rows_att, columns=["t", "qw", "qx", "qy", "qz"]),
        gnss=pd.DataFrame(rows_gnss, columns=["t", "px", "py", "pz", "vx", "vy", "vz"]),
        motors=pd.DataFrame(rows_mot, columns=["t", "u0", "u1", "u2", "u3"]),
        battery=pd.DataFrame(rows_bat, columns=["t", "V"]),
        truth=pd.DataFrame(rows_truth, columns=truth_cols),
        meta=meta,
    )


def _log_so3(Rm):
    """Rotation matrix -> rotation vector."""
    c = (np.trace(Rm) - 1.0) / 2.0
    c = min(max(c, -1.0), 1.0)
    th = math.acos(c)
    w = np.array([Rm[2, 1] - Rm[1, 2], Rm[0, 2] - Rm[2, 0], Rm[1, 0] - Rm[0, 1]])
    if th < 1e-9:
        return 0.5 * w
    return th / (2.0 * math.sin(th)) * w
