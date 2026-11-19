"""Bench identification (plan P3): the priors the filters start from, found the way a real bench test would.

    python -m efe.tools.bench                     # thrust sweep + step test, writes config/priors_bench.yaml + figures

One motor on a load cell, driven directly (no flight controller), powered from the same battery model the simulator
uses. The thrust and voltage the simulator applies come from efe.truth (Isaac only integrates the resulting forces), so
this is the same plant the Tier 1 flights see. The load cell samples at --cell-hz with noise, like an HX711.

  1. Thrust sweep u = 0.10 ... 1.00 at full and at low battery: fit k_b and u0 of
         T = k_b (V/V0)^2 ((u + u0)/(1 + u0))^2
  2. Step u 0.4 -> 0.7: fit tau of T(t) = T0 + dT (1 - exp(-(t - t_s)/tau))
  3. Perturb the fitted values (k_b +5 %, tau +20 %) as a bench-to-flight mismatch -> config/priors.yaml values.
Drag is fitted from force-free flight data with --drag-from <run dir> (efe.tools.bench drag).
"""

from __future__ import annotations

import argparse
import math
from pathlib import Path

import numpy as np
import yaml
from scipy.optimize import least_squares

from ..scenario import CONFIG, REPO, load_yaml
from ..truth.battery import BatteryModel
from ..truth.motors import MotorModel


def one_motor(vehicle):
    m = MotorModel.from_config({**vehicle, "k_f_spread": 0.0})
    m.n = 1
    m.k_f = m.k_f[:1]
    m.omega = np.zeros(1)
    return m


def sweep(vehicle, battery_cfg, soc, rng, cell_hz=80.0, cell_noise=0.02, settle=1.5, hold=1.0):
    """Stepped sweep; returns per-step means of (u, V, T_measured)."""
    mot = one_motor(vehicle)
    bat = BatteryModel.from_config({**battery_cfg, "soc0": soc})
    dt = 1e-3
    rows = []
    for u in np.round(np.arange(0.10, 1.001, 0.05), 3):
        samples_T, samples_V = [], []
        t, next_sample = 0.0, settle
        while t < settle + hold:
            T = mot.step(np.array([u]), bat.voltage, dt)
            # the bench powers one motor: battery sees one motor's current x 4 (as if all four were spinning)
            bat.step(np.repeat(T, 4), dt * 0.0)            # voltage under load, without draining between steps
            t += dt
            if t >= next_sample:
                samples_T.append(float(T[0]) + rng.normal(0, cell_noise))
                samples_V.append(bat.voltage + rng.normal(0, 0.01))
                next_sample += 1.0 / cell_hz
        rows.append((u, np.mean(samples_V), np.mean(samples_T)))
    return np.array(rows)


def fit_thrust_curve(rows, V0):
    u, V, T = rows.T

    def model(th):
        k, u0 = th
        return k * (V / V0) ** 2 * ((u + u0) / (1 + u0)) ** 2

    r = least_squares(lambda th: model(th) - T, x0=[8.0, 0.0], bounds=([0.1, -0.5], [100.0, 1.0]))
    return float(r.x[0]), float(r.x[1]), float(np.sqrt(np.mean(r.fun ** 2)))


def step_test(vehicle, battery_cfg, rng, cell_hz=200.0, cell_noise=0.02, u_lo=0.4, u_hi=0.7):
    mot = one_motor(vehicle)
    bat = BatteryModel.from_config(battery_cfg)
    dt = 1e-4
    mot.reset(np.array([u_lo]), bat.voltage)
    t_s, t_end = 0.2, 0.6
    ts, Ts = [], []
    t, next_sample = 0.0, 0.0
    while t < t_end:
        u = u_hi if t >= t_s else u_lo
        T = mot.step(np.array([u]), bat.voltage, dt)
        t += dt
        if t >= next_sample:
            ts.append(t)
            Ts.append(float(T[0]) + rng.normal(0, cell_noise))
            next_sample += 1.0 / cell_hz
    ts, Ts = np.array(ts), np.array(Ts)

    def model(th):
        T0, dT, tau, delay = th
        x = np.clip(ts - t_s - delay, 0, None)
        return T0 + dT * (1 - np.exp(-x / tau))

    r = least_squares(lambda th: model(th) - Ts, x0=[Ts[0], Ts[-1] - Ts[0], 0.03, 0.0],
                      bounds=([-np.inf, -np.inf, 1e-3, -0.01], [np.inf, np.inf, 1.0, 0.05]))
    return float(r.x[2]), ts, Ts, model(r.x)


def fit_drag(run, mass, settle=5.0):
    """Least-squares body drag from a force-free flight: m * (f_meas - f_model_without_drag) = -D R^T v (body)."""
    from ..estimation.frames import quat_to_rot
    from ..estimation.inputs import zoh_index

    tr = run.truth
    t = run.t_imu
    i = np.clip(zoh_index(tr["t"].to_numpy(), t), 0, len(tr) - 1)
    keep = (t - t[0]) > settle
    R = quat_to_rot(tr[["qw", "qx", "qy", "qz"]].to_numpy()[i][keep])
    v = tr[["vx", "vy", "vz"]].to_numpy()[i][keep]
    acc = run.acc()[keep]
    v_body = np.einsum("nji,nj->ni", R, v)
    b = tr[["bax", "bay", "baz"]].to_numpy()[i][keep]
    d = []
    for ax in (0, 1):                       # x, y: f = -d v_body / m + bias + noise (thrust is along z)
        y = -(acc[:, ax] - b[:, ax]) * mass
        d.append(float(np.dot(v_body[:, ax], y) / np.dot(v_body[:, ax], v_body[:, ax])))
    return [d[0], d[1], 0.0]


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--k-error", type=float, default=0.05, help="bench-to-flight error added to k_b")
    ap.add_argument("--tau-error", type=float, default=0.20, help="bench-to-flight error added to tau")
    ap.add_argument("--drag-from", default=None, help="force-free run directory (RunData) for the drag fit")
    ap.add_argument("--out", default=str(CONFIG / "priors_bench.yaml"))
    args = ap.parse_args(argv)
    rng = np.random.default_rng(args.seed)
    veh = load_yaml(CONFIG / "vehicle_iris.yaml")
    bat = load_yaml(CONFIG / "battery_4s.yaml")
    V0 = float(veh["V0"])

    full = sweep(veh, bat, 1.0, rng)
    low = sweep(veh, bat, 0.25, rng)
    k_b, u0, rms = fit_thrust_curve(np.vstack([full, low]), V0)
    tau, ts, Ts, fit = step_test(veh, bat, rng)
    truth = MotorModel.from_config(veh)
    print(f"[bench] k_b {k_b:.3f} N (truth {truth.k_b():.3f}, {100 * (k_b / truth.k_b() - 1):+.2f} %), "
          f"u0 {u0:.4f} (truth {truth.u0:.4f}), fit rms {rms:.3f} N")
    print(f"[bench] tau {tau * 1e3:.1f} ms (truth {truth.tau * 1e3:.1f} ms)")
    out = {
        "k_b_fit": k_b, "u0_fit": u0, "tau_fit": tau, "V0": V0,
        "k0": k_b * (1 + args.k_error), "lam0": 1.0 / (tau * (1 + args.tau_error)), "u0": u0,
        "mass": float(veh["mass"]), "drag": list(veh["drag"]),
        "perturbation": {"k": args.k_error, "tau": args.tau_error},
    }
    if args.drag_from:
        from ..io.rundata import RunData

        run = RunData.load(args.drag_from)
        out["drag"] = fit_drag(run, float(run.meta.get("mass", veh["mass"])))
        print(f"[bench] drag {np.round(out['drag'], 3)} (truth {veh['drag']})")
    Path(args.out).write_text("# written by efe.tools.bench\n" + yaml.safe_dump(out, sort_keys=False))

    from ..evaluation import plots

    fig, (a1, a2) = plots.plt.subplots(1, 2, figsize=(7.5, 3.0))
    for rows, lab, ls in [(full, "full battery", "-"), (low, "25 % charge", "--")]:
        u_, V_, T_ = rows.T
        a1.plot(u_, T_, "o", ms=3.5, color=plots.color("EKF-B+bias") if ls == "-" else plots.color("EKF-B"),
                label=f"{lab} ({V_.max():.1f}-{V_.min():.1f} V under load)")
        a1.plot(u_, k_b * (V_ / V0) ** 2 * ((u_ + u0) / (1 + u0)) ** 2, ls, color=plots.INK2, lw=1.0)
    a1.plot([], [], color=plots.INK2, lw=1.0, label="fit at each point's voltage")
    a1.set_xlabel("motor command $u$")
    a1.set_ylabel("thrust per motor [N]")
    a1.legend(loc="upper left")
    a2.plot((ts - 0.2) * 1e3, Ts, ".", ms=2.5, color=plots.color("EKF-B+bias"), label="load cell")
    a2.plot((ts - 0.2) * 1e3, fit, color=plots.INK, lw=1.2, label=f"fit, $\\tau$ = {tau * 1e3:.0f} ms")
    a2.set_xlabel("time after step [ms]")
    a2.set_ylabel("thrust [N]")
    a2.legend(loc="lower right")
    plots.save(fig, REPO / "results" / "bench" / "bench_identification")
    print(f"[bench] wrote {args.out} and results/bench/bench_identification.pdf")


if __name__ == "__main__":
    main()
