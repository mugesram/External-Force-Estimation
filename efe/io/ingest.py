"""Tier 1 ingest: PX4 ULog + the simulator's truth log -> RunData (the format the filters and evaluation read).

    python -m efe.io.ingest runs/tier1/S1_main/seed1          # writes imu/att/gnss/motors/battery/truth.parquet there

Sources:
  sensor_combined       IMU (specific force FRD, gyro), full rate            ULog
  vehicle_attitude      attitude quaternion (FRD -> NED)                     ULog (full rate; the DDS stream is 50 Hz,
                                                                             decimated here with --att-rate to match it)
  vehicle_gps_position  lat/lon/alt + NED velocity -> local NED via the datum ULog
  actuator_motors       motor commands u_i                                   ULog (not on DDS in this PX4)
  battery voltage       the simulator's battery (PX4 SITL simulates its own, unrelated one) + noise, at a set rate
  truth                 truth_raw.parquet from sim/efe_sim.py
Clock: PX4 timestamps are the simulator's time_usec in lockstep. The offset is not assumed: it is measured by
cross-correlating the accelerometer with the true specific force, written to meta.yaml, and applied to the truth.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

from ..estimation.frames import geodetic_to_ned
from .rundata import RunData


def find_ulog(run_dir):
    logs = sorted(Path(run_dir).glob("px4/log/**/*.ulg"), key=lambda p: p.stat().st_mtime)
    if not logs:
        raise FileNotFoundError(f"no ULog under {run_dir}/px4/log")
    return logs[-1]


def read_ulog(path, topics):
    from pyulog import ULog

    ulog = ULog(str(path), list(topics))
    out = {}
    for d in ulog.data_list:
        if d.multi_id == 0:
            out[d.name] = pd.DataFrame(d.data)
    return out, ulog


def _t(df, sample_col=None):
    t = df["timestamp"].to_numpy(dtype=np.int64)
    if sample_col and sample_col in df:
        s = df[sample_col].to_numpy(dtype=np.int64)
        t = np.where(s > 0, s, t)
    return t * 1e-6


def _cols(df, base, n):
    return np.column_stack([df[f"{base}[{i}]"].to_numpy(dtype=float) for i in range(n)])


def measure_clock_offset(t_imu, az, t_truth, fz, max_lag=0.05):
    """Lag [s] to ADD to truth times so that truth fz best matches the accelerometer az (least squares)."""
    dt = float(np.median(np.diff(t_imu)))
    lags = np.arange(-max_lag, max_lag + dt / 2, dt / 4)
    a = az - np.mean(az)
    cost = []
    for lag in lags:
        f = np.interp(t_imu, t_truth + lag, fz)
        cost.append(np.mean((a - (f - np.mean(f))) ** 2))
    return float(lags[int(np.argmin(cost))]), float(np.min(cost))


def ingest(run_dir, att_rate=None, warmup=10.0, seed=None):
    run_dir = Path(run_dir)
    meta = yaml.safe_load((run_dir / "meta.yaml").read_text())
    truth_file = run_dir / meta.get("truth_file", "truth_raw.parquet")
    truth_raw = pd.read_parquet(truth_file) if truth_file.suffix == ".parquet" else pd.read_csv(truth_file)
    ulog_path = find_ulog(run_dir)
    d, _ = read_ulog(ulog_path, ["sensor_combined", "vehicle_attitude", "vehicle_gps_position", "actuator_motors"])

    sc = d["sensor_combined"]
    t_imu = (sc["timestamp"].to_numpy(dtype=np.int64) + sc["accelerometer_timestamp_relative"].to_numpy(dtype=np.int64)) * 1e-6
    acc = _cols(sc, "accelerometer_m_s2", 3)
    gyro = _cols(sc, "gyro_rad", 3)

    va = d["vehicle_attitude"]
    t_att = _t(va, "timestamp_sample")
    q = _cols(va, "q", 4)
    if att_rate:
        keep = np.concatenate([[True], np.diff(np.floor(t_att * att_rate)) > 0])
        t_att, q = t_att[keep], q[keep]

    g = d["vehicle_gps_position"]
    t_g = _t(g, "timestamp_sample")
    if "latitude_deg" in g:
        lat, lon, alt = g["latitude_deg"].to_numpy(), g["longitude_deg"].to_numpy(), g["altitude_msl_m"].to_numpy()
    else:
        lat, lon, alt = g["lat"].to_numpy() * 1e-7, g["lon"].to_numpy() * 1e-7, g["alt"].to_numpy() * 1e-3
    datum = meta["datum"]
    p_g = geodetic_to_ned(lat, lon, alt, *datum)
    v_g = np.column_stack([g["vel_n_m_s"], g["vel_e_m_s"], g["vel_d_m_s"]]).astype(float)
    # PX4 publishes the same fix repeatedly between receiver updates: keep each fix once
    fresh = np.concatenate([[True], np.any(np.diff(np.column_stack([p_g, v_g]), axis=0) != 0, axis=1)])
    t_g, p_g, v_g = t_g[fresh], p_g[fresh], v_g[fresh]

    am = d["actuator_motors"]
    t_m = _t(am, "timestamp_sample")
    u = _cols(am, "control", 4)

    # truth on the PX4 clock
    t_tr = truth_raw["t_usec"].to_numpy(dtype=np.int64) * 1e-6
    t_s = float(meta["t_scenario_start"])
    w = (t_imu > t_s) & (t_imu < t_s + 20)
    offset, cost = measure_clock_offset(t_imu[w], acc[w, 2], t_tr, truth_raw["fz"].to_numpy())
    t_tr = t_tr + offset

    # command latency check: which shift of the logged commands reproduces what the simulator applied
    ut = truth_raw["u0"].to_numpy()
    wm = (t_tr > t_s) & (t_tr < t_s + 60)
    lags = np.arange(0, 5) * 0.004
    rms = [np.sqrt(np.mean((u[np.clip(np.searchsorted(t_m, t_tr[wm] - lg, side="right") - 1, 0, len(t_m) - 1), 0]
                            - ut[wm]) ** 2)) for lg in lags]
    measured_delay = float(lags[int(np.argmin(rms))])

    # the applied force must follow the scenario's schedule on PX4's clock (catches clock mix-ups in the injector)
    from ..scenario import load_scenario
    from ..truth.forces import ForceSchedule

    sched = ForceSchedule(load_scenario(meta["scenario"]).get("forces"))
    t_px4 = truth_raw["t_usec"].to_numpy(dtype=np.int64)[::5] * 1e-6      # the injector's own clock, unshifted
    F_sched = np.array([sched(tt - t_s) for tt in t_px4])
    sched_err = float(np.max(np.abs(F_sched - truth_raw[["Fx", "Fy", "Fz"]].to_numpy()[::5]))) if len(F_sched) else 0.0

    # battery voltage stream for the filter
    bv = meta.get("battery_voltage_stream", {"rate_hz": 50.0, "noise": 0.05})
    rng = np.random.default_rng(meta.get("seed", 0) if seed is None else seed)
    t_b = np.arange(t_tr[0], t_tr[-1], 1.0 / float(bv["rate_hz"]))
    V_b = np.interp(t_b, t_tr, truth_raw["V"].to_numpy()) + rng.normal(0, float(bv["noise"]), len(t_b))

    # keep from (scenario start - warm-up) to the end of the scenario
    t_end = t_s + float(meta["duration"])
    lo = t_s - warmup

    def win(t):
        return (t >= lo) & (t <= t_end)

    truth = truth_raw.rename(columns={}).copy()
    truth.insert(0, "t", t_tr)
    truth = truth.drop(columns=["t_usec"])
    truth = truth[win(t_tr)].reset_index(drop=True)

    run = RunData(
        imu=pd.DataFrame(np.column_stack([t_imu, acc, gyro])[win(t_imu)], columns=["t", "ax", "ay", "az", "gx", "gy", "gz"]),
        att=pd.DataFrame(np.column_stack([t_att, q])[win(t_att)], columns=["t", "qw", "qx", "qy", "qz"]),
        gnss=pd.DataFrame(np.column_stack([t_g, p_g, v_g])[win(t_g)], columns=["t", "px", "py", "pz", "vx", "vy", "vz"]),
        motors=pd.DataFrame(np.column_stack([t_m, u])[win(t_m)], columns=["t", "u0", "u1", "u2", "u3"]),
        battery=pd.DataFrame({"t": t_b[win(t_b)], "V": V_b[win(t_b)]}),
        truth=truth,
        meta=dict(meta),
    )
    run.meta.update({
        "ulog": str(ulog_path.relative_to(run_dir)),
        "clock_offset_s": offset, "clock_fit_rms": float(np.sqrt(cost)),
        "motor_delay_measured_s": measured_delay,
        "force_schedule_max_err": sched_err,
        "mass": float(meta.get("mass") or meta.get("mass_body") or 1.5),
        "rates_hz": {k: float(1.0 / np.median(np.diff(t))) for k, t in
                     [("imu", t_imu), ("att", t_att), ("gnss", t_g), ("motors", t_m), ("battery", t_b)]},
    })
    run.meta.pop("scenario_cfg", None)
    return run


def check_bag(run_dir):
    """Gate P4: every DDS sample equals the ULog sample with the same timestamp. Returns {topic: summary}."""
    run_dir = Path(run_dir)
    csv_dir = run_dir / "bag_csv"
    if not csv_dir.exists():
        raise FileNotFoundError(f"{csv_dir}: run tools/bag_to_csv.py with ROS 2 sourced first")
    pairs = {"fmu__out__sensor_combined": ("sensor_combined", ["accelerometer_m_s2", "gyro_rad"]),
             "fmu__out__vehicle_attitude": ("vehicle_attitude", ["q"]),
             "fmu__out__vehicle_gps_position": ("vehicle_gps_position", ["latitude_deg", "longitude_deg", "vel_n_m_s"]),
             "fmu__out__actuator_motors": ("actuator_motors", ["control"])}
    d, _ = read_ulog(find_ulog(run_dir), [v[0] for v in pairs.values()])
    out = {}
    for csv_name, (ulog_name, fields) in pairs.items():
        f = csv_dir / f"{csv_name}.csv"
        if not f.exists() or ulog_name not in d:
            out[ulog_name] = "not in both"
            continue
        bag = pd.read_csv(f)
        ul = d[ulog_name]
        m = bag.merge(ul, on="timestamp", suffixes=("_dds", "_ulog"))
        cols = [c for c in bag.columns if any(c == fl or c.startswith(fl + "[") for fl in fields)]
        diff = max(float(np.max(np.abs(m[c + "_dds"] - m[c + "_ulog"]))) for c in cols) if len(m) else np.nan
        t = bag["timestamp"].to_numpy() * 1e-6
        out[ulog_name] = {"dds_msgs": len(bag), "matched_in_ulog": len(m), "max_abs_diff": diff,
                          "dds_rate_hz": float(1 / np.median(np.diff(t))) if len(t) > 1 else np.nan}
    return out


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("run_dirs", nargs="+")
    ap.add_argument("--att-rate", type=float, default=50.0, help="decimate attitude to the DDS rate (0 = full)")
    ap.add_argument("--check-bag", action="store_true", help="only compare the DDS recording with the ULog")
    args = ap.parse_args(argv)
    for rd in args.run_dirs:
        if args.check_bag:
            for k, v in check_bag(rd).items():
                print(f"[check-bag] {rd} {k}: {v}")
            continue
        run = ingest(rd, att_rate=args.att_rate or None)
        meta_backup = Path(rd) / "meta_sim.yaml"
        if not meta_backup.exists():
            (Path(rd) / "meta.yaml").rename(meta_backup)
        run.save(rd)
        r = run.meta["rates_hz"]
        if run.meta["force_schedule_max_err"] > 1e-6:
            print(f"[ingest] WARNING {rd}: applied force deviates from the scenario schedule by up to "
                  f"{run.meta['force_schedule_max_err']:.2f} N (clock mismatch between injector and offboard node?)")
        print(f"[ingest] {rd}: motor delay measured {run.meta['motor_delay_measured_s'] * 1e3:.0f} ms "
              f"(configured {run.meta.get('motor_delay_s', 0) * 1e3:.0f} ms)")
        print(f"[ingest] {rd}: clock offset {run.meta['clock_offset_s'] * 1e3:+.1f} ms (fit rms "
              f"{run.meta['clock_fit_rms']:.3f} m/s^2), rates " + ", ".join(f"{k} {v:.0f} Hz" for k, v in r.items()))


if __name__ == "__main__":
    main()
