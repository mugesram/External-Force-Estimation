"""Tier 1 simulator: Isaac Sim + Pegasus (Iris) + PX4 SITL in lockstep, with our truth model and truth log.

    ~/isaacsim/python.sh sim/efe_sim.py --scenario S1_main --seed 1 --headless
    (then, in other terminals: the XRCE agent and the offboard node; see tools/run_tier1.sh, which does all of it)

What it does:
  * the Iris from Pegasus with EfeMultirotor (lockstep ordering), VoltageLaggedThrustCurve (open-loop ESC, motor lag,
    battery), EfeIMU (an accelerometer bias that reaches PX4), an RTK-grade GPS, linear drag;
  * the scenario's force schedule applied every physics step, starting at --t0 seconds of simulated time (the offboard
    node starts the scenario's trajectory at the same simulated time: both read the one lockstep clock);
  * PX4 SITL launched by sim/pegasus_ext/px4_launch.py with config/px4_params.yaml, working directory and ULog in
    runs/tier1/<scenario>/seed<N>/px4/;
  * truth at every physics step to runs/tier1/<scenario>/seed<N>/truth_raw.parquet, plus meta.yaml.
Runs until --t0 + scenario duration + 5 s of simulated time, or the window closes.
"""

import argparse
import os
import sys
import time
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(REPO / "sim"))

import yaml  # noqa: E402

from efe.scenario import load_scenario, load_yaml  # noqa: E402

ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
ap.add_argument("--scenario", default="S1_main")
ap.add_argument("--seed", type=int, default=1)
ap.add_argument("--headless", action="store_true")
ap.add_argument("--t0", type=float, default=None, help="scenario start [s of sim time]; default config/sim_tier1.yaml")
ap.add_argument("--duration", type=float, default=None, help="override total simulated time [s]")
ap.add_argument("--out", default=None, help="run directory; default runs/tier1/<scenario>/seed<N>")
ap.add_argument("--px4-dir", default=os.path.expanduser("~/Documents/Repos/PX4-Autopilot"))
ap.add_argument("--set", action="append", default=[], metavar="KEY=VALUE",
                help="scenario override, e.g. --set vehicle.tau=0.05 (repeatable)")

ap.add_argument("--realtime", action="store_true", help="hold the loop to real time (needed when DDS nodes pace on wall time)")
args = ap.parse_args()


def parse_sets(items):
    out = {}
    for item in items:
        key, val = item.split("=", 1)
        d = out
        *path, last = key.split(".")
        for p in path:
            d = d.setdefault(p, {})
        d[last] = yaml.safe_load(val)
    return out


sim_cfg = load_yaml(REPO / "config" / "sim_tier1.yaml")
px4_cfg = load_yaml(REPO / "config" / "px4_params.yaml")
sc = load_scenario(args.scenario, {"battery": dict(sim_cfg.get("battery_overrides") or {}), **parse_sets(args.set)})
t0 = float(args.t0 if args.t0 is not None else sim_cfg["t_scenario_start"])
duration = float(args.duration if args.duration is not None else t0 + float(sc["duration"]) + 5.0)
run_dir = Path(args.out or REPO / "runs" / "tier1" / sc["name"] / f"seed{args.seed}")
run_dir.mkdir(parents=True, exist_ok=True)
physics_hz = float(sim_cfg["physics_hz"])

sys.stdout.reconfigure(line_buffering=True)
_stdout = sys.stdout

from isaacsim import SimulationApp  # noqa: E402

simulation_app = SimulationApp({"headless": args.headless})

import numpy as np  # noqa: E402
import omni.timeline  # noqa: E402
from isaacsim.core.api.objects import GroundPlane  # noqa: E402
from isaacsim.core.api.world import World  # noqa: E402
from pegasus.simulator.logic.backends.px4_mavlink_backend import PX4MavlinkBackend, PX4MavlinkBackendConfig  # noqa: E402
from pegasus.simulator.logic.dynamics import LinearDrag  # noqa: E402
from pegasus.simulator.logic.interface.pegasus_interface import PegasusInterface  # noqa: E402
from pegasus.simulator.logic.sensors import GPS, Barometer, Magnetometer  # noqa: E402
from pegasus.simulator.logic.vehicles.multirotor import MultirotorConfig  # noqa: E402
from pegasus.simulator.params import ROBOTS  # noqa: E402
from pxr import UsdLux  # noqa: E402

from efe.truth.battery import BatteryModel  # noqa: E402
from efe.truth.forces import ForceSchedule  # noqa: E402
from pegasus_ext import px4_launch  # noqa: E402
from pegasus_ext.imu import EfeIMU  # noqa: E402
from pegasus_ext.multirotor import EfeMultirotor  # noqa: E402
from pegasus_ext.thrust import VoltageLaggedThrustCurve  # noqa: E402
from pegasus_ext.truth_logger import TruthLogger  # noqa: E402


def say(msg):
    """Isaac redirects sys.stdout once the app is up; write to the original stream."""
    _stdout.write(msg + "\n")
    _stdout.flush()


def main():
    rng = np.random.default_rng(args.seed)
    np.random.seed(args.seed)                     # Pegasus' own sensors draw from numpy's global generator
    px4_launch.install(workdir=run_dir / "px4", build=px4_cfg["build"], airframe=px4_cfg["airframe"],
                       ros_domain_id=px4_cfg["ros_domain_id"], uxrce_dds_port=px4_cfg["uxrce_dds_port"],
                       params=px4_cfg["params"])

    pg = PegasusInterface()
    pg.set_world_settings(physics_dt=1.0 / physics_hz)
    pg._world = World(**pg._world_settings)
    world = pg.world
    GroundPlane(prim_path="/World/GroundPlane", size=200.0, color=np.array([0.45, 0.45, 0.47]))
    stage = world.stage
    UsdLux.DistantLight.Define(stage, "/World/Sun").CreateIntensityAttr(2500.0)
    UsdLux.DomeLight.Define(stage, "/World/Sky").CreateIntensityAttr(600.0)

    every_step = {"update_rate": 2.0 * physics_hz}   # sensors decide by summing dt: ask for twice the rate
    eb = sim_cfg["imu_extra_bias"]
    imu = EfeIMU(every_step, acc_bias_sigma=float(sc.get("sensors", {}).get("acc_bias_sigma", eb["acc_bias_sigma"])),
                 acc_bias_rw=eb["acc_bias_rw"], acc_bias_tau=eb["acc_bias_tau"], seed=args.seed)
    battery = BatteryModel.from_config(sc["battery"])
    thrust = VoltageLaggedThrustCurve(sc["vehicle"], battery, rng=np.random.default_rng(args.seed + 1000))
    logger = TruthLogger(run_dir / "truth_raw")

    cfg = MultirotorConfig()
    cfg.thrust_curve = thrust
    cfg.drag = LinearDrag(list(sc["vehicle"]["drag"]))
    cfg.sensors = [Barometer(every_step), imu, Magnetometer(every_step), GPS(dict(sim_cfg["gps"]))]
    bk = sim_cfg["pegasus_px4_backend"]
    cfg.backends = [PX4MavlinkBackend(PX4MavlinkBackendConfig({
        "vehicle_id": 0, "px4_autolaunch": True, "px4_dir": args.px4_dir, "px4_vehicle_model": px4_cfg["airframe"],
        "enable_lockstep": True, "update_rate": physics_hz,
        "input_offset": bk["input_offset"], "input_scaling": bk["input_scaling"],
        "zero_position_armed": bk["zero_position_armed"],
    }))]
    vehicle = EfeMultirotor("/World/quadrotor", ROBOTS["Iris"], 0, list(sim_cfg["spawn_enu"]), [0.0, 0.0, 0.0, 1.0],
                            config=cfg, injector=ForceSchedule(sc.get("forces")), logger=logger, imu=imu,
                            scenario_start=t0)
    world.reset()

    meta = {
        "tier": 1, "scenario": sc["name"], "seed": args.seed, "t_scenario_start": t0, "physics_hz": physics_hz,
        "motor_delay_s": float(sim_cfg.get("motor_delay_s", 0.0)),
        "mass": None, "drag": list(sc["vehicle"]["drag"]), "V0": float(sc["vehicle"]["V0"]), "u0": thrust.motors.u0,
        "duration": float(sc["duration"]), "phases": sc.get("phases", {}), "events": sc.get("events", {}),
        "datum": [float(pg.latitude), float(pg.longitude), float(pg.altitude)],
        "truth_params": {"k_b": thrust.motors.k_b(), "tau": thrust.motors.tau,
                         "voltage_exponent": thrust.motors.voltage_exponent},
        "sensors": sc.get("sensors", {}), "battery_voltage_stream": sim_cfg["battery_voltage_stream"],
        "scenario_cfg": sc,
    }
    try:
        dc = vehicle.get_dc_interface()
        art = dc.get_articulation("/World/quadrotor")
        bodies = [dc.get_articulation_body(art, i) for i in range(dc.get_articulation_body_count(art))]
        masses = {dc.get_rigid_body_name(b): float(dc.get_rigid_body_properties(b).mass) for b in bodies}
        meta["mass_bodies"] = masses
        meta["mass"] = float(sum(masses.values()))       # body + rotors: what the thrust has to carry
        say(f"[efe_sim] vehicle mass {meta['mass']:.4f} kg ({masses})")
    except Exception as e:
        say(f"[efe_sim] could not read the vehicle mass: {e}")

    timeline = omni.timeline.get_timeline_interface()
    timeline.play()
    say(f"[efe_sim] {sc['name']} seed {args.seed}: scenario starts at t = {t0:.1f} s, runs to {duration:.1f} s; "
          f"run dir {run_dir}")
    wall0, steps = time.perf_counter(), 0
    try:
        while simulation_app.is_running() and vehicle.px4_time() < duration:
            world.step(render=not args.headless)
            steps += 1
            if args.realtime:
                ahead = vehicle.sim_time - (time.perf_counter() - wall0)
                if ahead > 0.002:
                    time.sleep(ahead)
            if steps % 2500 == 0:
                wall = time.perf_counter() - wall0
                say(f"[efe_sim] sim {vehicle.sim_time:7.1f} s  wall {wall:7.1f} s  RTF {vehicle.sim_time / wall:.2f}  "
                      f"V {battery.voltage:5.2f}  z {-vehicle.state.position[2]:+.2f}")
    except KeyboardInterrupt:
        say("[efe_sim] interrupted")
    finally:
        path = logger.save()
        meta["truth_file"] = path.name
        with open(run_dir / "meta.yaml", "w") as fh:
            yaml.safe_dump(_plain(meta), fh, sort_keys=False)
        timeline.stop()
        say(f"[efe_sim] wrote {path} ({len(logger.rows)} steps) and meta.yaml")
    simulation_app.close()


def _plain(o):
    if isinstance(o, dict):
        return {str(k): _plain(v) for k, v in o.items()}
    if isinstance(o, (list, tuple)):
        return [_plain(v) for v in o]
    if isinstance(o, np.ndarray):
        return o.tolist()
    if isinstance(o, np.generic):
        return o.item()
    return o


if __name__ == "__main__":
    try:
        main()
    except Exception:
        import traceback

        traceback.print_exc()
        sys.stdout.flush()
        os._exit(1)
