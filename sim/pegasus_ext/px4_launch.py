"""Launching PX4 SITL for a run: our parameters, our working directory, our DDS settings.

Pegasus' own PX4LaunchTool runs PX4 from a throwaway temp directory with Isaac's environment. We need the run's
working directory (so the ULog lands in runs/<id>/px4/log), PX4_PARAM_* overrides, the DDS domain and agent port, and
optionally a different build. `install()` swaps Pegasus' launcher for this one before the vehicle starts.
"""

from __future__ import annotations

import os
import subprocess
from pathlib import Path

_settings: dict = {}


class RunPX4LaunchTool:
    """Same constructor and methods as pegasus' PX4LaunchTool."""

    def __init__(self, px4_dir, vehicle_id: int = 0, px4_model: str = "none_iris"):
        s = _settings
        self.px4_dir = str(px4_dir)
        self.vehicle_id = vehicle_id
        self.workdir = Path(s["workdir"])
        self.workdir.mkdir(parents=True, exist_ok=True)
        self.binary = Path(self.px4_dir) / "build" / s["build"] / "bin" / "px4"
        self.env = dict(os.environ)
        self.env.update({
            "PX4_SIM_MODEL": s.get("airframe", px4_model),
            "ROS_DOMAIN_ID": str(s["ros_domain_id"]),
            "PX4_UXRCE_DDS_PORT": str(s["uxrce_dds_port"]),
            "PX4_UXRCE_DDS_NS": "",
        })
        for k, v in s.get("params", {}).items():
            self.env[f"PX4_PARAM_{k}"] = str(v)
        self.px4_process = None

    def launch_px4(self):
        if not self.binary.exists():
            raise FileNotFoundError(f"{self.binary} not found: build it (make {_settings['build']}) in {self.px4_dir}")
        for saved in ("parameters.bson", "parameters_backup.bson"):   # every run starts from airframe defaults
            (self.workdir / saved).unlink(missing_ok=True)
        log = open(self.workdir / "px4.log", "ab")
        self.px4_process = subprocess.Popen(
            [str(self.binary), f"{self.px4_dir}/ROMFS/px4fmu_common/", "-s",
             f"{self.px4_dir}/ROMFS/px4fmu_common/init.d-posix/rcS", "-i", str(self.vehicle_id), "-d"],
            cwd=self.workdir, env=self.env, stdout=log, stderr=subprocess.STDOUT)

    def kill_px4(self):
        if self.px4_process is not None:
            self.px4_process.terminate()
            try:
                self.px4_process.wait(timeout=10)
            except subprocess.TimeoutExpired:
                self.px4_process.kill()
            self.px4_process = None

    def __del__(self):
        self.kill_px4()


def install(workdir, build, airframe, ros_domain_id, uxrce_dds_port, params):
    from pegasus.simulator.logic.backends import px4_mavlink_backend

    _settings.update(workdir=str(workdir), build=build, airframe=airframe, ros_domain_id=ros_domain_id,
                     uxrce_dds_port=uxrce_dds_port, params=dict(params))
    px4_mavlink_backend.PX4LaunchTool = RunPX4LaunchTool
