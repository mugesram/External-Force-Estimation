"""One flight's data, the same format for both simulation tiers and (later) real logs.

A run is a directory:
  imu.parquet      t, ax, ay, az (specific force, FRD), gx, gy, gz (rad/s, FRD)
  att.parquet      t, qw, qx, qy, qz (FRD -> NED)
  gnss.parquet     t, px, py, pz, vx, vy, vz (local NED)
  motors.parquet   t, u0 .. u3 (0..1)
  battery.parquet  t, V
  truth.parquet    (simulation only) t, px.., vx.., Fx, Fy, Fz, T, k, lam, bax.., V, SoC, qw..qz
  meta.yaml        mass, V0, scenario, seed, phases, events ...
All times are seconds on one clock (PX4's in Tier 1).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

STREAMS = ("imu", "att", "gnss", "motors", "battery", "truth")


@dataclass
class RunData:
    imu: pd.DataFrame
    att: pd.DataFrame
    gnss: pd.DataFrame
    motors: pd.DataFrame
    battery: pd.DataFrame
    truth: pd.DataFrame | None = None
    meta: dict = field(default_factory=dict)

    # convenience accessors -------------------------------------------------
    @property
    def t_imu(self):
        return self.imu["t"].to_numpy()

    def acc(self):
        return self.imu[["ax", "ay", "az"]].to_numpy()

    def gyro(self):
        return self.imu[["gx", "gy", "gz"]].to_numpy()

    def quats(self):
        return self.att[["qw", "qx", "qy", "qz"]].to_numpy()

    def motor_cmds(self):
        cols = [c for c in self.motors.columns if c.startswith("u")]
        return self.motors[sorted(cols)].to_numpy()

    def save(self, path):
        path = Path(path)
        path.mkdir(parents=True, exist_ok=True)
        for name in STREAMS:
            df = getattr(self, name)
            if df is not None:
                df.to_parquet(path / f"{name}.parquet", index=False)
        with open(path / "meta.yaml", "w") as fh:
            yaml.safe_dump(_plain(self.meta), fh, sort_keys=False)

    @classmethod
    def load(cls, path):
        path = Path(path)
        frames = {}
        for name in STREAMS:
            p = path / f"{name}.parquet"
            frames[name] = pd.read_parquet(p) if p.exists() else None
        meta = {}
        if (path / "meta.yaml").exists():
            meta = yaml.safe_load((path / "meta.yaml").read_text()) or {}
        return cls(meta=meta, **frames)

    def with_streams(self, **replace):
        """A shallow copy with some streams replaced (used by the replay Monte Carlo and ablations)."""
        d = dict(imu=self.imu, att=self.att, gnss=self.gnss, motors=self.motors, battery=self.battery,
                 truth=self.truth, meta=dict(self.meta))
        d.update(replace)
        return RunData(**d)


def _plain(obj):
    """numpy -> plain Python, so meta.yaml stays readable."""
    if isinstance(obj, dict):
        return {str(k): _plain(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_plain(v) for v in obj]
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, np.generic):
        return obj.item()
    return obj
