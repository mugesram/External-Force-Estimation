"""Scenario files (config/scenarios/*.yaml): the flight, the forces and the noise, shared by both tiers.

The trajectory is a list of segments, times in seconds from the scenario start, positions in local NED [m]:
  {t: [t0, t1], type: hold,    pos: [x, y, z]}
  {t: [t0, t1], type: zsteps,  pos: [x, y, z], amp: 0.5, period: 5}     z alternates pos.z -/+ amp every period
  {t: [t0, t1], type: figure8, center: [x, y, z], ax: 2.0, ay: 1.0, period: 10}
Between segments the last reference is held. Only numpy and yaml, so the ROS 2 offboard node can import it.
"""

from __future__ import annotations

import copy
import math
from pathlib import Path

import numpy as np
import yaml

REPO = Path(__file__).resolve().parents[1]
CONFIG = REPO / "config"


def load_yaml(path):
    return yaml.safe_load(Path(path).read_text())


def deep_merge(base, over):
    out = copy.deepcopy(base)
    for k, v in (over or {}).items():
        if isinstance(v, dict) and isinstance(out.get(k), dict):
            out[k] = deep_merge(out[k], v)
        else:
            out[k] = copy.deepcopy(v)
    return out


def load_scenario(name_or_path, overrides=None):
    """Scenario dict with the vehicle, battery and sensor defaults merged in (scenario keys win)."""
    p = Path(name_or_path)
    if not p.exists():
        p = CONFIG / "scenarios" / f"{name_or_path}.yaml"
    sc = load_yaml(p)
    base = {
        "vehicle": load_yaml(CONFIG / "vehicle_iris.yaml"),
        "battery": load_yaml(CONFIG / "battery_4s.yaml"),
        "sensors": load_yaml(CONFIG / "sensors.yaml"),
    }
    sc = deep_merge(base, sc)
    sc = deep_merge(sc, overrides or {})
    sc.setdefault("name", p.stem)
    return sc


class Trajectory:
    def __init__(self, segments):
        self.segments = sorted(segments, key=lambda s: s["t"][0])

    def _segment_ref(self, s, t):
        t0, t1 = s["t"]
        tt = min(max(t - t0, 0.0), t1 - t0)
        kind = s["type"]
        if kind == "hold":
            return np.asarray(s["pos"], dtype=float), np.zeros(3), np.zeros(3)
        if kind == "zsteps":
            p = np.asarray(s["pos"], dtype=float).copy()
            k = int(tt // float(s["period"]))
            p[2] += -float(s["amp"]) if k % 2 == 0 else float(s["amp"])
            return p, np.zeros(3), np.zeros(3)
        if kind == "figure8":
            c = np.asarray(s["center"], dtype=float)
            w = 2 * math.pi / float(s["period"])
            ax, ay = float(s["ax"]), float(s["ay"])
            p = c + np.array([ax * math.sin(w * tt), ay * math.sin(2 * w * tt), 0.0])
            v = np.array([ax * w * math.cos(w * tt), 2 * ay * w * math.cos(2 * w * tt), 0.0])
            a = np.array([-ax * w * w * math.sin(w * tt), -4 * ay * w * w * math.sin(2 * w * tt), 0.0])
            return p, v, a
        raise ValueError(f"unknown trajectory segment {kind!r}")

    def __call__(self, t):
        """(position, velocity, acceleration) reference at scenario time t."""
        current = None
        for s in self.segments:
            if t >= s["t"][0]:
                current = s
        if current is None:
            current = self.segments[0]
        if t > current["t"][1]:
            p, _, _ = self._segment_ref(current, current["t"][1])
            return p, np.zeros(3), np.zeros(3)
        return self._segment_ref(current, t)


def phases(sc):
    """Named time windows for per-phase metrics, from the scenario file (optional)."""
    return {k: tuple(v) for k, v in (sc.get("phases") or {}).items()}
