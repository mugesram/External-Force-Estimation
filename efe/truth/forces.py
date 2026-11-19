"""External-force schedule (the ground truth the filter must recover), world NED, in newtons.

NED: +z is DOWN, so a hanging mass pulls with +m g in z.

Event types (times relative to the scenario start, seconds):
  step     {type: step, t0, t1, F: [x, y, z]}
  ramp     {type: ramp, t0, t1, F: [x, y, z]}            linear 0 -> F over [t0, t1], held after until t_end
  sine     {type: sine, t0, t1, amp: [..], freq, bias: [..], phase: 0}
  impulse  {type: impulse, t0, dur, F: [..]}
  payload  {type: payload, t0, t1, mass, length, swing_deg, damping}   hanging load: weight + damped pendulum swing
numpy only (runs in Isaac Sim too).
"""

from __future__ import annotations

import math

import numpy as np

G = 9.80665


class ForceSchedule:
    def __init__(self, events=None):
        self.events = [dict(e) for e in (events or [])]
        for e in self.events:
            if e["type"] not in ("step", "ramp", "sine", "impulse", "payload"):
                raise ValueError(f"unknown force event type {e['type']!r}")

    def __call__(self, t):
        F = np.zeros(3)
        for e in self.events:
            F += _event(e, t)
        return F


def _event(e, t):
    kind = e["type"]
    if kind == "impulse":
        t0, dur = float(e["t0"]), float(e["dur"])
        return np.asarray(e["F"], dtype=float) if t0 <= t < t0 + dur else np.zeros(3)
    t0, t1 = float(e["t0"]), float(e.get("t1", math.inf))
    if kind == "ramp":
        t_end = float(e.get("t_end", math.inf))
        if t < t0 or t >= t_end:
            return np.zeros(3)
        return np.asarray(e["F"], dtype=float) * min((t - t0) / max(t1 - t0, 1e-9), 1.0)
    if not (t0 <= t < t1):
        return np.zeros(3)
    if kind == "step":
        return np.asarray(e["F"], dtype=float)
    if kind == "sine":
        amp = np.asarray(e["amp"], dtype=float)
        bias = np.asarray(e.get("bias", [0, 0, 0]), dtype=float)
        return bias + amp * math.sin(2 * math.pi * float(e["freq"]) * (t - t0) + float(e.get("phase", 0.0)))
    # payload: weight plus a damped pendulum swing in x and y (small-angle), as seen by the drone
    m, L = float(e["mass"]), float(e["length"])
    w = math.sqrt(G / L)
    th0 = math.radians(float(e.get("swing_deg", 10.0)))
    damp = float(e.get("damping", 0.05))
    tt = t - t0
    env = th0 * math.exp(-damp * w * tt)
    thx = env * math.cos(w * tt)
    thy = 0.5 * env * math.sin(w * tt)
    tension = m * G
    return np.array([tension * math.sin(thx), tension * math.sin(thy), tension * math.cos(thx) * math.cos(thy)])
