"""Frames and rotations, in one place.

Conventions used everywhere in the estimator (PX4's):
  world  NED (x north, y east, z down)
  body   FRD (x forward, y right, z down)
  quaternion  [w, x, y, z], body -> world (PX4 vehicle_attitude.q)
  R = quat_to_rot(q) maps body vectors into the world frame.

Isaac Sim / Pegasus use world ENU and body FLU, with quaternions [x, y, z, w]. The converters below are the only
place where the two meet.
"""

from __future__ import annotations

import math

import numpy as np

E3 = np.array([0.0, 0.0, 1.0])

# Vector maps between the two conventions (both are their own inverse).
NED_FROM_ENU = np.array([[0.0, 1.0, 0.0], [1.0, 0.0, 0.0], [0.0, 0.0, -1.0]])
FRD_FROM_FLU = np.diag([1.0, -1.0, -1.0])


def enu_to_ned(v):
    return np.asarray(v, dtype=float) @ NED_FROM_ENU.T


def flu_to_frd(v):
    return np.asarray(v, dtype=float) @ FRD_FROM_FLU.T


ned_to_enu = enu_to_ned
frd_to_flu = flu_to_frd


def quat_to_rot(q):
    """[w, x, y, z] (unit) -> 3x3 rotation matrix. Accepts (4,) or (N, 4)."""
    q = np.asarray(q, dtype=float)
    w, x, y, z = np.moveaxis(q, -1, 0)
    R = np.stack([
        np.stack([1 - 2 * (y * y + z * z), 2 * (x * y - w * z), 2 * (x * z + w * y)], axis=-1),
        np.stack([2 * (x * y + w * z), 1 - 2 * (x * x + z * z), 2 * (y * z - w * x)], axis=-1),
        np.stack([2 * (x * z - w * y), 2 * (y * z + w * x), 1 - 2 * (x * x + y * y)], axis=-1),
    ], axis=-2)
    return R


def rot_to_quat(R):
    """3x3 rotation matrix -> [w, x, y, z] with w >= 0."""
    R = np.asarray(R, dtype=float)
    tr = R[0, 0] + R[1, 1] + R[2, 2]
    if tr > 0:
        s = math.sqrt(tr + 1.0) * 2
        q = [0.25 * s, (R[2, 1] - R[1, 2]) / s, (R[0, 2] - R[2, 0]) / s, (R[1, 0] - R[0, 1]) / s]
    elif R[0, 0] > R[1, 1] and R[0, 0] > R[2, 2]:
        s = math.sqrt(1.0 + R[0, 0] - R[1, 1] - R[2, 2]) * 2
        q = [(R[2, 1] - R[1, 2]) / s, 0.25 * s, (R[0, 1] + R[1, 0]) / s, (R[0, 2] + R[2, 0]) / s]
    elif R[1, 1] > R[2, 2]:
        s = math.sqrt(1.0 + R[1, 1] - R[0, 0] - R[2, 2]) * 2
        q = [(R[0, 2] - R[2, 0]) / s, (R[0, 1] + R[1, 0]) / s, 0.25 * s, (R[1, 2] + R[2, 1]) / s]
    else:
        s = math.sqrt(1.0 + R[2, 2] - R[0, 0] - R[1, 1]) * 2
        q = [(R[1, 0] - R[0, 1]) / s, (R[0, 2] + R[2, 0]) / s, (R[1, 2] + R[2, 1]) / s, 0.25 * s]
    q = np.array(q)
    q /= np.linalg.norm(q)
    return q if q[0] >= 0 else -q


def quat_mul(a, b):
    """Hamilton product, [w, x, y, z]."""
    aw, ax, ay, az = a
    bw, bx, by, bz = b
    return np.array([
        aw * bw - ax * bx - ay * by - az * bz,
        aw * bx + ax * bw + ay * bz - az * by,
        aw * by - ax * bz + ay * bw + az * bx,
        aw * bz + ax * by - ay * bx + az * bw,
    ])


def quat_exp(rotvec):
    """Rotation vector [rad] -> quaternion [w, x, y, z]."""
    rotvec = np.asarray(rotvec, dtype=float)
    angle = float(np.linalg.norm(rotvec))
    if angle < 1e-12:
        return np.array([1.0, 0.5 * rotvec[0], 0.5 * rotvec[1], 0.5 * rotvec[2]])
    axis = rotvec / angle
    return np.concatenate([[math.cos(angle / 2)], axis * math.sin(angle / 2)])


def slerp(q0, q1, f):
    """Spherical interpolation between two [w, x, y, z] quaternions, f in [0, 1]."""
    q0 = np.asarray(q0, dtype=float)
    q1 = np.asarray(q1, dtype=float)
    d = float(np.dot(q0, q1))
    if d < 0:
        q1, d = -q1, -d
    if d > 0.9995:
        q = q0 + f * (q1 - q0)
        return q / np.linalg.norm(q)
    th = math.acos(min(d, 1.0))
    s = math.sin(th)
    return (math.sin((1 - f) * th) * q0 + math.sin(f * th) * q1) / s


def quat_xyzw_enu_flu_to_wxyz_ned_frd(q_xyzw):
    """Isaac/Pegasus attitude (body FLU -> world ENU, [x, y, z, w]) -> PX4 attitude (FRD -> NED, [w, x, y, z])."""
    x, y, z, w = q_xyzw
    R_enu_flu = quat_to_rot([w, x, y, z])
    return rot_to_quat(NED_FROM_ENU @ R_enu_flu @ FRD_FROM_FLU)


def skew(v):
    return np.array([[0.0, -v[2], v[1]], [v[2], 0.0, -v[0]], [-v[1], v[0], 0.0]])


# --- geodetic: PX4's azimuthal equidistant projection (lib/geo MapProjection), so local NED matches PX4 ---
EARTH_RADIUS = 6371000.0


def geodetic_to_ned(lat_deg, lon_deg, alt_m, ref_lat_deg, ref_lon_deg, ref_alt_m):
    """Latitude/longitude/altitude -> local NED [m] about a reference point. Vectorised over the first three args."""
    lat = np.radians(np.asarray(lat_deg, dtype=float))
    lon = np.radians(np.asarray(lon_deg, dtype=float))
    lat0, lon0 = math.radians(ref_lat_deg), math.radians(ref_lon_deg)
    sin_lat0, cos_lat0 = math.sin(lat0), math.cos(lat0)
    sin_lat, cos_lat = np.sin(lat), np.cos(lat)
    cos_d_lon = np.cos(lon - lon0)
    arg = np.clip(sin_lat0 * sin_lat + cos_lat0 * cos_lat * cos_d_lon, -1.0, 1.0)
    c = np.arccos(arg)
    k = np.where(np.abs(c) > 0, c / np.where(np.sin(c) == 0, 1.0, np.sin(c)), 1.0)
    north = k * (cos_lat0 * sin_lat - sin_lat0 * cos_lat * cos_d_lon) * EARTH_RADIUS
    east = k * cos_lat * np.sin(lon - lon0) * EARTH_RADIUS
    down = -(np.asarray(alt_m, dtype=float) - ref_alt_m)
    return np.stack([north, east, down], axis=-1)
