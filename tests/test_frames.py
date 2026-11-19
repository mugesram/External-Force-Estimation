import numpy as np

from efe.estimation import frames as Fr


def test_quat_roundtrip():
    rng = np.random.default_rng(0)
    for _ in range(50):
        q = rng.normal(size=4)
        q /= np.linalg.norm(q)
        q = q if q[0] >= 0 else -q
        assert np.allclose(Fr.rot_to_quat(Fr.quat_to_rot(q)), q, atol=1e-10)
        R = Fr.quat_to_rot(q)
        assert np.allclose(R @ R.T, np.eye(3), atol=1e-12)


def test_quat_mul_matches_matrix_product():
    rng = np.random.default_rng(1)
    a, b = rng.normal(size=4), rng.normal(size=4)
    a, b = a / np.linalg.norm(a), b / np.linalg.norm(b)
    assert np.allclose(Fr.quat_to_rot(Fr.quat_mul(a, b)), Fr.quat_to_rot(a) @ Fr.quat_to_rot(b))


def test_enu_flu_identity_attitude_maps_to_ned_frd():
    # Pegasus: level, nose pointing east (ENU x) = yaw 0 in ENU. In NED that is heading 90 deg (east).
    q = Fr.quat_xyzw_enu_flu_to_wxyz_ned_frd([0, 0, 0, 1])
    R = Fr.quat_to_rot(q)
    assert np.allclose(R @ [1, 0, 0], [0, 1, 0])     # body forward -> east
    assert np.allclose(R @ [0, 0, 1], [0, 0, 1])     # body down -> down


def test_vector_maps():
    assert np.allclose(Fr.enu_to_ned([1, 2, 3]), [2, 1, -3])
    assert np.allclose(Fr.flu_to_frd([1, 2, 3]), [1, -2, -3])


def test_geodetic_projection_small_offsets():
    lat0, lon0 = 47.397742, 8.545594
    ned = Fr.geodetic_to_ned(lat0 + 1e-4, lon0 + 1e-4, 500.0 + 2.0, lat0, lon0, 500.0)
    assert abs(ned[0] - 11.12) < 0.05
    assert abs(ned[1] - 11.12 * np.cos(np.radians(lat0))) < 0.05
    assert abs(ned[2] + 2.0) < 1e-9
