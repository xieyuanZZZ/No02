import numpy as np
from fhd.adapters.common_evaluator import evaluate_run


def _paths(n=2, h=2):
    return np.stack([np.array([[0, 0], [0, 0.]]), np.array([[1, 0], [1, 0.]])]) if n == 2 else None


def test_crossing_between_samples_is_detected():
    # endpoints at t=0,1 are far; middle is overlapping
    pos = np.array([[[0., 0.], [1., 0.]], [[0.5, 0.], [0.5, 0.]], [[1., 0.], [0., 0.]]])
    vel = np.zeros_like(pos)
    paths = np.stack([[[0, 0], [1, 0]], [[1, 0], [0, 0]]])
    r = evaluate_run(np.array([0., 1., 2.]), pos, vel, paths)
    assert r["inter_robot_violation"] is True
    assert r["min_pair_distance"] == 0.0


def test_exact_safe_boundary_not_violation():
    pos = np.array([[[0., 0.], [0.35, 0.]], [[0., 0.], [0.35, 0.]]])
    vel = np.zeros_like(pos)
    paths = np.stack([[[0, 0], [0, 0]], [[0.35, 0], [0.35, 0]]])
    r = evaluate_run(np.array([0., 0.1]), pos, vel, paths)
    assert r["min_pair_distance"] == 0.35
    assert r["inter_robot_violation"] is False


def test_static_obstacle_and_arrival():
    pos = np.array([[[0., 0.], [0.2, 0.]], [[1., 0.], [1.2, 0.]]])
    vel = np.zeros_like(pos)
    paths = np.stack([[[0, 0], [1, 0]], [[0.2, 0], [1.2, 0]]])
    r = evaluate_run(np.array([0., 0.1]), pos, vel, paths, rects=[(0.4, -1, 0.8, 1)])
    assert r["static_obstacle_violation"] is True
    assert r["static_status"] == "evaluated"
    assert r["all_arrived"] is True
    assert r["safe_completion"] is False
