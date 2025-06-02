import numpy as np
import pytest
from pyalphashape.SphereCap import (
    spherical_distance,
    cap_through,
    is_in_cap,
    welzl_spherical_cap,
    minimum_enclosing_spherical_cap,
    maximum_empty_spherical_cap
)


def normalize(v):
    return v / np.linalg.norm(v)


def test_spherical_distance_orthogonal():
    u = normalize(np.array([1.0, 0.0, 0.0]))
    v = normalize(np.array([0.0, 1.0, 0.0]))
    d = spherical_distance(u, v)
    assert np.isclose(d, np.pi / 2, atol=1e-8)


def test_cap_through_cases():
    p1 = normalize(np.array([1.0, 0.0, 0.0]))
    p2 = normalize(np.array([0.0, 1.0, 0.0]))
    p3 = normalize(np.array([0.0, 0.0, 1.0]))

    # 0 points
    center, radius = cap_through([])
    assert np.allclose(center, [1.0, 0.0, 0.0])
    assert np.isclose(radius, 0.0)

    # 1 point
    center, radius = cap_through([p1])
    assert np.allclose(center, p1)
    assert np.isclose(radius, 0.0)

    # 2 points
    center, radius = cap_through([p1, p2])
    assert np.isclose(spherical_distance(center, p1), radius)

    # 3 points
    center, radius = cap_through([p1, p2, p3])
    assert all(is_in_cap(p, center, radius) for p in [p1, p2, p3])


def test_is_in_cap_logic():
    center = normalize(np.array([1.0, 0.0, 0.0]))
    inside = normalize(np.array([0.999, 0.01, 0.0]))
    outside = normalize(np.array([-1.0, 0.0, 0.0]))
    radius = spherical_distance(center, inside)
    assert is_in_cap(inside, center, radius)
    assert not is_in_cap(outside, center, radius)


def test_welzl_spherical_cap_encloses_all():
    points = [normalize(np.random.randn(3)) for _ in range(10)]
    center, radius = welzl_spherical_cap(points)
    assert all(is_in_cap(p, center, radius + 1e-8) for p in points)


def test_minimum_enclosing_spherical_cap_reproducible():
    points = np.array([normalize(np.random.randn(3)) for _ in range(10)])
    c1, r1 = minimum_enclosing_spherical_cap(points, seed=42)
    c2, r2 = minimum_enclosing_spherical_cap(points, seed=42)
    assert np.allclose(c1, c2)
    assert np.isclose(r1, r2)


def test_maximum_empty_spherical_cap_empty_result():
    points = np.array([
        normalize(np.array([1, 0, 0])),
        normalize(np.array([0, 1, 0])),
        normalize(np.array([0, 0, 1])),
    ])
    center, radius = maximum_empty_spherical_cap(points, n_restarts=5)
    assert np.isfinite(radius)
    assert radius > 0
    assert all(not is_in_cap(p, center, radius - 1e-6) for p in points)
