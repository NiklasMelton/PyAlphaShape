import numpy as np
import pytest
from pyalphashape.sphere_utils import (
    normalize,
    latlon_to_unit_vectors,
    unit_vectors_to_latlon,
    spherical_triangle_area,
    spherical_incircle_check,
    unit_vector,
    gnomonic_projection,
    arc_distance,
    spherical_circumradius
)


def test_normalize_unit_length():
    v = np.array([3.0, 4.0, 0.0])
    v_norm = normalize(v)
    assert np.isclose(np.linalg.norm(v_norm), 1.0)


def test_latlon_unit_vector_round_trip():
    latlon = np.array([[0, 0], [90, 0], [0, 90], [-45, 45]])
    vecs = latlon_to_unit_vectors(latlon)
    latlon_roundtrip = unit_vectors_to_latlon(vecs)
    assert np.allclose(latlon, latlon_roundtrip, atol=1e-6)


def test_spherical_triangle_area_known_case():
    a = np.array([1.0, 0.0, 0.0])
    b = np.array([0.0, 1.0, 0.0])
    c = np.array([0.0, 0.0, 1.0])
    area = spherical_triangle_area(a, b, c)
    assert area > 0 and area < 2 * np.pi  # should be < hemisphere


def test_spherical_incircle_check_inclusion():
    a = np.array([1.0, 0.0, 0.0])
    b = np.array([0.0, 1.0, 0.0])
    c = np.array([0.0, 0.0, 1.0])
    triangle = np.stack([a, b, c])
    pt = normalize(a + b + c)
    assert spherical_incircle_check(triangle, pt)


def test_unit_vector_equivalent_to_normalize():
    v = np.random.randn(3)
    assert np.allclose(unit_vector(v), normalize(v))


def test_gnomonic_projection_dimensions_and_spacing():
    pts = latlon_to_unit_vectors(np.array([[0, 0], [0, 1], [1, 0], [-1, -1]]))
    center = normalize(np.array([1, 0, 0]))
    proj = gnomonic_projection(pts, center)
    assert proj.shape == (4, 2)
    assert np.isfinite(proj).all()


def test_arc_distance_behavior():
    a = normalize(np.array([1.0, 0.0, 0.0]))
    b = normalize(np.array([0.0, 1.0, 0.0]))
    p_on_arc = normalize(a + b)
    dist = arc_distance(p_on_arc, a, b)
    assert np.isclose(dist, 0.0, atol=1e-2)

    p_off_arc = normalize(np.array([0.0, 0.0, 1.0]))
    dist = arc_distance(p_off_arc, a, b)
    assert np.isclose(dist, np.pi / 2, atol=1e-2)



def test_spherical_circumradius_equilateral_triangle():
    a = normalize(np.array([1.0, 0.0, 0.0]))
    b = normalize(np.array([0.0, 1.0, 0.0]))
    c = normalize(np.array([1.0, 1.0, 1.0]))
    c = normalize(c)
    points = np.stack([a, b, c])
    radius = spherical_circumradius(points)
    assert 0 < radius < np.pi


def test_spherical_circumradius_degenerate():
    pt = normalize(np.array([1.0, 0.0, 0.0]))
    points = np.stack([pt, pt, pt])
    circumradius = spherical_circumradius(points)
    assert np.isclose(circumradius, 0, atol=1e-2)

    pt1 = normalize(np.array([1.0, 0.0, 0.0]))
    pt2 = normalize(np.array([-1.0, 0.0, 0.0]))
    pt3 = normalize(np.array([0.0, 1.0, 0.0]))
    points = np.stack([pt1, pt2, pt3])
    circumradius = spherical_circumradius(points)
    assert np.isclose(circumradius, np.pi, atol=1e-2)

