import numpy as np
import pytest
from pyalphashape.AlphaShape import AlphaShape, circumcenter, circumradius, alphasimplices


def test_circumcenter_2d_triangle():
    points = np.array([[0, 0], [1, 0], [0, 1]])
    center = circumcenter(points)
    expected = np.array([0.5, 0.5])
    assert np.allclose(np.dot(center, points), expected, atol=1e-8)


def test_circumradius_2d_triangle():
    points = np.array([[0, 0], [1, 0], [0, 1]])
    radius = circumradius(points)
    assert np.isclose(radius, np.sqrt(0.5), atol=1e-8)


def test_alphasimplices_valid_output():
    points = np.random.rand(10, 2)
    result = list(alphasimplices(points))
    for simplex, radius, simplex_points in result:
        assert isinstance(simplex, np.ndarray)
        assert isinstance(radius, float)
        assert isinstance(simplex_points, np.ndarray)
        assert simplex_points.shape[0] == len(simplex)


def test_alpha_shape_basic_perimeter():
    points = np.array([[0, 0], [1, 0], [0.5, 1], [0.5, 0.5]])
    alpha_shape = AlphaShape(points, alpha=1.0)
    assert len(alpha_shape.perimeter_points) > 0
    assert isinstance(alpha_shape.perimeter_edges, list)


def test_contains_point_inside():
    points = np.array([[0, 0], [1, 0], [0.5, 1]])
    shape = AlphaShape(points, alpha=0.0)
    assert shape.contains_point(np.array([0.5, 0.5]))

def test_contains_point_outside():
    points = np.array([[0, 0], [1, 0], [0.5, 1]])
    shape = AlphaShape(points, alpha=0.0)
    assert not shape.contains_point(np.array([2, 2]))

def test_distance_to_surface_various_cases():
    # Define a simple 2D triangle
    points = np.array([
        [0.0, 0.0],
        [1.0, 0.0],
        [0.5, 1.0],
    ])
    shape = AlphaShape(points, alpha=0.0)

    # 1) Interior point ⇒ distance = 0
    inside = np.array([0.5, 0.5])
    assert np.isclose(shape.distance_to_surface(inside), 0.0, atol=1e-9)

    # 2) Far above ⇒ closest to vertex (0.5,1.0):
    #    distance = sqrt((2-0.5)^2 + (2-1)^2) = sqrt(3.25)
    above = np.array([2.0, 2.0])
    expected_above = np.sqrt(3.25)
    got_above = shape.distance_to_surface(above)
    assert np.isclose(got_above, expected_above, atol=1e-9), (
        f"Above: expected {expected_above:.9f}, got {got_above:.9f}"
    )

    # 3) Directly below base edge ⇒ closest to edge y=0 between x=0 and x=1:
    #    point = (0.5, -1) ⇒ distance = 1.0
    below = np.array([0.5, -1.0])
    expected_below = 1.0
    got_below = shape.distance_to_surface(below)
    assert np.isclose(got_below, expected_below, atol=1e-9), (
        f"Below: expected {expected_below:.9f}, got {got_below:.9f}"
    )

    # 4) Down‑left corner ⇒ closest to vertex (0,0):
    #    point = (-1, -1) ⇒ distance = sqrt(2)
    down_left = np.array([-1.0, -1.0])
    expected_dl = np.sqrt(2)
    got_dl = shape.distance_to_surface(down_left)
    assert np.isclose(got_dl, expected_dl, atol=1e-9), (
        f"Down-left: expected {expected_dl:.9f}, got {got_dl:.9f}"
    )



def test_add_points_grows_shape():
    points = np.array([[0, 0], [1, 0], [0.5, 1]])
    shape = AlphaShape(points, alpha=3.0)
    n_perim_before = len(shape.perimeter_points)
    shape.add_points(np.array([[0.5, 1.5]]), perimeter_only=False)
    n_perim_after = len(shape.perimeter_points)
    assert n_perim_after > n_perim_before


def test_get_boundary_facets_returns_facets():
    points = np.random.rand(10, 3)
    shape = AlphaShape(points, alpha=1.0)
    facets = shape._get_boundary_facets()
    assert isinstance(facets, set)
    for f in facets:
        assert isinstance(f, tuple)
        assert len(f) == shape._dim


def test_centroid_reasonable():
    points = np.array([[0, 0], [1, 0], [0, 1]])
    shape = AlphaShape(points, alpha=0.0)
    centroid = shape.centroid
    assert centroid.shape == (2,)
    assert np.all(np.isfinite(centroid))


def test_alpha_shape_empty_with_single_point():
    pt = np.array([[1.0, 1.0]])
    shape = AlphaShape(pt)
    assert shape.vertices.shape == (1, 2)
