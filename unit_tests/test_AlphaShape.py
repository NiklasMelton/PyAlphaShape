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


def test_contains_point_inside_and_outside():
    points = np.array([[0, 0], [1, 0], [0.5, 1]])
    shape = AlphaShape(points, alpha=0.0)
    assert shape.contains_point(np.array([0.5, 0.5]))
    assert not shape.contains_point(np.array([2, 2]))


def test_distance_to_surface_inside_and_outside():
    points = np.array([[0, 0], [1, 0], [0.5, 1]])
    shape = AlphaShape(points, alpha=0.0)
    inside_point = np.array([0.5, 0.5])
    outside_point = np.array([2.0, 2.0])
    assert np.isclose(shape.distance_to_surface(inside_point), 0.0)
    assert shape.distance_to_surface(outside_point) > 0.0


def test_add_points_grows_shape():
    points = np.array([[0, 0], [1, 0], [0.5, 1]])
    shape = AlphaShape(points, alpha=3.0)
    n_perim_before = len(shape.perimeter_points)
    shape.add_points(np.array([[0.5, 1.5]]), perimeter_only=False)
    n_perim_after = len(shape.perimeter_points)
    assert n_perim_after > n_perim_before


def test_get_boundary_faces_returns_faces():
    points = np.random.rand(10, 3)
    shape = AlphaShape(points, alpha=1.0)
    faces = shape._get_boundary_faces()
    assert isinstance(faces, set)
    for f in faces:
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
