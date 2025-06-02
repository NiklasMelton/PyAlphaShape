import numpy as np
import pytest
from pyalphashape.SphericalAlphaShape import SphericalAlphaShape


def test_basic_initialization():
    points = np.array([
        [0, 0],
        [0, 90],
        [45, 45],
        [30, 10],
        [15, 60]
    ])
    shape = SphericalAlphaShape(points, alpha=1.0)
    assert not shape.is_empty
    assert shape.vertices.shape[1] == 3  # 3D unit vectors
    assert shape.perimeter_points_latlon.shape[1] == 2


def test_contains_point_inside_and_outside():
    points = np.array([
        [0, 0],
        [0, 90],
        [45, 45],
        [30, 10],
        [15, 60]
    ])
    shape = SphericalAlphaShape(points, alpha=1.0)
    assert shape.contains_point(np.array([20, 30]))  # should be inside
    assert not shape.contains_point(np.array([-60, 120]))  # should be outside


def test_distance_to_surface_inside_and_outside():
    points = np.array([
        [0, 0],
        [0, 90],
        [45, 45],
        [30, 10],
        [15, 60]
    ])
    shape = SphericalAlphaShape(points, alpha=1.0)
    assert np.isclose(shape.distance_to_surface(np.array([20, 30])), 0.0)
    assert shape.distance_to_surface(np.array([-60, 120])) > 0


def test_add_points_behavior():
    points = np.array([
        [0, 0],
        [0, 90],
        [45, 45],
    ])
    shape = SphericalAlphaShape(points, alpha=1.0)
    n_before = len(shape.perimeter_points)
    shape.add_points(np.array([[60, 60]]), perimeter_only=True)
    n_after = len(shape.perimeter_points)
    assert n_after > n_before


def test_triangle_faces_properties():
    points = np.array([
        [0, 0],
        [0, 90],
        [45, 45],
        [30, 10],
        [15, 60]
    ])
    shape = SphericalAlphaShape(points, alpha=1.0)
    faces_vec = shape.triangle_faces
    faces_ll = shape.triangle_faces_latlon
    for face in faces_vec:
        assert face.shape == (3, 3)  # 3 unit vectors
    for face in faces_ll:
        assert face.shape == (3, 2)  # 3 lat/lon pairs


def test_centroid_validity():
    points = np.array([
        [0, 0],
        [0, 90],
        [45, 45],
        [30, 10],
        [15, 60]
    ])
    shape = SphericalAlphaShape(points, alpha=1.0)
    centroid = shape.centroid
    assert centroid.shape == (2,)
    assert np.all(np.isfinite(centroid))


def test_degenerate_shape_returns_nan_centroid():
    points = np.array([[0, 0], [0, 90]])
    shape = SphericalAlphaShape(points)
    centroid = shape.centroid
    assert np.isnan(centroid).all()


def test_get_boundary_faces_valid_format():
    points = np.array([
        [0, 0],
        [0, 90],
        [45, 45],
        [30, 10],
        [15, 60]
    ])
    shape = SphericalAlphaShape(points, alpha=1.0)
    faces = shape._get_boundary_faces()
    for f in faces:
        assert isinstance(f, tuple)
        assert len(f) == 2  # spherical edges are between 2 points
