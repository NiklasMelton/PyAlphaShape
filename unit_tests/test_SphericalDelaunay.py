import numpy as np
import pytest
from pyalphashape.SphericalDelaunay import SphericalDelaunay


def make_test_points_hemisphere():
    """Points that lie in the same hemisphere."""
    return np.array([
        [0, 0],       # Equator
        [0, 90],      # Equator
        [45, 45],     # Northern Hemisphere
        [30, 10],     # Northern Hemisphere
        [15, 60],     # Northern Hemisphere
    ])


def make_test_points_global():
    """Points distributed over a large portion of the sphere."""
    return np.array([
        [0, 0],
        [90, 0],
        [-90, 0],
        [0, 120],
        [0, -120],
    ])


def test_hemisphere_method_auto():
    points = make_test_points_hemisphere()
    sd = SphericalDelaunay(points)
    assert sd.method == "hemisphere"
    assert sd.simplices.shape[1] == 3
    assert sd.projected_2d.shape[1] == 2


def test_hemisphere_method_forced():
    points = make_test_points_hemisphere()
    sd = SphericalDelaunay(points, assume_hemispheric=True)
    assert sd.method == "hemisphere"
    assert hasattr(sd, "delaunay")
    assert sd.simplices.shape[1] == 3


def test_global_method_forced():
    points = make_test_points_global()
    sd = SphericalDelaunay(points, assume_hemispheric=False)
    assert sd.method == "global"
    assert sd.center_vec is None
    assert sd.simplices.shape[1] == 3


def test_get_triangle_coords_shape():
    points = make_test_points_hemisphere()
    sd = SphericalDelaunay(points)
    coords = sd.get_triangle_coords()
    assert coords.ndim == 3
    assert coords.shape[1:] == (3, 2)


def test_repr_contains_method_and_count():
    points = make_test_points_hemisphere()
    sd = SphericalDelaunay(points)
    repr_str = repr(sd)
    assert f"method='{sd.method}'" in repr_str
    assert f"n_triangles={len(sd.simplices)}" in repr_str


def test_triangle_winding_consistency():
    """Ensure triangle normals point outward (dot(normal, centroid) > 0)."""
    points = make_test_points_global()
    sd = SphericalDelaunay(points, assume_hemispheric=False)
    for tri in sd.simplices:
        a, b, c = sd.points_xyz[tri]
        normal = np.cross(b - a, c - a)
        centroid = (a + b + c) / 3.0
        assert np.dot(normal, centroid) > 0


def test_get_triangles_matches_simplices():
    points = make_test_points_hemisphere()
    sd = SphericalDelaunay(points)
    assert np.array_equal(sd.get_triangles(), sd.simplices)
