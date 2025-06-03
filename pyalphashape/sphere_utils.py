"""
Spherical geometry helpers
==========================

This utility file collects small, self‑contained functions for working with
points on the unit sphere.  They cover common tasks such as converting between
latitude–longitude and 3‑D unit vectors, projecting to a tangent plane,
measuring distances and areas, checking spherical incircles, and computing the
circumradius of a spherical triangle.

Public functions
----------------
* normalize, unit_vector                     – return a vector with length 1.
* latlon_to_unit_vectors, unit_vectors_to_latlon
                                             – convert between geographic
                                               coordinates and Cartesian unit
                                               vectors.
* spherical_triangle_area                    – area of a triangle on the sphere
                                               (steradians).
* spherical_incircle_check                   – test if a point lies within the
                                               circumcircle of a spherical
                                               triangle.
* gnomonic_projection                        – map points to a 2‑D tangent
                                               plane via gnomonic projection.
* arc_distance                               – shortest angular distance from a
                                               point to a great‑circle arc.
* spherical_circumradius                     – angular radius of the
                                               circumcircle of a spherical
                                               triangle.

These helpers are used throughout the alpha‑shape and spherical Delaunay
modules but are written without external dependencies so they can be reused in
other spherical‑geometry projects.
"""


import numpy as np
def normalize(v: np.ndarray) -> np.ndarray:
    """
    Normalize a vector to unit length.

    Parameters
    ----------
    v : np.ndarray
        Input vector.

    Returns
    -------
    np.ndarray
        Unit-length version of the input vector.
    """

    return v / np.linalg.norm(v)
def latlon_to_unit_vectors(latlon: np.ndarray) -> np.ndarray:
    """
    Convert latitude and longitude (in degrees) to 3D unit vectors on the unit sphere.

    Parameters
    ----------
    latlon : np.ndarray
        An (N, 2) array of latitude and longitude in degrees.

    Returns
    -------
    np.ndarray
        An (N, 3) array of unit vectors in 3D Cartesian coordinates.
    """

    lat = np.radians(latlon[:, 0])
    lon = np.radians(latlon[:, 1])
    x = np.cos(lat) * np.cos(lon)
    y = np.cos(lat) * np.sin(lon)
    z = np.sin(lat)
    return np.stack([x, y, z], axis=1)


def unit_vectors_to_latlon(vectors: np.ndarray) -> np.ndarray:
    """
    Convert 3D unit vectors to latitude and longitude in degrees.

    Parameters
    ----------
    vectors : np.ndarray
        An (N, 3) array of unit vectors in 3D Cartesian coordinates.

    Returns
    -------
    np.ndarray
        An (N, 2) array of [latitude, longitude] in degrees.
    """
    x, y, z = vectors[:, 0], vectors[:, 1], vectors[:, 2]
    lat = np.degrees(np.arcsin(np.clip(z, -1.0, 1.0)))
    lon = np.degrees(np.arctan2(y, x))
    return np.stack([lat, lon], axis=1)


def spherical_triangle_area(a, b, c):
    # Angular area of spherical triangle via L'Huilier's formula
    a_, b_, c_ = np.arccos(np.clip(np.dot(b, c), -1, 1)), \
        np.arccos(np.clip(np.dot(c, a), -1, 1)), \
        np.arccos(np.clip(np.dot(a, b), -1, 1))
    s = 0.5 * (a_ + b_ + c_)
    tan_e = np.tan(s / 2) * np.tan((s - a_) / 2) * \
            np.tan((s - b_) / 2) * np.tan((s - c_) / 2)
    E = 4 * np.arctan(np.sqrt(np.abs(tan_e)))
    return E  # in steradians (on unit sphere)

def spherical_incircle_check(
        triangle: np.ndarray,
        test_point: np.ndarray
) -> bool:
    """
    Check if a test point lies inside the spherical circumcircle of a triangle on the unit sphere.

    Parameters
    ----------
    triangle : np.ndarray
        A (3, 3) array of 3D unit vectors representing triangle vertices.
    test_point : np.ndarray
        A 3D unit vector representing the point to test.

    Returns
    -------
    bool
        True if the test point lies within the circumcircle, False otherwise.
    """

    a, b, c = triangle
    a, b, c, d = [v / np.linalg.norm(v) for v in [a, b, c, test_point]]

    # Compute circumcenter as normalized sum of edge plane normals
    ab = np.cross(a, b)
    bc = np.cross(b, c)
    ca = np.cross(c, a)
    n = ab + bc + ca
    if np.linalg.norm(n) < 1e-10:
        return False  # Degenerate triangle
    center = n / np.linalg.norm(n)

    # Angular radius to triangle vertices
    radius = np.arccos(np.clip(np.dot(center, a), -1, 1))
    dist = np.arccos(np.clip(np.dot(center, d), -1, 1))

    return dist < radius

def unit_vector(v: np.ndarray) -> np.ndarray:
    """
    Normalize a vector to unit length (alias for `normalize`).

    Parameters
    ----------
    v : np.ndarray
        Input vector.

    Returns
    -------
    np.ndarray
        Unit-length vector.
    """

    return v / np.linalg.norm(v)

def gnomonic_projection(points_xyz: np.ndarray, center_vec: np.ndarray) -> np.ndarray:
    """
    Project points on the unit sphere to a 2D tangent plane using gnomonic projection.

    Parameters
    ----------
    points_xyz : np.ndarray
        An (N, 3) array of unit vectors to project.
    center_vec : np.ndarray
        The projection center (3D unit vector).

    Returns
    -------
    np.ndarray
        An (N, 2) array of 2D coordinates in the tangent plane.
    """

    # Ensure center_vec is unit length
    center_vec = unit_vector(center_vec)

    # Create a local tangent plane basis at center_vec
    # z-axis is center_vec
    # x-axis is arbitrary perpendicular (e.g., rotate north pole to center_vec)
    north_pole = np.array([0.0, 0.0, 1.0])
    x_axis = unit_vector(np.cross(north_pole, center_vec))
    if np.linalg.norm(x_axis) < 1e-6:  # handle pole case
        x_axis = np.array([1.0, 0.0, 0.0])
    y_axis = np.cross(center_vec, x_axis)

    # Project onto tangent plane
    dots = points_xyz @ center_vec
    proj = points_xyz / dots[:, np.newaxis]  # gnomonic projection
    x = proj @ x_axis
    y = proj @ y_axis
    return np.stack([x, y], axis=1)

def spherical_triangle_area(a: np.ndarray, b: np.ndarray, c: np.ndarray) -> float:
    """
    Compute the area of a spherical triangle on the unit sphere using the spherical excess formula.

    Parameters
    ----------
    a : np.ndarray
        First vertex (unit vector).
    b : np.ndarray
        Second vertex (unit vector).
    c : np.ndarray
        Third vertex (unit vector).

    Returns
    -------
    float
        Spherical area in steradians (radians^2).
    """

    def angle(u, v):
        return np.arccos(np.clip(np.dot(u, v), -1.0, 1.0))

    def tri_angle(a, b, c):
        ab = np.cross(a, b)
        ac = np.cross(a, c)
        return np.arccos(np.clip(np.dot(normalize(ab), normalize(ac)), -1.0, 1.0))

    alpha = tri_angle(a, b, c)
    beta = tri_angle(b, c, a)
    gamma = tri_angle(c, a, b)

    return (alpha + beta + gamma) - np.pi

def arc_distance(
        P: np.ndarray,
        A: np.ndarray,
        B: np.ndarray,
        tol: float = 1e-10
) -> float:
    """
    Compute the shortest angular distance from point P to the arc segment between A and B on the sphere.

    Parameters
    ----------
    P : np.ndarray
        Query point (3D unit vector).
    A : np.ndarray
        Arc start point (3D unit vector).
    B : np.ndarray
        Arc end point (3D unit vector).
    tol : float, optional
        Tolerance for degeneracy checks. Default is 1e-10.
    Returns
    -------
    float
        Angular distance in radians from P to the closest point on arc AB.
    """

    A = A / np.linalg.norm(A)
    B = B / np.linalg.norm(B)
    P = P / np.linalg.norm(P)

    n = np.cross(A, B)
    n_norm = np.linalg.norm(n)
    if n_norm < tol:  # degenerate edge  (A ≈ B)
        return min(np.arccos(np.clip(np.dot(P, A), -1.0, 1.0)),
                   np.arccos(np.clip(np.dot(P, B), -1.0, 1.0)))
    n /= n_norm

    # If P lies on the pole, distance is simply to the nearer endpoint
    if abs(np.dot(P, n)) > 1.0 - tol:
        return min(np.arccos(np.clip(np.dot(P, A), -1.0, 1.0)),
                   np.arccos(np.clip(np.dot(P, B), -1.0, 1.0)))

    perp = np.cross(n, np.cross(P, n))
    perp_norm = np.linalg.norm(perp)
    if perp_norm < tol:  # numerical safety net
        return min(np.arccos(np.clip(np.dot(P, A), -1.0, 1.0)),
                   np.arccos(np.clip(np.dot(P, B), -1.0, 1.0)))

    projected = perp / perp_norm
    angle_total = np.arccos(np.clip(np.dot(A, B), -1.0, 1.0))
    angle_ap = np.arccos(np.clip(np.dot(A, projected), -1.0, 1.0))
    angle_bp = np.arccos(np.clip(np.dot(B, projected), -1.0, 1.0))

    # Check if the projected foot lies between A and B along the great‑circle
    if abs((angle_ap + angle_bp) - angle_total) < 1e-8:
        return np.arccos(np.clip(np.dot(P, projected), -1.0, 1.0))
    else:
        return min(np.arccos(np.clip(np.dot(P, A), -1.0, 1.0)),
                   np.arccos(np.clip(np.dot(P, B), -1.0, 1.0)))


def spherical_circumradius(points: np.ndarray, tol: float = 1e-10) -> float:
    """
    Compute the spherical circumradius of a triangle on the unit sphere.

    Parameters
    ----------
    points : np.ndarray
        A (3, 3) array of unit vectors defining a triangle.
    tol : float, optional
        Tolerance for degeneracy checks. Default is 1e-10.

    Returns
    -------
    float
        Angular radius (in radians) of the triangle’s circumcircle.

    Raises
    ------
    ValueError
        If input does not have shape (3, 3).
    """

    if points.shape != (3, 3):
        raise ValueError("Input must be an array of shape (3, 3) representing 3 points in 3D.")

    A, B, C = points

    # Ensure all points are unit vectors
    assert np.allclose(np.linalg.norm(A), 1.0, atol=tol)
    assert np.allclose(np.linalg.norm(B), 1.0, atol=tol)
    assert np.allclose(np.linalg.norm(C), 1.0, atol=tol)

    # Compute normals to edges (great circles)
    n1 = np.cross(A, B)
    n2 = np.cross(B, C)

    # Normal to the plane containing the triangle's circumcenter
    center = np.cross(n1, n2)
    norm_center = np.linalg.norm(center)

    if norm_center < tol:
        # degeneracy: are all three vertices essentially the same?
        pairwise_sep = [
            np.arccos(np.clip(np.dot(u, v), -1.0, 1.0))
            for u, v in [(A, B), (B, C), (C, A)]
        ]
        if max(pairwise_sep) < tol:  # coincident vertices
            return 0.0
        else:  # colinear but distinct
            return np.pi

    center /= norm_center  # normalize to lie on the unit sphere

    # Angular radius is arc distance from center to any vertex (say A)
    radius = np.arccos(np.clip(np.dot(center, A), -1.0, 1.0))

    return radius