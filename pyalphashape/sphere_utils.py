import numpy as np
def normalize(v: np.ndarray) -> np.ndarray:
    return v / np.linalg.norm(v)
def latlon_to_unit_vectors(latlon):
    """Convert lat/lon in degrees to 3D Cartesian coordinates on unit sphere."""
    lat = np.radians(latlon[:, 0])
    lon = np.radians(latlon[:, 1])
    x = np.cos(lat) * np.cos(lon)
    y = np.cos(lat) * np.sin(lon)
    z = np.sin(lat)
    return np.stack([x, y, z], axis=1)

def spherical_incircle_check(triangle, test_point, epsilon=1e-6):
    """
    Check if test_point lies inside the spherical circumcircle of the triangle.
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

    return dist < radius - epsilon

def unit_vector(v):
    return v / np.linalg.norm(v)

def gnomonic_projection(points_xyz, center_vec):
    """
    Project points from unit sphere to 2D plane using gnomonic projection centered at center_vec.
    All inputs must be unit vectors.
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

def spherical_triangle_area(a, b, c):
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

def arc_distance(P, A, B):
    A = A / np.linalg.norm(A)
    B = B / np.linalg.norm(B)
    P = P / np.linalg.norm(P)

    n = np.cross(A, B)
    n_norm = np.linalg.norm(n)
    if n_norm < 1e-10:
        return min(np.arccos(np.clip(np.dot(P, A), -1.0, 1.0)),
                   np.arccos(np.clip(np.dot(P, B), -1.0, 1.0)))
    n /= n_norm

    perp = np.cross(n, np.cross(P, n))
    projected = perp / np.linalg.norm(perp)

    angle_total = np.arccos(np.clip(np.dot(A, B), -1.0, 1.0))
    angle_ap = np.arccos(np.clip(np.dot(A, projected), -1.0, 1.0))
    angle_bp = np.arccos(np.clip(np.dot(B, projected), -1.0, 1.0))

    if np.abs((angle_ap + angle_bp) - angle_total) < 1e-8:
        return np.arccos(np.clip(np.dot(P, projected), -1.0, 1.0))
    else:
        return min(np.arccos(np.clip(np.dot(P, A), -1.0, 1.0)),
                   np.arccos(np.clip(np.dot(P, B), -1.0, 1.0)))


def spherical_circumradius(points: np.ndarray, tol: float = 1e-10) -> float:
    """
    Compute the spherical circumradius (in radians) of a triangle formed by 3 points on the unit sphere.

    Args:
        points: (3, 3) array representing 3 points on the surface of a unit sphere.
        tol: Tolerance to detect degeneracy.

    Returns:
        Angular radius in radians of the spherical circumcircle.
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
        # Degenerate triangle (e.g., colinear points)
        return np.pi  # maximal uncertainty — half a great circle

    center /= norm_center  # normalize to lie on the unit sphere

    # Angular radius is arc distance from center to any vertex (say A)
    radius = np.arccos(np.clip(np.dot(center, A), -1.0, 1.0))

    return radius