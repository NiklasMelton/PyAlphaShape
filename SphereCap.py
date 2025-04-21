import numpy as np
from numpy.linalg import norm
from typing import List, Tuple, Optional
from scipy.optimize import minimize

def spherical_distance(u: np.ndarray, v: np.ndarray) -> float:
    """Great-arc distance between unit vectors u and v."""
    return np.arccos(np.clip(np.dot(u, v), -1.0, 1.0))

def normalize(v: np.ndarray) -> np.ndarray:
    return v / norm(v)

def cap_through(points: List[np.ndarray]) -> Tuple[np.ndarray, float]:
    """Return the center and angular radius of the smallest spherical cap through 0–3 points."""
    if len(points) == 0:
        return np.array([1.0, 0.0, 0.0]), 0.0
    elif len(points) == 1:
        return points[0], 0.0
    elif len(points) == 2:
        center = normalize(points[0] + points[1])
        radius = spherical_distance(center, points[0])
        return center, radius
    elif len(points) == 3:
        a, b, c = points
        u = normalize(np.cross(b, c))
        v = normalize(np.cross(c, a))
        w = normalize(np.cross(a, b))
        center = normalize(u + v + w)
        radius = max(spherical_distance(center, p) for p in points)
        return center, radius
    else:
        raise ValueError("Only up to 3 points allowed for support set")

def is_in_cap(p: np.ndarray, center: np.ndarray, radius: float, eps: float = 1e-10) -> bool:
    return spherical_distance(p, center) <= radius + eps

def welzl_spherical_cap(points: List[np.ndarray], R: List[np.ndarray] = []) -> Tuple[np.ndarray, float]:
    if len(points) == 0 or len(R) == 3:
        return cap_through(R)

    p = points[-1]
    rest = points[:-1]
    center, radius = welzl_spherical_cap(rest, R)

    if is_in_cap(p, center, radius):
        return center, radius
    else:
        return welzl_spherical_cap(rest, R + [p])

def minimum_enclosing_spherical_cap(points_xyz: np.ndarray, seed: Optional[int] = None) -> Tuple[np.ndarray, float]:
    """Compute the smallest enclosing spherical cap for points on the unit sphere."""
    if seed is not None:
        np.random.seed(seed)
    points = points_xyz.copy()
    np.random.shuffle(points)
    return welzl_spherical_cap(list(points))


def find_max_empty_spherical_cap(points_xyz: np.ndarray, n_restarts: int = 10) -> Tuple[np.ndarray, float]:
    """
    Finds the direction on the unit sphere that maximizes the minimum angular distance
    to all input points — i.e., the center of the largest spherical cap that excludes all points.

    Uses the center of the minimum enclosing spherical cap over the antipodes as a prime initialization.

    Returns:
        center: np.ndarray (3,) — unit vector pointing to cap center
        radius: float — angular radius (in radians) of the largest empty cap
    """
    def cost(center_flat):
        center = normalize(center_flat)
        return -np.min(np.dot(points_xyz, center))  # maximize minimum dot product (cos(theta))

    # Prime using the center of the min cap of the antipodes
    prime_center, _ = minimum_enclosing_spherical_cap(-points_xyz)

    seeds = [prime_center] + [normalize(np.random.randn(3)) for _ in range(n_restarts - 1)]

    best_center = None
    best_dot = -1.0

    for init in seeds:
        res = minimize(cost, init, method='BFGS')
        candidate = normalize(res.x)
        dot_val = np.min(np.dot(points_xyz, candidate))
        if dot_val > best_dot:
            best_dot = dot_val
            best_center = candidate

    max_radius = np.arccos(np.clip(best_dot, -1.0, 1.0))
    return best_center, max_radius

