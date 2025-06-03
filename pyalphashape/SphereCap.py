"""
Spherical cap geometry
======================

This module provides algorithms for working with spherical caps: the smallest
(or largest empty) circular regions on the unit sphere that satisfy a given
constraint.  Typical applications include coverage analysis, outlier detection,
and bounding‑volume hierarchy construction for data on the sphere.

Implemented tools
-----------------
- spherical_distance                 – great‑circle distance between two unit
  vectors.
- cap_through                        – minimal cap passing through up to three
  support points.
- is_in_cap                          – test whether a point lies inside a
  specified cap.
- welzl_spherical_cap                – Welzl’s recursive algorithm that yields
  the smallest enclosing cap of a point set.
- minimum_enclosing_spherical_cap    – convenience wrapper that shuffles the
  input, then calls the Welzl routine.
- maximum_empty_spherical_cap        – optimisation routine that finds the
  largest cap containing none of the input points.

Example
-------
```python
center, rad = minimum_enclosing_spherical_cap(points_xyz)
empty_c, empty_r = maximum_empty_spherical_cap(points_xyz)
```

All functions assume inputs are three‑dimensional unit vectors; use
normalize from pyalphashape.sphere_utils to ensure this when required.
"""

import numpy as np
from numpy.linalg import norm
from typing import List, Tuple, Optional
from scipy.optimize import minimize
from pyalphashape.sphere_utils import normalize

def spherical_distance(u: np.ndarray, v: np.ndarray) -> float:
    """
    Compute the great-arc (angular) distance between two unit vectors on the unit
    sphere.

    Parameters
    ----------
    u : np.ndarray
        A 3D unit vector.
    v : np.ndarray
        A 3D unit vector.

    Returns
    -------
    float
        The angular distance in radians between u and v.
    """

    return np.arccos(np.clip(np.dot(u, v), -1.0, 1.0))


def cap_through(points: List[np.ndarray]) -> Tuple[np.ndarray, float]:
    """
    Compute the smallest spherical cap that passes through 0 to 3 given points.

    Parameters
    ----------
    points : List[np.ndarray]
        A list of 0 to 3 unit vectors on the sphere.

    Returns
    -------
    Tuple[np.ndarray, float]
        A tuple containing:
        - center (np.ndarray): unit vector pointing to the center of the cap.
        - radius (float): angular radius of the cap in radians.

    Raises
    ------
    ValueError
        If more than 3 points are given.
    """

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


def is_in_cap(
        p: np.ndarray,
        center: np.ndarray,
        radius: float,
        eps: float = 1e-10
) -> bool:
    """
    Check if a point lies within (or on) a spherical cap.

    Parameters
    ----------
    p : np.ndarray
        The point to check (unit vector).
    center : np.ndarray
        The cap center (unit vector).
    radius : float
        The angular radius of the cap (in radians).
    eps : float, optional
        Tolerance to account for numerical precision. Default is 1e-10.

    Returns
    -------
    bool
        True if the point lies within the cap, False otherwise.
    """

    return spherical_distance(p, center) <= radius + eps


def welzl_spherical_cap(
        points: List[np.ndarray],
        R: List[np.ndarray] = []
) -> Tuple[np.ndarray, float]:
    """
    Recursive implementation of Welzl’s algorithm for smallest enclosing spherical cap.

    Parameters
    ----------
    points : List[np.ndarray]
        A list of unit vectors to enclose.
    R : List[np.ndarray], optional
        A list of up to 3 support points defining the current cap (default empty).

    Returns
    -------
    Tuple[np.ndarray, float]
        - center (np.ndarray): unit vector pointing to cap center.
        - radius (float): angular radius in radians.
    """

    if len(points) == 0 or len(R) == 3:
        return cap_through(R)

    p = points[-1]
    rest = points[:-1]
    center, radius = welzl_spherical_cap(rest, R)

    if is_in_cap(p, center, radius):
        return center, radius
    else:
        return welzl_spherical_cap(rest, R + [p])


def minimum_enclosing_spherical_cap(
        points_xyz: np.ndarray,
        seed: Optional[int] = None
) -> Tuple[np.ndarray, float]:
    """
    Compute the smallest enclosing spherical cap for a set of 3D unit vectors.

    Parameters
    ----------
    points_xyz : np.ndarray
        An (N, 3) array of unit vectors on the sphere.
    seed : Optional[int], optional
        Random seed for reproducibility of the point shuffle. Default is None.

    Returns
    -------
    Tuple[np.ndarray, float]
        - center (np.ndarray): unit vector pointing to cap center.
        - radius (float): angular radius in radians.
    """

    if seed is not None:
        np.random.seed(seed)
    points = points_xyz.copy()
    np.random.shuffle(points)
    return welzl_spherical_cap(list(points))


def maximum_empty_spherical_cap(
        points_xyz: np.ndarray,
        n_restarts: int = 10
) -> Tuple[np.ndarray, float]:
    """
    Find the largest spherical cap that does not contain any input point.

    This is done by maximizing the minimum angular distance to all input points
    on the unit sphere. The result is the center and radius of the largest
    empty spherical cap.

    Parameters
    ----------
    points_xyz : np.ndarray
        An (N, 3) array of unit vectors on the sphere.
    n_restarts : int, optional
        Number of random restarts for optimization. Default is 10.

    Returns
    -------
    Tuple[np.ndarray, float]
        - center (np.ndarray): unit vector pointing to cap center.
        - radius (float): angular radius in radians of the largest empty cap.
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

