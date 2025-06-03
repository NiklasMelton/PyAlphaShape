"""
Spherical Delaunay triangulation
================================

This class constructs a Delaunay triangulation on the unit sphere from
latitude‑longitude data.  Two complementary strategies are available:

* **Hemispheric method** – if all points lie on one hemisphere, points are
  projected by the gnomonic map to a 2‑D tangent plane and triangulated with
  SciPy’s planar Delaunay routine.
* **Global method** – otherwise a 3‑D convex hull is built and its surface
  facets are interpreted as spherical Delaunay triangles.  The equivalence
  between convex‑hull facets of points on the sphere and their Delaunay
  triangulation is detailed by Caroli et al.,
  *Robust and Efficient Delaunay Triangulations of Points on or Close to a
  Sphere*, Research Report RR‑7004, 2009.

You can force either route with the ``assume_hemispheric`` flag; by default the
constructor chooses automatically.

Public API
----------
- ``simplices`` (property) or ``get_triangles`` — triangle indices.
- ``get_triangle_coords`` — triangle vertices in (lat, lon) degrees.

Example
-------
```python
coords = np.array([
    [37.8, -122.4],   # San Francisco
    [34.0, -118.2],   # Los Angeles
    [40.7,  -74.0],   # New York
    [41.9,   12.5],   # Rome
])

tri = SphericalDelaunay(coords)
print(tri)  # <SphericalDelaunay(method='global', n_triangles=...)>
indices = tri.simplices
```

All triangles are outward‑facing and consistently wound (right‑hand rule),
ready for downstream tasks such as spherical alpha‑shape construction.
"""


import numpy as np
from scipy.spatial import ConvexHull, Delaunay
from typing import Optional
from pyalphashape.sphere_utils import (
    latlon_to_unit_vectors,
    unit_vector, gnomonic_projection,
    spherical_triangle_area
)


class SphericalDelaunay:
    """
    Computes a spherical Delaunay triangulation from latitude–longitude points.

    Automatically selects between a hemispheric gnomonic projection method and a
    global convex hull approach to construct valid triangulations on the unit sphere.

    Parameters
    ----------
    latlon_coords : np.ndarray
        An (N, 2) array of latitude and longitude values in degrees.
    assume_hemispheric : Optional[bool], optional
        - If True: force hemispheric (gnomonic) method.
        - If False: force convex hull method.
        - If None (default): decide automatically via dot-product with centroid vector.
    """

    def __init__(
        self,
        latlon_coords: np.ndarray,
        assume_hemispheric: Optional[bool] = None
    ):
        """
        Accepts latlon_coords: np.ndarray of shape (n_points, 2) in degrees.
        Automatically detects whether a gnomonic (hemisphere) or convex hull method should be used.
        If assume_hemispheric is:
            - True: force gnomonic method
            - False: skip hemisphere check and use convex hull
            - None: use mean vector and dot product test
        """
        self.latlon = latlon_coords
        self.points_xyz = latlon_to_unit_vectors(latlon_coords)

        # Try to use hemisphere method
        try_hemisphere = assume_hemispheric if assume_hemispheric is not None else True

        if try_hemisphere:
            # Compute geometric center and check dot products
            center_vec = unit_vector(np.mean(self.points_xyz, axis=0))
            if assume_hemispheric is None:
                dots = self.points_xyz @ center_vec
                try_hemisphere = np.all(dots >= -1e-8)

            if try_hemisphere:
                self.method = 'hemisphere'
                self.center_vec = center_vec
                self.projected_2d = gnomonic_projection(self.points_xyz, center_vec)
                self.delaunay = Delaunay(self.projected_2d)
                self.triangles = self._ensure_consistent_winding(self.delaunay.simplices)
                return

        # Fallback to global method
        self.method = 'global'
        self.center_vec = None
        self.triangles = self._compute_spherical_delaunay()

    def _ensure_consistent_winding(self, triangles: np.ndarray) -> np.ndarray:
        """
        Ensure triangle vertex order produces outward-facing normals from the sphere origin.

        Parameters
        ----------
        triangles : np.ndarray
            Array of triangle indices.

        Returns
        -------
        np.ndarray
            Array of triangles with consistent vertex winding (right-hand rule).
        """

        corrected = []
        for tri in triangles:
            i, j, k = tri
            A, B, C = self.points_xyz[i], self.points_xyz[j], self.points_xyz[k]

            # Compute triangle normal (un-normalized)
            normal = np.cross(B - A, C - A)

            # Check if normal points outward from origin (dot with centroid)
            centroid = (A + B + C) / 3.0
            if np.dot(normal, centroid) < 0:
                # Flip triangle to ensure consistent outward-facing normal
                corrected.append([i, k, j])
            else:
                corrected.append([i, j, k])
        return np.array(corrected, dtype=int)

    def _compute_spherical_delaunay(self) -> np.ndarray:
        """
        Construct a spherical Delaunay triangulation using a 3D convex hull.

        The largest triangle (which covers the convex cap) is excluded to maintain
        consistency with surface triangulations.

        Returns
        -------
        np.ndarray
            Array of triangle indices representing the surface triangulation.
        """

        hull = ConvexHull(self.points_xyz)
        triangles = hull.simplices

        max_area = -1.0
        max_index = -1
        for i, simplex in enumerate(triangles):
            a, b, c = self.points_xyz[simplex]
            area = spherical_triangle_area(a, b, c)
            if area > max_area:
                max_area = area
                max_index = i

        valid_triangles = np.delete(triangles, max_index, axis=0)
        return self._ensure_consistent_winding(valid_triangles)

    @property
    def simplices(self) -> np.ndarray:
        """
        Return the triangle indices of the spherical Delaunay triangulation.

        Returns
        -------
        np.ndarray
            Array of shape (M, 3) containing indices of triangle corners.
        """

        return self.triangles

    def get_triangles(self) -> np.ndarray:
        """
        Alias for `self.triangles`. Mimics `scipy.spatial.Delaunay.simplices`.

        Returns
        -------
        np.ndarray
            Triangle indices of the triangulation.
        """

        return self.triangles

    def get_triangle_coords(self) -> np.ndarray:
        """
        Return triangle vertex coordinates in (latitude, longitude) degrees.

        Returns
        -------
        np.ndarray
            Array of shape (M, 3, 2), where M is the number of triangles and
            each triangle contains 3 (lat, lon) pairs.
        """

        return self.latlon[self.triangles]

    def __repr__(self) -> str:
        """
        Return a string representation of the triangulation object.

        Returns
        -------
        str
            String indicating the triangulation method and number of triangles.
        """

        return f"<SphericalDelaunay(method='{self.method}', n_triangles={len(self.triangles)})>"

