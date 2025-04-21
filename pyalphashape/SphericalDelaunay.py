# Re-import after environment reset
import numpy as np
from scipy.spatial import ConvexHull, Delaunay
from typing import Optional
from sphere_utils import (
    latlon_to_unit_vectors,
    unit_vector, gnomonic_projection,
    spherical_triangle_area
)



class SphericalDelaunay:
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
        Ensure all triangles are consistently wound so their normal vector points outward
        from the origin. This is important for geodesic containment and triangle-based tests.
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
        Compute the spherical Delaunay triangulation by computing the 3D convex hull of the unit vectors
        and removing the single largest-area triangle (which corresponds to the cap over the convex hull).
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
    def simplices(self):
        return self.triangles

    def get_triangles(self):
        """Alias for compatibility with scipy.spatial.Delaunay."""
        return self.triangles

    def get_triangle_coords(self):
        """Returns Nx3x2 array of lat/lon coordinates of triangle corners."""
        return self.latlon[self.triangles]

    def __repr__(self):
        return f"<SphericalDelaunay(method='{self.method}', n_triangles={len(self.triangles)})>"

