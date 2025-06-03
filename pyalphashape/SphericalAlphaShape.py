"""
Spherical alpha shapes
======================

This module implements :class:`SphericalAlphaShape`, an extension of the
Euclidean α‑shape concept to data that lie on the surface of a unit sphere
(e.g. geographic coordinates).  By adjusting the parameter α the boundary
tightens or relaxes, capturing cavities and concavities that a convex hull
misses while always respecting spherical geometry.

Core features
-------------
- Converts input latitude–longitude pairs to 3‑D unit vectors and back.
- Uses a spherical Delaunay triangulation and circumradius filter to select
  simplices that satisfy the α criterion.
- Supports *strict* and *relaxed* connectivity modes, plus optional hole
  patching to guarantee a watertight surface.
- Provides geometric queries:
  * `contains_point` – point‑in‑shape test.
  * `distance_to_surface` – shortest arc distance to the boundary.
  * `triangle_faces`, `triangle_faces_latlon` – access to all retained
    triangles in either coordinate system.
  * `centroid` – area‑weighted centre of the shape.
- Accepts incremental point insertion via `add_points`.

Example
-------
```python
import numpy as np
from pyalphashape.SphericalAlphaShape import SphericalAlphaShape

# Sample latitude–longitude data (degrees)
coords = np.array([
    [ 40.0, -105.0],
    [ 41.0, -104.0],
    [ 39.5, -103.5],
    [ 40.5, -106.0],
])

sph_shape = SphericalAlphaShape(coords, alpha=2.0)
inside = sph_shape.contains_point(np.array([40.3, -104.7]))
dist   = sph_shape.distance_to_surface(np.array([42.0, -107.0]))
```

All computations use plain NumPy; no external geometry libraries are required.
"""

import numpy as np
import itertools
from collections import defaultdict
from typing import Literal, Set, Tuple, List, Optional
from pyalphashape.SphericalDelaunay import SphericalDelaunay
from pyalphashape.sphere_utils import (
    latlon_to_unit_vectors,
    unit_vectors_to_latlon,
    arc_distance,
    spherical_circumradius,
    spherical_triangle_area
)
from pyalphashape.GraphClosure import GraphClosureTracker


class SphericalAlphaShape:
    """
    Compute the α-shape (concave hull) of points defined on the surface of a unit sphere.

    Parameters
    ----------
    points : np.ndarray
        An (N, 2) array of latitude and longitude coordinates in degrees.
    alpha : float, optional
        The α parameter controlling shape detail. Smaller values yield tighter shapes.
    connectivity : {"strict", "relaxed"}, optional
        Rule for keeping connected components during filtering.
    ensure_closure : bool, default True
        If True, triangles that are otherwise too large but are fully enclosed by
        accepted triangles will be included. This prevents holes
    """

    def __init__(self,
                 points: np.ndarray,
                 alpha: float = 0.,
                 connectivity: Literal["strict", "relaxed"] = "strict",
                 ensure_closure: bool = True
    ):
        self._dim = points.shape[1]
        if self._dim < 2:
            raise ValueError("dimension must be ≥ 2")

        self.alpha = float(alpha)
        if connectivity not in {"strict", "relaxed"}:
            raise ValueError("connectivity must be 'strict' or 'relaxed'")
        self.connectivity = connectivity
        self.ensure_closure = ensure_closure

        self.points_latlon = np.asarray(points, dtype=float)
        self.points = latlon_to_unit_vectors(self.points_latlon)

        self.simplices: Set[Tuple[int, ...]] = set()
        self.perimeter_edges: List[Tuple[np.ndarray, np.ndarray]] = []
        self.perimeter_points: np.ndarray | None = None
        self.perimeter_points_latlon: np.ndarray | None = None
        self.GCT = GraphClosureTracker(len(points))

        # build once
        self._build_batch()

    @property
    def vertices(self) -> Optional[np.ndarray]:
        """
        Return the perimeter vertices of the spherical alpha shape (in 3D unit vector form).

        Returns
        -------
        np.ndarray or None
            The perimeter points of the alpha shape, or None if uninitialized.
        """

        return self.perimeter_points

    def contains_point(self, pt_latlon: np.ndarray, tol: float = 1e-8) -> bool:
        """
        Return ``True`` iff the latitude/longitude point lies inside *or on*
        the spherical α‑shape.

        Parameters
        ----------
        pt_latlon : (2,) array‑like
            [latitude, longitude] **degrees**.
        tol : float, default 1e‑8
            Angular tolerance (radians) for vertex proximity and half‑space test.
        """
        # ── 0. quick outs ──────────────────────────────────────────────────
        if len(self.perimeter_points) == 0:
            return False

        P = latlon_to_unit_vectors(pt_latlon[None, :])[0]  # unit vector

        # close to a perimeter vertex?
        if np.any(
                np.arccos(np.clip(self.perimeter_points @ P, -1.0, 1.0)) < tol
        ):
            return True

        # no faces yet  → no enclosed area
        if len(self.simplices) == 0:
            return False

        # ── 1. build (or fetch) oriented edge normals ─────────────────────
        # We cache them so subsequent calls are O(#edges) only.
        if not hasattr(self, "_edge_normals"):
            # (a) choose an interior reference vector
            centroid_latlon = getattr(self, "centroid", np.array([np.nan, np.nan]))
            if np.isnan(centroid_latlon).any():
                inside_vec = self.perimeter_points.mean(axis=0)
                inside_vec /= np.linalg.norm(inside_vec)
            else:
                inside_vec = latlon_to_unit_vectors(centroid_latlon[None])[0]

            normals = []
            for A, B in self.perimeter_edges:
                n = np.cross(A, B)  # normal to great‑circle AB
                if np.linalg.norm(n) < 1e-12:  # degenerate edge
                    continue
                # orient inward so that n·inside_vec  >  0
                if np.dot(n, inside_vec) < 0:
                    n = -n
                normals.append(n / np.linalg.norm(n))

            self._edge_normals = np.vstack(normals)  # shape (E, 3)

        # ── 2. half‑space test for all perimeter edges ─────────────────────
        # Point is inside iff it lies in the "positive" half‑space of **every**
        # edge after orientation.
        if np.all(self._edge_normals @ P >= -tol):
            return True

        return False

    def add_points(self, new_pts: np.ndarray, perimeter_only: bool = False) -> None:
        """
        Add new latitude-longitude points and rebuild the spherical alpha shape.

        Parameters
        ----------
        new_pts : np.ndarray
            An (M, 2) array of new points in degrees [lat, lon].
        perimeter_only: bool
            If True, only pass perimeter points to new shape. Otherwise, pass all points
        """
        if perimeter_only:
            pts = np.vstack([self.perimeter_points_latlon, new_pts])
        else:
            pts = np.vstack([self.points_latlon, new_pts])
        self.__init__(pts, alpha=self.alpha, connectivity=self.connectivity,
                      ensure_closure=self.ensure_closure)

    def _get_boundary_faces(self) -> Set[Tuple[int, ...]]:
        """
        Return the set of (d-1)-faces that form the boundary of the alpha shape.

        Returns
        -------
        Set[Tuple[int, ...]]
            Set of index tuples representing the boundary faces.
        """

        if hasattr(self, "_boundary_faces"):
            return self._boundary_faces

        faces: Set[Tuple[int, ...]] = set()
        for s in self.simplices:
            for f in itertools.combinations(s, 2):
                f = tuple(sorted(f))
                if f in faces:
                    faces.remove(f)
                else:
                    faces.add(f)
        # cache
        self._boundary_faces = faces
        return faces

    def distance_to_surface(self, point: np.ndarray) -> float:
        """
        Compute the angular arc distance from a (lat, lon) point to the alpha shape surface.

        Parameters
        ----------
        point : np.ndarray
            A (2,) array representing a point in [latitude, longitude] degrees.

        Returns
        -------
        float
            The arc distance in radians from the point to the alpha shape surface.
            Returns 0 if the point lies inside or on the surface.
        """

        if point.shape[-1] != 2:
            raise ValueError("Input point must be (lat, lon) in degrees")

        if self.contains_point(point):
            return 0.0

        # Convert (lat, lon) to 3D unit vector
        p = latlon_to_unit_vectors(point[None, :])[0]

        faces = self._get_boundary_faces()
        if not faces:
            # Fallback to nearest perimeter point
            return float(np.min([
                np.arccos(np.clip(np.dot(p, q), -1.0, 1.0))
                for q in self.perimeter_points
            ]))

        # Compute minimum arc distance to all boundary edges
        dists = []
        for f in faces:
            idx = list(f)
            if len(idx) != 2:
                raise ValueError(
                    "Expected boundary face with 2 vertices for spherical surface"
                )
            A, B = self.points[idx[0]], self.points[idx[1]]
            dists.append(arc_distance(p, A, B))

        return float(min(dists))

    def _build_batch(self) -> None:
        """
        Construct the spherical alpha shape by computing Delaunay triangles
        and filtering them by circumradius and connectivity.
        This method is automatically called on initialization.
        """

        pts, pts_latlon = self.points, self.points_latlon
        n = len(pts)
        if n < 3:
            self.perimeter_points = pts
            self.perimeter_points_latlon = pts_latlon
            return

        r_filter = np.inf if self.alpha <= 0 else 1.0 / self.alpha
        try:
            tri = SphericalDelaunay(pts_latlon)
        except Exception as E:
            print(pts_latlon)
            raise E

        # ---------- 1.  main sweep ---------------------------------------
        simplices = []
        for s in tri.simplices:
            r = spherical_circumradius(pts[s])
            simplices.append((tuple(s), r))

        simplices.sort(key=lambda t: t[1])  # radius ascending
        kept = []
        uf = GraphClosureTracker(n)  # temp tracker

        for simp, r in simplices:
            root_set = {uf.find(v) for v in simp}
            keep = (r <= r_filter) or \
                   (self.connectivity == "relaxed" and len(root_set) > 1)
            if not keep:
                continue
            uf.add_fully_connected_subgraph(list(simp))
            kept.append(simp)

        # ---------- 2.  strict‑mode pruning ------------------------------
        if self.connectivity == "strict":
            # Build full graph from all triangles that satisfy the alpha condition
            all_passed = [simp for simp, r in simplices if r <= r_filter]

            # Track edge-connected components
            gct = GraphClosureTracker(n)
            for simp in all_passed:
                gct.add_fully_connected_subgraph(simp)

            # Identify the largest connected component
            comp_sizes = {root: len(nodes) for root, nodes in gct.components.items()}
            main_root = max(comp_sizes, key=comp_sizes.get)
            main_verts = gct.components[main_root]

            # Build edge-to-triangle map
            edge_to_triangles = defaultdict(list)
            for simp in all_passed:
                for edge in itertools.combinations(simp, 2):
                    edge = tuple(sorted(edge))
                    edge_to_triangles[edge].append(simp)


            # Keep only triangles that:
            # (a) have all vertices in the main component, and
            # (b) share at least one edge with another triangle
            kept = []
            for simp in all_passed:
                if not set(simp) <= main_verts:
                    continue
                shares_edge = any(
                    len(edge_to_triangles[tuple(sorted(edge))]) > 1
                    for edge in itertools.combinations(simp, 2)
                )
                if shares_edge:
                    kept.append(simp)

            # ---------- 2.5 patch triangle holes -----------------------------
            if self.ensure_closure:
                # Re-add triangles that were excluded but all their edges are shared
                # (likely fully enclosed and cause small holes)
                existing = set(kept)
                for simp, r in simplices:
                    if simp in existing:
                        continue  # already included
                    if not set(simp) <= main_verts:
                        continue  # not in main component
                    edge_shared = all(
                        len(edge_to_triangles[tuple(sorted(edge))]) > 0
                        for edge in itertools.combinations(simp, 2)
                    )
                    if edge_shared:
                        kept.append(simp)

        # ---------- 3.  rebuild perimeter from *kept* simplices ----------
        self.simplices = set(kept)
        self.GCT = GraphClosureTracker(n)  # final tracker

        edge_counts = defaultdict(int)
        for s in self.simplices:
            self.GCT.add_fully_connected_subgraph(list(s))
            for edge in itertools.combinations(s, 2):  # triangle edges
                edge = tuple(sorted(edge))
                edge_counts[edge] += 1

        # ---------- 4.  store perimeter ----------------------------------
        # Only edges that appear once are on the perimeter
        perimeter_edges_idx = [e for e, count in edge_counts.items() if count == 1]
        perim_idx = set(i for e in perimeter_edges_idx for i in e)

        self.perimeter_points = pts[list(sorted(perim_idx))]
        self.perimeter_points_latlon = pts_latlon[list(sorted(perim_idx))]
        self.perimeter_edges = [(pts[i], pts[j]) for i, j in perimeter_edges_idx]
        self.perimeter_edges_latlon = [(pts_latlon[i], pts_latlon[j]) for i, j in
                                       perimeter_edges_idx]

    @property
    def is_empty(self) -> bool:
        """
        Check whether the alpha shape has any perimeter points.

        Returns
        -------
        bool
            True if no perimeter has been constructed, False otherwise.
        """

        return len(self.perimeter_points) == 0

    @property
    def triangle_faces(self) -> List[np.ndarray]:
        """
        Return all triangles (simplices) of the alpha shape in unit vector coordinates.

        Returns
        -------
        List[np.ndarray]
            List of (3, 3) arrays representing triangle vertices as 3D unit vectors.
        """

        return [self.points[list(s)] for s in self.simplices]

    @property
    def triangle_faces_latlon(self) -> List[np.ndarray]:
        """
        Return all triangles (simplices) of the alpha shape in (latitude, longitude) degrees.

        Returns
        -------
        List[np.ndarray]
            List of (3, 2) arrays representing triangle vertices in (lat, lon).
        """

        return [self.points_latlon[list(s)] for s in self.simplices]

    @property
    def centroid(self) -> np.ndarray:
        """
        Compute the center of area of the spherical alpha shape.

        Returns
        -------
        np.ndarray
            A (2,) array representing the centroid in [latitude, longitude] degrees.
        """
        if len(self.simplices) == 0:
            return np.array([np.nan, np.nan])

        total_area = 0.0
        centroid_vec = np.zeros(3)

        for s in self.simplices:
            A, B, C = self.points[list(s)]
            area = spherical_triangle_area(A, B, C)
            centroid = (A + B + C) / np.linalg.norm(A + B + C)
            centroid_vec += area * centroid
            total_area += area

        if total_area == 0.0 or np.linalg.norm(centroid_vec) == 0:
            return np.array([np.nan, np.nan])

        centroid_vec /= np.linalg.norm(centroid_vec)
        return unit_vectors_to_latlon(centroid_vec[None])[0]


