"""
Alpha Shape module
==================

An *α‑shape* is a family of “concave hulls” that generalise the convex hull:
for a point cloud in **d** dimensions, decreasing the parameter α gradually
shrinks the hull so that cavities and concavities appear once their radius
exceeds 1/α.  The resulting simplicial complex captures the true boundary of
scattered data far more faithfully than a convex hull, making it useful for
shape reconstruction, outlier removal, mesh generation, cluster delineation and
morphological statistics.

This module provides:

* low‑level helpers ``circumcenter`` and ``circumradius`` for arbitrary‑dimensional
  simplices;
* an ``alphasimplices`` generator that streams Delaunay simplices with their
  circumradii; and
* the high‑level :class:`~pyalphashape.AlphaShape` class, which builds an
  α‑shape in *any* dimension, supports strict/relaxed connectivity rules,
  optional hole‑patching, incremental point insertion, inside/point‑to‑surface
  queries, centroid computation, and access to perimeter vertices, edges and
  faces.

In practice you construct an α‑shape from an ``(N,d)`` array of points,
tune α until the boundary is as tight as required, and then use the resulting
object for geometric queries or for exporting a watertight simplicial mesh.
"""

import itertools
import logging
import math
from collections import defaultdict
from scipy.spatial import Delaunay
import numpy as np
from typing import Tuple, Set, List, Literal, Optional
from pyalphashape.GraphClosure import GraphClosureTracker


def circumcenter(points: np.ndarray) -> np.ndarray:
    """
    Compute the circumcenter of a simplex in arbitrary dimensions.

    Parameters
    ----------
    points : np.ndarray
        An (N, K) array of coordinates defining an (N-1)-simplex in K-dimensional space.
        Must satisfy 1 <= N <= K and K >= 1.

    Returns
    -------
    np.ndarray
        The barycentric coordinates of the circumcenter of the simplex.
    """

    n, _ = points.shape

    # build the (d+1) × (d+1) system with plain ndarrays
    A = np.block([
        [2 * points @ points.T, np.ones((n, 1))],
        [np.ones((1, n)), np.zeros((1, 1))]
    ])
    b = np.concatenate([np.sum(points * points, axis=1), np.array([1.0])])

    return np.linalg.solve(A, b)[:-1]


def circumradius(points: np.ndarray) -> float:
    """
    Compute the circumradius of a simplex in arbitrary dimensions.

    Parameters
    ----------
    points : np.ndarray
        An (N, K) array of coordinates defining an (N-1)-simplex in K-dimensional space.

    Returns
    -------
    float
        The circumradius of the simplex.
    """

    return np.linalg.norm(points[0, :] - np.dot(circumcenter(points), points))



def alphasimplices(points: np.ndarray) -> Tuple[np.ndarray, float, np.ndarray]:
    """
    Generate all simplices in the Delaunay triangulation along with their circumradii.

    Parameters
    ----------
    points : np.ndarray
        An (N, K) array of points to triangulate.

    Yields
    ------
    Tuple[np.ndarray, float, np.ndarray]
        A tuple containing:
        - The simplex as an array of indices,
        - Its circumradius,
        - The coordinates of the simplex vertices.
    """

    coords = np.asarray(points)
    tri = Delaunay(coords, qhull_options="Qz")

    for simplex in tri.simplices:
        simplex_points = coords[simplex]
        try:
            yield simplex, circumradius(simplex_points), simplex_points
        except np.linalg.LinAlgError:
            logging.warn('Singular matrix. Likely caused by all points '
                         'lying in an N-1 space.')


class AlphaShape:
    """
    Compute the α-shape (concave hull) of a point cloud in arbitrary dimensions.

    Parameters
    ----------
    points : np.ndarray
        An (N, d) array of points.
    alpha : float, optional
        The α parameter controlling the "tightness" of the shape. Default is 0.
    connectivity : {"strict", "relaxed"}, optional
        Connectivity rule for filtering simplices. Default is "strict".
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

        self.points = np.asarray(points, dtype=float)

        self.simplices: Set[Tuple[int, ...]] = set()
        self.perimeter_edges: List[Tuple[np.ndarray, np.ndarray]] = []
        self.perimeter_points: np.ndarray | None = None
        self.GCT = GraphClosureTracker(len(points))

        # build once
        self._build_batch()

    @property
    def vertices(self) -> Optional[np.ndarray]:
        """
        Get the perimeter vertices of the alpha shape.

        Returns
        -------
        np.ndarray or None
            Array of perimeter points, or None if not computed.
        """

        return self.perimeter_points

    def contains_point(self, pt: np.ndarray, tol: float = 1e-8) -> bool:
        """
        Check whether a given point lies inside or on the alpha shape.

        Parameters
        ----------
        pt : np.ndarray
            A point of shape (d,) to test for inclusion.
        tol : float
            Tolerance used for numerical comparisons.

        Returns
        -------
        bool
            True if the point lies inside or on the alpha shape; False otherwise.
        """

        if len(self.perimeter_points) == 0:
            return False

        # 1. Close to any perimeter vertex?
        if np.any(np.linalg.norm(self.perimeter_points - pt, axis=1) < tol):
            return True

        if len(self.simplices) == 0:
            return False

        # 2. On any perimeter edge?
        for a, b in self.perimeter_edges:
            ab = b - a
            ap = pt - a
            proj_len = np.dot(ap, ab) / np.dot(ab, ab)
            if -tol <= proj_len <= 1.0 + tol:
                closest = a + proj_len * ab
                if np.linalg.norm(closest - pt) < tol:
                    return True

        # 3. Inside a simplex?  (barycentric‑coordinate test)
        for s in self.simplices:
            verts = self.points[list(s)]
            try:
                A = np.vstack([verts.T, np.ones(len(verts))])  # (d+1, d+1)
                b = np.append(pt, 1.0)

                # Full‑rank simplex → solve directly; otherwise fall back to least‑squares
                if A.shape[0] == A.shape[1]:
                    bary = np.linalg.solve(A, b)
                else:
                    bary, *_ = np.linalg.lstsq(A, b, rcond=None)

                if np.all(bary >= -tol):
                    return True
            except np.linalg.LinAlgError:
                continue

        return False

    def add_points(self, new_pts: np.ndarray, perimeter_only: bool = False) -> None:
        """
        Add new points to the alpha shape (batch rebuild).

        Parameters
        ----------
        new_pts : np.ndarray
            A (N, d) array of new points to add. The alpha shape is rebuilt.
        perimeter_only: bool
            If True, only pass perimeter points to new shape. Otherwise, pass all points
        """

        if perimeter_only:
            pts = np.vstack([self.points, new_pts])
        else:
            pts = np.vstack([self.perimeter_points, new_pts])
        self.__init__(pts, alpha=self.alpha, connectivity=self.connectivity,
                      ensure_closure=self.ensure_closure)

    def _get_boundary_faces(self) -> Set[Tuple[int, ...]]:
        """
        Identify and return the boundary (d-1)-faces of the alpha shape.

        Returns
        -------
        Set[Tuple[int, ...]]
            A set of index tuples representing the boundary faces.
        """

        if hasattr(self, "_boundary_faces"):
            return self._boundary_faces

        dim = self._dim
        faces: Set[Tuple[int, ...]] = set()
        for s in self.simplices:
            for f in itertools.combinations(s, dim):
                f = tuple(sorted(f))
                if f in faces:
                    faces.remove(f)
                else:
                    faces.add(f)
        # cache
        self._boundary_faces = faces
        return faces

    def distance_to_surface(self, point: np.ndarray, tol: float = 1e-9) -> float:
        """
        Compute the shortest Euclidean distance from a point to the alpha shape surface.

        Parameters
        ----------
        point : np.ndarray
            A point of shape (d,) in the same ambient space as the alpha shape.
        tol : float, optional
            Tolerance for barycentric coordinate test. Default is 1e-9.

        Returns
        -------
        float
            Distance from the point to the alpha shape surface. Returns 0 if inside
            or on surface.
        """

        p = np.asarray(point, dtype=float)
        if p.shape[-1] != self._dim:
            raise ValueError("point dimensionality mismatch")

        # 1. inside / on‑surface test
        if self.contains_point(p):
            return 0.0

        # 2. gather boundary faces and vertices
        faces = self._get_boundary_faces()
        if not faces:
            # degenerate case (e.g. only 1–2 input points)
            # fall back to nearest perimeter vertex
            return np.min(np.linalg.norm(self.perimeter_points - p, axis=1))

        dists = []

        for f in faces:
            verts = self.points[list(f)]  # shape (d, d)
            base = verts[0]
            A = verts[1:] - base  # (d‑1, d)

            # orthogonal projection of p onto the face’s affine span
            # Solve A x = (p - base)  →  least‑squares because A is tall
            x_hat, *_ = np.linalg.lstsq(A.T, (p - base), rcond=None)
            proj = base + A.T @ x_hat

            # barycentric coordinates to test if proj is inside the simplex
            # coords = [1 - sum(x_hat), *x_hat]
            bary = np.concatenate(([1.0 - x_hat.sum()], x_hat))
            if np.all(bary >= -tol):  # inside (or on) the face
                dists.append(np.linalg.norm(p - proj))
            else:
                # outside → distance to nearest vertex of this face
                dists.extend(np.linalg.norm(verts - p, axis=1))

        return float(min(dists))

    def _build_batch(self) -> None:
        """
        Construct the alpha shape using Delaunay triangulation and filtering by alpha.

        This method is automatically called upon initialization.
        """

        dim, pts = self._dim, self.points
        n = len(pts)
        if n < dim + 1:
            self.perimeter_points = pts
            return

        r_filter = np.inf if self.alpha <= 0 else 1.0 / self.alpha
        tri = Delaunay(pts, qhull_options="Qz")

        # ---------- 1.  main sweep ---------------------------------------
        simplices = []
        for s in tri.simplices:
            r = circumradius(pts[s])
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

            # Keep only triangles in the main component that share an edge with another
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
            if getattr(self, "ensure_closure", False):
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
        perimeter_edges_idx = [e for e, count in edge_counts.items() if count == 1]
        perim_idx = set(i for e in perimeter_edges_idx for i in e)

        self.perimeter_points = pts[list(sorted(perim_idx))]
        self.perimeter_edges = [(pts[i], pts[j]) for i, j in perimeter_edges_idx]

    @property
    def is_empty(self) -> bool:
        """
        Check whether the alpha shape is empty (i.e., no perimeter points).

        Returns
        -------
        bool
            True if empty; False otherwise.
        """

        return len(self.perimeter_points) == 0

    @property
    def triangle_faces(self) -> List[np.ndarray]:
        """
        Get the triangle faces (simplices) that make up the alpha shape.

        Returns
        -------
        List[np.ndarray]
            List of arrays containing vertex coordinates of each simplex.
        """

        return [self.points[list(s)] for s in self.simplices]


    @property
    def centroid(self) -> np.ndarray:
        """
        Compute the hyper-volumetric centroid of the Euclidean alpha shape
        using direct determinant-based simplex volume.

        Returns
        -------
        np.ndarray
            A (d,) array representing the centroid in Euclidean space.
        """
        if len(self.simplices) == 0:
            return np.full(self._dim, np.nan)

        d = self._dim
        total_volume = 0.0
        weighted_sum = np.zeros(d)

        for s in self.simplices:
            verts = self.points[list(s)]
            if len(verts) != d + 1:
                continue  # not a full-dimensional simplex

            # Form matrix of edge vectors
            mat = verts[1:] - verts[0]
            try:
                vol = np.abs(np.linalg.det(mat)) / math.factorial(d)
            except np.linalg.LinAlgError:
                continue

            centroid = np.mean(verts, axis=0)
            total_volume += vol
            weighted_sum += vol * centroid

        if total_volume == 0.0:
            return np.mean(self.points, axis=0)

        return weighted_sum / total_volume

