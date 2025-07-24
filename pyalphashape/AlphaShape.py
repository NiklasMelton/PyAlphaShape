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
  facets.

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
from typing import Tuple, Set, List, Literal, Optional, Generator, Union, Dict, Any
from pyalphashape.GraphClosure import GraphClosureTracker


def circumcenter(points: np.ndarray) -> np.ndarray:
    """Compute the circumcenter of a simplex in arbitrary dimensions.

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
    A = np.block(
        [[2 * points @ points.T, np.ones((n, 1))], [np.ones((1, n)), np.zeros((1, 1))]]
    )
    b = np.concatenate([np.sum(points * points, axis=1), np.array([1.0])])

    return np.linalg.solve(A, b)[:-1]


def circumradius(points: np.ndarray) -> float:
    """Compute the circumradius of a simplex in arbitrary dimensions.

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


def alphasimplices(
    points: np.ndarray,
) -> Generator[Tuple[np.ndarray, float, np.ndarray], None, None]:
    """Generate all simplices in the Delaunay triangulation along with their
    circumradii.

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
            logging.warn(
                "Singular matrix. Likely caused by all points " "lying in an N-1 space."
            )


def point_to_simplex_distance(
    p: np.ndarray, simplex: np.ndarray, tol: float = 1e-9
) -> float:
    """Compute the shortest Euclidean distance from a point to a k‑simplex in
    n‑dimensional space.

    Parameters
    ----------
    p : np.ndarray of shape (n,)
        A query point in the ambient n‑dimensional space.
    simplex : np.ndarray of shape (k+1, n)
        An array of k+1 vertices defining a k‑dimensional simplex embedded in n‑space.
    tol : float, optional
        Tolerance for the barycentric coordinate “inside” test. Defaults to 1e-9.

    Returns
    -------
    float
        The Euclidean distance from `p` to the simplex. Returns 0.0 if `p` projects
        inside or on the simplex (within the given tolerance).

    """
    k = len(simplex) - 1
    if k == 0:
        # 0-simplex: just a point
        return np.linalg.norm(p - simplex[0])

    base = simplex[0]
    A = (simplex[1:] - base).T  # shape (d, k)
    λ, *_ = np.linalg.lstsq(A, p - base, rcond=None)
    proj = base + A @ λ
    bary = np.empty(k + 1)
    bary[0] = 1 - λ.sum()
    bary[1:] = λ

    # inside (or on) the simplex?
    if np.all(bary >= -tol):
        return np.linalg.norm(p - proj)

    # otherwise, recurse on all k facets (remove one vertex at a time)
    return min(
        point_to_simplex_distance(p, np.delete(simplex, i, axis=0), tol)
        for i in range(k + 1)
    )


class AlphaShape:
    """Compute the α-shape (concave hull) of a point cloud in arbitrary dimensions."""

    def __init__(
        self,
        points: np.ndarray,
        alpha: float = 0.0,
        connectivity: Literal["strict", "relaxed"] = "strict",
        ensure_closure: bool = True,
    ):
        """Compute the α-shape (concave hull) of a point cloud in arbitrary dimensions.

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
        self.perimeter_points: Union[np.ndarray, None] = None
        self.GCT = GraphClosureTracker(len(points))

        self._delaunay: Union[Delaunay, None] = None

        # build once
        self._build_batch()

    @property
    def vertices(self) -> Optional[np.ndarray]:
        """Get the perimeter vertices of the alpha shape.

        Returns
        -------
        np.ndarray or None
            Array of perimeter points, or None if not computed.

        """

        return self.perimeter_points

    def contains_point(self, pt: np.ndarray, tol: float = 1e-8) -> bool:
        """Check whether a given point lies inside or on the alpha shape.

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
        assert (
            self._delaunay is not None
        ), "Delaunay triangulation must be performed first"
        # fast hull‐check
        simplex_idx = self._delaunay.find_simplex(pt)
        if simplex_idx < 0:
            return False  # outside convex hull -> outside α‐shape

        # map back to vertex‐indices & test membership
        tri = tuple(sorted(self._delaunay.simplices[simplex_idx]))
        if tri in self.simplices:
            return True  # inside one of the retained simplices
        else:
            return False  # lies in a Delaunay simplex that was carved away

    def add_points(self, new_pts: np.ndarray, perimeter_only: bool = False) -> None:
        """Add new points to the alpha shape (batch rebuild).

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
        self.__init__(  # type: ignore[misc]
            pts,
            alpha=self.alpha,
            connectivity=self.connectivity,
            ensure_closure=self.ensure_closure,
        )

    def _get_boundary_facets(self) -> Set[Tuple[int, ...]]:
        """Identify and return the boundary (d-1)-facets of the alpha shape, and cache
        per-facet filter data for fast distance queries."""
        if hasattr(self, "_boundary_facets"):
            return self._boundary_facets

        dim = self._dim
        facets: Set[Tuple[int, ...]] = set()
        for s in self.simplices:
            for f in itertools.combinations(s, dim):
                f = tuple(sorted(f))
                if f in facets:
                    facets.remove(f)
                else:
                    facets.add(f)
        # cache boundary facets
        self._boundary_facets: Set[Tuple[int, ...]] = facets

        # cache filter data: (normal, base_point, sphere_center, sphere_radius)
        self._facet_filter_data: Dict[
            Tuple[int, ...], Tuple[np.ndarray, np.ndarray, np.ndarray, float]
        ] = {}
        for facet in facets:
            verts = self.points[list(facet)]  # (d, d)
            base = verts[0]
            A = (verts[1:] - base).T  # (d, d-1)
            # unit normal via nullspace of A^T
            _, _, Vt = np.linalg.svd(A.T)
            normal = Vt[-1]
            normal /= np.linalg.norm(normal)
            # bounding sphere: centroid + max vertex distance
            center = verts.mean(axis=0)
            radius = np.max(np.linalg.norm(verts - center, axis=1))
            self._facet_filter_data[facet] = (normal, base, center, radius)

        return facets

    def distance_to_surface(self, point: np.ndarray, tol: float = 1e-9) -> float:
        """Compute the shortest Euclidean distance from a point to the alpha shape
        surface, using plane & sphere filters for speed."""
        p = np.asarray(point, dtype=float)
        if p.shape[-1] != self._dim:
            raise ValueError("point dimensionality mismatch")

        # 1. inside / on‑surface test
        if self.contains_point(p):
            return 0.0

        # 2. boundary facets
        facets = self._get_boundary_facets()
        if not facets:
            return float(np.min(np.linalg.norm(self.perimeter_points - p, axis=1)))

        best = np.inf
        for facet in facets:
            normal, base, center, radius = self._facet_filter_data[facet]
            # plane-distance filter
            if abs(np.dot(p - base, normal)) >= best:
                continue
            # bounding-sphere filter
            if np.linalg.norm(p - center) - radius >= best:
                continue
            # exact distance on this facet
            verts = self.points[list(facet)]
            d = point_to_simplex_distance(p, verts, tol)
            if d < best:
                best = d

        return float(best)

    def _build_batch(self) -> None:
        """Construct the alpha shape using Delaunay triangulation and filtering by
        alpha.

        This method is automatically called upon initialization.

        """

        dim, pts = self._dim, self.points
        n = len(pts)
        if n < dim + 1:
            self.perimeter_points = pts
            return

        r_filter = np.inf if self.alpha <= 0 else 1.0 / self.alpha
        self._delaunay = Delaunay(pts, qhull_options="Qz")

        # ---------- 1.  main sweep ---------------------------------------
        simplices = []
        for s in self._delaunay.simplices:
            r = circumradius(pts[s])
            simplices.append((tuple(sorted(s)), r))

        simplices.sort(key=lambda t: t[1])  # radius ascending
        kept = []
        uf = GraphClosureTracker(n)  # temp tracker

        for simp, r in simplices:
            root_set = {uf.find(v) for v in simp}
            keep = (r <= r_filter) or (
                self.connectivity == "relaxed" and len(root_set) > 1
            )
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
            main_root = max(comp_sizes, key=lambda k: comp_sizes[k])
            main_verts = gct.components[main_root]

            # Build edge-to-triangle map
            edge_to_triangles = defaultdict(list)
            for simp in all_passed:
                for edge in itertools.combinations(simp, 2):
                    edge_ = tuple(sorted(edge))
                    edge_to_triangles[edge_].append(simp)

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
        edge_counts: Dict[Tuple[Any, ...], int] = defaultdict(int)

        for s in self.simplices:
            self.GCT.add_fully_connected_subgraph(list(s))
            for edge in itertools.combinations(s, 2):  # triangle edges
                edge_ = tuple(sorted(edge))
                edge_counts[edge_] += 1

        # ---------- 4.  store perimeter ----------------------------------
        perimeter_edges_idx = [e for e, count in edge_counts.items() if count == 1]
        perim_idx = set(i for e in perimeter_edges_idx for i in e)

        self.perimeter_points = pts[list(sorted(perim_idx))]
        self.perimeter_edges = [(pts[i], pts[j]) for i, j in perimeter_edges_idx]

    @property
    def is_empty(self) -> bool:
        """Check whether the alpha shape is empty (i.e., no perimeter points).

        Returns
        -------
        bool
            True if empty; False otherwise.

        """
        if self.perimeter_points is None:
            return True
        return len(self.perimeter_points) == 0

    @property
    def triangle_facets(self) -> List[np.ndarray]:
        """Get the triangle facets (simplices) that make up the alpha shape.

        Returns
        -------
        List[np.ndarray]
            List of arrays containing vertex coordinates of each simplex.

        """

        return [self.points[list(s)] for s in self.simplices]

    @property
    def centroid(self) -> np.ndarray:
        """Compute the hyper-volumetric centroid of the Euclidean alpha shape using
        direct determinant-based simplex volume.

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
