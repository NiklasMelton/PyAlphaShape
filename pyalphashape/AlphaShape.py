import itertools
import logging
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

    num_rows, num_columns = points.shape
    A = np.bmat([[2 * np.dot(points, points.T),
                  np.ones((num_rows, 1))],
                 [np.ones((1, num_rows)), np.zeros((1, 1))]])
    b = np.hstack((np.sum(points * points, axis=1),
                   np.ones((1))))
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
    """

    def __init__(self,
                 points: np.ndarray,
                 alpha: float = 0.,
                 connectivity: Literal["strict", "relaxed"] = "strict"
    ):
        self._dim = points.shape[1]
        if self._dim < 2:
            raise ValueError("dimension must be ≥ 2")

        self.alpha = float(alpha)
        if connectivity not in {"strict", "relaxed"}:
            raise ValueError("connectivity must be 'strict' or 'relaxed'")
        self.connectivity = connectivity

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

    def contains_point(self, pt: np.ndarray) -> bool:
        """
        Check whether a given point lies inside the alpha shape.

        Parameters
        ----------
        pt : np.ndarray
            A point of shape (d,) to test for inclusion.

        Returns
        -------
        bool
            True if the point lies inside or on the alpha shape; False otherwise.
        """

        if len(self.simplices) == 0:
            return False
        for s in self.simplices:
            verts = self.points[list(s)]
            try:
                A = np.vstack([verts.T, np.ones(len(verts))])
                bary = np.linalg.solve(A, np.append(pt, 1.0))
                if np.all(bary >= 0):
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
        self.__init__(pts, alpha=self.alpha)

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
            comp_sizes = {root: len(nodes) for root, nodes in uf.components.items()}
            main_root = max(comp_sizes, key=comp_sizes.get)
            main_verts = uf.components[main_root]
            kept = [s for s in kept if set(s) <= main_verts]

        # ---------- 3.  rebuild perimeter from *kept* simplices ----------
        self.simplices = set(kept)
        self.GCT = GraphClosureTracker(n)  # final tracker
        edges, perim_idx = set(), set()

        for s in self.simplices:
            self.GCT.add_fully_connected_subgraph(list(s))

            for f in itertools.combinations(s, dim):  # (d-1)-faces
                f = tuple(sorted(f))
                if f in edges:
                    edges.remove(f)
                else:
                    edges.add(f)
                    perim_idx.update(f)

        # ---------- 4.  store perimeter ----------------------------------
        self.perimeter_points = pts[list(sorted(perim_idx))]
        self.perimeter_edges = [(pts[i], pts[j]) for f in edges
                                for i, j in itertools.combinations(f, 2)]

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
