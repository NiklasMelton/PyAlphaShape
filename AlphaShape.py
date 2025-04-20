import itertools
import logging
from scipy.spatial import Delaunay
import numpy as np
import math
from typing import Tuple, Set, List, Literal
from matplotlib.axes import Axes
from GraphClosure import GraphClosureTracker



def circumcenter(points: np.ndarray) -> np.ndarray:
    """
    Calculate the circumcenter of a set of points in barycentric coordinates.

    Args:
      points: An `N`x`K` array of points which define an (`N`-1) simplex in K
        dimensional space.  `N` and `K` must satisfy 1 <= `N` <= `K` and
        `K` >= 1.

    Returns:
      The circumcenter of a set of points in barycentric coordinates.
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
    Calculte the circumradius of a given set of points.

    Args:
      points: An `N`x`K` array of points which define an (`N`-1) simplex in K
        dimensional space.  `N` and `K` must satisfy 1 <= `N` <= `K` and
        `K` >= 1.

    Returns:
      The circumradius of a given set of points.
    """
    return np.linalg.norm(points[0, :] - np.dot(circumcenter(points), points))



def alphasimplices(points: np.ndarray) -> np.ndarray:
    """
    Returns an iterator of simplices and their circumradii of the given set of
    points.

    Args:
      points: An `N`x`M` array of points.

    Yields:
      A simplex, and its circumradius as a tuple.
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

def volume_of_simplex(vertices: np.ndarray) -> float:
    """
    Calculates the n-dimensional volume of a simplex defined by its vertices.

    Parameters
    ----------
    vertices : np.ndarray
        An (n+1) x n array representing the coordinates of the simplex vertices.

    Returns
    -------
    float
        Volume of the simplex.

    """
    vertices = np.asarray(vertices)
    # Subtract the first vertex from all vertices to form a matrix
    matrix = vertices[1:] - vertices[0]
    # Calculate the absolute value of the determinant divided by factorial(n) for volume
    return np.abs(np.linalg.det(matrix)) / np.math.factorial(len(vertices) - 1)


def get_perimeter_simplices(perimeter_edges: Set[Tuple], simplices: Set[Tuple], n: int):

    perimeter_edges_set = set(tuple(sorted(edge)) for edge in perimeter_edges)

    # This will store the (n-1)-simplices composed of perimeter edges
    perimeter_simplices = []

    for simplex in simplices:
        # Generate all (n-1)-dimensional subsimplices (faces)
        for subsimplex in itertools.combinations(simplex, n - 1):
            # Check if all edges of this subsimplex are in the perimeter
            subsimplex_edges = itertools.combinations(
                subsimplex,
                2
            )  # get pairs of vertices
            if all(tuple(sorted(edge)) in perimeter_edges_set for edge in
                   subsimplex_edges):
                perimeter_simplices.append(subsimplex)

    return perimeter_simplices


def compute_surface_area(points: np.ndarray, perimeter_edges: Set[Tuple], simplices: Set[Tuple]):
    """
    Compute the surface area (or perimeter in 2D) of the polytope formed by the perimeter edges.

    Args:
      points (np.ndarray): The points of the polytope.
      perimeter_edges (set): The set of perimeter edges.
      simplices (set): the set of simplices

    Returns:
      float: The total surface "area" (or perimeter in 2D, hyper-volume in higher dimensions).
    """
    # Handle the 2D case (perimeter)
    if points.shape[-1] == 2:
        total_perimeter = 0.0
        for edge in perimeter_edges:
            p1, p2 = edge
            total_perimeter += np.linalg.norm(points[p1] - points[p2])
        return total_perimeter

    # Handle the 3D and higher-dimensional cases
    perimeter_simplices = get_perimeter_simplices(
        perimeter_edges,
        simplices,
        points.shape[-1]
    )
    total_area = 0.
    for perimeter_simplex in perimeter_simplices:
        simplex_points = np.array([points[p,:] for p in perimeter_simplex])
        total_area += volume_of_simplex(simplex_points)

    return total_area


def equalateral_simplex_volume(n: int, s: float):
    numerator = s ** n
    denominator = math.factorial(n)
    sqrt_term = math.sqrt((n + 1) / (2 ** n))
    volume = (numerator / denominator) * sqrt_term
    return volume


def plot_polygon_edges(
    edges: np.ndarray,
    ax: Axes,
    line_color: str = "b",
    line_width: float = 1.0,
):
    """
    Plots a convex polygon given its vertices using Matplotlib.

    Parameters
    ----------
    vertices : np.ndarray
        A list of edges representing a polygon.
    ax : matplotlib.axes.Axes
        A matplotlib Axes object to plot on.
    line_color : str, optional
        The color of the polygon lines, by default 'b'.
    line_width : float, optional
        The width of the polygon lines, by default 1.0.

    """
    for p1, p2 in edges:
        x = [p1[0], p2[0]]
        y = [p1[1], p2[1]]
        if len(p1) > 2:
            z = [p1[2], p2[2]]
            ax.plot(
                x,
                y,
                z,
                linestyle="-",
                color=line_color,
                linewidth=line_width,
            )
        else:
            ax.plot(
                x,
                y,
                linestyle="-",
                color=line_color,
                linewidth=line_width,
            )

class AlphaShape:
    """
    Batch α‑shape (concave hull) in arbitrary dimension.
    """

    # --------------------------------------------------------------------- #
    #  construction (unchanged from your paste, but wrapped in a method so  #
    #  subclasses can re‑use it)                                            #
    # --------------------------------------------------------------------- #
    def __init__(self,
                 points: np.ndarray,
                 alpha: float = 0.,
                 max_perimeter_length: float = np.inf,
                 connectivity: Literal["strict", "relaxed"] = "strict"
    ):
        self._dim = points.shape[1]
        if self._dim < 2:
            raise ValueError("dimension must be ≥ 2")

        self.alpha = float(alpha)
        if connectivity not in {"strict", "relaxed"}:
            raise ValueError("connectivity must be 'strict' or 'relaxed'")
        self.connectivity = connectivity

        self.max_perimeter_length = float(max_perimeter_length)
        self.points = np.asarray(points, dtype=float)

        self.simplices: Set[Tuple[int, ...]] = set()
        self.perimeter_edges: List[Tuple[np.ndarray, np.ndarray]] = []
        self.perimeter_points: np.ndarray | None = None
        self.centroid = np.zeros(self._dim)
        self.volume = 0.0
        self.GCT = GraphClosureTracker(len(points))

        # build once
        self._build_batch()

    # ------------------------------------------------------------------ #
    #  public helpers (unchanged)                                        #
    # ------------------------------------------------------------------ #

    @property
    def vertices(self):
        return self.perimeter_points
    @property
    def max_edge_length(self) -> float:
        if not self.perimeter_edges:
            return 0.0
        return max(np.linalg.norm(a - b) for a, b in self.perimeter_edges)

    def contains_point(self, pt: np.ndarray) -> bool:
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

    def add_points(self, new_pts: np.ndarray):
        """
        *Batch* version – simply rebuilds everything.
        Sub‑class overrides this for incremental behaviour.
        """
        pts = np.vstack([self.points, new_pts])
        self.__init__(pts, alpha=self.alpha,
                      max_perimeter_length=self.max_perimeter_length)


    # ---------- lazy accessor for boundary (d‑1)-faces ------------------ #
    def _get_boundary_faces(self) -> Set[Tuple[int, ...]]:
        """
        Return the set of boundary faces (index tuples of length d) and cache
        it in `self._boundary_faces` so the computation is done only once.

        For `iAlphaShape` this simply forwards the attribute it already keeps.
        For the batch version we reconstruct the set from `self.simplices`
        using the usual “flip once ↔ boundary” rule.
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


    def distance_to_surface(self,
                            point: np.ndarray,
                            tol: float = 1e-9) -> float:
        """
        Euclidean distance from `point` to the α‑shape surface.
        Returns 0 if the point lies inside or on the surface.

        Works for any ambient dimension d ≥ 2.
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


    # ------------------------------------------------------------------ #
    #  internal: one‑shot batch builder (exact logic you pasted,         #
    #  only lightly reorganised so subclasses can call it)               #
    # ------------------------------------------------------------------ #
    def _build_batch(self):
        dim, pts = self._dim, self.points
        n = len(pts)
        if n < dim + 1:
            self.perimeter_points = pts
            self.centroid = pts.mean(axis=0)
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
        self.volume = 0.0
        self.centroid = np.zeros(dim)

        for s in self.simplices:
            self.GCT.add_fully_connected_subgraph(list(s))
            vol = volume_of_simplex(pts[list(s)])
            self.centroid = (self.centroid * self.volume +
                             vol * pts[list(s)].mean(axis=0)) / (self.volume + vol)
            self.volume += vol

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
    def is_empty(self):
        return len(self.perimeter_points) == 0
