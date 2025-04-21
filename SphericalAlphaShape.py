import numpy as np
import itertools
from typing import Literal, Set, Tuple, List
from SphericalDelaunay import SphericalDelaunay
from sphere_utils import latlon_to_unit_vectors, arc_distance, spherical_circumradius
from GraphClosure import GraphClosureTracker


class SphericalAlphaShape:
    """
    Batch α‑shape (concave hull) in arbitrary dimension.
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

        self.points_latlon = np.asarray(points, dtype=float)
        self.points = latlon_to_unit_vectors(self.points_latlon)

        self.simplices: Set[Tuple[int, ...]] = set()
        self.perimeter_edges: List[Tuple[np.ndarray, np.ndarray]] = []
        self.perimeter_points: np.ndarray | None = None
        self.GCT = GraphClosureTracker(len(points))

        # build once
        self._build_batch()


    @property
    def vertices(self):
        return self.perimeter_points

    def contains_point(self, pt_latlon: np.ndarray, tol: float = 1e-10) -> bool:
        """
        Check whether a (lat, lon) point lies inside any spherical triangle in the alpha shape.

        Args:
            pt_latlon: (2,) array with [latitude, longitude] in degrees.
            tol: Numerical tolerance.

        Returns:
            True if the point lies inside any triangle, False otherwise.
        """
        if len(self.simplices) == 0:
            return False

        # Convert (1, 2) input to (1, 3) unit vector
        pt = latlon_to_unit_vectors(pt_latlon[None, :])[0]

        for s in self.simplices:
            verts = self.points[list(s)]
            A, B, C = verts

            nAB = np.cross(A, B)
            nBC = np.cross(B, C)
            nCA = np.cross(C, A)

            sign1 = np.sign(np.dot(nAB, pt))
            sign2 = np.sign(np.dot(nBC, pt))
            sign3 = np.sign(np.dot(nCA, pt))

            if (sign1 == sign2 == sign3) or (np.abs(sign1 + sign2 + sign3) >= 2.9):
                # Either all signs are equal or numerically very close
                return True

        return False

    def add_points(self, new_pts: np.ndarray):
        """
        *Batch* version – simply rebuilds everything.
        Sub‑class overrides this for incremental behaviour.
        """
        pts = np.vstack([self.points_latlon, new_pts])
        self.__init__(pts, alpha=self.alpha)


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
        Angular arc distance (radians) from `point` (given in [lat, lon] degrees)
        to the α‑shape surface on a unit sphere.

        Returns 0 if the point lies inside or on the surface.

        Only works for the 2D surface of a 3D sphere (dim=3).
        """
        if point.shape[-1] != 2:
            raise ValueError("Input point must be (lat, lon) in degrees")
        if self._dim != 3:
            raise NotImplementedError(
                "Only implemented for points on the 2D surface of a 3D sphere"
            )

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

    # ------------------------------------------------------------------ #
    def _build_batch(self):
        dim, pts, pts_latlon = self._dim, self.points, self.points_latlon
        n = len(pts)
        if n < dim + 1:
            self.perimeter_points = pts
            return

        r_filter = np.inf if self.alpha <= 0 else 1.0 / self.alpha
        tri = SphericalDelaunay(pts_latlon)

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
            from collections import defaultdict
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
    def is_empty(self):
        return len(self.perimeter_points) == 0

    @property
    def triangle_faces(self) -> List[np.ndarray]:
        return [self.points[list(s)] for s in self.simplices]

    @property
    def triangle_faces_latlon(self):
        return [self.points_latlon[list(s)] for s in self.simplices]

