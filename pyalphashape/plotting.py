import numpy as np
import matplotlib.pyplot as plt
from matplotlib.axes import Axes
import plotly.graph_objects as go
from pyalphashape.AlphaShape import AlphaShape
from typing import Any, Optional
from pyalphashape.SphericalAlphaShape import SphericalAlphaShape
from pyalphashape.SphericalDelaunay import SphericalDelaunay

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


def interpolate_great_arc(A, B, num_points=100):
    """
    Interpolate points along the great arc between unit vectors A and B on the sphere.
    Returns a (num_points, 3) array of 3D unit vectors.
    """
    A = A / np.linalg.norm(A)
    B = B / np.linalg.norm(B)
    dot = np.clip(np.dot(A, B), -1.0, 1.0)
    theta = np.arccos(dot)

    if theta < 1e-6:
        return np.repeat(A[None, :], num_points, axis=0)

    sin_theta = np.sin(theta)
    t_vals = np.linspace(0, 1, num_points)
    arc_points = (np.sin((1 - t_vals) * theta)[:, None] * A +
                  np.sin(t_vals * theta)[:, None] * B) / sin_theta
    return arc_points


def plot_spherical_triangulation(
        triangulation: SphericalDelaunay,
        title: str="Spherical Triangulation"
):
    # Sphere surface
    u, v = np.mgrid[0:2*np.pi:60j, 0:np.pi:30j]
    xs = np.cos(u) * np.sin(v)
    ys = np.sin(u) * np.sin(v)
    zs = np.cos(v)

    fig = go.Figure()

    # Add semi-transparent sphere
    fig.add_trace(go.Surface(
        x=xs, y=ys, z=zs,
        opacity=0.15,
        showscale=False,
        colorscale='Greys',
        name='Sphere'
    ))

    # Add points
    fig.add_trace(go.Scatter3d(
        x=triangulation.points_xyz[:, 0],
        y=triangulation.points_xyz[:, 1],
        z=triangulation.points_xyz[:, 2],
        mode='markers',
        marker=dict(size=5, color='black'),
        name='Points'
    ))

    # Draw great arc edges of triangles
    for tri in triangulation.triangles:
        indices = [0, 1, 2, 0]  # Loop back to start
        for i in range(3):
            A = triangulation.points_xyz[tri[indices[i]]]
            B = triangulation.points_xyz[tri[indices[i + 1]]]
            arc_pts = interpolate_great_arc(A, B, num_points=50)
            fig.add_trace(go.Scatter3d(
                x=arc_pts[:, 0],
                y=arc_pts[:, 1],
                z=arc_pts[:, 2],
                mode='lines',
                line=dict(color='blue', width=2),
                showlegend=False
            ))

    fig.update_layout(
        title=title,
        scene=dict(
            xaxis=dict(showgrid=False, zeroline=False),
            yaxis=dict(showgrid=False, zeroline=False),
            zaxis=dict(showgrid=False, zeroline=False),
            aspectmode='data'
        ),
        margin=dict(l=0, r=0, b=0, t=30)
    )

    fig.show()

def plot_alpha_shape(
        shape: AlphaShape,
        ax: Optional[Axes] = None,
        line_width: int = 1,
        line_color: Any = "r"
):
    edges = [(e1, e2) for e1, e2 in shape.perimeter_edges]

    if ax is None:
        fig, ax = plt.subplots()
    plot_polygon_edges(
        edges, ax, line_width=line_width, line_color=line_color
    )

def plot_sperical_alpha_shape(
        shape: SphericalAlphaShape,
        line_width: int = 1,
        line_color: Any = "r",
        title: str="Spherical Alpha Shape"
):
    edges = [(e1, e2) for e1, e2 in shape.perimeter_edges]

    # Sphere surface
    u, v = np.mgrid[0:2 * np.pi:60j, 0:np.pi:30j]
    xs = np.cos(u) * np.sin(v)
    ys = np.sin(u) * np.sin(v)
    zs = np.cos(v)

    fig = go.Figure()

    # Add semi-transparent sphere
    fig.add_trace(go.Surface(
        x=xs, y=ys, z=zs,
        opacity=0.15,
        showscale=False,
        colorscale='Greys',
        name='Sphere'
    ))

    for e in edges:
        A = shape.points[e[0]]
        B = shape.points[e[1]]
        arc_pts = interpolate_great_arc(A, B, num_points=50)
        fig.add_trace(go.Scatter3d(
            x=arc_pts[:, 0],
            y=arc_pts[:, 1],
            z=arc_pts[:, 2],
            mode='lines',
            line=dict(color=line_color, width=line_width),
            showlegend=False
        ))

    fig.update_layout(
        title=title,
        scene=dict(
            xaxis=dict(showgrid=False, zeroline=False),
            yaxis=dict(showgrid=False, zeroline=False),
            zaxis=dict(showgrid=False, zeroline=False),
            aspectmode='data'
        ),
        margin=dict(l=0, r=0, b=0, t=30)
    )

    fig.show()