import matplotlib.pyplot as plt
from pyalphashape.AlphaShape import AlphaShape
from pyalphashape.SphericalAlphaShape import SphericalAlphaShape
from pyalphashape.SphericalDelaunay import SphericalDelaunay
from typing import Any, Optional
from matplotlib.axes import Axes
from mpl_toolkits.mplot3d import Axes3D  # noqa
import numpy as np
import plotly.graph_objects as go

def plot_polygon_edges(
    edges: np.ndarray,
    ax: Axes,
    line_color: str = "b",
    line_width: float = 1.0,
):
    """
    Plot a set of polygon edges on a 2D or 3D Matplotlib axis.

    Parameters
    ----------
    edges : np.ndarray
        An array of shape (n, 2) where each entry is a pair of points (each a coordinate array)
        representing an edge of the polygon.
    ax : matplotlib.axes.Axes
        A Matplotlib Axes object (2D or 3D) to plot on.
    line_color : str, optional
        Color of the edges, by default 'b'.
    line_width : float, optional
        Width of the edge lines, by default 1.0.
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


def interpolate_great_arc(A: np.ndarray, B: np.ndarray, num_points: int = 100) -> np.ndarray:
    """
    Compute evenly spaced points along the great arc connecting two points on the unit sphere.

    Parameters
    ----------
    A : np.ndarray
        A 3D unit vector representing the starting point.
    B : np.ndarray
        A 3D unit vector representing the ending point.
    num_points : int, optional
        Number of interpolation points along the arc, by default 100.

    Returns
    -------
    np.ndarray
        Array of shape (num_points, 3) containing interpolated points on the unit sphere.
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
    title: str = "Spherical Triangulation",
    fig: Optional[go.Figure] = None,
    ax: Optional[Axes] = None,
    marker_size: float = 5,
    marker_color: Any = "black",
    marker_symbol: Optional[str] = "circle",
    line_width: float = 1.5,
    line_color: Any = "blue",
    line_style: str = "-"
):
    """
    Visualize a spherical Delaunay triangulation using either Matplotlib or Plotly.

    Parameters
    ----------
    triangulation : SphericalDelaunay
        The triangulation object containing unit vectors and triangle indices.
    title : str, optional
        Title of the plot. Default is "Spherical Triangulation".
    fig : plotly.graph_objects.Figure, optional
        A Plotly figure to add to. If None, a new figure is created.
    ax : matplotlib.axes.Axes, optional
        A Matplotlib 3D axis object to plot on. If provided, Matplotlib is used.
    marker_size : float, optional
        Size of the point markers. Default is 5.
    marker_color : Any, optional
        Color of the point markers. Default is "black".
    marker_symbol : str, optional
        Marker style for Plotly (e.g., "circle", "square"). Ignored in Matplotlib.
    line_width : float, optional
        Width of triangle edge lines. Default is 1.5.
    line_color : Any, optional
        Color of triangle edges. Default is "blue".
    line_style : str, optional
        Line style for Matplotlib (e.g., "-", "--"). Ignored in Plotly.

    Returns
    -------
    plotly.graph_objects.Figure or matplotlib.axes.Axes
        The figure or axis object used for plotting.
    """

    if ax is not None:
        ax.set_title(title)
        ax.set_box_aspect([1, 1, 1])
        ax.grid(False)

        # Sphere surface
        u, v = np.mgrid[0:2*np.pi:60j, 0:np.pi:30j]
        xs = np.cos(u) * np.sin(v)
        ys = np.sin(u) * np.sin(v)
        zs = np.cos(v)
        ax.plot_surface(xs, ys, zs, color="lightgrey", alpha=0.15, linewidth=0)

        # Points
        pts = triangulation.points_xyz
        ax.scatter(pts[:, 0], pts[:, 1], pts[:, 2], color=marker_color, s=marker_size**2)

        # Arcs
        for tri in triangulation.triangles:
            indices = [0, 1, 2, 0]
            for i in range(3):
                A = pts[tri[indices[i]]]
                B = pts[tri[indices[i + 1]]]
                arc_pts = interpolate_great_arc(A, B)
                ax.plot(arc_pts[:, 0], arc_pts[:, 1], arc_pts[:, 2],
                        color=line_color, linewidth=line_width, linestyle=line_style)

        return ax

    return _plot_spherical_triangulation_plotly(
        triangulation, title, fig,
        marker_size, marker_color, marker_symbol,
        line_width, line_color
    )


def _plot_spherical_triangulation_plotly(
    triangulation: SphericalDelaunay,
    title: str,
    fig: Optional[go.Figure],
    marker_size: float,
    marker_color: Any,
    marker_symbol: Optional[str],
    line_width: float,
    line_color: Any
):
    """
    Internal helper to render a spherical triangulation using Plotly.

    Parameters
    ----------
    triangulation : SphericalDelaunay
        Triangulation object containing vertices and triangle indices.
    title : str
        Title for the plot.
    fig : plotly.graph_objects.Figure or None
        Existing figure to modify, or None to create a new one.
    marker_size : float
        Size of the point markers.
    marker_color : Any
        Color of the point markers.
    marker_symbol : str or None
        Marker symbol for Plotly (e.g., "circle", "x").
    line_width : float
        Width of the triangle edge lines.
    line_color : Any
        Color of the triangle edges.

    Returns
    -------
    plotly.graph_objects.Figure
        The updated or newly created Plotly figure.
    """

    if fig is None:
        fig = go.Figure()

        u, v = np.mgrid[0:2*np.pi:60j, 0:np.pi:30j]
        xs = np.cos(u) * np.sin(v)
        ys = np.sin(u) * np.sin(v)
        zs = np.cos(v)
        fig.add_trace(go.Surface(
            x=xs, y=ys, z=zs,
            opacity=0.15,
            showscale=False,
            colorscale='Greys',
            name='Sphere'
        ))

    pts = triangulation.points_xyz
    fig.add_trace(go.Scatter3d(
        x=pts[:, 0], y=pts[:, 1], z=pts[:, 2],
        mode='markers',
        marker=dict(size=marker_size, color=marker_color, symbol=marker_symbol),
        name='Points'
    ))

    for tri in triangulation.triangles:
        indices = [0, 1, 2, 0]
        for i in range(3):
            A = pts[tri[indices[i]]]
            B = pts[tri[indices[i + 1]]]
            arc_pts = interpolate_great_arc(A, B)
            fig.add_trace(go.Scatter3d(
                x=arc_pts[:, 0], y=arc_pts[:, 1], z=arc_pts[:, 2],
                mode='lines',
                line=dict(color=line_color, width=line_width),
                showlegend=False
            ))

    fig.update_layout(
        title=title,
        scene=dict(xaxis=dict(showgrid=False),
                   yaxis=dict(showgrid=False),
                   zaxis=dict(showgrid=False),
                   aspectmode='data'),
        margin=dict(l=0, r=0, b=0, t=30)
    )
    return fig


def plot_alpha_shape(
    shape: AlphaShape,
    ax: Optional[Axes] = None,
    line_width: int = 1,
    line_color: Any = "r"
):
    """
    Plot a 2D alpha shape perimeter using Matplotlib.

    Parameters
    ----------
    shape : AlphaShape
        The alpha shape object whose perimeter edges will be plotted.
    ax : matplotlib.axes.Axes, optional
        An optional Matplotlib axis to plot on. A new figure is created if None.
    line_width : int, optional
        Width of the perimeter lines, by default 1.
    line_color : Any, optional
        Color of the perimeter lines, by default 'r'.
    """

    edges = [(e1, e2) for e1, e2 in shape.perimeter_edges]

    if ax is None:
        fig, ax = plt.subplots()
    plot_polygon_edges(
        edges, ax, line_width=line_width, line_color=line_color
    )


def plot_sperical_alpha_shape(
    shape: SphericalAlphaShape,
    line_width: float = 1.5,
    line_color: Any = "red",
    line_style: str = "-",
    marker_size: float = 5,
    marker_color: Any = "black",
    marker_symbol: Optional[str] = "circle",
    title: str = "Spherical Alpha Shape",
    fig: Optional[go.Figure] = None,
    ax: Optional[Axes] = None
):
    """
    Visualize the perimeter of a spherical alpha shape using either Matplotlib or Plotly.

    Parameters
    ----------
    shape : SphericalAlphaShape
        The spherical alpha shape object containing perimeter edges and points.
    line_width : float, optional
        Width of the perimeter arcs. Default is 1.5.
    line_color : Any, optional
        Color of the perimeter arcs. Default is "red".
    line_style : str, optional
        Line style for Matplotlib (e.g., "-", "--"). Ignored in Plotly.
    marker_size : float, optional
        Size of point markers. Default is 5.
    marker_color : Any, optional
        Color of the point markers. Default is "black".
    marker_symbol : str, optional
        Plotly marker style (e.g., "circle", "cross"). Ignored in Matplotlib.
    title : str, optional
        Plot title. Default is "Spherical Alpha Shape".
    fig : plotly.graph_objects.Figure, optional
        Existing Plotly figure to modify. A new one is created if None.
    ax : matplotlib.axes.Axes, optional
        Existing Matplotlib 3D axis to plot on. If provided, Matplotlib is used.

    Returns
    -------
    plotly.graph_objects.Figure or matplotlib.axes.Axes
        The figure or axis used for plotting.
    """

    if ax is not None:
        ax.set_title(title)
        ax.set_box_aspect([1, 1, 1])
        ax.grid(False)

        # Sphere surface
        u, v = np.mgrid[0:2*np.pi:60j, 0:np.pi:30j]
        xs = np.cos(u) * np.sin(v)
        ys = np.sin(u) * np.sin(v)
        zs = np.cos(v)
        ax.plot_surface(xs, ys, zs, color="lightgrey", alpha=0.15, linewidth=0)

        # Arcs
        for e1, e2 in shape.perimeter_edges:
            A = shape.points[e1]
            B = shape.points[e2]
            arc_pts = interpolate_great_arc(A, B)
            ax.plot(arc_pts[:, 0], arc_pts[:, 1], arc_pts[:, 2],
                    color=line_color, linewidth=line_width, linestyle=line_style)

        # Points
        pts = shape.points
        ax.scatter(pts[:, 0], pts[:, 1], pts[:, 2], color=marker_color, s=marker_size**2)

        return ax

    return _plot_sperical_alpha_shape_plotly(
        shape, line_width, line_color, marker_size, marker_color, marker_symbol, title, fig
    )


def _plot_sperical_alpha_shape_plotly(
    shape: SphericalAlphaShape,
    line_width: float,
    line_color: Any,
    marker_size: float,
    marker_color: Any,
    marker_symbol: Optional[str],
    title: str,
    fig: Optional[go.Figure]
):
    """
    Internal helper to render a spherical alpha shape perimeter using Plotly.

    Parameters
    ----------
    shape : SphericalAlphaShape
        The alpha shape object with perimeter edges and point cloud.
    line_width : float
        Width of perimeter arc lines.
    line_color : Any
        Color of perimeter arcs.
    marker_size : float
        Size of point markers.
    marker_color : Any
        Color of the markers.
    marker_symbol : str or None
        Marker style for Plotly (e.g., "circle", "square").
    title : str
        Title of the plot.
    fig : plotly.graph_objects.Figure or None
        Existing Plotly figure to modify or None to create a new one.

    Returns
    -------
    plotly.graph_objects.Figure
        The updated or newly created Plotly figure.
    """

    if fig is None:
        fig = go.Figure()
        u, v = np.mgrid[0:2*np.pi:60j, 0:np.pi:30j]
        xs = np.cos(u) * np.sin(v)
        ys = np.sin(u) * np.sin(v)
        zs = np.cos(v)
        fig.add_trace(go.Surface(
            x=xs, y=ys, z=zs,
            opacity=0.15,
            showscale=False,
            colorscale='Greys',
            name='Sphere'
        ))

    for e1, e2 in shape.perimeter_edges:
        A = shape.points[e1]
        B = shape.points[e2]
        arc_pts = interpolate_great_arc(A, B)
        fig.add_trace(go.Scatter3d(
            x=arc_pts[:, 0], y=arc_pts[:, 1], z=arc_pts[:, 2],
            mode='lines',
            line=dict(color=line_color, width=line_width),
            showlegend=False
        ))

    pts = shape.points
    fig.add_trace(go.Scatter3d(
        x=pts[:, 0], y=pts[:, 1], z=pts[:, 2],
        mode='markers',
        marker=dict(size=marker_size, color=marker_color, symbol=marker_symbol),
        name="Points"
    ))

    fig.update_layout(
        title=title,
        scene=dict(xaxis=dict(showgrid=False),
                   yaxis=dict(showgrid=False),
                   zaxis=dict(showgrid=False),
                   aspectmode='data'),
        margin=dict(l=0, r=0, b=0, t=30)
    )
    return fig
