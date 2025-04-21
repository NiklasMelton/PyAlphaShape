from pyalphashape import SphericalDelaunay, plot_spherical_triangulation
import numpy as np

# Hemispherical test dataset: cluster in northern hemisphere
latlon_hemisphere = np.array([
    [10, 0], [20, 20], [30, -30], [40, 60], [50, -60], [25, 120],
    [15, 150], [35, 170], [5, 90], [45, 10]
])

# Global test dataset: spread around the globe
latlon_global = np.array([
    [-45, -120], [60, -90], [30, 0], [-10, 60], [45, 90], [-60, 150],
    [20, -150], [-30, 170], [0, 180], [75, -45], [-75, 45]
])

# Run both triangulations
tri_hemisphere = SphericalDelaunay(latlon_hemisphere)
tri_global = SphericalDelaunay(latlon_global)

# Plot both examples
plot_spherical_triangulation(tri_hemisphere, title="Hemispherical Triangulation")
plot_spherical_triangulation(tri_global, title="Global Triangulation")
