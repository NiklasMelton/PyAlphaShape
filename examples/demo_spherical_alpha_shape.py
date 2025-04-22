import numpy as np
from pyalphashape import SphericalAlphaShape, plot_sperical_alpha_shape

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

# Generate both alpha shapes
alpha_hemisphere = SphericalAlphaShape(latlon_hemisphere, alpha=1.0)
alpha_global = SphericalAlphaShape(latlon_global, alpha=0.7)


# Plot both examples
fig = plot_sperical_alpha_shape(alpha_hemisphere, title="Hemispherical Alpha Shape")
fig.show()
fig = plot_sperical_alpha_shape(alpha_global, title="Global Alpha Shape")
fig.show()
