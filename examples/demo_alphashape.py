from sklearn.datasets import make_blobs
import matplotlib.pyplot as plt
from pyalphashape import AlphaShape, plot_alpha_shape

data, _ = make_blobs(
        n_samples=50,
        centers=1,
        cluster_std=0.50,
        random_state=0,
        shuffle=False,
    )

shape = AlphaShape(data, alpha=1.0)

fig, ax = plt.subplots()
plot_alpha_shape(shape, ax)

ax.scatter(
    data[:, 0],
    data[:, 1],
    color="k",
    marker=".",
    s=10,
)

plt.show()