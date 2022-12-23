import matplotlib.pyplot as plt
from santa_2022.original import *


def plot_configuration(config, image=None, ax=None, color='0.25', **figure_args):
    if ax is None:
        _, ax = plt.subplots(**figure_args)

    k = 2 ** (len(config) - 1) + 1
    X, Y = np.vstack([[(0, 0)], np.asarray(config).cumsum(axis=0)])[:-1].T
    U, V = np.asarray(config).T
    ax.quiver(
        X, Y, U, V,
        angles='xy', scale_units='xy', scale=1,
        color=color,
        width=0.005,
        zorder=10,
    )
    point = get_position(config)
    ax.plot(point[0], point[1], '.', color='k', zorder=11)
    l = k - 1 + 0.5
    if image is not None:
        ax.matshow(image, extent=[-l, l, -l, l])
    ax.set_xlim(-l, l)
    ax.set_ylim(-l, l)
    ax.set_aspect('equal')
    # ax.set_xticks(np.arange(-k, k + 1))
    # ax.set_yticks(np.arange(-k, k + 1))
    return ax


def plot_image(image):
    radius = (image.shape[0] - 1) // 2
    fig, ax = plt.subplots(figsize=(2, 2))
    ax.matshow(image, extent=(-radius - 1, radius, -radius - 1, radius + 1))
    ax.grid(None)
    return ax


def plot_neighbors(config):
    point = (0, 0)
    cs = get_neighbors(config)
    k = 2 ** (len(config) - 1) + 1
    colors = cycle(plt.rcParams["axes.prop_cycle"].by_key()["color"])

    fig, ax = plt.subplots(figsize=(10, 11))
    for c in cs:
        X, Y = (np.vstack([[(0, 0)], np.asarray(c).cumsum(axis=0)])[:-1].T - 0.5)
        U, V = np.asarray(c).T
        ax.quiver(
            X, Y, U, V,
            angles='xy', scale_units='xy', scale=1,
            color=next(colors),
            width=0.0025,
            zorder=10,
            alpha=0.25,
        )
        point = get_position(c)
        ax.plot(point[0] - 0.5, point[1] - 0.5, '.', color='k', zorder=11)
    point = get_position(config)
    ax.plot(point[0] - 0.5, point[1] - 0.5, 'o', color='C3', zorder=11)
    ax.set_xlim(-k - 1, k)
    ax.set_ylim(-k - 1, k)
    ax.set_aspect('equal')
    ax.set_xticks(np.arange(-k, k + 1))
    ax.set_yticks(np.arange(-k, k + 1))
    ax.grid(True, color='0.5')
    ax.set_title(f"Neighbors of {config}", fontsize=20)
    return ax


def plot_path(path):
    config = path[0]
    point = get_position(config)
    k = 2 ** (len(config) - 1) + 1
    colors = plt.cm.plasma(np.linspace(0, 1, len(path)))

    fig, ax = plt.subplots(figsize=(10, 11))
    for i, c in enumerate(path):
        prev_point = point
        point = get_position(c)
        ax.plot(point[0] - 0.5, point[1] - 0.5, '.', color="k", zorder=11)
        ax.arrow(prev_point[0] - 0.49, prev_point[1] - 0.49, point[0] - prev_point[0],
                 point[1] - prev_point[1],
                 width=0.0025, zorder=10, alpha=0.8, head_width=0.1,
                 length_includes_head=True, color=colors[i])
    point = get_position(config)
    ax.plot(point[0] - 0.5, point[1] - 0.5, 'o', color='C3', zorder=11)
    ax.set_xlim(-k - 1, k)
    ax.set_ylim(-k - 1, k)
    ax.set_aspect('equal')
    ax.set_xticks(np.arange(-k, k + 1))
    ax.set_yticks(np.arange(-k, k + 1))
    ax.grid(True, color='0.5')
    ax.set_title(f"Plot of path", fontsize=20)
    return ax
