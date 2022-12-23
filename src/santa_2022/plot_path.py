import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import pandas as pd

from santa_2022.plotting import *


def _to_hex(s):
    return '#' + s[2:]


def plot_path_over_image(config, arrows, save_path=None, image=None, ax=None,
                         **figure_args):
    if ax is None:
        _, ax = plt.subplots(figsize=(8, 8), **figure_args)

    k = 2 ** (len(config) - 1) + 1
    x = arrows.loc[:, 'x'].to_numpy()
    y = arrows.loc[:, 'y'].to_numpy()
    dx = arrows.loc[:, 'dx'].to_numpy()
    dy = arrows.loc[:, 'dy'].to_numpy()
    color = arrows.loc[:, 'color'].apply(_to_hex)
    color = color.apply(mcolors.to_rgb)
    color = color.to_numpy()
    ax.quiver(
        x, y, dx, dy,
        color=color,
        angles='xy', scale_units='xy', scale=1,
        alpha=0.5,
        width=0.0005,
    )
    l = k - 1 + 0.5
    if image is not None:
        ax.matshow(image, extent=[-l, l, -l, l])
    ax.set_xlim(-l-1, l+1)
    ax.set_ylim(-l-1, l+1)
    ax.set_aspect('equal')
    if save_path:
        plt.savefig(save_path,
                    dpi=1000,
                    # bbox_inches='tight',
                    # pad_inches=0.,
                    )
    else:
        plt.show()
    return ax


def main():
    arrows = pd.read_csv('../../data/arrowswcolor.csv')
    image = df_to_image(pd.read_csv('../../data/image.csv'))

    origin = [(64, 0), (-32, 0), (-16, 0), (-8, 0), (-4, 0), (-2, 0), (-1, 0), (-1, 0)]
    plot_path_over_image(origin,
                         arrows,
                         save_path='test.png',
                         image=image,
                         )


if __name__ == "__main__":
    main()
