from functools import *
from itertools import *
from math import sqrt

import numpy as np
import pandas as pd
from tqdm import tqdm
import matplotlib.pyplot as plt

import datetime
import logging


# Functions to map between cartesian coordinates and array indexes
def cartesian_to_array(x, y, shape):
    m, n = shape[:2]
    i = (n - 1) // 2 - y
    j = (n - 1) // 2 + x
    if i < 0 or i >= m or j < 0 or j >= n:
        raise ValueError("Coordinates not within given dimensions.")
    return i, j


def array_to_cartesian(i, j, shape):
    m, n = shape[:2]
    if i < 0 or i >= m or j < 0 or j >= n:
        raise ValueError("Coordinates not within given dimensions.")
    y = (n - 1) // 2 - i
    x = j - (n - 1) // 2
    return x, y


# Functions to map an image between array and record formats
def image_to_dict(image):
    image = np.atleast_3d(image)
    kv_image = {}
    for i, j in product(range(len(image)), repeat=2):
        kv_image[array_to_cartesian(i, j, image.shape)] = tuple(image[i, j])
    return kv_image


def image_to_df(image):
    return pd.DataFrame(
        [(x, y, r, g, b) for (x, y), (r, g, b) in image_to_dict(image).items()],
        columns=['x', 'y', 'r', 'g', 'b']
    )


def df_to_image(df):
    side = int(len(df) ** 0.5)  # assumes a square image
    return df.set_index(['x', 'y']).to_numpy().reshape(side, side, -1)


def get_position(config):
    return reduce(lambda p, q: (p[0] + q[0], p[1] + q[1]), config, (0, 0))


def rotate_link(vector, direction):
    x, y = vector
    if direction == 1:  # counter-clockwise
        if y >= x and y > -x:
            x -= 1
        elif x < y <= -x:
            y -= 1
        elif y <= x and y < -x:
            x += 1
        else:
            y += 1
    elif direction == -1:  # clockwise
        if y > x and y >= -x:
            x += 1
        elif x <= y < -x:
            y += 1
        elif y < x and y <= -x:
            x -= 1
        else:
            y -= 1
    return x, y


def rotate(config, i, direction):
    config = config.copy()
    config[i] = rotate_link(config[i], direction)
    return config


def get_square(link_length):
    link = (link_length, 0)
    coords = [link]
    for _ in range(8 * link_length - 1):
        link = rotate_link(link, direction=1)
        coords.append(link)
    return coords


def get_neighbors(config):
    nhbrs = (
        reduce(lambda x, y: rotate(x, *y), enumerate(directions), config)
        for directions in product((-1, 0, 1), repeat=len(config))
    )
    return list(filter(lambda c: c != config, nhbrs))


# Functions to compute the cost function

# Cost of reconfiguring the robotic arm: the square root of the number of links rotated
def reconfiguration_cost(from_config, to_config):
    # diffs = np.abs(np.asarray(from_config) - np.asarray(to_config)).sum(axis=1)
    # return np.sqrt(diffs.sum())
    return sqrt(sum(abs(x0 - x1) + abs(y0 - y1) for (x0, y0), (x1, y1) in
                    zip(from_config, to_config)))


# Cost of moving from one color to another: the sum of the absolute change in color components
def color_cost(from_position, to_position, image, color_scale=3.0):
    return np.abs(image[to_position] - image[from_position]).sum() * color_scale


def color_cost_from_config(from_config, to_config, image, color_scale=3.0):
    from_position = cartesian_to_array(*get_position(from_config), image.shape)
    to_position = cartesian_to_array(*get_position(to_config), image.shape)
    return color_cost(from_position, to_position,
                      image, color_scale=color_scale)


# Total cost of one step: the reconfiguration cost plus the color cost
def step_cost(from_config, to_config, image):
    from_position = cartesian_to_array(*get_position(from_config), image.shape)
    to_position = cartesian_to_array(*get_position(to_config), image.shape)
    return (
            reconfiguration_cost(from_config, to_config) +
            color_cost(from_position, to_position, image)
    )


def get_direction(u, v):
    """Returns the sign of the angle from u to v."""
    direction = np.sign(np.cross(u, v))
    if direction == 0 and np.dot(u, v) < 0:
        direction = 1
    return direction


# We don't use this elsewhere, but you might find it useful.
def get_angle(u, v):
    """Returns the angle (in degrees) from u to v."""
    return np.degrees(np.math.atan2(
        np.cross(u, v),
        np.dot(u, v),
    ))


def compress_path(path):
    r = [[] for _ in range(len(path[0]))]
    for p in path:
        for i in range(len(path[0])):
            if len(r[i]) == 0 or r[i][-1] != p[i]:
                r[i].append(p[i])
    mx = max([len(x) for x in r])

    for rr in r:
        while len(rr) < mx:
            rr.append(rr[-1])
    r = list(zip(*r))
    for i in range(len(r)):
        r[i] = list(r[i])
    return r


def get_path_to_point(config, point):
    """Find a path of configurations to `point` starting at `config`."""
    path = [config]
    # Rotate each link, starting with the largest, until the point can
    # be reached by the remaining links. The last link must reach the
    # point itself.
    for i in range(len(config)):
        link = config[i]
        base = get_position(config[:i])
        relbase = (point[0] - base[0], point[1] - base[1])
        position = get_position(config[:i + 1])
        relpos = (point[0] - position[0], point[1] - position[1])
        radius = reduce(lambda r, link: r + max(abs(link[0]), abs(link[1])),
                        config[i + 1:], 0)
        # Special case when next-to-last link lands on point.
        if radius == 1 and relpos == (0, 0):
            config = rotate(config, i, 1)
            if get_position(config) == point:  # Thanks @pgeiger
                path.append(config)
                break
            else:
                continue
        while np.max(np.abs(relpos)) > radius:
            direction = get_direction(link, relbase)
            config = rotate(config, i, direction)
            path.append(config)
            link = config[i]
            base = get_position(config[:i])
            relbase = (point[0] - base[0], point[1] - base[1])
            position = get_position(config[:i + 1])
            relpos = (point[0] - position[0], point[1] - position[1])
            radius = reduce(lambda r, link: r + max(abs(link[0]), abs(link[1])),
                            config[i + 1:], 0)
    path = compress_path(path)
    assert get_position(path[-1]) == point
    return path


def get_path_to_configuration(from_config, to_config):
    path = [from_config]
    config = from_config.copy()
    while config != to_config:
        for i in range(len(config)):
            config = rotate(config, i, get_direction(config[i], to_config[i]))
        path.append(config)
    path = compress_path(path)
    assert path[-1] == to_config
    return path


# Compute total cost of path over image
def total_cost(path, image):
    return reduce(
        lambda cost, pair: cost + step_cost(pair[0], pair[1], image),
        zip(path[:-1], path[1:]),
        0,
    )


def config_to_string(config):
    return ';'.join([' '.join(map(str, vector)) for vector in config])


# plotting
def plot_path_over_image(config, arrows, save_path=None, image=None, ax=None,
                         **figure_args):
    if ax is None:
        _, ax = plt.subplots(figsize=(8, 8), **figure_args)

    k = 2 ** (len(config) - 1) + 1
    x = arrows.loc[:, 'x'].to_numpy()
    y = arrows.loc[:, 'y'].to_numpy()
    dx = arrows.loc[:, 'dx'].to_numpy()
    dy = arrows.loc[:, 'dy'].to_numpy()

    replace_dict = {
        'down': 'b',
        'cheapest': 'g',
        'slow': 'r',
    }

    color = arrows.loc[:, 'path_type']
    color = color.replace(replace_dict)
    # color = color.apply(mcolors.to_rgb)
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


# my stuff
def get_neighbors_positions_costs(config, image):
    neighbors = get_neighbors(config)
    return list(zip(neighbors,
                    map(get_position, neighbors),
                    starmap(partial(step_cost, image=image),
                            zip(repeat(config), neighbors))
                    ))


def get_reachable_positions(neighbors):
    """Get configs that reach unique points"""
    reachable_positions = set()
    for move in neighbors:
        pos = get_position(move)
        if pos not in reachable_positions:
            reachable_positions.add(pos)
    return reachable_positions


def get_unvisited_neighbors(config, unvisited):
    nhbrs = (
        reduce(lambda x, y: rotate(x, *y), enumerate(directions), config)
        for directions in product((-1, 0, 1), repeat=len(config))
    )
    return list(filter(lambda c: c != config and get_position(c) in unvisited, nhbrs))


def get_all_cheapest_neighbors(config, image):
    neighbors = get_neighbors(config)
    cheapest = min(step_cost(config, c, image) for c in neighbors)
    return list(c for c in neighbors if step_cost(config, c, image) == cheapest)


def get_all_cheapest_unvisited_neighbors(config, unvisited, image):
    neighbors = get_unvisited_neighbors(config, unvisited)
    if len(neighbors) == 0:
        return list(), None
    cheapest = min(step_cost(config, c, image) for c in neighbors)
    return list(
        c for c in neighbors if step_cost(config, c, image) == cheapest), cheapest


def get_cheapest_next_unvisited_config(config, unvisited, image):
    """Goes two levels deep to break ties."""
    neighbors, cost = get_all_cheapest_unvisited_neighbors(config,
                                                           unvisited,
                                                           image)
    no_candidates = len(neighbors)
    if no_candidates == 0:
        return None
    elif no_candidates == 1:
        return neighbors[0]
    else:
        minimum = 1000.  # bigger than any possible step cost
        min_idx = 0
        for i, candidate in enumerate(neighbors):
            _, cand_cost = get_all_cheapest_unvisited_neighbors(candidate,
                                                                unvisited,
                                                                image)
            if cand_cost < minimum:
                minimum = cand_cost
                min_idx = i
        return neighbors[min_idx]


def get_visited_neighbor_with_cheapest_unvisited(config, unvisited, image):
    neighbors = get_neighbors(config)
    minimum = 1000.  # bigger than any possible step cost
    min_idx = None
    unvis_candidate = None
    for i, neigh in enumerate(neighbors):
        unvis, cost_to_unvisited = get_all_cheapest_unvisited_neighbors(neigh,
                                                                        unvisited,
                                                                        image)
        if cost_to_unvisited is None:
            continue
        else:
            cost_to_neighbor = step_cost(config, neigh, image)
            combined_cost = cost_to_neighbor + cost_to_unvisited
            if combined_cost < minimum:
                minimum = combined_cost
                min_idx = i
                unvis_candidate = unvis[0].copy()

    if min_idx is None:
        return None
    return neighbors[min_idx], unvis_candidate


def l1_dist(other, position):
    return abs(abs(other[0] - position[0]) + abs(other[1] - position[1]))


def get_cheapest_farthest_neighbor(config, image):
    position = get_position(config)
    l1_dist_partial = partial(l1_dist, position=position)

    candidates = max(get_neighbors_positions_costs(config, image),
                     key=lambda x: l1_dist_partial(other=x[1]))[:5]
    if isinstance(candidates[2], float):
        return candidates[0]
    else:
        return min(candidates, key=lambda x: x[-1])[0]


def get_closest_unvisited(config, unvisited):
    position = get_position(config)

    l1_dist_partial = partial(l1_dist, position=position)
    return min(unvisited, key=l1_dist_partial)


def get_cheapest_farther_neighbor_towards_unvisited(config, image, unvisited):
    closest_unvisited = get_closest_unvisited(config, unvisited)
    l1_dist_partial = partial(l1_dist, position=closest_unvisited)

    candidates = min(get_neighbors_positions_costs(config, image),
                     key=lambda x: l1_dist_partial(other=x[1]))[:5]

    if isinstance(candidates[2], float):
        return candidates[0]
    else:
        return min(candidates, key=lambda x: x[-1])[0]


def get_below(config):
    x, y = get_position(config)
    return x, y - 1


def rotate_n_links(config, link_idxs, directions):
    config = config.copy()
    assert len(link_idxs) == len(directions)
    for i, direction in zip(link_idxs, directions):
        config[i] = rotate_link(config[i], direction)
    return config


def get_n_link_rotations(config, n):
    rotations = []
    for comb in combinations(range(len(config)), r=n):
        for direction in product((-1, 1), repeat=n):
            rotations.append(rotate_n_links(config, comb, direction))
    return rotations


def get_one_two_link_neighbors(config):
    return get_n_link_rotations(config, 1) + get_n_link_rotations(config, 2)


def get_unvisited(neighbors, unvisited):
    return list(filter(lambda c: get_position(c) in unvisited, neighbors))


def sliced_image(config, image):
    n = config[0][0] * 2

    top_left = cartesian_to_array(*(-n, n), image.shape)
    bottom_right = cartesian_to_array(*(n, -n), image.shape)

    sliced = image[
             top_left[0]:bottom_right[0] + 1, top_left[1]:bottom_right[1] + 1, :]
    return sliced


def main(number_of_links=8):
    now = datetime.datetime.now()
    logging.basicConfig(filename=f"../../logging/{now}.log",
                        filemode='a',
                        format='%(asctime)s,%(msecs)d %(name)s %(levelname)s %(message)s',
                        datefmt='%H:%M:%S',
                        level=logging.DEBUG)

    submission = open(f'../../output/submissions/{now}.csv', 'a')
    submission.write(f'configuration\n')

    arrows = open(f'../../output/arrows/{now}.csv', 'a')
    arrows.write(f'x,y,dx,dy,path_type\n')

    df_image = pd.read_csv("../../data/image.csv")

    origin = [(64, 0), (-32, 0), (-16, 0), (-8, 0), (-4, 0), (-2, 0), (-1, 0), (-1, 0)]
    image = df_to_image(df_image)

    # setup if testing behavior on smaller version of image
    if number_of_links != len(origin):
        assert number_of_links < 8
        origin = origin[-number_of_links:]
        origin[0] = (abs(origin[0][0]), abs(origin[0][1]))
        assert get_position(origin) == (0, 0)
        image = sliced_image(origin, image)

    n = origin[0][0] * 2
    points = list(product(range(-n, n + 1), repeat=2))
    unvisited = set(points)

    path = [origin]
    submission.write(f'{config_to_string(origin)}\n')

    unvisited.remove(get_position(origin))
    for i in tqdm(range(len(points) - 1)):
        config = path[-1]
        below = get_below(config)

        one_two_rotations = get_one_two_link_neighbors(config)
        unvisited_rotations = get_unvisited(one_two_rotations, unvisited)

        below_candidates = list(
            filter(lambda c: get_position(c) == below, unvisited_rotations))

        if below_candidates:
            logging.info(f"{i}: {config=} moved down")
            next_config = min(below_candidates,
                              key=lambda x: step_cost(config, x, image))
            path.append(next_config)
            submission.write(f'{config_to_string(next_config)}\n')
            x, y = get_position(config)
            u, v = get_position(next_config)
            dx = u - x
            dy = v - y
            arrows.write(f'{x},{y},{dx},{dy},down\n')

        elif unvisited_rotations:
            next_config = min(unvisited_rotations,
                              key=lambda x: step_cost(config, x, image))
            logging.info(f"{i} {config=} moved cheapest")
            path.append(next_config)
            submission.write(f'{config_to_string(next_config)}\n')
            x, y = get_position(config)
            u, v = get_position(next_config)
            dx = u - x
            dy = v - y
            arrows.write(f'{x},{y},{dx},{dy},cheapest\n')

        else:
            closest_unvisited = get_closest_unvisited(config, unvisited)
            path_extension = get_path_to_point(config, closest_unvisited)[1:]
            path.extend(path_extension)
            logging.info(f"{i} {config=} moved slow")

            prev = config.copy()
            for c in path_extension:
                submission.write(f'{config_to_string(c)}\n')
                x, y = get_position(prev)
                u, v = get_position(c)
                dx = u - x
                dy = v - y
                arrows.write(f'{x},{y},{dx},{dy},slow\n')
                prev = c

        logging.info(f"{i}: visited {get_position(path[-1])}")
        unvisited.remove(get_position(path[-1]))

    assert unvisited == set()

    logging.debug(f"path length before going to origin: {len(path)}")
    path_extension = get_path_to_configuration(path[-1], origin)[1:]
    prev = path[-1].copy()
    for c in path_extension:
        submission.write(f'{config_to_string(c)}\n')
        x, y = get_position(prev)
        u, v = get_position(c)
        dx = u - x
        dy = v - y
        arrows.write(f'{x},{y},{dx},{dy},slow\n')
        prev = c

    path.extend(path_extension)
    assert path[0] == path[-1]
    logging.debug(f"path length after going to origin: {len(path)}")

    submission.close()
    arrows.close()

    arrcsv = pd.read_csv(f'../../output/arrows/{now}.csv')
    plot_path_over_image(origin, arrcsv, save_path=f'../../output/images/{now}.png',
                         image=image)


if __name__ == "__main__":
    main(number_of_links=8)
