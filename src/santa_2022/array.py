import numpy as np
from itertools import *


def powerset(iterable):
    "powerset([1,2,3]) --> () (1,) (2,) (3,) (1,2) (1,3) (2,3) (1,2,3)"
    s = list(iterable)
    return chain.from_iterable(combinations(s, r) for r in range(len(s)+1))


def config_to_array(config):
    return np.asarray(config, dtype=np.int8)


def array_get_position(config):
    return config.sum(axis=0)


def array_rotate_link(link, direction):
    if direction == 1:  # counter-clockwise
        if link[1] >= link[0] and link[1] > -link[0]:
            link[0] -= 1
        elif link[0] < link[1] <= -link[0]:
            link[1] -= 1
        elif link[1] <= link[0] and link[1] < -link[0]:
            link[0] += 1
        else:
            link[1] += 1
    elif direction == -1:  # clockwise
        if link[1] > link[0] and link[1] >= -link[0]:
            link[0] += 1
        elif link[0] <= link[1] < -link[0]:
            link[1] += 1
        elif link[1] < link[0] and link[1] <= -link[0]:
            link[0] -= 1
        else:
            link[1] -= 1
    return link


def array_rotate(config, link_ids, directions):
    config = config.copy()
    for i, direc in zip(link_ids, directions):
        config[i] = array_rotate_link(config[i], direc)
    return config


def array_get_neighbors(config):
    no_links, point = config.shape
    neighbors = np.zeros(shape=(3**no_links, no_links, point), dtype=np.int8)
    outer_index = 0
    for i, comb in enumerate(powerset(range(no_links))):
        if outer_index == 0:
            neighbors[0] = config
            outer_index += 1
            continue
        no_combs = len(comb)
        for j, directions in enumerate(product((1, -1), repeat=no_combs)):
            neighbors[outer_index+j] = array_rotate(config, comb, directions)
        outer_index += 2**no_combs
    return neighbors

