from tqdm import tqdm

import datetime
import logging

from santa_2022.common import *
from santa_2022.post_processing import run_remove, save_submission, \
    save_descriptive_stats, path_to_arrows, plot_path_over_image


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


def l2_dist(other, position):
    return sqrt((other[0] - position[0]) ** 2 + (other[1] - position[1]) ** 2)


def lmax_dist(other, position):
    return max(abs(other[0] - position[0]), abs(other[1] - position[1]))


def get_cheapest_farthest_neighbor(config, image):
    position = get_position(config)
    l1_dist_partial = partial(l1_dist, position=position)

    candidates = max(get_neighbors_positions_costs(config, image),
                     key=lambda x: l1_dist_partial(other=x[1]))[:5]
    if isinstance(candidates[2], float):
        return candidates[0]
    else:
        return min(candidates, key=lambda x: x[-1])[0]


def get_closest_unvisited_l1(config, unvisited):
    position = get_position(config)

    dist_partial = partial(l1_dist, position=position)
    return min(unvisited, key=dist_partial)


def get_closest_unvisited_l2(config, unvisited):
    position = get_position(config)

    dist_partial = partial(l2_dist, position=position)
    return min(unvisited, key=dist_partial)


def get_closest_unvisited_lmax(config, unvisited):
    position = get_position(config)

    dist_partial = partial(lmax_dist, position=position)
    return min(unvisited, key=dist_partial)


def get_cheapest_farther_neighbor_towards_unvisited(config, image, unvisited):
    closest_unvisited = get_closest_unvisited_l1(config, unvisited)
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


def get_above(config):
    x, y = get_position(config)
    return x, y + 1


def get_left(config):
    x, y = get_position(config)
    return x - 1, y


def get_right(config):
    x, y = get_position(config)
    return x + 1, y


def get_diagonal(config):
    x, y = get_position(config)
    return x - 1, y - 1


def rotate_n_links(config, link_idxs, directions):
    config = config.copy()
    assert len(link_idxs) == len(directions)
    for i, direction in zip(link_idxs, directions):
        config[i] = rotate_link(config[i], direction)
    return config


def get_n_link_rotations(config, n):
    rotations = []
    for comb in reversed(list(combinations(range(len(config)), r=n))):
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


def merge_path_and_information(path, info):
    assert len(path) == len(info)
    return [[config_to_string(config)] + info for config, info in zip(path, info)]


def main(number_of_links=8):
    assert number_of_links <= 8

    now = datetime.datetime.now()
    logging.basicConfig(
        filename=f"../../logging/{now}.log",
        filemode='a',
        format='%(asctime)s,%(msecs)d %(name)s %(levelname)s %(message)s',
        datefmt='%H:%M:%S',
        level=logging.ERROR
    )
    corner = False
    max_links = 8
    move_direction = 'down'

    direction_functions = {
        'down': get_below,
        'up': get_above,
        'left': get_left,
        'right': get_right,
    }
    direction_function = direction_functions.get(move_direction)

    df_image = pd.read_csv("../../data/image.csv")
    image = df_to_image(df_image)

    origin = [(64, 0), (-32, 0), (-16, 0), (-8, 0), (-4, 0), (-2, 0), (-1, 0), (-1, 0)]

    # setup if testing behavior on smaller version of image
    if number_of_links < 8:
        origin = origin[-number_of_links:]
        origin[0] = (abs(origin[0][0]), abs(origin[0][1]))
        assert get_position(origin) == (0, 0)
        image = sliced_image(origin, image)

    n = origin[0][0] * 2
    points = list(product(range(-n, n + 1), repeat=2))
    unvisited = set(points)

    path = [origin]
    additional_info = []

    slow_counter = 0

    if corner:
        corners = [(-128, -128), (-128, 128), (128, -128), (128, 128)]
    else:
        corners = []

    unvisited.remove(get_position(origin))
    for i in tqdm(range(len(points) - 1)):
        config = path[-1]
        x, y = get_position(config)

        nearby = direction_function(config)

        rotations = get_n_link_rotations(config, 1)
        unvisited_rotations = get_unvisited(rotations, unvisited)

        corner_candidates = list(
            filter(lambda x: get_position(x) in corners, unvisited_rotations))

        nearby_candidates = list(
            filter(lambda x: get_position(x) == nearby, unvisited_rotations))

        if corner_candidates:
            logging.info(f"{i}: {config=} moved to corner")
            next_config = min(corner_candidates,
                              key=lambda x: step_cost(config, x, image))
            path.append(next_config)

            u, v = get_position(next_config)
            dx = u - x
            dy = v - y
            additional_info.append([x, y, dx, dy, 'corner', pd.NA])

        elif nearby_candidates:
            logging.info(f"{i}: {config=} moved down")
            next_config = min(nearby_candidates,
                              key=lambda x: step_cost(config, x, image))
            path.append(next_config)

            u, v = get_position(next_config)
            dx = u - x
            dy = v - y
            additional_info.append([x, y, dx, dy, move_direction, pd.NA])

        else:
            for n in range(2, max_links+1):
                rotations += get_n_link_rotations(config, n)
                unvisited_rotations = get_unvisited(rotations, unvisited)
                if unvisited_rotations:
                    break
            else:
                unvisited_rotations = []

            if unvisited_rotations:
                next_config = min(unvisited_rotations,
                                  key=lambda x: step_cost(config, x, image))
                logging.info(f"{i} {config=} moved cheapest")
                path.append(next_config)

                u, v = get_position(next_config)
                dx = u - x
                dy = v - y
                additional_info.append([x, y, dx, dy, 'cheapest', pd.NA])
            else:
                closest_unvisited = get_closest_unvisited_l1(config, unvisited)
                path_extension = get_path_to_point(config, closest_unvisited)[1:]
                path.extend(path_extension)
                logging.info(f"{i} {config=} moved slow")

                prev = config.copy()
                for c in path_extension:
                    x, y = get_position(prev)
                    u, v = get_position(c)
                    dx = u - x
                    dy = v - y
                    additional_info.append([x, y, dx, dy, 'slow', slow_counter])
                    prev = c
                slow_counter += 1

        logging.info(f"{i}: visited {get_position(path[-1])}")
        unvisited.remove(get_position(path[-1]))

    assert unvisited == set()

    logging.debug(f"path length before going to origin: {len(path)}")
    path_extension = get_path_to_configuration(path[-1], origin)[1:]
    prev = path[-1].copy()
    for c in path_extension:
        x, y = get_position(prev)
        u, v = get_position(c)
        dx = u - x
        dy = v - y
        additional_info.append([x, y, dx, dy, 'return_to_origin', slow_counter])
        prev = c

    path.extend(path_extension)
    assert path[0] == path[-1]

    x, y = get_position(path[-1])
    additional_info.append([x, y, 0, 0, 'origin', pd.NA])

    logging.debug(f"path length after going to origin: {len(path)}")

    df = pd.DataFrame(data=merge_path_and_information(path, additional_info),
                      columns=['configuration', 'x', 'y', 'dx', 'dy', 'move_type',
                               'slow_counter'])

    cost = total_cost(path, image)
    print(f'Total cost: {cost}')

    file_name = f'{corner=}_{max_links=}_{move_direction}_{cost}'
    save_descriptive_stats(df, file_name)
    save_submission(path, file_name)
    plot_path_over_image(origin, df, save_path=f'../../output/images/{file_name}.png',
                         image=image)

    dedup = True
    if dedup:
        for _ in range(2):
            path = run_remove(path)
            cost = total_cost(path, image)
            print(f'Deduplicated total cost: {cost}')
        file_name = f'{corner=}_{max_links=}_{move_direction}_{cost}_deduped'
        save_submission(path, file_name)
        df = path_to_arrows(path)
        plot_path_over_image(origin, df,
                             save_path=f'../../output/images/{file_name}.png',
                             image=image)


if __name__ == "__main__":
    main(number_of_links=8)
