from multiprocessing import Pool

from tqdm import tqdm

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


def get_closest_unvisited(config, unvisited, dist_func):
    position = get_position(config)
    dist_partial = partial(dist_func, position=position)
    return min(unvisited, key=dist_partial)


def get_cheapest_farther_neighbor_towards_unvisited(config, image, unvisited):
    closest_unvisited = get_closest_unvisited(config, unvisited, l1_dist)
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


def get_unvisited(neighbors, unvisited):
    return list(filter(lambda c: get_position(c) in unvisited, neighbors))


def get_unvisited_with_costs(current_config, unvisited_neighbors, image):
    costs = [step_cost(current_config, c, image) for c in unvisited_neighbors]
    return list(zip(unvisited_neighbors, costs))


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


def search(corner, max_links, nearby_direction, nearby_threshold, tag, dedup=True,
           save=False, number_of_links=8):
    assert 2 <= number_of_links <= 8

    direction_functions = {
        'down': get_below,
        'up': get_above,
        'left': get_left,
        'right': get_right,
    }
    direction_function = direction_functions.get(nearby_direction)
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
    for _ in tqdm(range(len(points) - 1)):
        config = path[-1]

        slow_counter = one_step(config, unvisited, path, image, additional_info,
                                direction_function, corners, nearby_threshold,
                                max_links,
                                nearby_direction,
                                slow_counter)

        unvisited.remove(get_position(path[-1]))
    assert unvisited == set()
    path_extension = get_path_to_configuration(path[-1], origin)[1:]

    prev = path[-1].copy()
    for c in path_extension:
        log_additional_info(prev, c, 'return_to_origin', 10000,
                            additional_info)
        prev = c

    path.extend(path_extension)
    assert path[0] == path[-1]
    x, y = get_position(path[-1])
    additional_info.append([x, y, 0, 0, 'origin', pd.NA])
    df = pd.DataFrame(data=merge_path_and_information(path, additional_info),
                      columns=['configuration', 'x', 'y', 'dx', 'dy', 'move_type',
                               'slow_counter'])
    cost = total_cost(path, image)
    print(f'Total cost: {cost}')

    if save:
        file_name = f'{tag}-{cost}'
        save_descriptive_stats(df, file_name)
        save_submission(path, file_name)
        plot_path_over_image(origin, df,
                             save_path=f'../../output/images/{file_name}.png',
                             image=image)
    if dedup:
        for _ in range(2):
            path = run_remove(path)
            cost = total_cost(path, image)
            print(f'Deduplicated total cost: {cost}')

        if save:
            file_name = f'{tag}-{cost}-deduped'
            save_submission(path, file_name)
            df = path_to_arrows(path)
            plot_path_over_image(origin, df,
                                 save_path=f'../../output/images/{file_name}.png',
                                 image=image)

    return tag, cost


def one_step(config, unvisited, path, image, additional_info, direction_function,
             corners, nearby_threshold, max_links, move_direction, slow_counter):
    rotations = get_n_link_rotations(config, 1)
    unvisited_rotations = get_unvisited(rotations, unvisited)
    corner_candidates = list(
        filter(lambda x: get_position(x) in corners, unvisited_rotations))
    if corner_candidates:
        assert len(corner_candidates) == 1
        next_config = corner_candidates[0]
        path.append(next_config)
        log_additional_info(config, next_config, 'corner', pd.NA, additional_info)

    else:
        nearby = direction_function(config)
        nearby_candidates = list(
            filter(lambda x: get_position(x) == nearby, unvisited_rotations))

        if nearby_candidates:
            unvisited_with_costs = get_unvisited_with_costs(config, nearby_candidates,
                                                            image)
            cheapest_down, nearby_cost = min(unvisited_with_costs, key=lambda x: x[1])
        else:
            cheapest_down, nearby_cost = None, 1000.

        if nearby_cost <= nearby_threshold:
            next_config = cheapest_down
            path.append(next_config)
            log_additional_info(config, next_config, move_direction, pd.NA,
                                additional_info)

        else:
            move_costs = [sqrt(k) for k in range(1, max_links + 1)] + [0.]
            for n in range(2, max_links + 1):
                rotations += get_n_link_rotations(config, n)
                unvisited_rotations = get_unvisited(rotations, unvisited)
                if unvisited_rotations:
                    unvisited_with_costs = get_unvisited_with_costs(config,
                                                                    unvisited_rotations,
                                                                    image)
                    cheapest_unvisited, cost = min(unvisited_with_costs,
                                                   key=lambda x: x[1])
                    if cost <= move_costs[n]:
                        next_config = cheapest_unvisited
                        path.append(next_config)
                        log_additional_info(config, next_config, 'cheapest', pd.NA,
                                            additional_info)
                        break
            else:
                if unvisited_rotations:
                    unvisited_with_costs = get_unvisited_with_costs(config,
                                                                    unvisited_rotations,
                                                                    image)
                    cheapest_unvisited, cost = min(unvisited_with_costs,
                                                   key=lambda x: x[1])
                else:
                    cheapest_unvisited, cost = [], 1000.

                closest_unvisited = get_closest_unvisited(config, unvisited, l1_dist)
                path_extension = get_path_to_point(config, closest_unvisited)
                extension_cost = total_cost(path_extension, image)
                path_extension = path_extension[1:]

                if cost <= extension_cost:
                    next_config = cheapest_unvisited
                    path.append(next_config)
                    log_additional_info(config, next_config, 'cheapest', pd.NA,
                                        additional_info)

                else:
                    path.extend(path_extension)

                    prev = config.copy()
                    for c in path_extension:
                        log_additional_info(prev, c, 'slow', slow_counter,
                                            additional_info)
                        prev = c
                    slow_counter += 1
    return slow_counter


def log_additional_info(config, next_config, move_direction, slow_counter,
                        additional_info):
    x, y = get_position(config)
    u, v = get_position(next_config)
    dx = u - x
    dy = v - y
    additional_info.append([x, y, dx, dy, move_direction, slow_counter])


def grid_search():
    corner = [False]
    links = [8]
    directions = ['up']
    thresholds = [4.0, 4.25, 4.5, 4.75, 5.0, 5.25, 5.5, 5.75]
    grid = [(corner, max_links, nd, nt, f'{corner=}-{max_links=}-{nd}-{nt:4.2f}') for
            corner, max_links, nd, nt in
            product(corner, links, directions, thresholds)
            ]

    with Pool() as pool:
        results = pool.starmap(search, grid)
        for tag, cost in results:
            print(f'{tag} has deduped cost of {cost:.0f}')


def single_search():
    corner = False
    max_links = 8
    nearby_direction = 'down'
    nearby_threshold = 6.0
    save = True

    tag = f'{corner=}-{max_links=}-{nearby_direction}-{nearby_threshold:4.2f}'
    tag, cost = search(corner, max_links, nearby_direction, nearby_threshold, tag,
                       save=False, number_of_links=2)
    print(tag, cost)


def main():
    grid_search()


if __name__ == "__main__":
    main()
