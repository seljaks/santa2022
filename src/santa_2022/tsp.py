import pandas as pd

from santa_2022.common import *
from santa_2022.post_processing import *

from tqdm import tqdm
from math import isqrt
from math import log2


def generate_point_map(n):
    """Makes a list of points from without (0, 0), starts at (0, -1) and ends there"""
    points = []
    # go down for n steps to reach edge (0, -n)
    for i in range(1, n + 1):
        p = (0, -i)
        points.append(p)

    for y in range(-n, 1):
        for x in range(1, n + 1):
            if y % 2 == 0:
                p = (-x, y)
            else:
                p = (-(n + 1) + x, y)
            points.append(p)

    for y in range(-n, 1):
        for x in range(1, n + 1):
            if y % 2 == 0:
                p = (y, x)
            else:
                p = (y, (n + 1) - x)
            points.append(p)

    for y in range(n + 1):
        for x in range(1, n + 1):
            if y % 2 == 0:
                p = (x, n - y)
            else:
                p = ((n + 1) - x, n - y)
            points.append(p)

    for y in range(n):
        for x in range(1, n + 1):
            if y % 2 == 0:
                p = (n - y, -x)
            else:
                p = (n - y, -(n + 1) + x)
            points.append(p)

    assert len(points) == (2 * n + 1) ** 2 - 1
    return points


def point_map_to_path(n, point_map=None):
    origin = [(64, 0), (-32, 0), (-16, 0), (-8, 0), (-4, 0), (-2, 0), (-1, 0), (-1, 0)]
    # setup if testing behavior on smaller version of image
    if n < 8:
        origin = origin[-n:]
        origin[0] = (abs(origin[0][0]), abs(origin[0][1]))

    assert origin[0][0] == 2 ** (n - 2)
    no_points = origin[0][0] * 2

    if point_map is None:
        point_map = generate_point_map(no_points)

    path = [origin]
    # initial_position = bot_right_point_to_config(*point_map[0], n=n)
    # start_move = get_path_to_configuration(origin, initial_position)[1:-1]
    # path.extend(start_move)
    for x, y in tqdm(point_map):
        config = path[-1]
        if x == 0 and y == 0:
            candidate = origin.copy()
        elif x <= -1 and y <= 0:
            candidate = bot_left_point_to_config(x, y, n=n)
        elif x <= 0 and y >= 1:
            candidate = top_left_point_to_config(x, y, n=n)
        elif x >= 1 and y >= 0:
            candidate = top_right_point_to_config(x, y, n=n)
        elif x >= 0 and y <= -1:
            candidate = bot_right_point_to_config(x, y, n=n)
        else:
            candidate = None
            raise ValueError('unreachable')
        if config != origin:
            # assert candidate in get_n_link_rotations(config, 1), f'{config=}, {candidate=}, {get_position(candidate)=}'
            if candidate not in get_n_link_rotations(config, 1) + get_n_link_rotations(
                    config, 2):
                raise ValueError('weird path')
                # extension = get_path_to_configuration(config, candidate)[1:]
                # path.extend(extension)
            # else:
        path.append(candidate)

    # ending_position = path[-1]
    # ending_move = get_path_to_configuration(ending_position, origin)[1:]
    # path.extend(ending_move)
    return path


def bot_left_point_to_config(x, y, n=8):
    """main link is (x, -L), others are (-L, x)."""
    assert x <= -1 and y <= 0, "This function is only for the bot left quadrant"
    assert n >= 2, "Not enough links"
    r = 2 ** (n - 2)
    config = [(x + r, -r)]
    y = y - config[0][1]
    while r > 1:
        r = r // 2
        free_arm = np.clip(y, -r, r)
        config.append((-r, free_arm))
        y -= free_arm
    free_arm = np.clip(y, -r, r)
    config.append((-r, free_arm))
    assert y == free_arm, f'{y=}, {free_arm=}'
    return config


def top_left_point_to_config(x, y, n=8):
    """main link is (-L, x), others are (x, L)."""
    assert x <= 0 and y >= 1, "This function is only for the bot left quadrant"
    assert n >= 2, "Not enough links"
    r = 2 ** (n - 2)
    config = [(-r, y - r)]
    x = x - config[0][0]
    while r > 1:
        r = r // 2
        free_arm = np.clip(x, -r, r)
        config.append((free_arm, r))
        x -= free_arm
    free_arm = np.clip(x, -r, r)
    config.append((free_arm, r))
    assert x == free_arm, f'{x=}, {free_arm=}'
    return config


def top_right_point_to_config(x, y, n=8):
    """main link is (x, L), others are (L, x)."""
    assert x >= 1 and y >= 0, "This function is only for the bot left quadrant"
    assert n >= 2, "Not enough links"
    r = 2 ** (n - 2)
    config = [(x - r, r)]
    y = y - config[0][1]
    while r > 1:
        r = r // 2
        free_arm = np.clip(y, -r, r)
        config.append((r, free_arm))
        y -= free_arm
    free_arm = np.clip(y, -r, r)
    config.append((r, free_arm))
    assert y == free_arm, f'{y=}, {free_arm=}'
    return config


def bot_right_point_to_config(x, y, n=8):
    """main link is (L, x), others are (x, -L)."""
    assert x >= 0 and y <= -1, "This function is only for the bot left quadrant"
    assert n >= 2, "Not enough links"
    r = 2 ** (n - 2)
    config = [(r, y + r)]
    x = x - config[0][0]
    while r > 1:
        r = r // 2
        free_arm = np.clip(x, -r, r)
        config.append((free_arm, -r))
        x -= free_arm
    free_arm = np.clip(x, -r, r)
    config.append((free_arm, -r))
    assert x == free_arm, f'{x=}, {free_arm=}'
    return config


def hash_to_xy(i, no_rows):
    return i // no_rows, i % no_rows


def xy_to_hash(x, y, no_rows):
    return x * no_rows + y


def reconf_cost_from_position(from_position, to_position):
    return np.sqrt((np.abs(np.asarray(from_position) - np.asarray(to_position))).sum())


def tsp_cost(from_position, to_position, image, color_scale=3.0):
    color_part = color_cost(from_position, to_position, image, color_scale=color_scale)
    reconf_part = reconf_cost_from_position(from_position, to_position)
    assert reconf_part <= sqrt(2)
    return color_part + reconf_part


def generate_lkh_file(number_of_links=8):
    assert 2 <= number_of_links <= 8
    origin = [(64, 0), (-32, 0), (-16, 0), (-8, 0), (-4, 0), (-2, 0), (-1, 0), (-1, 0)]
    image = df_to_image(pd.read_csv("../../data/image.csv"))
    if number_of_links < 8:
        origin = origin[-number_of_links:]
        origin[0] = (abs(origin[0][0]), abs(origin[0][1]))
        assert get_position(origin) == (0, 0)
        image = sliced_image(origin, image)

    cartesian_limit = origin[0][0] * 2
    no_rows = origin[0][0] * 4 + 1
    assert no_rows == image.shape[0] == image.shape[1]
    print(no_rows)
    no_nodes = no_rows ** 2 - 1
    hashed_origin = xy_to_hash(*cartesian_to_array(0, 0, image.shape), no_rows)
    print(hashed_origin)
    edge_count = 0
    with open(f'santa2022-{number_of_links}-edge_list-diagonal.tsp', 'w') as file:
        file.write(f'NAME : santa2022-{number_of_links}\n')
        file.write('TYPE : TSP\n')
        file.write('COMMENT : TEST\n')
        file.write(f'DIMENSION : {no_nodes}\n')
        file.write('EDGE_WEIGHT_TYPE : SPECIAL\n')
        file.write('EDGE_DATA_FORMAT : EDGE_LIST\n')
        file.write('EDGE_DATA_SECTION\n')
        for i in tqdm(range(no_nodes)):
            x, y = hash_to_xy(i, no_rows)
            cart_x, cart_y = array_to_cartesian(x, y, image.shape)
            ignore_right = False
            ignore_bottom = False
            ignore_bot_left = False
            ignore_bot_right = False

            if cart_x == 0 and cart_y == 0:
                continue
            if cart_x == 0 and cart_y == 1:
                ignore_bottom = True
                ignore_bot_right = True
                ignore_bot_left = True
            if cart_x == -1 and cart_y == 0:
                ignore_right = True
                ignore_bot_right = True
            if cart_x == -1 and cart_y == 1:
                ignore_bottom = True
                ignore_bot_right = True
            if cart_x == 1 and cart_y == 1:
                ignore_bot_left = True
            if cart_x == 1 and cart_y == 0:
                ignore_bottom = True
                ignore_bot_left = True
                ignore_bot_right = True
            if cart_x == -1 and cart_y == -1:
                ignore_right = True
            if cart_x == 0 and cart_y == -1:
                ignore_bot_left = True

            if cart_x == 0 and 0 < cart_y < cartesian_limit:
                ignore_right = True
                ignore_bot_right = True
            if cart_x == -1 and 0 > cart_y > -(cartesian_limit - 1):
                ignore_right = True
                ignore_bot_right = True
            if cart_x == -1 and cart_y == -(cartesian_limit - 1):
                ignore_right = True
            if cart_x == 1 and 0 <= cart_y <= cartesian_limit:
                ignore_bot_left = True
            if cart_x == 0 and 0 > cart_y >= -cartesian_limit:
                ignore_bot_left = True

            if cart_y == 1 and 0 > cart_x > -(cartesian_limit - 1):
                ignore_bottom = True
                ignore_bot_left = True
                ignore_bot_right = True
            if cart_y == 1 and cart_x == -(cartesian_limit - 1):
                ignore_bottom = True
                ignore_bot_right = True
            if cart_y == 1 and cart_x == -cartesian_limit:
                ignore_bot_right = True
            if cart_y == 0 and 0 < cart_x < cartesian_limit:
                ignore_bottom = True
                ignore_bot_left = True
                ignore_bot_right = True

            if cart_x == -cartesian_limit:
                ignore_bot_left = True
            if cart_x == cartesian_limit:
                ignore_right = True
                ignore_bot_right = True
            if cart_y == -cartesian_limit:
                ignore_bottom = True
                ignore_bot_left = True
                ignore_bot_right = True

            hashed_pos = i
            if hashed_pos <= hashed_origin:
                hashed_pos += 1

            if ignore_right and ignore_bottom and ignore_bot_left and ignore_bot_right:
                pass

            edges = [ignore_right,
                     ignore_bottom,
                     ignore_bot_left,
                     ignore_bot_right]
            nodes = [(x, y + 1),
                     (x + 1, y),
                     (x + 1, y - 1),
                     (x + 1, y + 1)]

            for j, edge in enumerate(edges):
                if not edge:
                    edge_count += 1
                    node = nodes[j]
                    hashed_node = xy_to_hash(*node, no_rows)
                    if hashed_node <= hashed_origin:
                        hashed_node += 1
                    cost = tsp_cost((x, y), node, image) * 1000
                    file.write(f'{hashed_pos} {hashed_node} {int(cost)}\n')

            # if ignore_bottom:
            #     right = x, y + 1
            #     hashed_right = xy_to_hash(*right, no_rows)
            #     if hashed_right <= hashed_origin:
            #         hashed_right += 1
            #     cost_right = tsp_cost((x, y), right, image) * 1000000
            #     file.write(f'{hashed_pos} {hashed_right} {int(cost_right)}\n')
            # elif ignore_right:
            #     bot = x + 1, y
            #     hashed_bot = xy_to_hash(*bot, no_rows)
            #     if hashed_bot <= hashed_origin:
            #         hashed_bot += 1
            #     cost_bot = tsp_cost((x, y), bot, image) * 1000000
            #     file.write(f'{hashed_pos} {hashed_bot} {int(cost_bot)}\n')
            # else:
            #     right = x, y + 1
            #     hashed_right = xy_to_hash(*right, no_rows)
            #     if hashed_right <= hashed_origin:
            #         hashed_right += 1
            #     cost_right = tsp_cost((x, y), right, image) * 1000000
            #     file.write(f'{hashed_pos} {hashed_right} {int(cost_right)}\n')
            #
            #     bot = x + 1, y
            #     hashed_bot = xy_to_hash(*bot, no_rows)
            #     if hashed_bot <= hashed_origin:
            #         hashed_bot += 1
            #     cost_bot = tsp_cost((x, y), bot, image) * 1000000
            #     file.write(f'{hashed_pos} {hashed_bot} {int(cost_bot)}\n')
        print(edge_count)
        file.write('EOF')


def generate_lkh_initial_tour(number_of_links=8):
    assert 2 <= number_of_links <= 8
    origin = [(64, 0), (-32, 0), (-16, 0), (-8, 0), (-4, 0), (-2, 0), (-1, 0), (-1, 0)]
    # setup if testing behavior on smaller version of image
    if number_of_links < 8:
        origin = origin[-number_of_links:]
        origin[0] = (abs(origin[0][0]), abs(origin[0][1]))

    assert origin[0][0] == 2 ** (number_of_links - 2)
    no_points = origin[0][0] * 2
    no_rows = origin[0][0] * 4 + 1
    shape = no_rows, no_rows

    points = generate_point_map(no_points)[:-1]
    hashed_origin = xy_to_hash(*cartesian_to_array(0, 0, shape), no_rows)
    hashed_points = (
        xy_to_hash(*cartesian_to_array(*point, shape), no_rows) for point
        in points)
    hashed_points = [hp + 1 if hp <= hashed_origin else hp for hp in hashed_points]

    one_idx = hashed_points.index(1)
    ordered_points = hashed_points[one_idx:] + hashed_points[:one_idx]

    return ordered_points


def lkh_solution_to_path(file_name):
    hashed_solution = []
    tour_section = False
    tour_length = None
    with open(file_name, 'r') as file:
        for line in file.readlines():
            stripped = line.strip()
            if stripped.startswith('DIMENSION'):
                tour_length = int(stripped.partition(':')[2].strip())
            if stripped == '-1':
                break
            if tour_section:
                hashed_solution.append(int(stripped))
            if stripped == 'TOUR_SECTION':
                tour_section = True
                continue
    assert tour_length == len(hashed_solution)

    image_size = tour_length + 1
    no_rows = isqrt(image_size)
    assert image_size == no_rows ** 2
    image_shape = no_rows, no_rows
    hashed_origin = xy_to_hash(*cartesian_to_array(0, 0, image_shape), no_rows)
    print(hashed_origin)

    origin_zz = (isqrt(image_size) - 1) // 4
    check_n = log2(origin_zz)
    assert check_n == int(check_n)
    number_of_links = int(check_n) + 2

    hashed_solution = [hp - 1 if hp <= hashed_origin else hp for hp in hashed_solution]
    check_hash = list(range(image_size))
    check_hash.remove(hashed_origin)
    assert sorted(hashed_solution) == sorted(
        check_hash), f'{len(hashed_solution)=}, {len(check_hash)=}, {min(hashed_solution), max(hashed_solution)}, {len(set(hashed_solution))=}'

    cartesian_solution = [array_to_cartesian(*hash_to_xy(hp, no_rows), image_shape) for
                          hp in hashed_solution]

    start_index = cartesian_solution.index((0, -1))
    ordered_cartesian = cartesian_solution[start_index:] + cartesian_solution[
                                                           :start_index]

    check_cartesian = generate_point_map(origin_zz * 2)
    assert sorted(ordered_cartesian) == sorted(
        check_cartesian), f'{len(check_cartesian)=}, {len(ordered_cartesian)=}'

    path = point_map_to_path(number_of_links, ordered_cartesian)

    return path, number_of_links


def single_search():
    df_image = pd.read_csv("../../data/image.csv")
    image = df_to_image(df_image)
    origin = [(64, 0), (-32, 0), (-16, 0), (-8, 0), (-4, 0), (-2, 0), (-1, 0), (-1, 0)]

    path = point_map_to_path(8)
    print(total_cost(path, image))

    file_name = 'initial_tsp_approach'
    save_submission(path, file_name)
    df = path_to_arrows(path)
    plot_path_over_image(origin, df,
                         save_path=f'../../output/images/{file_name}.png',
                         image=image)


def lkh_search():
    df_image = pd.read_csv("../../data/image.csv")
    image = df_to_image(df_image)
    origin = [(64, 0), (-32, 0), (-16, 0), (-8, 0), (-4, 0), (-2, 0), (-1, 0), (-1, 0)]

    lkh_output = 'santa2022-8-output-diag.txt'
    path, number_of_links = lkh_solution_to_path(lkh_output)

    if number_of_links < 8:
        origin = origin[-number_of_links:]
        origin[0] = (abs(origin[0][0]), abs(origin[0][1]))
        assert get_position(origin) == (0, 0)
        image = sliced_image(origin, image)

    print(total_cost(path, image))

    # path = run_remove(path)
    # path = run_remove(path)
    # print(total_cost(path, image))

    file_name = lkh_output[:-4] + '-no_start_or_finish-latest'
    save_submission(path, file_name)
    df = path_to_arrows(path)
    plot_path_over_image(origin, df,
                         save_path=f'../../output/images/{file_name}.png',
                         image=image)


def main():
    lkh_search()


if __name__ == '__main__':
    main()
