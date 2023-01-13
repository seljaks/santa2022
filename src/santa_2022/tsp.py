import pandas as pd

from santa_2022.common import *
from santa_2022.post_processing import *


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
    points.append((0, -1))
    return points


def point_map_to_path(n):
    origin = [(64, 0), (-32, 0), (-16, 0), (-8, 0), (-4, 0), (-2, 0), (-1, 0), (-1, 0)]
    # setup if testing behavior on smaller version of image
    if n < 8:
        origin = origin[-n:]
        origin[0] = (abs(origin[0][0]), abs(origin[0][1]))

    assert origin[0][0] == 2 ** (n - 2)
    no_points = origin[0][0] * 2

    path = [origin]
    initial_position = [(64, 63), (-32, -32), (-16, -16), (-8, -8), (-4, -4), (-2, -2),
                        (-1, -1),
                        (-1, -1)]
    start_move = get_path_to_configuration(origin, initial_position)[1:-1]
    path.extend(start_move)
    for x, y in generate_point_map(no_points):
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
        assert candidate in get_n_link_rotations(config,
                                                 1), f'{config=}, {candidate=}, {get_position(candidate)=}'
        path.append(candidate)
    ending_position = path[-1]
    assert get_position(ending_position) == (0, -1)
    ending_move = get_path_to_configuration(ending_position, origin)[1:]
    path.extend(ending_move)
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
    """main link is (-L, x), others are (x, L)."""
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


def generate_lkh_file(number_of_links=8):
    assert 2 <= number_of_links <= 8
    origin = [(64, 0), (-32, 0), (-16, 0), (-8, 0), (-4, 0), (-2, 0), (-1, 0), (-1, 0)]
    image = df_to_image(pd.read_csv("../../data/image.csv"))
    if number_of_links < 8:
        origin = origin[-number_of_links:]
        origin[0] = (abs(origin[0][0]), abs(origin[0][1]))
        assert get_position(origin) == (0, 0)
        image = sliced_image(origin, image)


def main():
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


if __name__ == '__main__':
    main()
