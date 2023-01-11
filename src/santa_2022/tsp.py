from santa_2022.common import *


def generate_point_map(n):
    points = []
    # go down for n steps to reach edge (0, -n)
    for i in range(n + 1):
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

    assert len(points) == (2 * n + 1) ** 2
    points.append((0, -1))
    points.append((0, 0))
    return points


def point_map_to_path(n):
    origin = [(64, 0), (-32, 0), (-16, 0), (-8, 0), (-4, 0), (-2, 0), (-1, 0), (-1, 0)]
    # setup if testing behavior on smaller version of image
    if n < 8:
        origin = origin[-n:]
        origin[0] = (abs(origin[0][0]), abs(origin[0][1]))

    assert n == origin[0][0] * 2

    path = [origin]
    for x, y in generate_point_map(n)[1:]:
        config = path[-1]
        if x == 0 and y == 0:
            path.append(origin)
        elif x <= -1 and y <= 0:
            path.append(bot_left_point_to_config(x, y))
        elif x <= 0 and y >= 1:
            path.append(top_left_point_to_config(x, y))
        elif x >= 1 and y >= 0:
            path.append(top_right_point_to_config(x, y))
        elif x >= 0 and y <= -1:
            path.append(bot_right_point_to_config(x, y))
        else:
            raise ValueError('unreachable')
    return path


def bot_left_point_to_config(x, y):
    pass


def top_left_point_to_config(x, y):
    pass


def top_right_point_to_config(x, y):
    pass


def bot_right_point_to_config(x, y):
    pass
