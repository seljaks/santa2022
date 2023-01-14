import pytest

from santa_2022.tsp import *


def test_generate_point_map():
    expected = [
        (0, -1),
        (0, -2),
        (-1, -2),
        (-2, -2),
        (-2, -1),
        (-1, -1),
        (-1, 0),
        (-2, 0),
        (-2, 1),
        (-2, 2),
        (-1, 2),
        (-1, 1),
        (0, 1),
        (0, 2),
        (1, 2),
        (2, 2),
        (2, 1),
        (1, 1),
        (1, 0),
        (2, 0),
        (2, -1),
        (2, -2),
        (1, -2),
        (1, -1),
        (0, -1),
    ]
    result = generate_point_map(2)
    assert expected == result


def test_points_from_configs():
    expected = [
        [(1, 0), (-1, 0)],
        [(1, 0), (-1, -1)],
        [(1, -1), (-1, -1)],
        [(0, -1), (-1, -1)],
        [(-1, -1), (-1, -1)],
        [(-1, -1), (-1, 0)],
        [(0, -1), (-1, 0)],
        [(0, -1), (-1, 1)],
        [(-1, -1), (-1, 1)],
        [(-1, 0), (-1, 1)],
        [(-1, 1), (-1, 1)],
        [(-1, 1), (0, 1)],
        [(-1, 0), (0, 1)],
        [(-1, 0), (1, 1)],
        [(-1, 1), (1, 1)],
        [(0, 1), (1, 1)],
        [(1, 1), (1, 1)],
        [(1, 1), (1, 0)],
        [(0, 1), (1, 0)],
        [(0, 1), (1, -1)],
        [(1, 1), (1, -1)],
        [(1, 0), (1, -1)],
        [(1, -1), (1, -1)],
        [(1, -1), (0, -1)],
        [(1, 0), (0, -1)],
        [(1, 0), (-1, -1)],
        [(1, 0), (-1, 0)],
    ]
    points = generate_point_map(2)
    assert points == [get_position(c) for c in expected]


def test_point_map_to_path():
    expected = [
        [(1, 0), (-1, 0)],
        [(1, 0), (-1, -1)],
        [(1, -1), (-1, -1)],
        [(0, -1), (-1, -1)],
        [(-1, -1), (-1, -1)],
        [(-1, -1), (-1, 0)],
        [(0, -1), (-1, 0)],
        [(0, -1), (-1, 1)],
        [(-1, -1), (-1, 1)],
        [(-1, 0), (-1, 1)],
        [(-1, 1), (-1, 1)],
        [(-1, 1), (0, 1)],
        [(-1, 0), (0, 1)],
        [(-1, 0), (1, 1)],
        [(-1, 1), (1, 1)],
        [(0, 1), (1, 1)],
        [(1, 1), (1, 1)],
        [(1, 1), (1, 0)],
        [(0, 1), (1, 0)],
        [(0, 1), (1, -1)],
        [(1, 1), (1, -1)],
        [(1, 0), (1, -1)],
        [(1, -1), (1, -1)],
        [(1, -1), (0, -1)],
        [(1, 0), (0, -1)],
        [(1, 0), (-1, -1)],
        [(1, 0), (-1, 0)],
    ]
    result = point_map_to_path(2)
    assert result == expected


@pytest.mark.parametrize(
    ('x', 'y', 'n', 'expected'),
    (
            (-2, -2, 2, [(-1, -1), (-1, -1)]),
            (-2, -1, 2, [(-1, -1), (-1, 0)]),
            (-2, 0, 2, [(-1, -1), (-1, 1)]),
            (-1, -2, 2, [(0, -1), (-1, -1)]),
            (-1, -1, 2, [(0, -1), (-1, 0)]),
            (-1, 0, 2, [(0, -1), (-1, 1)]),

            (-1, 0, 8,
             [(63, -64), (-32, 32), (-16, 16), (-8, 8), (-4, 4), (-2, 2), (-1, 1),
              (-1, 1)]),
            (-4, 0, 8,
             [(60, -64), (-32, 32), (-16, 16), (-8, 8), (-4, 4), (-2, 2), (-1, 1),
              (-1, 1)]),
            (-4, -4, 8,
             [(60, -64), (-32, 32), (-16, 16), (-8, 8), (-4, 4), (-2, 0), (-1, 0),
              (-1, 0)]),
    )
)
def test_bot_left_point_to_config(x, y, n, expected):
    result = bot_left_point_to_config(x, y, n=n)
    assert result == expected


@pytest.mark.parametrize(
    ('x', 'y', 'n', 'expected'),
    (
            (2, 2, 2, [(1, 1), (1, 1)]),
            (2, 1, 2, [(1, 1), (1, 0)]),
            (2, 0, 2, [(1, 1), (1, -1)]),
            (1, 2, 2, [(0, 1), (1, 1)]),
            (1, 1, 2, [(0, 1), (1, 0)]),
            (1, 0, 2, [(0, 1), (1, -1)]),

            (1, 0, 8,
             [(-63, 64), (32, -32), (16, -16), (8, -8), (4, -4), (2, -2), (1, -1),
              (1, -1)]),
            (4, 0, 8,
             [(-60, 64), (32, -32), (16, -16), (8, -8), (4, -4), (2, -2), (1, -1),
              (1, -1)]),
            (4, 4, 8,
             [(-60, 64), (32, -32), (16, -16), (8, -8), (4, -4), (2, 0), (1, 0),
              (1, 0)]),
            (4, 120, 8,
             [(-60, 64), (32, 32), (16, 16), (8, 8), (4, 0), (2, 0), (1, 0),
              (1, 0)]),
    )
)
def test_top_right_point_to_config(x, y, n, expected):
    result = top_right_point_to_config(x, y, n=n)
    assert result == expected


@pytest.mark.parametrize(
    ('x', 'y', 'n', 'expected'),
    (
            (-2, 2, 2, [(-1, 1), (-1, 1)]),
            (-1, 2, 2, [(-1, 1), (0, 1)]),
            (0, 2, 2, [(-1, 1), (1, 1)]),
            (-2, 1, 2, [(-1, 0), (-1, 1)]),
            (-1, 1, 2, [(-1, 0), (0, 1)]),
            (0, 1, 2, [(-1, 0), (1, 1)]),
            (0, 1, 8,
             [(-64, -63), (32, 32), (16, 16), (8, 8), (4, 4), (2, 2), (1, 1),
              (1, 1)]),
            (0, 2, 8,
             [(-64, -62), (32, 32), (16, 16), (8, 8), (4, 4), (2, 2), (1, 1),
              (1, 1)]),
            (-2, 2, 8,
             [(-64, -62), (32, 32), (16, 16), (8, 8), (4, 4), (2, 2), (0, 1),
              (0, 1)]),
    )
)
def test_top_left_point_to_config(x, y, n, expected):
    result = top_left_point_to_config(x, y, n=n)
    assert result == expected


@pytest.mark.parametrize(
    ('x', 'y', 'n', 'expected'),
    (
            (2, -2, 2, [(1, -1), (1, -1)]),
            (1, -2, 2, [(1, -1), (0, -1)]),
            (0, -2, 2, [(1, -1), (-1, -1)]),
            (2, -1, 2, [(1, 0), (1, -1)]),
            (1, -1, 2, [(1, 0), (0, -1)]),
            (0, -1, 2, [(1, 0), (-1, -1)]),

            (0, -1, 4, [(4, 3), (-2, -2), (-1, -1), (-1, -1)]),
            (2, -2, 4, [(4, 2), (-2, -2), (0, -1), (0, -1)]),

            (0, -1, 8,
             [(64, 63), (-32, -32), (-16, -16), (-8, -8), (-4, -4), (-2, -2), (-1, -1),
              (-1, -1)]),
            (0, -2, 8,
             [(64, 62), (-32, -32), (-16, -16), (-8, -8), (-4, -4), (-2, -2), (-1, -1),
              (-1, -1)]),
            (2, -2, 8,
             [(64, 62), (-32, -32), (-16, -16), (-8, -8), (-4, -4), (-2, -2), (0, -1),
              (0, -1)]),
    )
)
def test_bot_right_point_to_config(x, y, n, expected):
    result = bot_right_point_to_config(x, y, n=n)
    assert result == expected


def test_xy_to_hash():
    no_rows = 5
    h = no_rows // 2
    points = []
    for y in reversed(range(-h, h + 1)):
        for x in range(-h, h + 1):
            points.append((cartesian_to_array(x, y, (no_rows, no_rows))))
    hashed = [xy_to_hash(x, y, no_rows) for x, y in points]
    assert hashed == list(range(no_rows ** 2))


def test_hast_to_xy():
    no_rows = 5
    hashed = list(range(no_rows ** 2))

    h = no_rows // 2
    points = []
    for y in reversed(range(-h, h + 1)):
        for x in range(-h, h + 1):
            points.append((x, y))

    unhashed = [array_to_cartesian(*hash_to_xy(i, no_rows), (no_rows, no_rows)) for i in
                hashed]
    assert unhashed == points


def test_hashed_points():
    expected = [1,
                2,
                7,
                8,
                3,
                4,
                5,
                10,
                9,
                13,
                14,
                19,
                24,
                23,
                18,
                17,
                22,
                21,
                20,
                15,
                16,
                12,
                11,
                6]
    result = generate_lkh_initial_tour(number_of_links=2)
    assert result == expected
