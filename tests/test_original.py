import pytest

from santa_2022.original import *


def test_get_position(origin):
    expected = (0, 0)
    assert get_position(origin) == expected


@pytest.mark.parametrize(
    ("direction", "link", "expected"),
    (
            (1, 0, [(2, 1), (-1, 0), (-1, 0)]),
            (-1, 0, [(2, -1), (-1, 0), (-1, 0)]),
            (1, 1, [(2, 0), (-1, -1), (-1, 0)]),
            (-1, 1, [(2, 0), (-1, 1), (-1, 0)]),
            (1, 2, [(2, 0), (-1, 0), (-1, -1)]),
            (-1, 2, [(2, 0), (-1, 0), (-1, 1)]),
    )
)
def test_rotate(direction, link, expected, mini_arm):
    rotated = rotate(mini_arm, link, direction)
    assert rotated == expected


def test_get_neighbors(mini_arm, mini_arm_neighbors):
    expected = mini_arm_neighbors
    l = len(mini_arm)
    assert len(expected) == 3 ** l
    neighbors = sorted(get_neighbors(mini_arm))
    assert len(neighbors) == 3 ** l - 1
    expected = sorted(expected[1:])
    assert expected == neighbors


def test_get_all_cheapest_neighbor(mini_arm, image):
    mini_image = sliced_image(mini_arm, image)
    expected = [
        [(2, -1), (-1, 0), (-1, 0)],
        [(2, 0), (-1, 0), (-1, -1)],
        [(2, 0), (-1, -1), (-1, 0)],
    ]
    expected.sort()
    assert expected == sorted(
        get_all_cheapest_neighbors(mini_arm, mini_image))


def test_get_all_cheapest_unvisited_neighbor(mini_arm, image):
    mini_image = sliced_image(mini_arm, image)

    n = mini_arm[0][0] * 2
    points = list(product(range(-n, n + 1), repeat=2))
    unvisited = set(points)
    unvisited.remove((0, 0))
    unvisited.remove((0, -1))

    expected = [
        [(2, 0), (-1, 1), (-1, 0)],
        [(2, 0), (-1, 0), (-1, 1)],
        [(2, 1), (-1, 0), (-1, 0)],
    ]
    expected.sort()
    assert expected == sorted(
        get_all_cheapest_unvisited_neighbors(mini_arm, unvisited, mini_image)[0])


def test_get_cheapest_next_config(mini_arm, image):
    mini_image = sliced_image(mini_arm, image)

    n = mini_arm[0][0] * 2
    points = list(product(range(-n, n + 1), repeat=2))
    unvisited = set(points)
    unvisited.remove((0, 0))

    expected = [(2, 0), (-1, 0), (-1, -1)]

    assert expected == get_cheapest_next_unvisited_config(mini_arm, unvisited, image)


def test_get_one_link_rotations(mini_arm):
    expected = [
        # rotate link 0
        [(2, 1), (-1, 0), (-1, 0)],
        [(2, -1), (-1, 0), (-1, 0)],
        # rotate link 1
        [(2, 0), (-1, 1), (-1, 0)],
        [(2, 0), (-1, -1), (-1, 0)],
        # rotate link 2
        [(2, 0), (-1, 0), (-1, 1)],
        [(2, 0), (-1, 0), (-1, -1)],
    ]

    one_link_neighbors = get_n_link_rotations(mini_arm, 1)
    assert sorted(expected) == sorted(one_link_neighbors)


def test_get_two_link_rotations(mini_arm):
    expected = [
        # rotate link 0, 1
        [(2, 1), (-1, 1), (-1, 0)],
        [(2, 1), (-1, -1), (-1, 0)],
        [(2, -1), (-1, 1), (-1, 0)],
        [(2, -1), (-1, -1), (-1, 0)],
        # rotate link 0, 2
        [(2, 1), (-1, 0), (-1, 1)],
        [(2, 1), (-1, 0), (-1, -1)],
        [(2, -1), (-1, 0), (-1, 1)],
        [(2, -1), (-1, 0), (-1, -1)],
        # rotate link 1, 2
        [(2, 0), (-1, 1), (-1, 1)],
        [(2, 0), (-1, 1), (-1, -1)],
        [(2, 0), (-1, -1), (-1, 1)],
        [(2, 0), (-1, -1), (-1, -1)],
    ]

    two_link_neighbors = get_n_link_rotations(mini_arm, 2)
    assert sorted(expected) == sorted(two_link_neighbors)


def test_one_two_link_neighbors(mini_arm):
    expected = [  # rotate link 0
        [(2, 1), (-1, 0), (-1, 0)],
        [(2, -1), (-1, 0), (-1, 0)],
        # rotate link 1
        [(2, 0), (-1, 1), (-1, 0)],
        [(2, 0), (-1, -1), (-1, 0)],
        # rotate link 2
        [(2, 0), (-1, 0), (-1, 1)],
        [(2, 0), (-1, 0), (-1, -1)],
        # rotate link 0, 1
        [(2, 1), (-1, 1), (-1, 0)],
        [(2, 1), (-1, -1), (-1, 0)],
        [(2, -1), (-1, 1), (-1, 0)],
        [(2, -1), (-1, -1), (-1, 0)],
        # rotate link 0, 2
        [(2, 1), (-1, 0), (-1, 1)],
        [(2, 1), (-1, 0), (-1, -1)],
        [(2, -1), (-1, 0), (-1, 1)],
        [(2, -1), (-1, 0), (-1, -1)],
        # rotate link 1, 2
        [(2, 0), (-1, 1), (-1, 1)],
        [(2, 0), (-1, 1), (-1, -1)],
        [(2, 0), (-1, -1), (-1, 1)],
        [(2, 0), (-1, -1), (-1, -1)],
    ]

    assert sorted(expected) == sorted(get_one_two_link_neighbors(mini_arm))
