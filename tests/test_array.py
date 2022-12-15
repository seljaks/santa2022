import numpy as np
import pytest

from santa_2022.array import *


def test_config_to_array(mini_arm):
    expected = np.array([
        (2, 0),
        (-1, 0),
        (-1, 0),
    ], dtype=np.int8)
    assert np.array_equal(config_to_array(mini_arm), expected)


def test_array_get_position(origin):
    assert np.array_equal(array_get_position(config_to_array(origin)),
                          np.zeros(shape=(2, ), dtype=np.int8))


@pytest.mark.parametrize(
    ("link_ids", "directions", "expected"),
    (
        ([0], [1], np.array([(2, 1), (-1, 0), (-1, 0)], dtype=np.int8)),
        ([2], [1], np.array([(2, 0), (-1, 0), (-1, -1)], dtype=np.int8)),
        ([0, 1, 2], [1, 1, 1], np.array([[2, 1], [-1, -1], [-1, -1]], dtype=np.int8)),
        ([0, 1, 2], [1, -1, -1], np.array([[2, 1], [-1, 1], [-1, 1]], dtype=np.int8)),
    )
)
def test_array_rotate(link_ids, directions, expected, mini_arm):
    rotated = array_rotate(config_to_array(mini_arm), link_ids, directions)
    assert np.array_equal(rotated, expected)


def test_array_get_neighbors(mini_arm, mini_arm_neighbors):
    expected = np.array(mini_arm_neighbors, dtype=np.int8)
    expected.sort(axis=0)
    result = array_get_neighbors(config_to_array(mini_arm))
    result.sort(axis=0)
    assert np.array_equal(result, expected)
