import numpy as np

from santa_2022.original import *
from santa_2022.array import *


def test_get_neighbors_integration(origin):
    original = np.array(list((get_neighbors(origin))), dtype=np.int8)
    array = array_get_neighbors(config_to_array(origin))[1:]
    assert original.shape == array.shape
    original.sort(axis=0)
    array.sort(axis=0)
    assert np.array_equal(original, array)
