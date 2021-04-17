from wavespectra.core.utils import bins_from_frequency_grid
import numpy as np
from numpy.testing import assert_array_almost_equal


def test_constant():
    grid = np.linspace(0.1, 10, 100)
    left, right, size,centers = bins_from_frequency_grid(grid)
    assert_array_almost_equal(np.max(size), np.min(size))

    assert_array_almost_equal(left[1:], right[:-1])

def test_constant_with_small_imperfection():
    grid = np.linspace(0.1, 10, 100)
    grid[2] += 1e-4
    left, right, size,centers = bins_from_frequency_grid(grid)
    assert_array_almost_equal(np.max(size), np.min(size))

def test_exponential():
    # exponential data with limited significant number of digits
    exp_data = [0.0345,0.038,0.0418,0.0459,0.0505,0.0556,0.0612,0.0673,0.074,0.0814,0.0895,0.0985,0.1083,0.1192,0.1311,0.1442,0.1586,0.1745,0.1919,0.2111,0.2323,0.2555,0.281,0.3091,0.34,0.374,0.4114,0.4526,0.4979,0.5476,0.6024,0.6626,0.7289,0.8018,0.882,0.9702]
    left, right, size, centers = bins_from_frequency_grid(exp_data)
    assert_array_almost_equal(left[1:], right[:-1])



