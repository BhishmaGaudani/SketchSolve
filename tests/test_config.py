import numpy as np

from src.config import ROOT, set_seed


def test_root_points_at_repo():
    assert (ROOT / "src" / "config.py").exists()


def test_set_seed_makes_random_numbers_repeatable():
    set_seed(123)
    first = np.random.rand(3)
    set_seed(123)
    second = np.random.rand(3)
    assert np.array_equal(first, second)
