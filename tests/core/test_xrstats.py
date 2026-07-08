import os
import pytest

from wavespectra import read_swan
from wavespectra.core.xrstats import (
    peak_wave_direction,
    mean_direction_at_peak_wave_period,
    peak_wave_period,
    peak_directional_spread,
)


FILES_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "../sample_files")


@pytest.fixture(autouse=True)
def dataset_transforms():
    """Run under the future dataset transforms behaviour in this module."""
    import wavespectra

    with wavespectra.set_options(dataset_transforms=True):
        yield


@pytest.fixture(scope="module")
def dset():
    """Load SpecDset but skip test if matplotlib is not installed."""
    pytest.importorskip("matplotlib")
    dset = read_swan(os.path.join(FILES_DIR, "swanfile.spec"), as_site=True)
    return dset


def test_peak_wave_direction(dset):
    peak_wave_direction(dset)
    peak_wave_direction(dset.efth)
    with pytest.raises(ValueError):
        peak_wave_direction(dset.spec.oned())


def test_mean_direction_at_peak_wave_period(dset):
    mean_direction_at_peak_wave_period(dset)
    mean_direction_at_peak_wave_period(dset.efth)
    with pytest.raises(ValueError):
        mean_direction_at_peak_wave_period(dset.spec.oned())


def test_peak_wave_period(dset):
    peak_wave_period(dset)
    peak_wave_period(dset.efth)


def test_peak_directional_spread(dset):
    peak_directional_spread(dset)
    peak_directional_spread(dset.efth)
