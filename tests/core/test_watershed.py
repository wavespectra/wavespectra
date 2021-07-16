"""Unit testing for stats methods in Partitioned SpecArray."""
import os
import pytest
import pandas as pd

from wavespectra import read_swan
from wavespectra.core.watershed import inflection, partition

FILES_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "../sample_files")


@pytest.fixture(scope="module")
def dset():
    """Load SpecDset but skip test if matplotlib is not installed."""
    pytest.importorskip("matplotlib")
    dset = read_swan(os.path.join(FILES_DIR, "swanfile.spec"), as_site=True)
    return dset


def test_inflection(dset):
    ds = dset.isel(time=0, site=0, drop=True)
    imax, imin = inflection(ds.efth.values, ds.freq.values, dfres=0.01, fmin=0.05)
    assert int(imax) == 5
    assert imin.size == 0
    imax, imin = inflection(ds.efth.values, ds.freq.values, dfres=0.015, fmin=0.05)
    assert int(imax) == 6
    assert imin.size == 0
    ds = dset.isel(freq=[0])
    imax, imin = inflection(ds.efth.values, ds.freq.values, dfres=0.01, fmin=0.05)
    assert imax == 0
    assert imin == 0


def test_partition_function(dset):
    dsp = partition(dset=dset, wspd="wspd", wdir="wdir", dpt="dpt")


class TestSpecArray(object):
    """Test methods from watershed."""

    @classmethod
    def setup_class(self):
        """Read test spectra and pre-calculated stats from file."""
        self.swanspec = read_swan(os.path.join(FILES_DIR, "swanfile.spec"))
        self.wshed = self.swanspec.spec.partition(
            self.swanspec.wspd, self.swanspec.wdir, self.swanspec.dpt
        )

    def test_watershed_dims(self):
        """Assert that extra dimension has been defined."""
        assert "part" in self.wshed.dims

    @pytest.mark.parametrize(
        "stat_name",
        [
            ("hs"),
            ("hmax"),
            ("tp"),
            ("tm01"),
            ("tm02"),
            ("dm"),
            ("dp"),
            ("dpm"),
            ("swe"),
            ("sw"),
        ],
    )
    def test_stat(self, stat_name):
        """Check that all stats can be calculated from watershed.

        Args:
            - stat_name (str): name of wave spectral statistic to compare,
              must be the name of a valid method in SpecArray class.

        """
        stat = getattr(self.wshed.spec, stat_name)()


