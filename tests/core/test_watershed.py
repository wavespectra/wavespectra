"""Unit testing for stats methods in Partitioned SpecArray."""
import os
import pytest
import pandas as pd

from wavespectra import read_swan

FILES_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "../sample_files")


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
