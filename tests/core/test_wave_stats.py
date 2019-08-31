"""Unit testing for stats methods in SpecArray."""
import os
import pytest
import pandas as pd

from wavespectra import read_swan

FILES_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "../sample_files")


class TestSpecArray(object):
    """Test methods from SpecArray class."""

    @classmethod
    def setup_class(self):
        """Read test spectra and pre-calculated stats from file."""

        self.control = pd.read_csv(os.path.join(FILES_DIR, "swanfile.txt"), sep="\t")
        self.swanspec = read_swan(os.path.join(FILES_DIR, "swanfile.spec"))

    @pytest.mark.parametrize(
        "stat_name, rel",
        [
            ("hs", 1e-3),
            ("hmax", 1e-3),
            ("tp", 1e-3),
            ("tm01", 1e-3),
            ("tm02", 1e-3),
            ("dm", 1e-3),
            ("dp", 1e-3),
            ("dpm", 1e-3),
            ("swe", 1e-3),
            ("sw", 1e-3),
        ],
    )
    def test_stat(self, stat_name, rel):
        """Compare stat between SpecArray and control file.

        Args:
            stat_name (str): name of wave spectral statistic to compare. Must
                be the name of a valid method in SpecArray class.
            rel (float): relative tolerance for comparing two values as
                described in pytest.approx() method.

        Asserts:
            values calculated from SpecArray method must be equal to those from
                control file swanfile.txt within the relative tolerance rel.

        """
        ctrl = self.control[stat_name].values
        calc = getattr(self.swanspec.spec, stat_name)().values.ravel()
        assert calc == pytest.approx(ctrl, rel=rel)
