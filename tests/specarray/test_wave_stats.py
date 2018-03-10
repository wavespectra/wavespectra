"""Unit testing for stats methods in SpecArray."""
import os
import inspect
import pytest
import pandas as pd

from wavespectra import read_swan
from wavespectra.core.swan import read_tab

class TestSpecArray(object):
    """Test methods from SpecArray class."""

    @classmethod
    def setup_class(self):
        """Read test spectra and control statistics from file."""
        here = os.path.dirname(os.path.abspath(__file__))
        self.control = pd.read_csv(os.path.join(here, '../swanfile.txt'), sep='\t')
        self.swanspec = read_swan(os.path.join(here, '../swanfile.spec'))

    @classmethod
    def teardown_class(self):
        """teardown tests."""
        pass

    def _assert_stat(self, stat_name, rel=1e-3):
        """Compare stat between SpecArray control file.
        
        Args:
            stat_name (str): name of wave spectral statistic to compare. Must
                be the name of a valid method in SpecArray class.
            rel (float): relative tolerance for comparing two values as
                described in pytest.approx() method.

        Asserts:
            values calculated from SpecArray method must be equal to those from
                control file swanfile.txt within a relative tolerance of 1e-3.

        """
        ctrl = self.control[stat_name].values
        calc = getattr(self.swanspec.spec, stat_name)().values.ravel()
        assert calc == pytest.approx(ctrl, rel=rel)

    def _stat_name(self):
        """Returns the name of the stat to be tested.

        Stat is inferred from the caller method which should be named as:
            test_{stat}

        """
        return inspect.stack()[1][3].split('test_')[-1]

    def test_hs(self):
        """Test hs method."""
        self._assert_stat(self._stat_name())

    def test_hmax(self):
        """Test hmax method."""
        self._assert_stat(self._stat_name())

    def test_tp(self):
        """Test tp method."""
        self._assert_stat(self._stat_name())

    def test_tm01(self):
        """Test tm01 method."""
        self._assert_stat(self._stat_name())

    def test_tm02(self):
        """Test tm02 method."""
        self._assert_stat(self._stat_name())

    def test_dm(self):
        """Test dm method."""
        self._assert_stat(self._stat_name())

    def test_dp(self):
        """Test dp method."""
        self._assert_stat(self._stat_name())

    def test_dpm(self):
        """Test dpm method."""
        self._assert_stat(self._stat_name())

    def test_swe(self):
        """Test swe method."""
        self._assert_stat(self._stat_name())

    def test_sw(self):
        """Test sw method."""
        self._assert_stat(self._stat_name())
