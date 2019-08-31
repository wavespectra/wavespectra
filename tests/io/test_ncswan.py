import os
import xarray as xr
import shutil
import pytest
from tempfile import mkdtemp

from wavespectra import read_ncswan
from wavespectra.core.attributes import attrs

FILES_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "../sample_files")


class TestNcSwan(object):
    """Test output from read_ncswan function looks sound."""

    @classmethod
    def setup_class(self):
        """Setup class."""
        self.tmp_dir = mkdtemp()
        self.filename = os.path.join(FILES_DIR, "swanfile.nc")

    @classmethod
    def teardown_class(self):
        shutil.rmtree(self.tmp_dir)

    def test_ncswan(self):
        ds_wavespectra = read_ncswan(self.filename)
        ds_xarray = xr.open_dataset(self.filename)
        assert ds_xarray.hs.values == pytest.approx(
            ds_wavespectra.spec.hs().values, rel=0.01
        )
