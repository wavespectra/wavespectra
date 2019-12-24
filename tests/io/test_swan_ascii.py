import os
import xarray as xr
import shutil
import pytest
from tempfile import mkdtemp

from wavespectra.input.swan import read_swan, read_swans, read_hotswan, read_swanow
from wavespectra.core.attributes import attrs

FILES_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "../sample_files")


class TestNcSwan(object):
    """Test read swan ascii functions."""

    @classmethod
    def setup_class(self):
        """Setup class."""
        self.tmp_dir = mkdtemp()
        self.filename = os.path.join(FILES_DIR, "swanfile.spec")
        self.ds = read_swan(self.filename, as_site=True)

    @classmethod
    def teardown_class(self):
        shutil.rmtree(self.tmp_dir)

    def test_read_swans(self):
        ds = read_swans([self.filename])
        assert self.ds.spec.hs().values == pytest.approx(
            ds.isel(site=[0]).spec.hs().values, rel=1e-2
        )

    def test_read_swans(self):
        ds = read_hotswan([self.filename])
        assert self.ds.spec.hs().values.ravel() == pytest.approx(
            ds.isel(lon=0, lat=0).spec.hs().values, rel=1e-2
        )

    def test_read_swanow(self):
        ds = read_swanow([self.filename])
        assert self.ds.isel(site=0).spec.hs().values == pytest.approx(
            ds.isel(lon=0, lat=0, drop=True).spec.hs().values, rel=1e-2
        )