import os
import xarray as xr
import shutil
import pytest
from tempfile import mkdtemp

from wavespectra import read_ncswan, read_octopus
from wavespectra.core.attributes import attrs
from wavespectra.core.timer import Timer

FILES_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "../sample_files")


# TODO: Write reading test once implemented read_octopus


class TestOctopus(object):
    """Test Octopus writer."""

    @classmethod
    def setup_class(self):
        """Setup class."""
        self.tmp_dir = mkdtemp()
        self.filename = os.path.join(FILES_DIR, "swanfile.nc")

    @classmethod
    def teardown_class(self):
        shutil.rmtree(self.tmp_dir)

    def test_read_octopus(self):
        with pytest.raises(NotImplementedError):
            dset = read_octopus(os.path.join(FILES_DIR, "octopusfile.oct"))

    def test_write_octopus(self):
        with Timer("Testing Octopus writer"):
            ds = read_ncswan(self.filename)
            ds.spec.to_octopus(os.path.join(self.tmp_dir, "spectra.oct"))

    def test_write_octopus_missing_winds_depth(self):
        ds = read_ncswan(self.filename)
        ds = ds.drop_vars([attrs.WSPDNAME, attrs.WDIRNAME, attrs.DEPNAME])
        ds.spec.to_octopus(os.path.join(self.tmp_dir, "spec_no_winds_depth.oct"))

    def test_write_octopus_missing_lonlat(self):
        ds = read_ncswan(self.filename)
        ds = ds.rename({"lon": "x", "lat": "y"})
        with pytest.raises(NotImplementedError):
            ds.spec.to_octopus(os.path.join(self.tmp_dir, "spec_no_lonlat.oct"))

    def test_write_octopus_one_time(self):
        ds = read_ncswan(self.filename)
        ds = ds.isel(time=[0])
        ds.spec.to_octopus(os.path.join(self.tmp_dir, "spec_one_time.oct"))
