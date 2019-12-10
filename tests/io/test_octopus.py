import os
import xarray as xr
import shutil
import pytest
from tempfile import mkdtemp

from wavespectra import read_ncswan
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

    def test_octopus(self):
        with Timer("Testing Octopus writer"):
            ds = read_ncswan(self.filename)
            ds.spec.to_octopus(os.path.join(self.tmp_dir, "spectra.oct"))

