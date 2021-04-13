import os
import shutil
import pytest
from tempfile import mkdtemp
import numpy as np
import xarray as xr
from zipfile import ZipFile

from wavespectra import read_funwave, read_ww3


FILES_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "../sample_files")


class TestFunwave:
    """Test Funwave input / output functions."""

    @classmethod
    def setup_class(self):
        """Setup class."""
        self.tmp_dir = mkdtemp()
        self.filename = os.path.join(FILES_DIR, "funwavefile.txt")
        self.dsww3 = read_ww3(os.path.join(FILES_DIR, "ww3file.nc"))

    @classmethod
    def teardown_class(self):
        shutil.rmtree(self.tmp_dir)

    def _files_in_zip(self, zipname):
        with ZipFile(zipname) as zstream:
            count = len(zstream.infolist())
        return count

    def test_read(self):
        dset = read_funwave(self.filename)

    def test_write_single(self):
        """
        * Write single funwave spectrum from ww3 file
        * Compare stats from original and new datasets.
        """
        filename = os.path.join(self.tmp_dir, "fw.txt")
        dset0 = self.dsww3.isel(time=0, site=0, drop=True)
        dset0.spec.to_funwave(filename)
        dset1 = read_funwave(filename)
        xr.testing.assert_allclose(
            dset0.spec.stats(["hs", "tp", "dpm"]),
            dset1.spec.stats(["hs", "tp", "dpm"]),
            rtol=1e-3,
        )

    def test_write_multi(self):
        """Write multiple funwave files from ww3 dataset."""
        filename = os.path.join(self.tmp_dir, "fw.txt")
        zipname = filename.replace(".txt", ".zip")
        self.dsww3.spec.to_funwave(filename)
        assert os.path.isfile(zipname)
        assert self._files_in_zip(zipname) == np.prod(
            self.dsww3.efth.isel(freq=0, dir=0).shape
        )
