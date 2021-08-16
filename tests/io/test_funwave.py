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
        dset0.spec.to_funwave(filename, clip=False)
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

    def test_clip(self):
        """Test directions are / are not clipped."""
        filename1 = os.path.join(self.tmp_dir, "fw1.txt")
        filename2 = os.path.join(self.tmp_dir, "fw2.txt")
        dset0 = self.dsww3.isel(time=0, site=0, drop=True)
        dset0.spec.to_funwave(filename1)
        dset0.spec.to_funwave(filename2, clip=False)

        dset1 = read_funwave(filename1)
        dset2 = read_funwave(filename2)

        dir1 = (270 - dset1.dir.values) % 360
        dir2 = (270 - dset2.dir.values) % 360
        dir1[dir1 > 180] = dir1[dir1 > 180] - 360
        dir2[dir2 > 180] = dir2[dir2 > 180] - 360

        assert dir1.min() >= -90 and dir1.max() <= 90
        assert dir2.min() < -90 and dset2.dir.max() > 90

    def test_1d(self):
        """
        * Write oned funwave spectrum from ww3 file.
        * Read oned funwave spectrum written.
        * Compare stats from original and new datasets.
        * Check for dimension consistency.
        """
        filename = os.path.join(self.tmp_dir, "fw1d.txt")
        dset0 = self.dsww3.isel(time=0, site=0, drop=True).spec.oned().to_dataset()
        dset0.spec.to_funwave(filename, clip=False)
        dset1 = read_funwave(filename)
        assert dset1.spec.dd == 1
        assert dset1.spec.dir is None
        xr.testing.assert_allclose(
            dset0.spec.stats(["hs", "tp"]),
            dset1.spec.stats(["hs", "tp"]),
            rtol=1e-3,
        )