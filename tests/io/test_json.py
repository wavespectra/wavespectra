import os
import shutil
import pytest
from tempfile import mkdtemp
import xarray as xr

from wavespectra.core.attributes import attrs
from wavespectra import (
    read_swan,
    read_netcdf,
    read_ww3,
    read_octopus,
    read_ncswan,
    read_triaxys,
    read_wwm,
    read_dataset,
    read_era5,
    read_wavespectra,
    read_json
)

FILES_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "../sample_files")


class TestIO:
    """Test json serialisation."""

    @classmethod
    def setup_class(self):
        """Setup class."""
        self.tmp_dir = mkdtemp()

    @classmethod
    def teardown_class(self):
        shutil.rmtree(self.tmp_dir)

    @pytest.mark.parametrize(
        "filename, read_func",
        [
            ("swanfile.spec", read_swan),
            ("ww3file.nc", read_ww3),
            ("swanfile.nc", read_ncswan),
            ("triaxys.DIRSPEC", read_triaxys),
            ("triaxys.NONDIRSPEC", read_triaxys),
            ("wavespectra.nc", read_netcdf),
            ("era5file.nc", read_era5),
            ("jsonfile.json", read_json)
        ],
    )
    def test_to_json(self, filename, read_func):
        """Test that all test files can be written as json."""
        self.filename = filename
        self.read_func = read_func
        self._read()
        self._write()
        self._check()

    def _read(self):
        self.infile = os.path.join(FILES_DIR, self.filename)
        self.ds = self.read_func(self.infile)

    def _write(self):
        self.outfile = os.path.join(self.tmp_dir, self.filename)
        self.ds.spec.to_json(self.outfile)

    def _check(self):
        """Check stats from output json dataset is equal to original input one."""
        self.ds2 = read_json(self.outfile)
        stats = ["hs", "tp"]
        ds = self.ds.spec.stats(stats)
        ds2 = self.ds2.spec.stats(stats)
        for stat in stats:
            assert ds[stat].values == pytest.approx(
                ds2[stat].values, rel=1e-3, nan_ok=True
            )
