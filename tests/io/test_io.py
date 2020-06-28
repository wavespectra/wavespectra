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
    read_wavespectra
)

FILES_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "../sample_files")


class TestIO(object):
    """Test reading and writing of different file formats.
    
    Extend IO tests by adding tuple to parametrize, e.g.:
        ('filename', read_{filetype}, 'to_{filetype}')
    Use None for 'to_{filetype} if there is no output method defined'.

    """

    @classmethod
    def setup_class(self):
        """Setup class."""
        self.tmp_dir = mkdtemp()

    @classmethod
    def teardown_class(self):
        shutil.rmtree(self.tmp_dir)

    @pytest.mark.parametrize(
        "filename, read_func, write_method_name",
        [
            ("swanfile.spec", read_swan, "to_swan"),
            ("ww3file.nc", read_ww3, "to_ww3"),
            ("swanfile.nc", read_ncswan, None),
            ("triaxys.DIRSPEC", read_triaxys, None),
            ("triaxys.NONDIRSPEC", read_triaxys, None),
            ("wavespectra.nc", read_netcdf, "to_netcdf"),
            ("era5file.nc", read_era5, None)
        ],
    )
    def test_io(self, filename, read_func, write_method_name):
        self.filename = filename
        self.read_func = read_func
        self.write_method_name = write_method_name
        # Execute io tests in order
        self._read()
        if self.write_method_name is not None:
            try:
                self._write()
                self._check()
            except NotImplementedError:
                pytest.skip("Writing function {} not implemented yet".format(
                    write_method_name
                    )
                )
        else:
            print(
                "No output method defined for {}, "
                "skipping output tests".format(filename)
            )

    @pytest.mark.parametrize(
        "reader, filename",
        [
            (read_ww3, "ww3file.nc"),
            (read_wwm, "wwmfile.nc"),
            (read_ncswan, "swanfile.nc"),
            (read_wavespectra, "wavespectra.nc"),
        ],
    )
    def test_read_dataset(self, reader, filename):
        """Check that read_dataset returns same object as read_{function}."""
        filepath = os.path.join(FILES_DIR, filename)
        dset1 = reader(filepath)
        dset2 = read_dataset(xr.open_dataset(filepath))
        assert dset1.equals(dset2)

    def test_do_not_read_unknown_dataset(self):
        dset = xr.open_dataset(os.path.join(FILES_DIR, "ww3file.nc"))
        dset = dset.rename({"efth": "DUMMY"})
        with pytest.raises(ValueError):
            read_dataset(dset)

    def test_zarr(self):
        """Check reading of zarr dataset in ww3 format."""
        filename = os.path.join(FILES_DIR, "ww3file.zarr")
        dset = read_ww3(filename, file_format="zarr")
        outfile = os.path.join(self.tmp_dir, "tmp.nc")
        dset.spec.to_netcdf(outfile)
        dset2 = read_netcdf(outfile)
        dset.equals(dset2)

    def _read(self):
        self.infile = os.path.join(FILES_DIR, self.filename)
        self.ds = self.read_func(self.infile)

    def _write(self):
        self.write_method = getattr(self.ds.spec, self.write_method_name, None)
        self.outfile = os.path.join(self.tmp_dir, self.filename)
        self.write_method(self.outfile)

    def _check(self):
        self.ds2 = self.read_func(self.outfile)
        stats = ["hs", "tp", "dpm"]
        ds = self.ds.spec.stats(stats)
        ds2 = self.ds2.spec.stats(stats)
        for stat in stats:
            assert ds[stat].values == pytest.approx(ds2[stat].values, rel=1e-3)
