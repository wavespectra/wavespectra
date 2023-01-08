from pathlib import Path
import datetime
import shutil
import pytest
import pandas as pd
from tempfile import mkdtemp

from wavespectra import read_ndbc, read_ndbc_ascii
from wavespectra.core.attributes import attrs

FILES_DIR = Path(__file__).parent.parent / "sample_files/ndbc"


def test_ndbc_hs_equals_1d_2d():
    url = "https://dods.ndbc.noaa.gov/thredds/dodsC/data/swden/42098/42098w9999.nc"
    dset_1d = read_ndbc(url, directional=False, chunks={"time": 1}).isel(time=100)
    dset_2d = read_ndbc(url, directional=True, chunks={"time": 1}).isel(time=100)
    assert dset_1d.spec.hs().values == pytest.approx(dset_2d.spec.hs().values)


def test_ndbc_netcdf_2d():
    dd = 5
    dset = read_ndbc(
        url="https://dods.ndbc.noaa.gov/thredds/dodsC/data/swden/32012/32012w2007.nc",
        directional=True,
        dd=dd,
    )
    assert dset.spec.dd == dd


def test_ndbc_netcdf_1d():
    dset = read_ndbc(
        url="https://dods.ndbc.noaa.gov/thredds/dodsC/data/swden/32012/32012w2007.nc",
        directional=False,
    )
    assert dset.spec.dir is None


class TestNDBCASCII(object):
    """Test parameters from 1D and 2D spectra are consistent and similar to NDBC summary values."""

    @classmethod
    def setup_class(self):
        """Setup class."""
        self.tmp_dir = mkdtemp()
        self.ds_realtime_1d = read_ndbc_ascii(FILES_DIR / "41010.data_spec")
        self.ds_realtime_2d = read_ndbc_ascii(
            [
                FILES_DIR / "41010.data_spec",
                FILES_DIR / "41010.swdir",
                FILES_DIR / "41010.swdir2",
                FILES_DIR / "41010.swr1",
                FILES_DIR / "41010.swr2"
            ]
        )
        self.ds_realtime_params = pd.read_csv(
            FILES_DIR / "41010.spec",
            delimiter=r"\s+",
            engine="python",
            header=[0, 1],
            parse_dates={"time": [0, 1, 2, 3, 4]},
            date_parser=lambda x: datetime.datetime.strptime(x, "%Y %m %d %H %M"),
            index_col=0,
        ).sort_values(by="time", ascending=True)
        self.ds_history_1d = read_ndbc_ascii(FILES_DIR / "41010w2019part.txt.gz")
        self.ds_history_2d = read_ndbc_ascii(
            [
                FILES_DIR / "41010w2019part.txt.gz",  # swden
                FILES_DIR / "41010d2019part.txt.gz",  # swdir
                FILES_DIR / "41010i2019part.txt.gz",  # swdir2
                FILES_DIR / "41010j2019part.txt.gz",  # swr1
                FILES_DIR / "41010k2019part.txt.gz",  # swr2
            ]
        )

    @classmethod
    def teardown_class(self):
        shutil.rmtree(self.tmp_dir)

    def test_realtime_hs_equals_params(self):
        assert self.ds_realtime_1d.spec.hs().values == pytest.approx(
            self.ds_realtime_params[("WVHT", "m")].values, rel=0.11
        )

    def test_realtime_1d_equals_2d(self):
        assert self.ds_realtime_1d.spec.hs().values == pytest.approx(
            self.ds_realtime_2d.spec.hs().values, rel=0.01
        )

    def test_history_1d_equals_2d(self):
        assert self.ds_history_1d.spec.hs().values == pytest.approx(
            self.ds_history_2d.spec.hs().values, rel=0.01
        )

    def test_history_1d_no_minutes(self):
        ds = read_ndbc_ascii(FILES_DIR / "44004w2000.txt")
        assert hasattr(ds, "spec")