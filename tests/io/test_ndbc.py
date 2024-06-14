from pathlib import Path
import shutil
import pytest
import pandas as pd
from tempfile import mkdtemp

from wavespectra import read_ndbc, read_ndbc_ascii
from wavespectra.core.attributes import attrs

FILES_DIR = Path(__file__).parent.parent / "sample_files/ndbc"


@pytest.mark.xfail
def test_ndbc_hs_equals_1d_2d():
    url = "https://dods.ndbc.noaa.gov/thredds/dodsC/data/swden/42098/42098w9999.nc"
    dset_1d = read_ndbc(url, directional=False, chunks={"time": 1}).isel(time=100)
    dset_2d = read_ndbc(url, directional=True, chunks={"time": 1}).isel(time=100)
    assert dset_1d.spec.hs().values == pytest.approx(dset_2d.spec.hs().values)


@pytest.mark.xfail
def test_ndbc_netcdf_2d():
    dd = 5
    dset = read_ndbc(
        url="https://dods.ndbc.noaa.gov/thredds/dodsC/data/swden/32012/32012w2007.nc",
        directional=True,
        dd=dd,
    )
    assert dset.spec.dd == dd


@pytest.mark.xfail
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
        f = open(FILES_DIR / "41010.spec", "r")
        header = f.readline()
        date_columns = {0: "year", 1: "month", 2: "day", 3: "hour", 4: "minute"}
        df = pd.read_csv(
            f,
            delimiter=r"\s+",
            header=None,
            skiprows=1,
        )
        df.index = pd.to_datetime(df[date_columns.keys()].rename(columns=date_columns))
        df.columns = header.split()
        f.close()
        self.ds_realtime_params = df.sort_index(ascending=True)
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
            self.ds_realtime_params["WVHT"].values, rel=0.11
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
