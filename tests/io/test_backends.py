"""Test backend entrypoints"""
from pathlib import Path
import xarray as xy
import pytest


DATADIR = Path(__file__).parent.parent / "sample_files"


@pytest.mark.parametrize(
    "backend, filename",
    [
        ("era5", DATADIR / "era5file.nc"),
        ("ncswan", DATADIR / "swanfile.nc"),
        # ("ndbc", "https://dods.ndbc.noaa.gov/thredds/dodsC/data/swden/42098/42098w9999.nc"),
        (
            "ndbc_ascii",
            [
                DATADIR / "ndbc/41010w2019part.txt.gz",
                DATADIR / "ndbc/41010d2019part.txt.gz",
                DATADIR / "ndbc/41010i2019part.txt.gz",
                DATADIR / "ndbc/41010j2019part.txt.gz",
                DATADIR / "ndbc/41010k2019part.txt.gz",
            ],
        ),
        ("netcdf", DATADIR / "wavespectra.nc"),
        ("octopus", DATADIR / "octopusfile.oct"),
        ("spotter", DATADIR / "spotter_20210929b.csv"),
        ("triaxys", DATADIR / "triaxys.DIRSPEC"),
        ("wavespectra", DATADIR / "wavespectra.nc"),
        ("wwm", DATADIR / "wwmfile.nc"),
        ("ww3", DATADIR / "ww3file.nc"),
        ("ww3_station", DATADIR / "ww3station.spec"),
        ("swan", DATADIR / "swanfile.spec"),
        ("funwave", DATADIR / "funwavefile.txt"),
        ("json", DATADIR / "jsonfile.json"),
    ],
)
def test_backend(backend, filename):
    """Test backend entrypoints."""
    ds = xy.open_dataset(filename, engine=backend)
    assert hasattr(ds, "spec")
