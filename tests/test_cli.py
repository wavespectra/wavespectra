"""Test model wrapper."""
from pathlib import Path
import pytest
import xarray as xr
from click.testing import CliRunner

from wavespectra import cli


DATADIR = Path(__file__).parent / "sample_files"
INFILE = str(DATADIR / "ww3file.nc")
ENGINE = "ww3"


@pytest.fixture(scope="module")
def runner():
    instance = CliRunner()
    yield instance


def test_main(runner):
    result = runner.invoke(cli.main)
    assert result.exit_code == 0
    assert "main" in result.output


def test_main_convert(runner):
    result = runner.invoke(cli.main, "convert")
    assert result.exit_code == 0
    assert "convert" in result.output


def test_main_convert_format(runner, tmpdir):
    OUTFILE = str(tmpdir / "outspec.nc")

    result = runner.invoke(
        cli.main, ["convert", "format", INFILE, ENGINE, OUTFILE, "swan"]
    )
    assert result.exit_code == 0

    xr.open_dataset(OUTFILE, engine="swan")


def test_main_convert_stats(runner, tmpdir):
    OUTFILE = str(tmpdir / "outstats.nc")

    result = runner.invoke(
        cli.main, ["convert", "stats", INFILE, ENGINE, OUTFILE, "-p", "hs", "-p", "tp"]
    )
    assert result.exit_code == 0

    dset = xr.open_dataset(OUTFILE)
    assert "hs" in dset.data_vars and "tp" in dset.data_vars


def test_main_reconstruct(runner):
    result = runner.invoke(cli.main, "reconstruct")
    assert result.exit_code == 0
    assert "reconstruct" in result.output


def test_main_reconstruct_spectra(runner, tmpdir):
    OUTFILE = str(tmpdir / "outspec.nc")

    result = runner.invoke(
        cli.main, ["reconstruct", "spectra", INFILE, ENGINE, OUTFILE]
    )
    assert result.exit_code == 0

    result = runner.invoke(cli.main, ["reconstruct", "spectra", INFILE, ENGINE, INFILE])
    assert isinstance(result.exception, ValueError)

    result = runner.invoke(
        cli.main,
        [
            "reconstruct",
            "spectra",
            INFILE,
            ENGINE,
            OUTFILE,
            "-p",
            "3",
            "-d",
            "cartwright,cartwright,cartwright",
        ],
    )
    assert result.exit_code == 0

    result = runner.invoke(
        cli.main,
        [
            "reconstruct",
            "spectra",
            INFILE,
            ENGINE,
            OUTFILE,
            "-p",
            "3",
            "-f",
            "jonswap,jonswap,jonswap",
        ],
    )
    assert result.exit_code == 0
