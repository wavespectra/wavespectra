"""Test model wrapper."""
import os
import yaml
import glob
import pytest
from click.testing import CliRunner

from wavespectra import cli


TESTDIR = os.path.dirname(os.path.abspath(__file__))


@pytest.fixture(scope="module")
def runner():
    instance = CliRunner()
    yield instance


def test_main(runner):
    result = runner.invoke(cli.main)
    assert result.exit_code == 0
    assert "main" in result.output


def test_main_reconstruct(runner):
    result = runner.invoke(cli.main, "reconstruct")
    assert result.exit_code == 0
    assert "reconstruct" in result.output


def test_main_reconstruct_spectra(runner, tmpdir):
    INFILE = os.path.join(TESTDIR, "sample_files/ww3file.nc")
    OUTFILE = str(tmpdir / "outspec.nc")
    CONFIG = os.path.join(TESTDIR, "reconstruct.yml")
    result = runner.invoke(cli.main, ["reconstruct", "spectra", INFILE, OUTFILE, CONFIG])
    assert result.exit_code == 0
