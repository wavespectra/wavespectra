from pathlib import Path
import shutil
import pytest
from tempfile import mkdtemp

from wavespectra import read_triaxys
from wavespectra.core.attributes import attrs


FILES_DIR = Path(__file__).parent.parent / 'sample_files'


@pytest.fixture(scope="module")
def dset_1d():
    _ds = read_triaxys(str(FILES_DIR / 'triaxys.NONDIRSPEC'))
    yield _ds


@pytest.fixture(scope="module")
def dset_2d():
    _ds = read_triaxys(str(FILES_DIR / 'triaxys.DIRSPEC'))
    yield _ds


def test_hs(dset_1d, dset_2d):
    assert dset_1d.spec.hs().values == pytest.approx(
        dset_2d.spec.hs().values, rel=0.01
    )


def test_magnetic_variation_only_2d():
    filename = FILES_DIR / "triaxys.NONDIRSPEC"
    dset1 = _ds = read_triaxys(filename)
    dset2 = _ds = read_triaxys(filename, magnetic_variation=10)
    assert dset1.identical(dset2)


def test_regrid_dir(dset_2d):
    filename = FILES_DIR / "triaxys.DIRSPEC"
    dset1 = _ds = read_triaxys(filename, magnetic_variation=None)
    dset2 = _ds = read_triaxys(filename, magnetic_variation=10, regrid_dir=False)
    dset3 = _ds = read_triaxys(filename, magnetic_variation=10, regrid_dir=True)
    assert dset_2d.dir.identical(dset1.dir)
    assert dset_2d.dir.identical(dset3.dir)
    assert not dset_2d.dir.identical(dset2.dir)


def test_declination_corrected_directional_stats(dset_2d):
    filename = FILES_DIR / "triaxys.DIRSPEC"
    dset1 = read_triaxys(filename, magnetic_variation=10, regrid_dir=False)
    dset2 = read_triaxys(filename, magnetic_variation=10, regrid_dir=True)
    assert dset1.spec.dm().values == pytest.approx(dset2.spec.dm().values, rel=1e-2)
    assert dset1.spec.dpm().values == pytest.approx(dset2.spec.dpm().values, rel=1e-2)
