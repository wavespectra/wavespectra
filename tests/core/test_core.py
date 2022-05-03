"""Unit testing for stats methods in SpecArray."""
import os
import pytest
import datetime
import xarray as xr

from wavespectra.core.utils import dnum_to_datetime, to_nautical, unique_times, spddir_to_uv, uv_to_spddir, flatten_list
from wavespectra import read_swan

FILES_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "../sample_files")


@pytest.fixture(scope="module")
def dset():
    filename = os.path.join(FILES_DIR, "swanfile.spec")
    _dset = read_swan(filename)
    yield _dset


def test_dnum_to_datetime():
    assert dnum_to_datetime(367) == datetime.datetime(1, 1, 1)


def test_to_nautical():
    assert to_nautical(0) == 270
    assert to_nautical(90) == 180
    assert to_nautical(180) == 90
    assert to_nautical(270) == 0


def test_unique_times(dset):
    ds = unique_times(xr.concat([dset, dset], dim="time"))
    ds.time.equals(dset.time)


def test_uv_spddir(dset):
    u, v = spddir_to_uv(dset.wspd, dset.wdir)
    wspd, wdir = uv_to_spddir(u, v)
    dset.wspd.equals(wspd)
    dset.wdir.equals(wdir)


def test_uv_to_dir():
    # Going to
    assert uv_to_spddir(0, 1, False)[1] == 0
    assert uv_to_spddir(1, 1, False)[1] == 45
    assert uv_to_spddir(1, 0, False)[1] == 90
    assert uv_to_spddir(1, -1, False)[1] == 135
    assert uv_to_spddir(0, -1, False)[1] == 180
    assert uv_to_spddir(-1, -1, False)[1] == 225
    assert uv_to_spddir(-1, 0, False)[1] == 270
    assert uv_to_spddir(-1, 1, False)[1] == 315
    # Coming from
    assert uv_to_spddir(0, 1, True)[1] == 180
    assert uv_to_spddir(1, 1, True)[1] == 225
    assert uv_to_spddir(1, 0, True)[1] == 270
    assert uv_to_spddir(1, -1, True)[1] == 315
    assert uv_to_spddir(0, -1, True)[1] == 0
    assert uv_to_spddir(-1, -1, True)[1] == 45
    assert uv_to_spddir(-1, 0, True)[1] == 90
    assert uv_to_spddir(-1, 1, True)[1] == 135


def test_flatten_list():
    l = [1, [2, 3], [4], [5, 6, 7]]
    assert flatten_list(l, []) == list(range(1, 8))

# class TestSpecArray(object):
#     """Test methods from SpecArray class."""

#     @classmethod
#     def setup_class(self):
#         """Read test spectra and pre-calculated stats from file."""

#         self.control = pd.read_csv(os.path.join(FILES_DIR, "swanfile.txt"), sep="\t")
#         self.swanspec = read_swan(os.path.join(FILES_DIR, "swanfile.spec"))

#     @pytest.mark.parametrize(
#         "stat_name, rel",
#         [
#             ("hs", 1e-3),
#             ("hmax", 1e-3),
#             ("tp", 1e-3),
#             ("tm01", 1e-3),
#             ("tm02", 1e-3),
#             ("dm", 1e-3),
#             ("dp", 1e-3),
#             ("dpm", 1e-3),
#             ("swe", 1e-3),
#             ("sw", 1e-3),
#         ],
#     )
#     def test_stat(self, stat_name, rel):
#         """Compare stat between SpecArray and control file.

#         Args:
#             stat_name (str): name of wave spectral statistic to compare. Must
#                 be the name of a valid method in SpecArray class.
#             rel (float): relative tolerance for comparing two values as
#                 described in pytest.approx() method.

#         Asserts:
#             values calculated from SpecArray method must be equal to those from
#                 control file swanfile.txt within the relative tolerance rel.

#         """
#         ctrl = self.control[stat_name].values
#         calc = getattr(self.swanspec.spec, stat_name)().values.ravel()
#         assert calc == pytest.approx(ctrl, rel=rel)
