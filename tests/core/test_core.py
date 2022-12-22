"""Unit testing for stats methods in SpecArray."""
import os
import pytest
import datetime
import numpy as np
import xarray as xr

from wavespectra.core.utils import (
    angle,
    to_nautical,
    unique_times,
    spddir_to_uv,
    uv_to_spddir,
    flatten_list,
    check_same_coordinates,
    scaled,
    wavelen,
    celerity,
    interp_spec,
    load_function,
    smooth_spec,
    is_overlap,
)
from wavespectra import read_swan

FILES_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "../sample_files")


@pytest.fixture(scope="module")
def dset():
    filename = os.path.join(FILES_DIR, "swanfile.spec")
    _dset = read_swan(filename)
    yield _dset


def test_angle():
    assert angle(0, 10) == 10
    assert angle(10, 0) == 10
    assert angle(0, 190) == 170
    assert angle(170, 361) == 169
    assert angle(10, 350) == 20


def test_is_overlap():
    rect1 = [0.1, 0, 0.2, 345]
    assert is_overlap(rect1, [0.15, 0, 0.3, 345])
    assert is_overlap(rect1, [0.0, 50, 0.11, 60])
    assert not is_overlap(rect1, [0.2, 0, 0.3, 345])
    assert not is_overlap(rect1, [0.1, 345, 0.2, 360])


def test_smooth_spectra(dset):
    ds = dset.isel(lon=0, lat=0, time=0, drop=True)
    dsmooth3 = smooth_spec(ds, freq_window=3, dir_window=3)
    dsmooth5 = smooth_spec(ds, freq_window=5, dir_window=5)
    assert ds.coords.to_index().equals(dsmooth3.coords.to_index())
    assert ds.efth.max() > dsmooth3.efth.max() > dsmooth5.efth.max()
    # Non-circular directions
    smooth_spec(ds.isel(dir=slice(None, -3)))


def test_smooth_window_odd(dset):
    with pytest.raises(ValueError):
        smooth_spec(dset, freq_window=2)


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


def test_scaled(dset):
    dset2 = scaled(dset, hs=2)
    assert dset2.spec.hs().values == pytest.approx(2)


def test_check_same_coordinates(dset):
    ds = dset.isel(lat=0, lon=0, drop=True)
    check_same_coordinates(dset.efth, dset.efth)
    with pytest.raises(ValueError):
        check_same_coordinates(dset.efth, ds.efth)
    with pytest.raises(TypeError):
        check_same_coordinates(dset, ds)


def test_wavelen(dset):
    wlen1 = wavelen(dset.freq, depth=10)
    wlen2 = wavelen(dset.freq, depth=50)
    wlen3 = wavelen(dset.freq, depth=None)
    assert np.sum(wlen1) < np.sum(wlen2) < np.sum(wlen3)
    assert wlen3.values == pytest.approx(1.56 / dset.freq ** 2)


def test_celerity(dset):
    clen1 = celerity(dset.freq, depth=10)
    clen2 = celerity(dset.freq, depth=50)
    clen3 = celerity(dset.freq, depth=None)
    assert np.sum(clen1) < np.sum(clen2) < np.sum(clen3)
    assert clen3.values == pytest.approx(1.56 / dset.freq)


def test_interp_spec(dset):
    inspec = dset.isel(lat=0, lon=0, time=0).efth.values
    inf = dset.freq.values
    ind = dset.dir.values
    outf = np.arange(inf[0], inf[-1], 0.01)
    outd = np.arange(0, 360, 5)
    # 2D spectra
    interp_spec(inspec, inf, ind, outfreq=None, outdir=None, method="linear")
    interp_spec(inspec, inf, ind, outfreq=outf, outdir=None, method="linear")
    interp_spec(inspec, inf, ind, outfreq=None, outdir=outd, method="linear")
    interp_spec(inspec, inf, ind, outfreq=outf, outdir=outd, method="linear")
    interp_spec(inspec, inf, ind, outfreq=outf, outdir=outd, method="cubic")
    interp_spec(inspec, inf, ind, outfreq=outf, outdir=outd, method="nearest")
    # 1D spectra
    inspec = dset.isel(lat=0, lon=0, time=0).spec.oned().values
    interp_spec(inspec, inf, indir=None, outfreq=None, outdir=None, method="linear")
    interp_spec(inspec, inf, indir=None, outfreq=outf, outdir=None, method="linear")

    with pytest.raises(ValueError):
        inspec = dset.efth.values
        interp_spec(inspec, inf, ind, outfreq=outf, outdir=outd, method="nearest")


def test_load_function():
    func = load_function("wavespectra", "fit_jonswap", prefix="fit_")
    func = load_function("wavespectra.directional", "cartwright")
    with pytest.raises(AttributeError):
        func = load_function("wavespectra", "fit_not_defined_spectrum", prefix="fit_")
    with pytest.raises(AttributeError):
        func = load_function("wavespectra.directional", "not_defined_distribution")
