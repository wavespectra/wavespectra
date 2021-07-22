import os
import pytest

from wavespectra import read_swan
from wavespectra.plot import WavePlot, LOG_CONTOUR_LEVELS


FILES_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "../sample_files")


@pytest.fixture(scope="module")
def load_specdataset():
    """Load SpecDset but skip test if matplotlib is not installed."""
    pytest.importorskip("matplotlib")
    dset = read_swan(os.path.join(FILES_DIR, "swanfile.spec"), as_site=True)
    return dset


# def teardown_function():
#     plt = pytest.importorskip("matplotlib.pyplot")
#     plt.show()


def test_single_pcolormesh(load_specdataset):
    dset = load_specdataset.isel(time=0, site=0)
    dset.efth.spec.plot(kind="pcolormesh")


def test_single_contour(load_specdataset):
    dset = load_specdataset.isel(time=0, site=0)
    dset.efth.spec.plot(kind="contour")


def test_single_contourf(load_specdataset):
    dset = load_specdataset.isel(time=0, site=0)
    dset.efth.spec.plot(kind="contourf")


def test_single_as_period(load_specdataset):
    dset = load_specdataset.isel(time=0, site=0)
    dset.efth.spec.plot(kind="contourf", as_period=True)


def test_single_no_logradius_frequency(load_specdataset):
    dset = load_specdataset.isel(time=0, site=0)
    dset.efth.spec.plot(kind="contourf", as_period=False, logradius=False)


def test_single_no_logradius_period(load_specdataset):
    dset = load_specdataset.isel(time=0, site=0)
    dset.efth.spec.plot(kind="contourf", as_period=True, logradius=False)


def test_single_not_normalised(load_specdataset):
    dset = load_specdataset.isel(site=0, time=0)
    pobj = dset.efth.spec.plot(kind="contourf", normalised=False)
    assert pobj.levels != pytest.approx(LOG_CONTOUR_LEVELS)


def test_single_sliced(load_specdataset):
    fmin = 0.04
    fmax = 0.2
    darray = load_specdataset.isel(time=0, site=0).efth
    darray_sliced1 = darray.sel(freq=slice(fmin, fmax))
    darray_sliced2 = darray.spec.split(fmin=fmin, fmax=fmax)
    darray_sliced1.spec.plot(kind="contourf")
    darray_sliced2.spec.plot(kind="contourf")
    darray.spec.plot(kind="contourf", rmin=fmin, rmax=fmax)


def test_single_set_properties_xarr_mpl(load_specdataset):
    dset = load_specdataset.isel(time=0, site=0)
    dset.efth.spec.plot(
        kind="contourf", cmap="viridis", vmin=0, vmax=1, levels=15, add_colorbar=False
    )


def test_multi(load_specdataset):
    dset = load_specdataset.isel(site=0)
    dset.efth.spec.plot(
        kind="contourf", col="time", col_wrap=3, levels=15, figsize=(15, 8)
    )


def test_with_ticklabels(load_specdataset):
    dset = load_specdataset.isel(site=0, time=0)
    pobj = dset.efth.spec.plot(
        kind="contourf", show_theta_labels=True, show_radii_labels=True
    )
    xlabels_pos = [tick.get_position() for tick in pobj.axes.get_xticklabels()]
    ylabels_pos = [tick.get_position() for tick in pobj.axes.get_yticklabels()]
    # assert set(xlabels_pos) == {(0, 0)}
    # assert set(ylabels_pos) == {(0, 0)}


def test_without_ticklabels(load_specdataset):
    dset = load_specdataset.isel(site=0, time=0)
    pobj = dset.efth.spec.plot(
        kind="contourf", show_theta_labels=False, show_radii_labels=False
    )
    xlabels_pos = [tick.get_position() for tick in pobj.axes.get_xticklabels()]
    ylabels_pos = [tick.get_position() for tick in pobj.axes.get_yticklabels()]
    # assert set(xlabels_pos) != {(0, 0)}
    # assert set(ylabels_pos) != {(0, 0)}


def test_default_contourf_log_colour(load_specdataset):
    dset = load_specdataset.isel(site=0, time=0)
    pobj = dset.efth.spec.plot(kind="contourf", normalised=True)
    assert pobj.levels == pytest.approx(LOG_CONTOUR_LEVELS)


def test_implemented(load_specdataset):
    dset = load_specdataset.isel(site=0, time=0)
    with pytest.raises(NotImplementedError):
        dset.efth.spec.plot(kind="imshow")


def test_radii_in_range(load_specdataset):
    dset = load_specdataset.isel(site=0, time=0)
    with pytest.raises(ValueError):
        dset.efth.spec.plot(kind="contourf", rmin=0.05, rmax=0.4, as_period=True)


def test_repr(load_specdataset):
    dset = load_specdataset.isel(site=0, time=0)
    wp = WavePlot(dset.efth)
    assert f"<Waveplot {wp.kind}" in repr(wp)


def test_rlim(load_specdataset):
    dset = load_specdataset.isel(site=0, time=0)
    with pytest.raises(ValueError):
        dset.spec.plot(kind="contourf", rmin=-1, rmax=0)
        dset.spec.plot(kind="contourf", rmin=-1, rmax=0, logradius=False)
