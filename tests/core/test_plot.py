import os
import pytest

from wavespectra import read_swan

FILES_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "../sample_files")


@pytest.fixture(scope='module')
def skip_missing_matplotlib():
    """Load SpecDset but skip test if matplotlib is not installed."""
    pytest.importorskip("matplotlib")
    dset = read_swan(os.path.join(FILES_DIR, "swanfile.spec"), as_site=True)
    return dset


def test_single_defaults(skip_missing_matplotlib):
    dset = skip_missing_matplotlib.isel(time=0, site=0)
    dset.efth.spec.plot.contourf()


def test_single_as_period(skip_missing_matplotlib):
    dset = skip_missing_matplotlib.isel(time=0, site=0)
    dset.efth.spec.plot.contourf(as_period=True)


def test_single_no_log10(skip_missing_matplotlib):
    dset = skip_missing_matplotlib.isel(time=0, site=0)
    dset.efth.spec.plot.contourf(as_period=True, as_log10=False)


def test_single_sliced(skip_missing_matplotlib):
    plt = pytest.importorskip("matplotlib.pyplot")
    fmin = 0.04
    fmax = 0.2
    darray = skip_missing_matplotlib.isel(time=0, site=0).efth
    darray_sliced1 = darray.sel(freq=slice(fmin, fmax))
    darray_sliced2 = darray.spec.split(fmin=fmin, fmax=fmax)

    darray_sliced1.spec.plot.contourf()
    darray_sliced2.spec.plot.contourf()
    darray.spec.plot.contourf()
    ax = plt.gca()
    ax.set_rmin(fmin)
    ax.set_rmax(fmax)


def test_single_set_properties_xarr_mpl(skip_missing_matplotlib):
    dset = skip_missing_matplotlib.isel(time=0, site=0)
    dset.efth.spec.plot.contourf(
        cmap="viridis",
        vmin=-5,
        vmax=-2,
        levels=15,
        add_colorbar=False,
    )


def test_multi(skip_missing_matplotlib):
    dset = skip_missing_matplotlib.isel(site=0)
    dset.efth.spec.plot.contourf(
        col="time",
        col_wrap=3,
        levels=15,
        figsize=(15,8),
        vmax=-1
    )


def test_multi_clean_axis(skip_missing_matplotlib):
    dset = skip_missing_matplotlib.isel(site=0)
    dset.efth.spec.plot.contourf(
        col="time",
        col_wrap=3,
        clean_radius=True,
        clean_sector=True
    )

