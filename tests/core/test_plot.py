import os
import pytest

from wavespectra import read_swan

FILES_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "../sample_files")


@pytest.fixture(scope="module")
def load_specdataset():
    """Load SpecDset but skip test if matplotlib is not installed."""
    pytest.importorskip("matplotlib")
    dset = read_swan(os.path.join(FILES_DIR, "swanfile.spec"), as_site=True)
    return dset


# def teardown_module():
#     plt = pytest.importorskip("matplotlib.pyplot")
#     plt.show()


def test_single_pcolormesh(load_specdataset):
    dset = load_specdataset.isel(time=0, site=0)
    dset.efth.spec.plot.pcolormesh()


def test_single_contour(load_specdataset):
    dset = load_specdataset.isel(time=0, site=0)
    dset.efth.spec.plot.contour()


def test_single_contourf(load_specdataset):
    dset = load_specdataset.isel(time=0, site=0)
    dset.efth.spec.plot.contourf()


def test_single_as_period(load_specdataset):
    dset = load_specdataset.isel(time=0, site=0)
    dset.efth.spec.plot.contourf(as_period=True)


def test_single_no_log10(load_specdataset):
    dset = load_specdataset.isel(time=0, site=0)
    dset.efth.spec.plot.contourf(as_period=True, as_log10=False)


def test_single_sliced(load_specdataset):
    plt = pytest.importorskip("matplotlib.pyplot")
    fmin = 0.04
    fmax = 0.2
    darray = load_specdataset.isel(time=0, site=0).efth
    darray_sliced1 = darray.sel(freq=slice(fmin, fmax))
    darray_sliced2 = darray.spec.split(fmin=fmin, fmax=fmax)

    darray_sliced1.spec.plot.contourf()
    darray_sliced2.spec.plot.contourf()
    darray.spec.plot.contourf()
    ax = plt.gca()
    ax.set_rmin(fmin)
    ax.set_rmax(fmax)


def test_single_set_properties_xarr_mpl(load_specdataset):
    dset = load_specdataset.isel(time=0, site=0)
    dset.efth.spec.plot.contourf(
        cmap="viridis", vmin=-5, vmax=-2, levels=15, add_colorbar=False
    )


def test_multi(load_specdataset):
    dset = load_specdataset.isel(site=0)
    dset.efth.spec.plot.contourf(
        col="time", col_wrap=3, levels=15, figsize=(15, 8), vmax=-1
    )


def test_multi_clean_axis(load_specdataset):
    dset = load_specdataset.isel(site=0)
    dset.efth.spec.plot.contourf(
        col="time", col_wrap=3, clean_radius=True, clean_sector=True
    )
