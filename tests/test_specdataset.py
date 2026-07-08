"""Testing the dataset-aware behaviour of the SpecDataset accessor."""

import os

import numpy as np
import pytest
import xarray as xr

from wavespectra import read_ww3, set_options, get_options
from wavespectra.partition.partition import Partition


FILES_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "sample_files")


@pytest.fixture(autouse=True)
def dataset_transforms():
    """Opt in to the future dataset transforms behaviour in this module."""
    with set_options(dataset_transforms=True):
        yield


TRANSFORMS = {
    "oned": {},
    "split": {"fmin": 0.05, "fmax": 0.4},
    "smooth": {},
    "rotate": {"angle": 30},
    "interp": {"freq": np.arange(0.04, 0.4, 0.02)},
    "to_energy": {},
    "scale_by_hs": {"expr": "2 * hs"},
}


@pytest.fixture(scope="module")
def dset():
    yield read_ww3(os.path.join(FILES_DIR, "ww3file.nc"))


@pytest.mark.parametrize("method, kwargs", TRANSFORMS.items())
def test_transform_returns_dataset_with_variables(dset, method, kwargs):
    """Dataset accessor transforms return Dataset with non-spectral variables."""
    dsout = getattr(dset.spec, method)(**kwargs)
    assert isinstance(dsout, xr.Dataset)
    for name in ["wspd", "wdir", "dpt", "lat", "lon"]:
        assert name in dsout.data_vars


@pytest.mark.parametrize("method, kwargs", TRANSFORMS.items())
def test_transform_dataarray_accessor_unchanged(dset, method, kwargs):
    """DataArray accessor transforms still return a DataArray with same values."""
    darr = getattr(dset.efth.spec, method)(**kwargs)
    assert isinstance(darr, xr.DataArray)
    dsout = getattr(dset.spec, method)(**kwargs)
    specname = "energy" if method == "to_energy" else "efth"
    assert dsout[specname].values == pytest.approx(darr.values, nan_ok=True)


def test_interp_like_returns_dataset(dset):
    dsout = dset.spec.interp_like(dset.efth)
    assert isinstance(dsout, xr.Dataset)
    assert "wspd" in dsout.data_vars
    assert isinstance(dset.efth.spec.interp_like(dset.efth), xr.DataArray)


def test_transform_drops_spectral_variables(dset):
    """Extra variables that depend on spectral dims are dropped."""
    ds = dset.copy()
    ds["extra"] = ds.efth * 2
    dsout = ds.spec.smooth()
    assert "extra" not in dsout
    assert "wspd" in dsout.data_vars


def test_transform_keeps_dataset_attrs(dset):
    ds = dset.copy()
    ds.attrs["title"] = "test title"
    assert ds.spec.oned().attrs["title"] == "test title"


def test_scalar_params_still_dataarray(dset):
    assert isinstance(dset.spec.hs(), xr.DataArray)
    assert isinstance(dset.spec.stats(["hs", "tp"]), xr.Dataset)


def test_transform_chaining(dset):
    dsout = dset.spec.split(fmax=0.2).spec.oned()
    assert isinstance(dsout, xr.Dataset)
    assert "wspd" in dsout.data_vars


class TestPartitionDatasetAware:
    def test_partition_returns_dataset(self, dset):
        dspart = dset.spec.partition.ptm1(swells=2)
        assert isinstance(dspart, xr.Dataset)
        for name in ["wspd", "wdir", "dpt"]:
            assert name in dspart.data_vars
        assert "part" in dspart.efth.dims

    def test_partition_wind_defaults_from_dataset(self, dset):
        implicit = dset.spec.partition.ptm1(swells=2)
        explicit = dset.spec.partition.ptm1(
            wspd=dset.wspd, wdir=dset.wdir, dpt=dset.dpt, swells=2
        )
        xr.testing.assert_allclose(implicit, explicit)

    @pytest.mark.parametrize("method", ["ptm2", "ptm4"])
    def test_partition_wind_defaults_other_methods(self, dset, method):
        dspart = getattr(dset.spec.partition, method)()
        assert isinstance(dspart, xr.Dataset)

    def test_partition_dataarray_returns_dataarray(self, dset):
        dspart = dset.efth.spec.partition.ptm1(
            wspd=dset.wspd, wdir=dset.wdir, dpt=dset.dpt, swells=2
        )
        assert isinstance(dspart, xr.DataArray)

    def test_partition_wind_required_without_dataset(self, dset):
        with pytest.raises(ValueError):
            dset.efth.spec.partition.ptm1()

    def test_partition_no_wind_dims(self, dset):
        dspart = dset.spec.partition.ptm5(fcut=0.1)
        assert isinstance(dspart, xr.Dataset)
        assert "wspd" in dspart.data_vars

    def test_track_returns_dataset_with_variables(self, dset):
        dsout = dset.spec.partition.track(swells=2)
        assert isinstance(dsout, xr.Dataset)
        for name in ["efth", "track_id", "ntracks", "wspd", "wdir", "dpt"]:
            assert name in dsout.data_vars

    def test_track_dataarray_no_extra_variables(self, dset):
        dsout = Partition(dset.efth).track(
            wspd=dset.wspd, wdir=dset.wdir, dpt=dset.dpt, swells=2
        )
        assert isinstance(dsout, xr.Dataset)
        assert "wspd" not in dsout.data_vars


class TestDefaultBehaviour:
    """Without the dataset_transforms option the old behaviour is preserved."""

    def test_transform_warns_and_returns_dataarray(self, dset):
        with set_options(dataset_transforms=False):
            with pytest.warns(FutureWarning, match="oned"):
                dsout = dset.spec.oned()
        assert isinstance(dsout, xr.DataArray)

    def test_transform_values_unchanged(self, dset):
        with set_options(dataset_transforms=False):
            with pytest.warns(FutureWarning):
                darr = dset.spec.smooth()
        xr.testing.assert_allclose(darr, dset.efth.spec.smooth())

    def test_dataarray_accessor_does_not_warn(self, dset):
        import warnings

        with set_options(dataset_transforms=False):
            with warnings.catch_warnings():
                warnings.simplefilter("error", FutureWarning)
                darr = dset.efth.spec.oned()
        assert isinstance(darr, xr.DataArray)

    def test_partition_warns_and_returns_dataarray(self, dset):
        with set_options(dataset_transforms=False):
            with pytest.warns(FutureWarning, match="Partitioning"):
                dspart = dset.spec.partition.ptm1(
                    wspd=dset.wspd, wdir=dset.wdir, dpt=dset.dpt, swells=2
                )
        assert isinstance(dspart, xr.DataArray)

    def test_hp01_does_not_default_wind_from_dataset(self, dset):
        with set_options(dataset_transforms=False):
            with pytest.warns((FutureWarning, UserWarning)):
                dspart = dset.spec.partition.hp01(swells=2)
        assert isinstance(dspart, xr.DataArray)
        # No wind classification, partition 0 is null
        assert float(dspart.isel(part=0).sum()) == 0


class TestSetOptions:
    def test_option_toggles_and_restores(self, dset):
        with set_options(dataset_transforms=False):
            assert get_options()["dataset_transforms"] is False
            with set_options(dataset_transforms=True):
                assert get_options()["dataset_transforms"] is True
                assert isinstance(dset.spec.oned(), xr.Dataset)
            assert get_options()["dataset_transforms"] is False

    def test_invalid_option_raises(self):
        with pytest.raises(ValueError):
            set_options(not_an_option=True)
