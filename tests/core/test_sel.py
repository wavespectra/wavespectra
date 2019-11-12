"""Unit testing for SpecDataset wrapper around DataArray."""
import os
import pytest
import numpy as np

from wavespectra.core.attributes import attrs
from wavespectra import read_ww3

FILES_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "../sample_files")


class TestDatasetWrapper(object):
    """Test SpecDataset wrapper."""

    @classmethod
    def setup_class(self):
        """Read test spectra from file."""
        self.dset = read_ww3(os.path.join(FILES_DIR, "ww3file.nc"))
        # First two sites are exact matches, third site is in between
        self.lons = [92.00, 92.10, 92.05]
        self.lats = [19.80, 19.95, 19.88]
        self.lons_exact = self.lons[:2]
        self.lats_exact = self.lats[:2]
        self.lons_inexact = self.lons[-1:]
        self.lats_inexact = self.lats[-1:]

    def test_dset_sel_idw(self):
        """Assert that sel/idw method runs."""
        dset = self.dset.spec.sel(
            lons=self.lons, lats=self.lats, method="idw", tolerance=1.0
        )
        assert dset[attrs.SITENAME].size == len(self.lons)

    def test_dset_sel_bbox(self):
        """Assert that sel/bbox method runs."""
        dset = self.dset.spec.sel(
            lons=self.dset.lon.values,
            lats=self.dset.lat.values,
            method="bbox",
            tolerance=0.0,
        )
        dset[attrs.SITENAME].size == self.dset[attrs.SITENAME].size

    def test_dset_sel_nearest(self):
        """Assert that sel/nearest method runs."""
        dset = self.dset.spec.sel(
            lons=self.lons, lats=self.lats, method="nearest", tolerance=1.0
        )
        assert dset[attrs.SITENAME].size == len(self.lons)

    def test_dset_sel_nearest_unique(self):
        """Assert duplicated sites with same neighbours are removed."""
        dset = self.dset.spec.sel(
            lons=self.lons, lats=self.lats, method="nearest", unique=True
        )
        assert dset[attrs.SITENAME].size == len(self.lons_exact)

    def test_dset_sel_none(self):
        """Assert that sel/none method runs."""
        dset = self.dset.spec.sel(
            lons=self.lons_exact, lats=self.lats_exact, method=None
        )
        with pytest.raises(AssertionError):
            dset = self.dset.spec.sel(lons=self.lons, lats=self.lats, method=None)

    def test_tolerance(self):
        """Test tolerance is working as expected."""
        # With idw no sites within tolerance should result in masked output
        dset = self.dset.spec.sel(
            lons=self.lons_inexact, lats=self.lats_inexact, method="idw", tolerance=1.0
        )
        assert dset.spec.hs().values.all()
        dset = self.dset.spec.sel(
            lons=self.lons_inexact, lats=self.lats_inexact, method="idw", tolerance=0.01
        )
        assert np.isnan(dset.spec.hs().values).any()
        # With nearest no sites within tolerance should raise an exception
        with pytest.raises(AssertionError):
            dset = self.dset.spec.sel(
                lons=self.lons_inexact,
                lats=self.lats_inexact,
                method="nearest",
                tolerance=0.01,
            )

    def test_exact_matches(self):
        """Test that exact matches are the same regardless of method."""
        idw = self.dset.spec.sel(
            lons=self.lons_exact, lats=self.lats_exact, method="idw"
        )
        nearest = self.dset.spec.sel(
            lons=self.lons_exact, lats=self.lats_exact, method="nearest"
        )
        assert abs(idw.efth - nearest.efth).max() == 0

    def test_weighting(self):
        """Assert that stats in interpolated site are constrained within neighbours."""
        dset = self.dset.spec.sel(
            lons=self.lons_inexact, lats=self.lats_inexact, method="idw"
        )
        for stat in ["hs", "tp"]:
            idw = dset.spec.stats([stat])[stat].values
            site0 = self.dset.isel(site=[0]).spec.stats([stat])[stat].values
            site1 = self.dset.isel(site=[1]).spec.stats([stat])[stat].values
            lower = np.array([min(s1, s2) for s1, s2 in zip(site0, site1)])
            upper = np.array([max(s1, s2) for s1, s2 in zip(site0, site1)])
            assert (upper - idw > 0).all() and (idw - lower > 0).all()
