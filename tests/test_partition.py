import pytest
from pathlib import Path
import numpy as np
import xarray as xr

from wavespectra import read_ww3
from wavespectra.partition.partition import Partition, np_ptm1, np_ptm2, np_ptm3
from wavespectra.core.npstats import hs


HERE = Path(__file__).parent


class BasePTM:

    def setup_class(self):
        self.dset = read_ww3(HERE / "sample_files/ww3file.nc")
        self.efth = self.dset.isel(time=0, site=0).efth.values
        self.freq = self.dset.freq.values
        self.dir = self.dset.dir.values
        self.pt = Partition(self.dset)

    @property
    def hs_full(self):
        return hs(self.efth, self.freq, self.dir)

    @property
    def hs_from_partitions(self):
        hs_partitions = [hs(efth, self.freq, self.dir) for efth in self.out]
        return np.sqrt(np.sum(np.power(hs_partitions, 2)))


class TestPTM1(BasePTM):

    def setup_class(self):
        super().setup_class(self)
        self.wspd = self.dset.isel(time=0, site=0).wspd.values
        self.wdir = self.dset.isel(time=0, site=0).wdir.values
        self.dpt = self.dset.isel(time=0, site=0).dpt.values
        self.agefac = 1.7
        self.wscut = 0.3333
        self.swells = 3

    def _exec(self, **kwargs):
        out = np_ptm1(
            spectrum=self.efth,
            spectrum_smooth=self.efth,
            freq=self.freq,
            dir=self.dir,
            wspd=self.wspd,
            wdir=self.wdir,
            dpt=self.dpt,
            agefac=kwargs.get("agefac", self.agefac),
            wscut=kwargs.get("wscut", self.wscut),
            swells=kwargs.get("swells", self.swells),
            combine=kwargs.get("combine", False),
        )
        return out

    def test_defaults(self):
        self.out = self._exec()
        assert self.out.shape[0] == self.swells + 1
        assert self.hs_full == pytest.approx(self.hs_from_partitions)

    def test_request_all_partitions(self):
        self.out = self._exec(swells=None)
        assert self.out.shape[0] == 3
        assert self.hs_full == pytest.approx(self.hs_from_partitions)

    def test_request_less_partitions(self):
        swells = 2
        self.out = self._exec(swells=swells, combine=False)
        assert self.hs_from_partitions < self.hs_full
        assert len(self.out) == swells + 1

    def test_less_partitions_combined(self):
        swells = 2
        self.out = self._exec(swells=swells, combine=True)
        assert self.hs_full == pytest.approx(self.hs_from_partitions)
        assert len(self.out) == swells + 1

    def test_partition_class(self):
        swells = 2
        combine = False
        self.out = self._exec(swells=swells, combine=combine)
        dspart = self.pt.ptm1(
            wspd=self.dset.wspd,
            wdir=self.dset.wdir,
            dpt=self.dset.dpt,
            swells=swells,
            combine=combine,
        )
        hs_dspart = np.sqrt(np.sum(dspart.isel(time=0, site=0).spec.hs().values ** 2))
        assert hs_dspart == pytest.approx(self.hs_from_partitions, rel=1e-2)

    def test_smoothing(self):
        swells = 2
        combine = True
        ds_nosmoothing = self.pt.ptm1(
            wspd=self.dset.wspd,
            wdir=self.dset.wdir,
            dpt=self.dset.dpt,
            swells=swells,
            combine=combine,
        )
        ds_smoothing = self.pt.ptm1(
            wspd=self.dset.wspd,
            wdir=self.dset.wdir,
            dpt=self.dset.dpt,
            swells=swells,
            combine=combine,
            smooth=True,
        )
        hs_nosmoothing = np.sqrt((ds_nosmoothing.spec.hs() ** 2).sum("part")).values
        hs_smoothing = np.sqrt((ds_smoothing.spec.hs() ** 2).sum("part")).values
        assert not ds_nosmoothing.spec.hs().equals(ds_smoothing.spec.hs())
        assert np.allclose(hs_nosmoothing, hs_smoothing, rtol=1e-5)

    def test_compare_legacy(self):
        kwargs = {"agefac": self.agefac, "wscut": self.wscut, "swells": self.swells}
        old = self.dset.spec.partition_deprecated(
            wsp_darr=self.dset.wspd,
            wdir_darr=self.dset.wdir,
            dep_darr=self.dset.dpt,
            **kwargs,
        )
        new = self.pt.ptm1(
            wspd=self.dset.wspd, wdir=self.dset.wdir, dpt=self.dset.dpt, **kwargs,
        )
        assert  np.array_equal(old.spec.hs().values, new.spec.hs().values)


class TestPTM2(BasePTM):

    def setup_class(self):
        super().setup_class(self)
        self.wspd = self.dset.isel(time=0, site=0).wspd.values
        self.wdir = self.dset.isel(time=0, site=0).wdir.values
        self.dpt = self.dset.isel(time=0, site=0).dpt.values
        self.agefac = 1.7
        self.wscut = 0.3333
        self.swells = 3

    def _exec(self, **kwargs):
        out = np_ptm2(
            spectrum=self.efth,
            spectrum_smooth=self.efth,
            freq=self.freq,
            dir=self.dir,
            wspd=self.wspd,
            wdir=self.wdir,
            dpt=self.dpt,
            agefac=kwargs.get("agefac", self.agefac),
            wscut=kwargs.get("wscut", self.wscut),
            swells=kwargs.get("swells", self.swells),
            combine=kwargs.get("combine", False),
        )
        return out

    def test_defaults(self):
        self.out = self._exec()
        assert self.out.shape[0] == self.swells + 2
        assert self.hs_full == pytest.approx(self.hs_from_partitions)

    def test_request_all_partitions(self):
        self.out = self._exec(swells=None)
        assert self.out.shape[0] == 4
        assert self.hs_full == pytest.approx(self.hs_from_partitions)

    def test_request_less_partitions(self):
        swells = 2
        self.out = self._exec(swells=swells, combine=False)
        assert self.hs_from_partitions < self.hs_full
        assert len(self.out) == swells + 2

    def test_less_partitions_combined(self):
        swells = 2
        self.out = self._exec(swells=swells, combine=True)
        assert self.hs_full == pytest.approx(self.hs_from_partitions)
        assert len(self.out) == swells + 2

    def test_partition_class(self):
        swells = 2
        combine = False
        self.out = self._exec(swells=swells, combine=combine)
        dspart = self.pt.ptm2(
            wspd=self.dset.wspd,
            wdir=self.dset.wdir,
            dpt=self.dset.dpt,
            swells=swells,
            combine=combine,
        )
        hs_dspart = np.sqrt(np.sum(dspart.isel(time=0, site=0).spec.hs().values ** 2))
        assert hs_dspart == pytest.approx(self.hs_from_partitions, rel=1e-2)

    def _test_smoothing(self):
        swells = 2
        combine = True
        ds_nosmoothing = self.pt.ptm1(
            wspd=self.dset.wspd,
            wdir=self.dset.wdir,
            dpt=self.dset.dpt,
            swells=swells,
            combine=combine,
        )
        ds_smoothing = self.pt.ptm1(
            wspd=self.dset.wspd,
            wdir=self.dset.wdir,
            dpt=self.dset.dpt,
            swells=swells,
            combine=combine,
            smooth=True,
        )
        hs_nosmoothing = np.sqrt((ds_nosmoothing.spec.hs() ** 2).sum("part")).values
        hs_smoothing = np.sqrt((ds_smoothing.spec.hs() ** 2).sum("part")).values
        assert not ds_nosmoothing.spec.hs().equals(ds_smoothing.spec.hs())
        assert np.allclose(hs_nosmoothing, hs_smoothing, rtol=1e-5)


class TestPTM3(BasePTM):

    def _exec(self, parts, combine):
        out = np_ptm3(
            spectrum=self.dset.efth.isel(time=0, site=0).values,
            spectrum_smooth=self.dset.efth.isel(time=0, site=0).values,
            freq=self.dset.freq.values,
            dir=self.dset.dir.values,
            parts=parts,
            combine=combine,
        )
        return out

    def test_request_all_partitions(self):
        self.out = self._exec(parts=None, combine=False)
        assert self.hs_full == pytest.approx(self.hs_from_partitions)

    def test_request_less_partitions(self):
        parts = 2
        self.out = self._exec(parts=parts, combine=False)
        assert self.hs_from_partitions < self.hs_full
        assert len(self.out) == parts

    def test_less_partitions_combined(self):
        parts = 2
        self.out = self._exec(parts=parts, combine=True)
        assert self.hs_full == pytest.approx(self.hs_from_partitions)
        assert len(self.out) == parts

    def test_more_partitions(self):
        parts = 5
        self.out = self._exec(parts=parts, combine=False)
        assert self.hs_full == pytest.approx(self.hs_from_partitions)
        assert len(self.out) == parts
        assert self.out[3:4].sum() == 0

    def test_partition_class(self):
        parts = 2
        combine = False
        self.out = self._exec(parts=parts, combine=combine)
        dspart = self.pt.ptm3(parts=parts, smooth=False, combine=combine)
        hs_dspart = np.sqrt(np.sum(dspart.isel(time=0, site=0).spec.hs().values ** 2))
        assert hs_dspart == pytest.approx(self.hs_from_partitions)

    def test_partition_class_smoothing(self):
        parts = 3
        combine = True
        ds_nosmoothing = self.pt.ptm3(parts=parts, combine=combine)
        ds_smoothing = self.pt.ptm3(parts=parts, combine=combine, smooth=True)
        hs_nosmoothing = np.sqrt((ds_nosmoothing.spec.hs() ** 2).sum("part")).values
        hs_smoothing = np.sqrt((ds_smoothing.spec.hs() ** 2).sum("part")).values
        assert not ds_nosmoothing.spec.hs().equals(ds_smoothing.spec.hs())
        assert np.allclose(hs_nosmoothing, hs_smoothing, rtol=1e-5)


class TestPTM4(BasePTM):

    def test_agefac(self):
        ds_agefac15 = self.pt.ptm4(
            wspd=self.dset.wspd,
            wdir=self.dset.wdir,
            dpt=self.dset.dpt,
            agefac=1.5
        )
        ds_agefac17 = self.pt.ptm4(
            wspd=self.dset.wspd,
            wdir=self.dset.wdir,
            dpt=self.dset.dpt,
        )
        sea_agefac15 = ds_agefac15.isel(part=0).spec.hs()
        sea_agefac17 = ds_agefac17.isel(part=0).spec.hs()
        assert np.all((sea_agefac17.values - sea_agefac15.values) > 0)


class TestPTM5(BasePTM):

    def test_default(self):
        fcut = 0.1
        ds = self.pt.ptm5(fcut=0.1)
        assert fcut in ds.freq
        assert ds.isel(part=0).sel(freq=slice(None, fcut - 0.001)).spec.hs().max() == 0
        assert ds.isel(part=1).sel(freq=slice(fcut + 0.001, None)).spec.hs().max() == 0

    def test_not_interpolate(self):
        fcut = 0.1
        ds = self.pt.ptm5(fcut=0.1, interpolate=False)
        assert fcut not in ds.freq
