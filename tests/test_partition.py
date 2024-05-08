import pytest
from pathlib import Path
import numpy as np
import xarray as xr

from wavespectra import read_ww3
from wavespectra.partition.partition import (
    Partition,
    np_ptm1,
    np_ptm2,
    np_ptm3,
    np_hp01,
)
from wavespectra.core.utils import waveage
from wavespectra.core import npstats
from wavespectra.partition.tracking import track_partitions


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
        return npstats.hs(self.efth, self.freq, self.dir)

    @property
    def hs_from_partitions(self):
        hs_partitions = [npstats.hs(efth, self.freq, self.dir) for efth in self.out]
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
        self.out = self._exec(swells=swells)
        assert self.hs_from_partitions < self.hs_full
        assert len(self.out) == swells + 1

    def test_partition_class(self):
        swells = 2
        self.out = self._exec(swells=swells)
        dspart = self.pt.ptm1(
            wspd=self.dset.wspd,
            wdir=self.dset.wdir,
            dpt=self.dset.dpt,
            swells=swells,
        )
        hs_dspart = np.sqrt(np.sum(dspart.isel(time=0, site=0).spec.hs().values ** 2))
        assert hs_dspart == pytest.approx(self.hs_from_partitions, rel=1e-2)

    def test_smoothing(self):
        swells = 2
        ds_nosmoothing = self.pt.ptm1(
            wspd=self.dset.wspd,
            wdir=self.dset.wdir,
            dpt=self.dset.dpt,
            swells=swells,
        )
        ds_smoothing = self.pt.ptm1(
            wspd=self.dset.wspd,
            wdir=self.dset.wdir,
            dpt=self.dset.dpt,
            swells=swells,
            smooth=True,
        )
        assert not ds_nosmoothing.spec.hs().equals(ds_smoothing.spec.hs())


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
        self.out = self._exec(swells=swells)
        assert self.hs_from_partitions < self.hs_full
        assert len(self.out) == swells + 2

    def test_less_partitions_combined(self):
        swells = 2
        self.out = self._exec(swells=swells)
        assert self.hs_full == pytest.approx(self.hs_from_partitions)
        assert len(self.out) == swells + 2

    def test_partition_class(self):
        swells = 2
        combine = False
        self.out = self._exec(swells=swells)
        dspart = self.pt.ptm2(
            wspd=self.dset.wspd,
            wdir=self.dset.wdir,
            dpt=self.dset.dpt,
            swells=swells,
        )
        hs_dspart = np.sqrt(np.sum(dspart.isel(time=0, site=0).spec.hs().values ** 2))
        assert hs_dspart == pytest.approx(self.hs_from_partitions, rel=1e-2)

    def test_smoothing(self):
        swells = 2
        ds_nosmoothing = self.pt.ptm1(
            wspd=self.dset.wspd,
            wdir=self.dset.wdir,
            dpt=self.dset.dpt,
            swells=swells,
        )
        ds_smoothing = self.pt.ptm1(
            wspd=self.dset.wspd,
            wdir=self.dset.wdir,
            dpt=self.dset.dpt,
            swells=swells,
            smooth=True,
        )
        assert not ds_nosmoothing.spec.hs().equals(ds_smoothing.spec.hs())


class TestPTM3(BasePTM):
    def _exec(self, parts):
        out = np_ptm3(
            spectrum=self.dset.efth.isel(time=0, site=0).values,
            spectrum_smooth=self.dset.efth.isel(time=0, site=0).values,
            freq=self.dset.freq.values,
            dir=self.dset.dir.values,
            parts=parts,
        )
        return out

    def test_request_all_partitions(self):
        self.out = self._exec(parts=None)
        assert self.hs_full == pytest.approx(self.hs_from_partitions)

    def test_request_less_partitions(self):
        parts = 2
        self.out = self._exec(parts=parts)
        assert self.hs_from_partitions < self.hs_full
        assert len(self.out) == parts

    def test_more_partitions(self):
        parts = 5
        self.out = self._exec(parts=parts)
        assert self.hs_full == pytest.approx(self.hs_from_partitions)
        assert len(self.out) == parts
        assert self.out[3:4].sum() == 0

    def test_partition_class(self):
        parts = 2
        self.out = self._exec(parts=parts)
        dspart = self.pt.ptm3(parts=parts, smooth=False)
        hs_dspart = np.sqrt(np.sum(dspart.isel(time=0, site=0).spec.hs().values ** 2))
        assert hs_dspart == pytest.approx(self.hs_from_partitions)

    def test_partition_class_smoothing(self):
        parts = 3
        ds_nosmoothing = self.pt.ptm3(parts=parts)
        ds_smoothing = self.pt.ptm3(parts=parts, smooth=True)
        assert not ds_nosmoothing.spec.hs().equals(ds_smoothing.spec.hs())


class TestPTM4(BasePTM):
    def test_agefac(self):
        ds_agefac15 = self.pt.ptm4(
            wspd=self.dset.wspd, wdir=self.dset.wdir, dpt=self.dset.dpt, agefac=1.5
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


class TestHP01(BasePTM):
    def setup_class(self):
        super().setup_class(self)
        self.agefac = 1.7
        self.wscut = 0.3333
        self.swells = 3
        wspd = self.dset.wspd.isel(time=0, site=0)
        wdir = self.dset.wdir.isel(time=0, site=0)
        dpt = self.dset.dpt.isel(time=0, site=0)
        self.windseamask = waveage(
            self.dset.freq, self.dset.dir, wspd, wdir, dpt, self.agefac
        )

    def _exec(self, **kwargs):
        out = np_hp01(
            spectrum=self.efth,
            spectrum_smooth=self.efth,
            windseamask=self.windseamask,
            freq=self.freq,
            dir=self.dir,
            wscut=kwargs.get("wscut", self.wscut),
            swells=kwargs.get("swells", self.swells),
        )
        return out

    def test_defaults(self):
        self.out = self._exec()
        # assert self.out.shape[0] == self.swells + 1
        # assert self.hs_full == pytest.approx(self.hs_from_partitions)

    # def test_request_all_partitions(self):
    #     self.out = self._exec(swells=None)
    #     assert self.out.shape[0] == 3
    #     assert self.hs_full == pytest.approx(self.hs_from_partitions)

    # def test_request_less_partitions(self):
    #     swells = 2
    #     self.out = self._exec(swells=swells)
    #     assert self.hs_from_partitions < self.hs_full
    #     assert len(self.out) == swells + 1

    def test_partition_class(self):
        swells = 2
        self.out = self._exec(swells=swells)
        dspart = self.pt.hp01(
            wspd=self.dset.wspd,
            wdir=self.dset.wdir,
            dpt=self.dset.dpt,
            agefac=self.agefac,
            swells=swells,
        )
        hs_dspart = np.sqrt(np.sum(dspart.isel(time=0, site=0).spec.hs().values ** 2))
        # assert hs_dspart == pytest.approx(self.hs_from_partitions, rel=1e-2)

    # def test_smoothing(self):
    #     swells = 2
    #     ds_nosmoothing = self.pt.ptm1(
    #         wspd=self.dset.wspd,
    #         wdir=self.dset.wdir,
    #         dpt=self.dset.dpt,
    #         swells=swells,
    #     )
    #     ds_smoothing = self.pt.ptm1(
    #         wspd=self.dset.wspd,
    #         wdir=self.dset.wdir,
    #         dpt=self.dset.dpt,
    #         swells=swells,
    #         smooth=True,
    #     )
    #     assert not ds_nosmoothing.spec.hs().equals(ds_smoothing.spec.hs())

    # def test_compare_legacy(self):
    #     kwargs = {"agefac": self.agefac, "wscut": self.wscut, "swells": self.swells}
    #     old = self.dset.spec.partition_deprecated(
    #         wsp_darr=self.dset.wspd,
    #         wdir_darr=self.dset.wdir,
    #         dep_darr=self.dset.dpt,
    #         **kwargs,
    #     )
    #     new = self.pt.ptm1(
    #         wspd=self.dset.wspd, wdir=self.dset.wdir, dpt=self.dset.dpt, **kwargs,
    #     )
    #     assert  np.array_equal(old.spec.hs().values, new.spec.hs().values)


class TestBbox(BasePTM):
    def test_default(self):
        bboxes = [
            dict(fmin=0.1, fmax=0.2, dmin=None, dmax=None),
            dict(fmin=0.2, fmax=0.3, dmin=None, dmax=None),
        ]
        ds = self.pt.bbox(bboxes=bboxes)
        assert ds.part.size == 3

    def test_ignore_dlimits(self):
        bboxes = [dict(fmin=0.1, fmax=0.2), dict(fmin=0.2, fmax=0.3)]
        ds = self.pt.bbox(bboxes=bboxes)
        assert ds.part.size == 3

    def test_overlap(self):
        bboxes = [
            dict(fmin=0.1, fmax=0.2, dmin=None, dmax=None),
            dict(fmin=0.15, fmax=0.3, dmin=None, dmax=None),
        ]
        with pytest.raises(ValueError):
            ds = self.pt.bbox(bboxes=bboxes)

    def test_freq_increasing(self):
        bboxes = [dict(fmin=0.2, fmax=0.1, dmin=None, dmax=None)]
        with pytest.raises(ValueError):
            ds = self.pt.bbox(bboxes=bboxes)


class TestPartitionAndTrack(BasePTM):
    def setup_class(self):
        super().setup_class(self)

    def test_partition_class(self):
        swells = 2
        dspart = self.pt.ptm1(
            wspd=self.dset.wspd,
            wdir=self.dset.wdir,
            dpt=self.dset.dpt,
            swells=swells,
        )
        stats = dspart.spec.stats(["fp", "dpm"]).load()
        part_ids = track_partitions(stats, wspd=self.dset.wspd)

    def test_class(self):
        swells = 2
        dspart = self.pt.ptm1_track(
            wspd=self.dset.wspd,
            wdir=self.dset.wdir,
            dpt=self.dset.dpt,
            swells=swells,
        )
        assert "part_id" in dspart
        assert "npart_id" in dspart
