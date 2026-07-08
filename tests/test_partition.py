import pytest
from pathlib import Path
import numpy as np

import wavespectra
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


@pytest.fixture(autouse=True)
def dataset_transforms():
    """Run under the future dataset transforms behaviour in this module."""
    with wavespectra.set_options(dataset_transforms=True):
        yield


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

    def test_request_all_partitions_class(self):
        """swells=None detects the required number of partitions."""
        dspart = self.pt.ptm1(
            wspd=self.dset.wspd,
            wdir=self.dset.wdir,
            dpt=self.dset.dpt,
            swells=None,
        )
        hs_dspart = np.sqrt((dspart.spec.hs() ** 2).sum("part"))
        assert np.allclose(hs_dspart.values, self.dset.spec.hs().values, rtol=1e-2)
        # No swell slot is null across all spectra
        hs_swells = dspart.isel(part=slice(1, None)).spec.hs()
        assert (hs_swells > 0).any(["time", "site"]).all()

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

    def test_request_all_partitions_class(self):
        """swells=None detects the required number of partitions."""
        dspart = self.pt.ptm2(
            wspd=self.dset.wspd,
            wdir=self.dset.wdir,
            dpt=self.dset.dpt,
            swells=None,
        )
        hs_dspart = np.sqrt((dspart.spec.hs() ** 2).sum("part"))
        assert np.allclose(hs_dspart.values, self.dset.spec.hs().values, rtol=1e-2)
        # No swell slot is null across all spectra
        hs_swells = dspart.isel(part=slice(2, None)).spec.hs()
        assert (hs_swells > 0).any(["time", "site"]).all()

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

    def test_request_all_partitions_class(self):
        """parts=None detects the required number of partitions."""
        dspart = self.pt.ptm3(parts=None, smooth=False)
        hs_dspart = np.sqrt((dspart.spec.hs() ** 2).sum("part"))
        assert np.allclose(hs_dspart.values, self.dset.spec.hs().values, rtol=1e-2)
        # No partition slot is null across all spectra
        assert (dspart.spec.hs() > 0).any(["time", "site"]).all()

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
        assert self.out.shape[0] == self.swells + 1
        assert self.hs_full == pytest.approx(self.hs_from_partitions)

    def test_request_all_partitions(self):
        """swells=None detects the required number of partitions."""
        dspart = self.pt.hp01(
            wspd=self.dset.wspd,
            wdir=self.dset.wdir,
            dpt=self.dset.dpt,
            agefac=self.agefac,
            swells=None,
        )
        hs_dspart = np.sqrt((dspart.spec.hs() ** 2).sum("part"))
        assert np.allclose(hs_dspart.values, self.dset.spec.hs().values, rtol=1e-2)
        # No partition slot is null across all spectra
        assert (dspart.spec.hs() > 0).any(["time", "site"]).all()

    def test_request_less_partitions(self):
        swells = 1
        self.out = self._exec(swells=swells)
        assert self.hs_full == pytest.approx(self.hs_from_partitions)
        assert len(self.out) == swells + 1

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
        assert hs_dspart == pytest.approx(self.hs_from_partitions, rel=1e-2)

    def test_smoothing(self):
        kwargs = dict(
            wspd=self.dset.wspd, wdir=self.dset.wdir, dpt=self.dset.dpt, swells=2
        )
        ds_nosmoothing = self.pt.hp01(**kwargs)
        ds_smoothing = self.pt.hp01(smooth=True, **kwargs)
        assert not ds_nosmoothing.spec.hs().equals(ds_smoothing.spec.hs())
        # Smoothing only changes the partition boundaries, energy is conserved
        hs = np.sqrt((ds_smoothing.spec.hs() ** 2).sum("part"))
        assert np.allclose(hs.values, self.dset.spec.hs().values, rtol=1e-2)


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
            self.pt.bbox(bboxes=bboxes)

    def test_freq_increasing(self):
        bboxes = [dict(fmin=0.2, fmax=0.1, dmin=None, dmax=None)]
        with pytest.raises(ValueError):
            self.pt.bbox(bboxes=bboxes)


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
        track_partitions(stats, wspd=self.dset.wspd)

    def _track(self, method, **kwargs):
        dspart = self.pt.track(method=method, swells=2, **kwargs)
        assert "track_id" in dspart
        assert "ntracks" in dspart
        # Force the computation: the tracking kernel runs lazily under dask
        # and never executes if only variable existence is checked.
        dspart = dspart.load()
        assert (dspart.ntracks.values > 0).all()
        # Partitions assigned to a track are all non-null
        assert ((dspart.track_id < 0) | (dspart.spec.hs() > 0)).all()
        return dspart

    def test_track_ptm1(self):
        self._track("ptm1", wspd=self.dset.wspd, wdir=self.dset.wdir, dpt=self.dset.dpt)

    def test_track_ptm2(self):
        self._track("ptm2", wspd=self.dset.wspd, wdir=self.dset.wdir, dpt=self.dset.dpt)

    def test_track_ptm3_no_wind(self):
        dspart = self.pt.track(method="ptm3", parts=3)
        assert "track_id" in dspart
        dspart = dspart.load()
        assert (dspart.ntracks.values > 0).all()

    def test_track_hp01(self):
        self._track("hp01", wspd=self.dset.wspd, wdir=self.dset.wdir, dpt=self.dset.dpt)

    def test_track_hp01_no_wind(self):
        # Wind variables are resolved from the dataset so partitioning from
        # the DataArray is required to exercise the no-wind path
        pt = Partition(self.dset.efth)
        with pytest.warns(UserWarning):
            dspart = pt.track(method="hp01", swells=2)
        dspart = dspart.load()
        assert (dspart.ntracks.values > 0).all()

    def test_track_hp01_wind_from_dataset(self):
        dspart = self.pt.track(method="hp01", swells=2).load()
        assert (dspart.ntracks.values > 0).all()

    def test_track_invalid_method(self):
        with pytest.raises(ValueError):
            self.pt.track(method="ptm4")

    def test_track_requires_wspd_for_sea(self):
        dspart = self.pt.ptm1(
            wspd=self.dset.wspd,
            wdir=self.dset.wdir,
            dpt=self.dset.dpt,
            swells=2,
        )
        stats = dspart.spec.stats(["fp", "dpm"]).load()
        with pytest.raises(ValueError):
            track_partitions(stats, wspd=None, nsea=1)
        track_partitions(stats, wspd=None, nsea=0)

    def test_track_unsorted_times_raise(self):
        from wavespectra.partition.tracking import np_track_partitions

        dspart = self.pt.ptm1(
            wspd=self.dset.wspd,
            wdir=self.dset.wdir,
            dpt=self.dset.dpt,
            swells=2,
        )
        stats = dspart.spec.stats(["fp", "dpm"]).isel(site=0).load()
        with pytest.raises(ValueError):
            np_track_partitions(
                times=stats.time.values[::-1],
                fp=stats.fp.values.T,
                dpm=stats.dpm.values.T,
                wspd=self.dset.wspd.isel(site=0).values,
            )

    def _track_kwargs(self):
        return dict(
            wspd=self.dset.wspd, wdir=self.dset.wdir, dpt=self.dset.dpt, swells=2
        )

    def test_track_systems(self):
        compact = self.pt.track(method="ptm1", **self._track_kwargs()).load()
        systems = self.pt.track(
            method="ptm1", systems=True, **self._track_kwargs()
        ).load()
        assert "wave_system" in systems.dims
        assert "part" not in systems.dims
        assert systems.wave_system.size == int(compact.ntracks.max())
        # The total energy at each time step is preserved by the remapping
        hs_systems = (systems.spec.hs() ** 2).sum("wave_system") ** 0.5
        hs_compact = (compact.spec.hs() ** 2).sum("part") ** 0.5
        assert np.allclose(hs_systems.values, hs_compact.values, rtol=1e-5)
        # Each system exists over a single contiguous time window
        alive = systems.efth.notnull().any(["freq", "dir"])
        for a in alive.stack(k=["site", "wave_system"]).transpose("k", "time").values:
            if a.any():
                assert np.all(np.diff(np.flatnonzero(a)) == 1)
        # Padding entries for sites with fewer systems are null with id -999
        null_systems = ~alive.any("time")
        assert bool(((systems.track_id == -999) == null_systems).all())

    def test_track_systems_min_duration(self):
        systems = self.pt.track(
            method="ptm1", systems=True, **self._track_kwargs()
        ).load()
        filtered = self.pt.track(
            method="ptm1", systems=True, min_duration=3, **self._track_kwargs()
        ).load()
        assert filtered.wave_system.size < systems.wave_system.size
        alive = filtered.efth.notnull().any(["freq", "dir"]).sum("time")
        assert bool(((filtered.track_id < 0) | (alive >= 3)).all())

    def test_track_systems_lazy(self):
        dset = self.dset.chunk({"time": 3})
        systems = dset.spec.partition.track(
            method="ptm1",
            systems=True,
            wspd=dset.wspd,
            wdir=dset.wdir,
            dpt=dset.dpt,
            swells=2,
        )
        # The spectra remapping is lazy on dask data
        assert hasattr(systems.efth.data, "dask")
        assert bool(systems.load().efth.notnull().any())


class TestSpecpartKernel:
    """Regression tests for the specpart C extension (GH issue #142).

    The original C wrapper read the input buffer assuming C-contiguous
    float32 data, silently producing wrong partition maps for transposed
    or otherwise strided inputs (e.g. datasets stored with dims
    (..., dir, freq)). The wrapper now converts any input to a
    C-contiguous float32 array, matching the behaviour of the f2py
    wrapper around the original Fortran code.
    """

    @classmethod
    def setup_class(cls):
        from wavespectra.partition import specpart

        cls.specpart = specpart
        cls.dset = read_ww3(HERE / "sample_files/ww3file.nc")
        cls.spec = np.ascontiguousarray(
            cls.dset.isel(time=0, site=0).efth.values, dtype=np.float32
        )
        cls.expected = specpart.partition(cls.spec, 200)

    def test_input_layout_and_dtype_invariance(self):
        """Same values in any layout / real dtype must give the same map."""
        variants = {
            "float64": self.spec.astype(np.float64),
            "F-ordered float32": np.asfortranarray(self.spec),
            "F-ordered float64": np.asfortranarray(self.spec.astype(np.float64)),
            "transposed view": np.ascontiguousarray(self.spec.T).T,
            "sliced view": np.pad(self.spec, ((1, 1), (1, 1)))[1:-1, 1:-1],
        }
        for name, variant in variants.items():
            assert np.array_equal(variant, self.spec), name
            result = self.specpart.partition(variant, 200)
            assert np.array_equal(result, self.expected), name

    def test_ptm1_dataset_layout_invariance(self):
        """ptm1 must give the same result for (..., dir, freq) datasets."""
        ds1 = self.dset
        ds2 = ds1.transpose("time", "site", "dir", "freq")
        kwargs = dict(wspd=ds1.wspd, wdir=ds1.wdir, dpt=ds1.dpt, ihmax=200)
        hs1 = ds1.spec.partition.ptm1(**kwargs).spec.hs().load()
        hs2 = ds2.spec.partition.ptm1(**kwargs).spec.hs().load()
        assert np.allclose(hs1.values, hs2.values, equal_nan=True)

    def test_invalid_inputs_raise(self):
        with pytest.raises(ValueError):
            self.specpart.partition(np.zeros((2, 3, 4), dtype=np.float32), 100)
        with pytest.raises(ValueError):
            self.specpart.partition(self.spec, 1)

    def test_flat_spectrum_no_partitions(self):
        ipart = self.specpart.partition(np.zeros((25, 24), dtype=np.float32), 100)
        assert ipart.max() == 0

    def test_thread_consistency(self):
        """The kernel is reentrant; concurrent calls must agree."""
        from concurrent.futures import ThreadPoolExecutor

        def work(_):
            return np.array_equal(
                self.specpart.partition(self.spec, 200), self.expected
            )

        with ThreadPoolExecutor(max_workers=8) as ex:
            assert all(ex.map(work, range(200)))
