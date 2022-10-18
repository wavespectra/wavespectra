import pytest
from pathlib import Path
import numpy as np
import xarray as xr

from wavespectra import read_ww3
from wavespectra.partition.watershed import np_ptm3
from wavespectra.partition.partition import Partition
from wavespectra.core.npstats import hs


HERE = Path(__file__).parent


@pytest.fixture(scope="module")
def dset():
    _ds = read_ww3(HERE / "sample_files/ww3file.nc")
    yield _ds


class TestPTM3:
    def setup_class(self):
        self.dset = read_ww3(HERE / "sample_files/ww3file.nc")
        self.efth = self.dset.isel(time=0, site=0).efth.values
        self.freq = self.dset.freq.values
        self.dir = self.dset.dir.values

    @property
    def hs_full(self):
        return hs(self.efth, self.freq, self.dir)

    @property
    def hs_from_partitions(self):
        hs_partitions = [hs(efth, self.freq, self.dir) for efth in self.out]
        return np.sqrt(np.sum(np.power(hs_partitions, 2)))

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
        pt = Partition()
        dspart = pt.ptm3(dset=self.dset, parts=parts, smooth=False, combine=combine)
        hs_dspart = np.sqrt(np.sum(dspart.isel(time=0, site=0).spec.hs().values ** 2))
        assert hs_dspart == pytest.approx(self.hs_from_partitions)
