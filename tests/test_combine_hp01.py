"""Tests for the Hanson and Phillips (2001) partition combining algorithm."""

from pathlib import Path

import numpy as np
import pytest

from wavespectra import read_datawell
from wavespectra.core import npstats
from wavespectra.partition import specpart
from wavespectra.partition.hanson_and_phillips_2001 import (
    _adjacency_saddles,
    _label_map,
    combine_partitions_hp01,
)


HERE = Path(__file__).parent

FREQ = np.arange(0.04, 0.4, 0.01)
DIR = np.arange(0, 360, 10.0)


def _gaussian_system(fp, dp, hs, fspread=0.02, dspread=20.0):
    """Synthetic Gaussian wave system in frequency-direction space."""
    f = FREQ[:, None]
    d = DIR[None, :]
    ddiff = np.minimum(np.abs(d - dp), 360 - np.abs(d - dp))
    spec = np.exp(-0.5 * ((f - fp) / fspread) ** 2 - 0.5 * (ddiff / dspread) ** 2)
    e = (spec * np.gradient(FREQ)[:, None] * (DIR[1] - DIR[0])).sum()
    return spec * (hs / 4) ** 2 / e


def _split(spectrum, ihmax=100):
    """Split a spectrum into its watershed partitions."""
    wmap = specpart.partition(spectrum.astype("float32"), ihmax)
    return [np.where(wmap == ip, spectrum, 0.0) for ip in range(1, wmap.max() + 1)]


def _hs(spectrum):
    return npstats.hs(spectrum, FREQ, DIR)


def _keys(partitions):
    """Hashable identifiers of combined partitions, independent of order."""
    return {tuple(np.flatnonzero(p.ravel())) for p in partitions}


@pytest.fixture
def noisy_two_systems():
    """Two distinct systems plus noise that over-segments the watershed."""
    rng = np.random.default_rng(42)
    spec = _gaussian_system(0.08, 270, 2.0) + _gaussian_system(0.20, 90, 1.5)
    spec *= rng.uniform(0.5, 1.5, spec.shape)
    return spec


class TestAdjacencySaddles:
    def test_adjacent_partitions_have_saddle(self):
        spec = _gaussian_system(0.08, 180, 2.0) + _gaussian_system(0.14, 180, 1.5)
        parts = _split(spec)
        assert len(parts) == 2
        lmap = _label_map(np.array(parts))
        saddles = _adjacency_saddles(lmap, spec, len(parts))
        assert saddles[1, 2] > 0
        assert saddles[1, 2] == saddles[2, 1]
        # The saddle between the two mutual peaks is a large fraction of the
        # smaller peak density
        assert saddles[1, 2] / min(p.max() for p in parts) > 0.5

    def test_distant_partitions_low_saddle(self):
        spec = _gaussian_system(0.06, 270, 2.0, 0.01, 10) + _gaussian_system(
            0.3, 90, 1.5, 0.01, 10
        )
        parts = _split(spec)
        lmap = _label_map(np.array(parts))
        saddles = _adjacency_saddles(lmap, spec, len(parts))
        # Basins touch only through near-zero tails so the saddle is
        # negligible relative to the peaks
        assert saddles[1, 2] / min(p.max() for p in parts) < 1e-3

    def test_wrap_across_direction_axis(self):
        lmap = np.zeros((3, 4), dtype=int)
        lmap[:, 0] = 1
        lmap[:, -1] = 2
        spec = np.ones((3, 4))
        saddles = _adjacency_saddles(lmap, spec, 2)
        assert saddles[1, 2] == 1


class TestCombine:
    def test_mutual_partitions_combined(self, noisy_two_systems):
        parts = _split(noisy_two_systems)
        assert len(parts) > 2  # noise over-segments
        combined = combine_partitions_hp01(parts, FREQ, DIR)
        assert len(combined) == 2

    def test_distinct_systems_not_combined(self, noisy_two_systems):
        parts = _split(noisy_two_systems)
        combined = combine_partitions_hp01(parts, FREQ, DIR)
        # Each combined partition contains exactly one of the two peaks
        fp = sorted(FREQ[np.argmax(p.sum(axis=1))] for p in combined)
        assert fp[0] == pytest.approx(0.08, abs=0.02)
        assert fp[1] == pytest.approx(0.20, abs=0.02)

    def test_energy_conserved(self, noisy_two_systems):
        parts = _split(noisy_two_systems)
        combined = combine_partitions_hp01(parts, FREQ, DIR)
        hs_combined = np.sqrt(sum(_hs(p) ** 2 for p in combined))
        assert hs_combined == pytest.approx(_hs(noisy_two_systems), rel=1e-6)

    def test_input_order_invariance(self, noisy_two_systems):
        """The main instability of the old code: results depended on ordering."""
        parts = _split(noisy_two_systems)
        expected = _keys(combine_partitions_hp01(parts, FREQ, DIR))
        rng = np.random.default_rng(7)
        for _ in range(5):
            shuffled = [parts[i] for i in rng.permutation(len(parts))]
            assert _keys(combine_partitions_hp01(shuffled, FREQ, DIR)) == expected

    def test_output_sorted_by_energy(self, noisy_two_systems):
        parts = _split(noisy_two_systems)
        combined = combine_partitions_hp01(parts, FREQ, DIR)
        hs = [_hs(p) for p in combined]
        assert hs == sorted(hs, reverse=True)

    def test_exact_number_of_swells_combined(self, noisy_two_systems):
        parts = _split(noisy_two_systems)
        combined = combine_partitions_hp01(parts, FREQ, DIR, swells=1)
        assert len(combined) == 1
        assert _hs(combined[0]) == pytest.approx(_hs(noisy_two_systems), rel=1e-6)

    def test_exact_number_of_swells_dropped(self, noisy_two_systems):
        parts = _split(noisy_two_systems)
        combined = combine_partitions_hp01(
            parts, FREQ, DIR, swells=1, combine_extra_swells=False
        )
        assert len(combined) == 1
        assert _hs(combined[0]) < _hs(noisy_two_systems)

    def test_fewer_partitions_than_swells(self, noisy_two_systems):
        parts = _split(noisy_two_systems)
        combined = combine_partitions_hp01(parts, FREQ, DIR, swells=5)
        # Combining does not pad, callers do
        assert len(combined) == 2

    def test_hs_min_forces_merging(self):
        spec = _gaussian_system(0.08, 270, 2.0) + _gaussian_system(0.25, 90, 0.3)
        parts = _split(spec)
        assert len(parts) == 2
        combined = combine_partitions_hp01(parts, FREQ, DIR, hs_min=0.5)
        assert len(combined) == 1
        assert _hs(combined[0]) == pytest.approx(_hs(spec), rel=1e-6)

    def test_noise_threshold(self):
        spec = _gaussian_system(0.08, 270, 2.0) + _gaussian_system(0.25, 90, 0.3)
        parts = _split(spec)
        assert len(parts) == 2
        e_small = min((_hs(p) / 4) ** 2 for p in parts)
        noise_a = 2 * e_small * 0.25**4
        combined = combine_partitions_hp01(parts, FREQ, DIR, noise_a=noise_a)
        assert len(combined) == 1
        assert _hs(combined[0]) == pytest.approx(_hs(spec), rel=1e-6)

    def test_angle_max_gate(self):
        # Two close systems at right angles are combined by the distance
        # criterion unless the angle gate is enabled
        spec = _gaussian_system(0.1, 180, 2.0, 0.01, 10) + _gaussian_system(
            0.12, 90, 1.5, 0.01, 10
        )
        parts = _split(spec)
        if len(parts) < 2:
            pytest.skip("watershed did not separate the test systems")
        with_gate = combine_partitions_hp01(parts, FREQ, DIR, angle_max=30)
        without_gate = combine_partitions_hp01(parts, FREQ, DIR, angle_max=None)
        assert len(with_gate) >= len(without_gate)

    def test_null_partitions_dropped(self, noisy_two_systems):
        parts = _split(noisy_two_systems) + [np.zeros_like(noisy_two_systems)]
        combined = combine_partitions_hp01(parts, FREQ, DIR)
        assert len(combined) == 2

    def test_all_null_partitions(self):
        parts = [np.zeros((FREQ.size, DIR.size)) for _ in range(3)]
        assert combine_partitions_hp01(parts, FREQ, DIR) == []

    def test_single_partition(self):
        spec = _gaussian_system(0.1, 180, 2.0)
        combined = combine_partitions_hp01([spec], FREQ, DIR)
        assert len(combined) == 1
        assert _hs(combined[0]) == pytest.approx(_hs(spec), rel=1e-6)

    def test_invalid_shape_raises(self):
        with pytest.raises(ValueError):
            combine_partitions_hp01(np.zeros((5, 5)), FREQ, DIR)


class TestMeasuredSpectra:
    """Sanity checks on real, noisy buoy spectra."""

    @classmethod
    def setup_class(cls):
        files = sorted((HERE / "sample_files/datawell").glob("*.spt"))
        cls.dset = read_datawell(files)

    def test_combining_reduces_and_conserves(self):
        for time in range(self.dset.time.size):
            d = self.dset.isel(time=time)
            spec = d.efth.values.astype("float64")
            freq = d.freq.values
            dirs = d.dir.values
            wmap = specpart.partition(spec.astype("float32"), 100)
            parts = [np.where(wmap == ip, spec, 0.0) for ip in range(1, wmap.max() + 1)]
            combined = combine_partitions_hp01(parts, freq, dirs, hs_min=0.2)
            assert 0 < len(combined) < len(parts)
            hs_combined = np.sqrt(sum(npstats.hs(p, freq, dirs) ** 2 for p in combined))
            assert hs_combined == pytest.approx(npstats.hs(spec, freq, dirs), rel=1e-6)
