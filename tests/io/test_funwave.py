import os
import shutil
import pytest
from tempfile import mkdtemp
import numpy as np
import xarray as xr
from zipfile import ZipFile

from wavespectra import read_funwave, read_funwave_new, read_ww3


FILES_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "../sample_files")


class TestFunwave:
    """Test Funwave input / output functions."""

    @classmethod
    def setup_class(self):
        """Setup class."""
        self.tmp_dir = mkdtemp()
        self.filename = os.path.join(FILES_DIR, "funwavefile.txt")
        self.filename_new = os.path.join(FILES_DIR, "funwave_new_file.txt")
        self.dsww3 = read_ww3(os.path.join(FILES_DIR, "ww3file.nc"))

    @classmethod
    def teardown_class(self):
        shutil.rmtree(self.tmp_dir)

    def _files_in_zip(self, zipname):
        with ZipFile(zipname) as zstream:
            count = len(zstream.infolist())
        return count

    def test_read(self):
        read_funwave(self.filename)

    def test_write_single(self):
        """
        * Write single funwave spectrum from ww3 file
        * Compare stats from original and new datasets.
        """
        filename = os.path.join(self.tmp_dir, "fw.txt")
        dset0 = self.dsww3.isel(time=0, site=0, drop=True)
        dset0.spec.to_funwave(filename, clip=False)
        dset1 = read_funwave(filename)
        xr.testing.assert_allclose(
            dset0.spec.stats(["hs", "tp", "dpm"]),
            dset1.spec.stats(["hs", "tp", "dpm"]),
            rtol=1e-3,
        )

    def test_write_multi(self):
        """Write multiple funwave files from ww3 dataset."""
        filename = os.path.join(self.tmp_dir, "fw.txt")
        zipname = filename.replace(".txt", ".zip")
        self.dsww3.spec.to_funwave(filename)
        assert os.path.isfile(zipname)
        assert self._files_in_zip(zipname) == np.prod(
            self.dsww3.efth.isel(freq=0, dir=0).shape
        )

    def test_clip(self):
        """Test directions are / are not clipped."""
        filename1 = os.path.join(self.tmp_dir, "fw1.txt")
        filename2 = os.path.join(self.tmp_dir, "fw2.txt")
        dset0 = self.dsww3.isel(time=0, site=0, drop=True)
        dset0.spec.to_funwave(filename1)
        dset0.spec.to_funwave(filename2, clip=False)

        dset1 = read_funwave(filename1)
        dset2 = read_funwave(filename2)

        dir1 = (270 - dset1.dir.values) % 360
        dir2 = (270 - dset2.dir.values) % 360
        dir1[dir1 > 180] = dir1[dir1 > 180] - 360
        dir2[dir2 > 180] = dir2[dir2 > 180] - 360

        assert dir1.min() >= -90 and dir1.max() <= 90
        assert dir2.min() < -90 and dset2.dir.max() > 90

    def _read_new_data2d(self, filename):
        """Parse blocks from a WK_NEW_DATA2D wave component file."""
        with open(filename) as fid:
            lines = fid.readlines()
        values = [float(line.split()[0]) for line in lines]
        nc = int(values[0])
        tp = values[1]
        blocks = values[2:]
        freq = np.array(blocks[:nc])
        dire = np.array(blocks[nc : 2 * nc])
        amp = np.array(blocks[2 * nc : 3 * nc])
        phase = np.array(blocks[3 * nc : 4 * nc])
        return nc, tp, freq, dire, amp, phase

    def test_new_data2d_write_single(self):
        """
        * Write single wave component file from ww3 file.
        * Check file structure and stats against the original dataset.
        """
        filename = os.path.join(self.tmp_dir, "fw_new.txt")
        dset0 = self.dsww3.isel(time=0, site=0, drop=True)
        dset0.spec.to_funwave_new(filename, clip=False)
        nc, tp, freq, dire, amp, phase = self._read_new_data2d(filename)

        assert nc == dset0.freq.size * dset0.dir.size
        assert len(freq) == len(dire) == len(amp) == len(phase) == nc
        assert tp == pytest.approx(float(dset0.spec.tp()), rel=1e-3)

        # Sum of component energies a^2 / 2 recovers m0
        hs = 4 * np.sqrt(np.sum(amp**2 / 2))
        assert hs == pytest.approx(float(dset0.spec.hs(tail=False)), rel=1e-4)

        # Directions in cartesian convention
        dir0 = np.sort(np.unique(dire))
        dir1 = (270 - dset0.dir.values) % 360
        dir1[dir1 > 180] = dir1[dir1 > 180] - 360
        assert np.allclose(dir0, np.sort(dir1))

    def test_new_data2d_clip(self):
        """Test directions outside [-90, 90] are clipped."""
        filename = os.path.join(self.tmp_dir, "fw_new_clip.txt")
        dset0 = self.dsww3.isel(time=0, site=0, drop=True)
        dset0.spec.to_funwave_new(filename)
        nc, tp, freq, dire, amp, phase = self._read_new_data2d(filename)
        assert dire.min() >= -90 and dire.max() <= 90

    def test_new_data2d_no_phases(self):
        """Test the phase block is omitted with phases=False."""
        filename = os.path.join(self.tmp_dir, "fw_new_nophase.txt")
        dset0 = self.dsww3.isel(time=0, site=0, drop=True)
        dset0.spec.to_funwave_new(filename, clip=False, phases=False)
        nc, tp, freq, dire, amp, phase = self._read_new_data2d(filename)
        assert phase.size == 0
        with open(filename) as fid:
            assert len(fid.readlines()) == 3 * nc + 2

    def test_new_data2d_write_multi(self):
        """Write multiple wave component files from ww3 dataset."""
        filename = os.path.join(self.tmp_dir, "fw_new.txt")
        zipname = filename.replace(".txt", ".zip")
        self.dsww3.spec.to_funwave_new(filename)
        assert os.path.isfile(zipname)
        assert self._files_in_zip(zipname) == np.prod(
            self.dsww3.efth.isel(freq=0, dir=0).shape
        )

    def test_new_data2d_read_official_example(self):
        """Read the two-wave example file from the Funwave repository.

        Two components with amplitude 0.3 m and directions +-12 deg cartesian,
        from simple_cases/wave_coherence/Case_DATA2D_2Waves.

        """
        dset = read_funwave_new(self.filename_new)
        assert dset.freq.size == 1
        assert sorted(dset.dir.values) == [258.0, 282.0]
        assert float(dset.tp) == 8.0
        hs = 4 * np.sqrt(2 * 0.3**2 / 2)
        assert float(dset.spec.hs(tail=False)) == pytest.approx(hs, rel=1e-6)

    def test_new_data2d_read(self):
        """
        * Write single wave component file from ww3 file.
        * Read it back and compare stats from original and new datasets.
        """
        filename = os.path.join(self.tmp_dir, "fw_new_roundtrip.txt")
        dset0 = self.dsww3.isel(time=0, site=0, drop=True)
        dset0.spec.to_funwave_new(filename, clip=False)
        dset1 = read_funwave_new(filename)
        xr.testing.assert_allclose(
            dset0.spec.stats(["hs", "tp", "dpm"]),
            dset1.spec.stats(["hs", "tp", "dpm"]),
            rtol=1e-3,
        )

    def test_new_data2d_read_1d(self):
        """
        * Write oned wave component file from ww3 file.
        * Read it back and check stats and dimension consistency.
        """
        filename = os.path.join(self.tmp_dir, "fw_new_roundtrip_1d.txt")
        dset0 = self.dsww3.isel(time=0, site=0, drop=True).spec.oned().to_dataset()
        dset0.spec.to_funwave_new(filename, clip=False)
        dset1 = read_funwave_new(filename)
        assert dset1.spec.dd == 1
        assert dset1.spec.dir is None
        xr.testing.assert_allclose(
            dset0.spec.stats(["hs", "tp"]),
            dset1.spec.stats(["hs", "tp"]),
            rtol=1e-3,
        )

    def test_new_data2d_1d(self):
        """Write oned wave component file with all directions set to zero."""
        filename = os.path.join(self.tmp_dir, "fw_new_1d.txt")
        dset0 = self.dsww3.isel(time=0, site=0, drop=True).spec.oned().to_dataset()
        dset0.spec.to_funwave_new(filename, clip=False)
        nc, tp, freq, dire, amp, phase = self._read_new_data2d(filename)
        assert nc == dset0.freq.size
        assert np.all(dire == 0)
        hs = 4 * np.sqrt(np.sum(amp**2 / 2))
        assert hs == pytest.approx(float(dset0.spec.hs(tail=False)), rel=1e-4)

    def test_negative_energy_no_nan(self):
        """Tiny negative energy must not produce NaN amplitudes.

        Interpolation / rotation of spectra can leave bins with tiny negative
        energy (~1e-21) from float cancellation. As amplitudes are computed via
        sqrt(efth ...), those must be clipped to avoid NaN in the output.
        """
        dset0 = self.dsww3.isel(time=0, site=0, drop=True).copy(deep=True)
        dset0["efth"][0, 0] = -1.0e-21

        # Gridded format
        filename = os.path.join(self.tmp_dir, "fw_neg.txt")
        dset0.spec.to_funwave(filename, clip=False)
        with open(filename) as fid:
            assert "nan" not in fid.read().lower()

        # Wave component (WK_NEW_DATA2D) format
        filename_new = os.path.join(self.tmp_dir, "fw_new_neg.txt")
        dset0.spec.to_funwave_new(filename_new, clip=False)
        _, _, _, _, amp, _ = self._read_new_data2d(filename_new)
        assert not np.isnan(amp).any()

    def test_1d(self):
        """
        * Write oned funwave spectrum from ww3 file.
        * Read oned funwave spectrum written.
        * Compare stats from original and new datasets.
        * Check for dimension consistency.
        """
        filename = os.path.join(self.tmp_dir, "fw1d.txt")
        dset0 = self.dsww3.isel(time=0, site=0, drop=True).spec.oned().to_dataset()
        dset0.spec.to_funwave(filename, clip=False)
        dset1 = read_funwave(filename)
        assert dset1.spec.dd == 1
        assert dset1.spec.dir is None
        xr.testing.assert_allclose(
            dset0.spec.stats(["hs", "tp"]),
            dset1.spec.stats(["hs", "tp"]),
            rtol=1e-3,
        )
