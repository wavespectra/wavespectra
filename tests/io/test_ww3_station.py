import os

import pytest

from wavespectra import read_ww3_station


FILES_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "../sample_files")


class TestWW3Station(object):
    """Test Funwave input / output functions."""

    @classmethod
    def setup_class(self):
        """Setup class."""
        self.filename = os.path.join(FILES_DIR, "ww3station.spec")

    def test_read(self):
        with open(self.filename, "r") as file:
            ds = read_ww3_station(file)

        assert len(ds.spec.freq) == 50
        assert len(ds.spec.direction) == 36
        assert len(ds.spec.time) == 4
        assert ds.spec.efth.shape == (4, 1, 1, 50, 36)

        assert ds.spec.hs().values[0] == pytest.approx(1.16, rel=1e-1)
        assert ds.spec.tp().values[0] == pytest.approx(13.6, rel=1e-1)
        assert ds.spec.dp().values[0] == pytest.approx(291.0, rel=1e-0)
