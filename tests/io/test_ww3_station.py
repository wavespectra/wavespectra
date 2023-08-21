import os

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
        assert len(ds.spec.dir) == 36
        assert len(ds.spec.time) == 385
        assert ds.spec.efth.shape == (385, 1, 1, 50, 36)

        print(ds.isel(time=0).spec.hs().values)