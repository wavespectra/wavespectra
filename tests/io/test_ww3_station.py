import os

from wavespectra import read_ww3_station


FILES_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "../sample_files")


class TestWW3Station(object):
    """Test Funwave input / output functions."""

    @classmethod
    def setup_class(self):
        """Setup class."""
        filename = os.path.join(FILES_DIR, "ww3station.spec")
        with open(filename, "r") as file:
            self.freq, self.dir = read_ww3_station(file)

    def test_parse_header(self):
        assert len(self.freq) == 50
        assert len(self.dir) == 36