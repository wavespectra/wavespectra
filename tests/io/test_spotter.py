import os
import shutil
import pytest
from tempfile import mkdtemp

from wavespectra import read_spotter
from wavespectra.input.spotter import read_spotter_csv
from wavespectra.input.spotter import Spotter
from wavespectra.core.attributes import attrs

FILES_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), 
                         "../sample_files")


class TestSpotterJson:
    """Test parameters Spotter JSON reader."""

    @classmethod
    def setup_class(self):
        """Setup class."""
        self.spot = Spotter(os.path.join(FILES_DIR, "spotter_20180214.json"))
        self.dset = self.spot.run()

    @pytest.mark.parametrize(
        "wavespectra_name, spotter_name, kwargs",
        [
            ("hs", "significantWaveHeight", {}),
            ("tp", "peakPeriod", {"smooth": False}),
            ("tm01", "meanPeriod", {}),
        ],
    )
    def test_stats(self, wavespectra_name, spotter_name, kwargs):
        """Assert that stat calculated from wavespectra matches the one read from spotter."""
        wavespectra = getattr(self.dset.spec, wavespectra_name)(**kwargs).values
        spotter = getattr(self.spot, spotter_name)
        assert wavespectra == pytest.approx(spotter, rel=1e-1)


# class TestSpotterCsv:
#     """Test parameters Spotter CSV reader."""

#     @classmethod
#     def setup_class(self):
#         """Setup class."""
#         self.dset = read_spotter_csv(os.path.join(FILES_DIR,'spotter_20210929.csv'))

#     @pytest.mark.parametrize(
#         "wavespectra_name, spotter_name, kwargs",
#         [
#             ("hs", "significantWaveHeight", {}),
#             ("tp", "peakPeriod", {"smooth": False}),
#             ("tm01", "meanPeriod", {}),
#         ],
#     )
#     def test_stats(self, wavespectra_name, spotter_name, kwargs):
#         """Assert that stat calculated from wavespectra matches the one read from spotter."""
#         wavespectra = getattr(self.dset.spec, wavespectra_name)(**kwargs).values
#         spotter = getattr(self.spot, spotter_name)
#         assert wavespectra == pytest.approx(spotter, rel=1e-1)

def test_read_spotter_csv():
    dset = read_spotter_csv(os.path.join(FILES_DIR,'spotter_20210929.csv'))
    pass