import os
import glob
import shutil
import pytest
from tempfile import mkdtemp

from wavespectra import read_spotter
from wavespectra.input.spotter import Spotter
from wavespectra.core.attributes import attrs

FILES_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "../sample_files")


class TestSpotterJson:
    """Test parameters Spotter JSON reader."""

    @classmethod
    def setup_class(self):
        """Setup class."""
        infile = os.path.join(FILES_DIR, "spotter_20180214.json")
        self.spot = Spotter(infile)
        self.spot.run()
        self.dset = read_spotter(infile)

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


@pytest.fixture(
    scope="module",
    params=[
        os.path.join(FILES_DIR, "spotter_20210929.csv"),
        glob.glob(os.path.join(FILES_DIR, "spotter*.csv"))[0],
    ],
)
def dset(request):
    return read_spotter(request.param)


@pytest.mark.parametrize(
    "wavespectra_name, spotter_name, kwargs",
    [
        ("hs", "Hm0", {}),
        ("tp", "Tp", {"smooth": False}),
        ("tm01", "Tm", {}),
    ],
)
def test_read_spotter_csv(dset, wavespectra_name, spotter_name, kwargs):
    wavespectra = getattr(dset.spec, wavespectra_name)(**kwargs).values
    spotter = getattr(dset, spotter_name)
    assert wavespectra == pytest.approx(spotter, rel=1e-1)
