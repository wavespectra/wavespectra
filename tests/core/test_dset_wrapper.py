"""Unit testing for SpecDataset wrapper around DataArray."""
import os
import pytest

from wavespectra.core.attributes import attrs
from wavespectra import read_swan

FILES_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "../sample_files")


class TestDatasetWrapper(object):
    """Test SpecDataset wrapper."""

    @classmethod
    def setup_class(self):
        """Read test spectra from file."""
        here = os.path.dirname(os.path.abspath(__file__))
        self.swanspec = read_swan(os.path.join(FILES_DIR, "swanfile.spec"))

    @pytest.mark.parametrize(
        "stat_name",
        [
            ("hs"),
            ("hmax"),
            ("tp"),
            ("tm01"),
            ("tm02"),
            ("dm"),
            ("dp"),
            ("dpm"),
            ("swe"),
            ("sw"),
        ],
    )
    def test_stat(self, stat_name):
        """Compare stat calculated from SpecArray and SpecDataset.

        Args:
            stat_name (str): name of wave spectral statistic to compare. Must
                be the name of a valid method in SpecArray class.

        Asserts:
            output must be identical regardless it is calculated from SpecArray
                or SpecDataset.

        """
        from_darray = getattr(self.swanspec[attrs.SPECNAME].spec, stat_name)()
        from_dset = getattr(self.swanspec.spec, stat_name)()
        assert from_darray.identical(from_dset)
