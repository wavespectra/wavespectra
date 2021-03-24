import os
import shutil
import pytest
from tempfile import mkdtemp

from wavespectra import read_argoss_csv
from wavespectra.core.attributes import attrs

FILES_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "../sample_files")


class TestToOrcaflex(object):
    """Test writing a spectrum to orcaflex. Test will only run if the orcaflex API can be used to create an new model.
    If not then the test will not run but will pass"""

    @classmethod
    def setup_class(self):
        """Setup class."""
        self.tmp_dir = mkdtemp()
        self.example = read_argoss_csv(os.path.join(FILES_DIR, "ARGOSS_example.csv"))

    def test_write_to_orcaflex(self):

        try:
            import OrcFxAPI
            m = OrcFxAPI.Model()
        except:
            return

        self.example.spec.to_orcaflex(m, 20)


