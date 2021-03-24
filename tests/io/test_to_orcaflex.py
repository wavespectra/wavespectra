import os
import shutil
import pytest
from tempfile import mkdtemp

from wavespectra import read_fugro_csv
from wavespectra.core.attributes import attrs

FILES_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "../sample_files")


class TestToOrcaflex(object):
    """Test parameters from fugro example files."""

    @classmethod
    def setup_class(self):
        """Setup class."""
        self.tmp_dir = mkdtemp()
        self.example = read_fugro_csv(os.path.join(FILES_DIR, "FUGRO_example.csv"))
        self.example.spec.to_orcaflex(20)


