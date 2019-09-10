import os
import shutil
import pytest
from tempfile import mkdtemp

from wavespectra import read_triaxys
from wavespectra.core.attributes import attrs

FILES_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "../sample_files")


class TestTriaxys(object):
    """Test parameters from 1D and 2D spectra from TRIAXYS are consistent."""

    @classmethod
    def setup_class(self):
        """Setup class."""
        self.tmp_dir = mkdtemp()
        self.ds_1d = read_triaxys(os.path.join(FILES_DIR, "triaxys.NONDIRSPEC"))
        self.ds_2d = read_triaxys(os.path.join(FILES_DIR, "triaxys.DIRSPEC"))

    @classmethod
    def teardown_class(self):
        shutil.rmtree(self.tmp_dir)

    def test_hs(self):
        assert self.ds_1d.spec.hs().values == pytest.approx(
            self.ds_2d.spec.hs().values, rel=0.01
        )
