import os
import shutil
import pytest
from tempfile import mkdtemp

from wavespectra import read_argoss_csv
from wavespectra.core.attributes import attrs

FILES_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "../sample_files")


EXAMPLE_HS = [1.09791853, 1.10582522, 1.26895423, 1.22835512, 1.3443963 ,
       1.68674831, 1.83061428, 1.74325149, 1.68883246, 1.53359874,
       1.53664742, 1.38083793, 1.23905557, 1.0167956 , 0.84510629,
       0.828673  , 0.85903418, 0.83382994, 0.76933194, 0.72165332,
       0.64487064, 0.58632282, 0.52292613, 0.47660438, 0.45281084,
       0.45049921, 0.50359588, 0.45945922, 0.42031421, 0.4057736 ,
       0.46883966, 0.78373774, 0.79986711, 0.74593252, 0.95702505,
       1.11418399, 1.02162518, 0.94379859, 0.85665397, 0.80262841,
       0.80487584, 0.79416205, 0.77493702, 0.74090162, 0.73359071,
       0.67742611, 0.60498101, 0.54155583, 0.47066372]

EXAMPLE_TP = [5.65754783, 5.80281025, 5.77072902, 5.98590133, 6.10143477,
       6.29810852, 6.46950931, 6.47411765, 6.311896  , 6.23586844,
       6.05801929, 5.99529243, 5.7780514 , 5.7329063 , 5.69170141,
       4.70254407, 4.39249543, 4.48989541, 4.44119058, 4.30381153,
       7.45898924, 7.47133003, 7.47129569, 7.44848383, 7.42757183,
       7.41056663, 7.37789481, 7.35741242, 7.26725282, 7.12859328,
       7.03622348, 3.80428349, 4.15593234, 4.14332916, 4.25490377,
       4.669465  , 4.83355457, 6.37367687, 6.50901127, 6.69939503,
       6.77632831, 5.56417361, 6.17359651, 6.33412728, 6.37372178,
       6.39199468, 6.41551901, 6.43430349, 6.42704584]

class TestArgossCSV(object):
    """Test parameters from fugro example files."""

    @classmethod
    def setup_class(self):
        """Setup class."""
        self.tmp_dir = mkdtemp()
        self.example = read_argoss_csv(os.path.join(FILES_DIR, "ARGOSS_example.csv"))


    def test_hs(self):
        assert self.example.spec.hs().values == pytest.approx(
            EXAMPLE_HS, rel=0.01
        )

    def test_hs_reported(self):

        assert self.example.Reported_Hs.values == pytest.approx(
             EXAMPLE_HS, abs=0.2
         )

    def test_TP(self):
        assert self.example.spec.tp().values == pytest.approx(
            EXAMPLE_TP, rel=0.01
        )