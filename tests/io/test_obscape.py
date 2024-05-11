from pathlib import Path
from datetime import datetime

from numpy.testing import assert_allclose

from wavespectra import read_obscape

FILES_DIR = Path(__file__).parent / "../sample_files/obscape"


def test_read_fine():
    s = read_obscape(FILES_DIR,
                     start_date=datetime(1990, 1, 1),
                     end_date=datetime(1999, 1, 2))
    assert_allclose(s.spec.hs().values, 1.73, rtol=1e-2)



def test_read_course():
    s = read_obscape(FILES_DIR,
                     start_date=datetime(1970, 1, 1),
                     end_date=datetime(1985, 1, 2))
    assert_allclose(s.spec.hs().values, 0.0625, rtol=1e-2)
