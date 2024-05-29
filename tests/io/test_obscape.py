from pathlib import Path
from datetime import datetime

import numpy as np
from numpy.testing import assert_allclose

from wavespectra import read_obscape
from wavespectra.input.obscape import read_obscape_dir

FILES_DIR = Path(__file__).parent / "../sample_files/obscape"


def test_read_fine():
    s = read_obscape(FILES_DIR / "*fine*.csv")
    assert_allclose(s.spec.hs().values, 1.73, rtol=1e-2)


def test_read_dir_fine_():
    s = read_obscape_dir(
        FILES_DIR,
        start_date=datetime(1990, 1, 1),
        end_date=datetime(1999, 1, 2),
    )
    assert_allclose(s.spec.hs().values, 1.73, rtol=1e-2)


def test_read_dir_course():
    s = read_obscape(FILES_DIR / "19800102_123456_Obscape2d_course.csv")
    assert_allclose(s.spec.hs().values, 0.0625, rtol=1e-2)
    assert isinstance(s.time.values[0], np.datetime64)
    assert s.time.values[0] == np.datetime64("2024-04-03T09:30:00")

