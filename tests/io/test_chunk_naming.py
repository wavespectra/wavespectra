import os
import pytest

from wavespectra import read_ncswan, read_wwm, read_ww3
from wavespectra.core.attributes import attrs

FILES_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "../sample_files")


@pytest.mark.parametrize(
    "read_func, filename, timename, sitename, freqname, dirname",
    [
        (read_ww3, "ww3file.nc", "time", "station", "frequency", "direction"),
        (read_wwm, "wwmfile.nc", "ocean_time", "nbstation", "nfreq", "ndir"),
        (read_ncswan, "swanfile.nc", "time", "points", "frequency", "direction"),
    ],
)
def test_chunk_naming(read_func, filename, timename, sitename, freqname, dirname):
    filename = os.path.join(FILES_DIR, filename)
    ds1 = read_func(
        filename, chunks={timename: None, sitename: None, freqname: None, dirname: None}
    )
    ds2 = read_func(
        filename,
        chunks={
            attrs.TIMENAME: None,
            attrs.SITENAME: None,
            attrs.FREQNAME: None,
            attrs.DIRNAME: None,
        },
    )
    assert ds1.identical(ds2)
