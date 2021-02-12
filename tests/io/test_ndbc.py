import os
import datetime
import shutil
import pytest
import pandas as pd
from tempfile import mkdtemp

from wavespectra import read_ndbc
from wavespectra.core.attributes import attrs

FILES_DIR = os.path.join(
    os.path.dirname(os.path.abspath(__file__)), "../sample_files/ndbc"
)


class TestNDBC(object):
    """Test parameters from 1D and 2D spectra are consistent and similar to NDBC summary values."""

    @classmethod
    def setup_class(self):
        """Setup class."""
        self.tmp_dir = mkdtemp()
        self.ds_realtime_1d = read_ndbc(os.path.join(FILES_DIR, "41010.data_spec"))
        self.ds_realtime_2d = read_ndbc(
            [
                os.path.join(FILES_DIR, "41010.data_spec"),
                os.path.join(FILES_DIR, "41010.swdir"),
                os.path.join(FILES_DIR, "41010.swdir2"),
                os.path.join(FILES_DIR, "41010.swr1"),
                os.path.join(FILES_DIR, "41010.swr2")
            ]
        )
        self.ds_realtime_params = pd.read_csv(
            os.path.join(FILES_DIR, "41010.spec"),
            delimiter="\s+",
            engine="python",
            header=[0, 1],
            parse_dates={"time": [0, 1, 2, 3, 4]},
            date_parser=lambda x: datetime.datetime.strptime(x, "%Y %m %d %H %M"),
            index_col=0,
        ).sort_values(by="time", ascending=True)
        self.ds_history_1d = read_ndbc(os.path.join(FILES_DIR, "41010w2019part.txt.gz"))
        self.ds_history_2d = read_ndbc(
            [
                os.path.join(FILES_DIR, "41010w2019part.txt.gz"),  # swden
                os.path.join(FILES_DIR, "41010d2019part.txt.gz"),  # swdir
                os.path.join(FILES_DIR, "41010i2019part.txt.gz"),  # swdir2
                os.path.join(FILES_DIR, "41010j2019part.txt.gz"),  # swr1
                os.path.join(FILES_DIR, "41010k2019part.txt.gz"),  # swr2
            ]
        )

    @classmethod
    def teardown_class(self):
        shutil.rmtree(self.tmp_dir)

    def test_realtime_hs_equals_params(self):
        assert self.ds_realtime_1d.spec.hs().values == pytest.approx(
            self.ds_realtime_params[("WVHT", "m")].values, rel=0.11
        )

    def test_realtime_1d_equals_2d(self):
        assert self.ds_realtime_1d.spec.hs().values == pytest.approx(
            self.ds_realtime_2d.spec.hs().values, rel=0.01
        )

    def test_history_1d_equals_2d(self):
        assert self.ds_history_1d.spec.hs().values == pytest.approx(
            self.ds_history_2d.spec.hs().values, rel=0.01
        )

    def test_history_1d_no_minutes(self):
        ds = read_ndbc(os.path.join(FILES_DIR, "44004w2000.txt"))
        assert hasattr(ds, "spec")