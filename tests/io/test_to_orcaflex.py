import os
import shutil
import pytest
from tempfile import mkdtemp
import xarray as xr
import numpy as np

from wavespectra import read_triaxys
from wavespectra.core.attributes import attrs
from wavespectra.construct import ochihubble

FILES_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "../sample_files")


class TestToOrcaflex(object):
    """Test writing a spectrum to orcaflex. Test will only run if the orcaflex API can be used to create an new model.
    If not then the test will not run but will pass"""

    @classmethod
    def setup_class(self):
        """Setup class."""

        # dataset with time
        self.example_single_time = read_triaxys(
            os.path.join(FILES_DIR, "triaxys.DIRSPEC")
        )

        # dataset without time
        # resampled to lower resolution for quicker run
        self.example = self.example_single_time.isel(time=0)
        self.example = self.example.interp(dir=np.linspace(0, 360, 19))

        # dataset with two times, to verify export error
        one_day_later = self.example_single_time.copy()
        one_day_later["time"] = one_day_later.time + np.timedelta64(1, "D")
        self.bad_example = xr.concat(
            [self.example_single_time, one_day_later], dim="time"
        )

    def test_write_to_orcaflex(self):

        try:
            import OrcFxAPI

            m = OrcFxAPI.Model()
        except:
            print("Orcaflex test skipped because no license or no api")
            return

        self.example.spec.to_orcaflex(m)

    def test_throw_error_for_multiple_spectra_inpput(self):
        with pytest.raises(ValueError):
            self.bad_example.spec.to_orcaflex(None)

    def test_example_single_time(self):
        try:
            import OrcFxAPI

            m = OrcFxAPI.Model()
        except:
            print("Orcaflex test skipped because no license or no api")
            return

        self.example_single_time.spec.to_orcaflex(m)

    def test_compare_spectra(self):

        import matplotlib.pyplot as plt

        try:
            import OrcFxAPI

            m = OrcFxAPI.Model()
        except:
            print("Orcaflex test skipped because no license or no api")
            return

        self.example.spec.to_orcaflex(m)
        m.general.StageDuration[1] = 10800  # 3 hours of simulations
        m.RunSimulation()

        t = m.general.TimeHistory("Time")
        e = m.environment.TimeHistory(
            "Elevation", None, objectExtra=OrcFxAPI.oeEnvironment(0, 0, 0)
        )

        plt.plot(t[1000:2000], e[1000:2000], linewidth=0.5)
        plt.show()

        from scipy.signal import welch, periodogram

        dt = t[1] - t[0]
        f, p = welch(e, 1 / dt)
        plt.plot(f, p, label="welch")

        self.example.spec.oned().plot(label="source spectrun")
        plt.legend()
        plt.show()

        plt.figure()
        self.example.spec.plot()
