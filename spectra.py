"""
New spectra object based on DataArray
"""
from collections import OrderedDict
from xray import DataArray
from datetime import datetime
import numpy as np

class Spectrum(DataArray):
    """
    Build on the top of DataArray object
    """
    def __init__(self, spec_array, freq_array, dir_array=[None], time_array=[None]):
        """
        spec_aray :: 1D, 2D or 3D array with spectrum
        freq_array :: 1D array with frequencies for spectrum
        dir_array :: 1D array with directions for spectrum
        time_array :: 1D array with datetimes
        """
        if time_array[0] is None:
            spec_array = np.expand_dims(spec_array, 0)
        if dir_array[0] is None:
            spec_array = np.expand_dims(spec_array, -1)

        coords = OrderedDict((('time', time_array), ('freq', freq_array), ('dir', dir_array)))
        super(Spectrum, self).__init__(data=spec_array, coords=coords, name='spec')

        self.df = abs(self.freq[1:].values - self.freq[:-1].values)
        self.dd = abs(self.dir[1].values - self.dir[0].values) if any(self.dir) else 1.0

    def oned(self):
        """
        Returns the one-dimensional frequency spectra
        The direction dimension is dropped after integrating
        """
        return self.dd * self.sum(dim='dir')

    def hs(self, tail=True, times=None):
        """
        Spectral significant wave height Hm0
        - tail ::
        - times :: list of datetimes to calculate hs over (all times calculated by default)
        """
        Sf = self.oned()
        if times:
            Sf = Sf.sel(time=times, method='nearest')
        E = 0.5 * (self.df * (Sf[{'freq': slice(1, None)}] + Sf[{'freq': slice(None, -1)}].values)).sum(dim='freq')
        if tail and self.freq[-1] > 0.333:
            E += 0.25 * Sf[{'freq': -1}].values * self.freq[-1].values
        return 4 * np.sqrt(E)


if __name__ == '__main__':
    import numpy as np
    from pymo.data.spectra import SwanSpecFile

    # Real spectra
    spectra = SwanSpecFile('/data/work/ops/prod/prelud.spec')
    spec_list = [s for s in spectra.readall()]
    spec_array = np.concatenate([np.expand_dims(s.S, 0) for s in spec_list])
    spec = Spectrum(spec_array=spec_array,
                    freq_array=spec_list[0].freqs,
                    dir_array=spec_list[0].dirs,
                    time_array=spectra.times)
    print 'Hs for 2015-07-21: %0.2f m' % (spec.hs(times=datetime(2015,07,21)))

    # freq_array = np.arange(0, 1.01, 0.1)
    # dir_array = np.arange(0, 360, 30)
    # time_array = [datetime(2015, 1, d) for d in [1,2,3]]
    #
    # # With time and directions
    # spec_array = np.random.randint(1, 10, (len(time_array), len(freq_array), len(dir_array)))
    # spec1 = Spectrum(spec_array, freq_array, dir_array, time_array)
    #
    # # Without time
    # spec_array = np.random.random((len(freq_array), len(dir_array)))
    # spec2 = Spectrum(spec_array, freq_array, dir_array)
    #
    # # Without directions
    # spec_array = np.random.random((len(time_array), len(freq_array)))
    # spec3 = Spectrum(spec_array, freq_array, time_array=time_array)
    #
    # # Without time and directions
    # spec_array = np.random.random(len(freq_array))
    # spec4 = Spectrum(spec_array, freq_array)
