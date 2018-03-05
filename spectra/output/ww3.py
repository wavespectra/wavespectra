"""
Native WAVEWATCH3 output plugin.

to_ww3 :: write spectra in ww3 netcdf format

"""
from wavespectra.core.attributes import attrs

def to_ww3(self, filename):
    """Save spectra in native WW3 netCDF format.

    Args:
        filename (str): name for output WW3 file

    """
    raise NotImplementedError('Cannot write to native WW3 format yet')