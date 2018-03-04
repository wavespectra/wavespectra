"""
MetOcean Solutions WAVEWATCH3 output plugin.

to_ww3_msl :: write spectra in custom ww3 netcdf format

"""
from spectra.core.attributes import attrs

def to_ww3_msl(self, filename):
    """Save spectra in custom WW3 netCDF format from MetOcean Solutions.

    Args:
        filename (str): name for output WW3 file

    """
    raise NotImplementedError('Cannot write to MSL WW3 format yet')