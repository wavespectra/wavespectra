"""MetOcean Solutions WAVEWATCH3 output plugin."""
from wavespectra.core.attributes import attrs

def to_ww3_msl(self, filename):
    """Save spectra in custom WW3 netCDF format from MetOcean Solutions.

    Not implemented!

    Args:
        - filename (str): name for output WW3 file.

    """
    raise NotImplementedError('Cannot write to MSL WW3 format yet')