"""Native WAVEWATCH3 output plugin."""
from wavespectra.core.attributes import attrs

def to_ww3(self, filename):
    """Save spectra in native WW3 netCDF format.

    Not implemented!

    Args:
        - filename (str): name for output WW3 file.

    """
    raise NotImplementedError('Cannot write to native WW3 format yet')