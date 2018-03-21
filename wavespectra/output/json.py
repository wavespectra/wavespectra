"""MetOcean Solutions WAVEWATCH3 output plugin."""
from wavespectra.core.attributes import attrs

try:
    from cfjson.xrdataset import CFJSONinterface
except ImportError:
    print('Warning: cannot import cf-json, install "metocean" dependencies for full functionality')

def to_json(self, filename, attributes={}):
    """Save spectra in CF-JSON format.

    Not implemented!

    Args:
        - filename (str): name for output WW3 file.
        - attributes (dict): add description.

    """
    strout = self.dset.cfjson.json_dumps(indent=2, attributes=attributes)
    with open(filename,'w') as f:
        f.write(strout)