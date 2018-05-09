"""Read cf-json spectra files."""
import json
from collections import OrderedDict
import numpy as np
import xarray as xr

from wavespectra.specdataset import SpecDataset
from wavespectra.core.attributes import attrs, set_spec_attributes

def read_cfjson(filename):
    """Read Spectra from cf-json format from MSL spectra API.

    Args:
        - filename (str): name of cf-json file to read.

    Returns:
        - dset (SpecDataset): spectra dataset object read from json file.

    """
    with open(filename, 'r') as stream:
        spec_dict = json.load(stream)

    dims = set()
    data = OrderedDict()
    for varname, vardata in spec_dict['variables'].items():
        shape = vardata.get('shape', [])
        dims.update(shape)
        data.update({varname: (shape, np.array(vardata['data']), vardata['attributes'])})

    return xr.Dataset(data)