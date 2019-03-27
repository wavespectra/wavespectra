"""Read cf-json spectra files."""
import json
import dateutil
import xarray as xr

from wavespectra.specdataset import SpecDataset
from wavespectra.core.attributes import attrs, set_spec_attributes

def read_cf_json(filename):
    """Read Spectra from cf-json format from MSL spectra API.

    Args:
        - filename (str): name of cf-json file to read.

    Returns:
        - dset (SpecDataset): spectra dataset object read from json file.

    """
    with open(filename, 'r') as stream:
        spec_dict = json.load(stream)

    data = {}
    for varname, vardata in spec_dict['variables'].items():
        dims = vardata.get('shape',[])
        if varname == 'time':
            attrs = {}
            vals = to_datetime(vardata['data'])
        else:
            attrs = vardata['attributes']
            vals = vardata['data']
        data.update({varname: (dims, vals, attrs)})
    return xr.Dataset(data)

def to_datetime(times):
    return [dateutil.parser.parse(t) for t in times]