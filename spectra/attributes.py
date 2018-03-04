"""Standarise SpecArray attributes.

attrs (dict): standarised names for spectral variables, standard_names and units
"""
# from collections import OrderedDict
import os
import ruamel.yaml as yaml
from attrdict import AttrDict

here = os.path.dirname(os.path.abspath(__file__))

with open(os.path.join(here, 'attributes.yml')) as fid:
    attrs =  AttrDict(yaml.load(fid, Loader=yaml.RoundTripLoader))

def set_spec_attributes(dset):
    """
    Standarise CF attributes in specarray variables
    """
    for varname, varattrs in attrs.ATTRS.items():
        try:
            dset[varname].attrs = varattrs
        except Exception as exc:
            pass
