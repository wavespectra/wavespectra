"""Standarise SpecArray attributes.

attrs (dict): standarised names for spectral variables, standard_names and units
"""
# from collections import OrderedDict
import os
import yaml
from attrdict import AttrDict

HERE = os.path.dirname(os.path.abspath(__file__))

with open(os.path.join(HERE, 'attributes.yml')) as stream:
    attrs =  AttrDict(yaml.load(stream, yaml.SafeLoader))

def set_spec_attributes(dset):
    """
    Standarise CF attributes in specarray variables
    """
    for varname, varattrs in attrs.ATTRS.items():
        try:
            dset[varname].attrs = varattrs
        except Exception as exc:
            pass
