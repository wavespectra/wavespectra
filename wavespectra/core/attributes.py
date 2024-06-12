"""Standarise SpecArray attributes.

attrs (dict): standarised names for spectral variables, standard_names and units
"""
import os
import yaml
import xarray as xr

HERE = os.path.dirname(os.path.abspath(__file__))


class AttrDict(dict):
    MARKER = object()

    def __init__(self, value=None):
        if value is None:
            pass
        elif isinstance(value, dict):
            for key in value:
                self.__setitem__(key, value[key])
        else:
            raise TypeError("expected dict")

    def __setitem__(self, key, value):
        if isinstance(value, dict) and not isinstance(value, AttrDict):
            value = AttrDict(value)
        super(AttrDict, self).__setitem__(key, value)

    def __getitem__(self, key):
        found = self.get(key, AttrDict.MARKER)
        if found is AttrDict.MARKER:
            found = AttrDict()
            super(AttrDict, self).__setitem__(key, found)
        return found

    __setattr__, __getattr__ = __setitem__, __getitem__


with open(os.path.join(HERE, "attributes.yml")) as stream:
    attrs = AttrDict(yaml.load(stream, yaml.SafeLoader))


def set_spec_attributes(dset):
    """
    Standarise CF attributes in specarray variables
    """
    if isinstance(dset, xr.DataArray):
        if dset.name in attrs.ATTRS:
            dset.attrs = attrs.ATTRS[dset.name]
    elif isinstance(dset, xr.Dataset):
        for varname in dset.data_vars:
            if varname in attrs.ATTRS:
                dset[varname].attrs = attrs.ATTRS[varname]
