"""Standarise SpecArray attributes.

attrs (dict): standarised names for spectral variables, standard_names and units
"""
# from collections import OrderedDict
import os
import yaml

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
            raise TypeError('expected dict')

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
    for varname, varattrs in attrs.ATTRS.items():
        try:
            dset[varname].attrs = varattrs
        except Exception:
            pass
