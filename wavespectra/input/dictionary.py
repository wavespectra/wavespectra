"""Read spectra from dictionary."""
import xarray as xr

from wavespectra.specdataset import SpecDataset
from wavespectra.core.attributes import set_spec_attributes

def read_dictionary(spcdict):
    """Read spectra from generic dictionary.

    Args:
        - spcdict (dict): information for defining SpecDataset. Keys define
          spectral coordinates and variables, and should be named using the
          attributes from :py:mod:`wavespectra.core.attributes.attrs`.
    
    Example:
        .. code:: python

            from wavespectra.core.attributes import attrs

            spcdict = {
                attrs.TIMENAME: {'dims': (attrs.TIMENAME), 'data': time},
                attrs.FREQNAME: {'dims': (attrs.FREQNAME), 'data': freq},
                attrs.DIRNAME: {'dims': (attrs.DIRNAME), 'data': dirs},
                attrs.SITENAME: {'dims': (attrs.SITENAME), 'data': site},
                attrs.SPECNAME: {'dims': (attrs.TIMENAME, attrs.DIRNAME, attrs.FREQNAME), 'data': efth},
                attrs.LONNAME: {'dims': (attrs.SITENAME), 'data': lon},
                attrs.LATNAME: {'dims': (attrs.SITENAME), 'data': lat},
                attrs.DEPNAME: {'dims': (attrs.SITENAME, attrs.TIMENAME), 'data': dpt},
                attrs.WDIRNAME: {'dims': (attrs.TIMENAME), 'data': wdir},
                attrs.WSPDNAME: {'dims': (attrs.TIMENAME), 'data': wnd},
                }

    """
    spcdict = {k: v for k, v in spcdict.items() if len(v['data'])}
    dset = xr.Dataset.from_dict(spcdict)
    set_spec_attributes(dset)
    return dset