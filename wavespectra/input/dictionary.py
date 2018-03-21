import xarray as xr
from wavespectra.specdataset import SpecDataset
from wavespectra.core.attributes import set_spec_attributes
def read_dictionary(spcdict):
    """Read spectra from generic dictionary.

    Args:
        - spcdict (dict): information for spectra. For example:
    
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
    units have to be as defined in attrs.

    """
    spcdict = {k: v for k, v in spcdict.items() if len(v['data'])}
    dset = xr.Dataset.from_dict(spcdict)
    set_spec_attributes(dset)
    return dset