from collections import OrderedDict

SPECNAME = 'efth'
TIMENAME = 'time'
SITENAME = 'site'
LATNAME = 'lat'
LONNAME = 'lon'
FREQNAME = 'freq'
DIRNAME = 'dir'
ATTRS = {
    SPECNAME: OrderedDict((
        ('standard_name', 'sea_surface_wave_directional_variance_spectral_density'),
        ('units', 'm2s')
        )),
    TIMENAME: OrderedDict((
        ('standard_name', 'time'),
        )),
    LATNAME: OrderedDict((
        ('standard_name', 'latitude'),
        ('units', 'degrees_north')
        )),
    LONNAME: OrderedDict((
        ('standard_name', 'longitude'),
        ('units', 'degrees_east')
        )),
    SITENAME: OrderedDict((
        ('standard_name', 'site'),
        ('units', '')
        )),
    FREQNAME: OrderedDict((
        ('standard_name', 'sea_surface_wave_frequency'),
        ('units', 'Hz')
        )),
    DIRNAME: OrderedDict((
        ('standard_name', 'sea_surface_wave_from_direction'),
        ('units', 'Hz')
        )),
}

def set_spec_attributes(dset):
    """
    Standarise CF attributes in specarray variables
    """
    for varname, varattrs in ATTRS.items():
        try:
            dset[varname].attrs = varattrs
        except Exception as exc:
            pass
