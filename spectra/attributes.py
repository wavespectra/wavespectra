from collections import OrderedDict

SPECNAME = 'efth'
TIMENAME = 'time'
SITENAME = 'site'
LATNAME = 'lat'
LONNAME = 'lon'
FREQNAME = 'freq'
DIRNAME = 'dir'
PARTNAME = 'part'
CYCLENAME = 'cycle'
WSPDNAME = 'wspd'
WDIRNAME = 'wdir'
DEPNAME = 'dpt'
ATTRS = {
    SPECNAME: OrderedDict((
        ('standard_name', 'sea_surface_wave_directional_variance_spectral_density'),
        ('units', 'm^{2}.s.deg^{-1}'),
        )),
    TIMENAME: OrderedDict((
        ('standard_name', 'time'),
        )),
    LATNAME: OrderedDict((
        ('standard_name', 'latitude'),
        ('units', 'degrees_north'),
        )),
    LONNAME: OrderedDict((
        ('standard_name', 'longitude'),
        ('units', 'degrees_east'),
        )),
    SITENAME: OrderedDict((
        ('standard_name', 'site'),
        ('units', ''),
        )),
    FREQNAME: OrderedDict((
        ('standard_name', 'sea_surface_wave_frequency'),
        ('units', 'Hz'),
        )),
    DIRNAME: OrderedDict((
        ('standard_name', 'sea_surface_wave_from_direction'),
        ('units', 'degree'),
        )),
    PARTNAME: OrderedDict((
        ('standard_name', 'spectral_partition_number'),
        ('units', ''),
        )),
    CYCLENAME: OrderedDict((
        ('standard_name', 'forecast_reference_time'),
        )),
    WSPDNAME: OrderedDict((
        ('standard_name', 'wind_speed'),
        ('units', 'm/s'),
        )),
    WDIRNAME: OrderedDict((
        ('standard_name', 'wind_from_direction'),
        ('units', 'degree'),
        )),
    DEPNAME: OrderedDict((
        ('standard_name', 'sea_floor_depth_below_sea_surface'),
        ('units', 'm'),
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
