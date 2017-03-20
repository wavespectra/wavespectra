from collections import OrderedDict

SPECNAME = 'efth'
TIMENAME = 'time'
SITENAME = 'site'
LATNAME = 'lat'
LONNAME = 'lon'
FREQNAME = 'freq'
DIRNAME = 'dir'
SPECATTRS = OrderedDict((
    ('standard_name', 'sea_surface_wave_directional_variance_spectral_density'),
    ('units', 'm2s')
    ))