"""
To make available to core SpecArray class just import here
Each module must provide a read_<format> function which returns a SpecArray object from a file
And a to_<format> function that writes the file format from a SpecArray object
Note that you should import the SpecArray object inside the read function to avoid a circular import
"""
import sys

from pyspectra.spectra.io.octopus import to_octopus, read_octopus
from pyspectra.spectra.io.swan import to_swan, read_swan
from pyspectra.spectra.io.attributes import *

try: # Check for dependencies (cfjson)
    from pyspectra.spectra.io.json import to_json, read_json
except:
    print "Cannot import json IO:", '%s: %s' % (sys.exc_info()[0], sys.exc_info()[1])

try: # Check for dependencies (xarray)
    from pyspectra.spectra.io.netcdf import to_netcdf, read_netcdf
except:
    print "Cannot import netcdf IO:", '%s: %s' % (sys.exc_info()[0], sys.exc_info()[1])

try: # Check for dependencies (xarray)
    from pyspectra.spectra.io.ww3 import to_ww3, read_ww3
except:
    print "Cannot import WW3 IO:", '%s: %s' % (sys.exc_info()[0], sys.exc_info()[1])

try: # Check for dependencies (xarray)
    from pyspectra.spectra.io.ww3_msl import to_ww3_msl, read_ww3_msl
except:
    print "Cannot import WW3_MSL IO:", '%s: %s' % (sys.exc_info()[0], sys.exc_info()[1])