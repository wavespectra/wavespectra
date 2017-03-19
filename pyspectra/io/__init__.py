#To make available to core SpecArray class just import here
#Each module must provide a read_<format> function which returns a SpecArray object from a file
#And a to_<format> function that writes the file format from a SpecArray object

from cfjson import to_cfjson,read_cfjson
from netcdf import to_netcdf,read_netcdf
from octopus import to_octopus,read_octopus
from swan import to_swan,read_swam
from ww3 import to_ww3,read_ww3
from ww3_msl import to_ww3_msl,read_ww3_msl