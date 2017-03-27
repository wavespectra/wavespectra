#To make available to core SpecArray class just import here
#Each module must provide a read_<format> function which returns a SpecArray object from a file
#And a to_<format> function that writes the file format from a SpecArray object
#Note that you should import the SpecArray object inside the read function to avoid a circular import
from json import to_json,read_json
from netcdf import read_netcdf #to_netcdf already in xarray
from octopus import to_octopus,read_octopus
from swan import to_swan,read_swan
from ww3 import to_ww3,read_ww3
from ww3_msl import to_ww3_msl,read_ww3_msl
from attributes import *