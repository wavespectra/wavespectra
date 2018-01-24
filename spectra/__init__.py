"""
Making all reading functions from readspec.py available at module level
Writing functions are defined in the specdataset module as methods of SpecDataset class
"""
import sys
from attributes import *

try:
    from readspec import read_swan
except:
    print("Cannot import reading read_swan:", sys.exc_info()[0])

try:
    from readspec import read_ww3
except:
    print("Cannot import reading read_ww3:", sys.exc_info()[0])

try:
    from readspec import read_ww3_msl
except:
    print("Cannot import reading read_ww3_msl:", sys.exc_info()[0])

try:
    from readspec import read_netcdf
except:
    print("Cannot import reading read_netcdf:", sys.exc_info()[0])

try:
    from readspec import read_octopus
except:
    print("Cannot import reading read_octopus:", sys.exc_info()[0])

try:
    from readspec import read_json
except:
    print("Cannot import reading read_json:", sys.exc_info()[0])

__version__ = '1.0'