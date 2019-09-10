"""Read SWAN spectra files.

Functions:
    read_spotter: Read Spectra from Spotter JSON file
    read_spotters: Read multiple spotter files into single Dataset

"""
import os
import glob
import datetime
import warnings
import pandas as pd
import xarray as xr
import numpy as np
import json



def read_spotter(filename, dirorder=True, as_site=None):
    """Read Spectra from spotter JSON file.

    Args:
        - dirorder (bool): If True reorder spectra so that directions are
          sorted.
        - as_site (bool): If True locations are defined by 1D site dimension.

    Returns:
        - dset (SpecDataset): spectra dataset object read from file.

    """
