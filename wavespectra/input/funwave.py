"""Read Funwave 2D Wave Maker files"""
import numpy as np
import xarray as xr

from wavespectra import SpecArray
from wavespectra.core.attributes import attrs, set_spec_attributes


def read_funwave(filename):
    """Read Spectra in Funwave format.

    Args:
        - filename (str): Funwave file to read.

    Returns:
        - dset (SpecDataset): spectra dataset object read from json file.

    Note:
        - Format description: https://fengyanshi.github.io/build/html/wavemaker_para.html.
        - Phases are ignored if present.

    """
    with open(filename, "r") as stream:
        data = stream.readlines()

    # Remove any empty rows
    data = [row for row in data if row != "\n"]

    # Shape
    nf, nd = [int(val) for val in data.pop(0).split()[:2]]

    # Tp
    tp = float(data.pop(0).split()[0])

    # Spectral coordinates
    freq = np.array([float(data.pop(0).split()[0]) for count in range(nf)])
    dir = np.array([float(data.pop(0).split()[0]) for count in range(nd)])

    # Amplitude spectrum
    amp = np.genfromtxt(data[:nd])
    darr = xr.DataArray(
        data=amp.transpose(),
        coords={attrs.FREQNAME: freq, attrs.DIRNAME: dir},
        dims=(attrs.FREQNAME, attrs.DIRNAME),
    )

    # Energy density spectrum
    darr = darr ** 2 / (darr.spec.dfarr * darr.spec.dd * 2)

    # Define output dataset
    dset = darr.to_dataset(name=attrs.SPECNAME)
    dset["tp"] = tp
    set_spec_attributes(dset)
    return dset

