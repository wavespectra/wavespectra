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
        - dset (SpecDataset): spectra dataset object read from funwave file.

    Note:
        - Format description: https://fengyanshi.github.io/build/html/wavemaker_para.html.
        - Both 2D E(f,d) and 1d E(f) spectra are supported.
        - Directions converted from Cartesian (0E, CCW, to) to wavespectra (0N, CW, from).
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

    # Spectral coordinates (convert dir from cartesian to wavespectra convention)
    freq = np.array([float(data.pop(0).split()[0]) for count in range(nf)])
    dir = np.array([float(data.pop(0).split()[0]) for count in range(nd)])
    dir = (270 - dir) % 360

    # Amplitude spectrum
    if nd == 1:
        amp = np.genfromtxt(data)
        coords = {attrs.FREQNAME: freq}
        dims = (attrs.FREQNAME)
    else:
        amp = np.genfromtxt(data[:nd])
        coords = {attrs.FREQNAME: freq, attrs.DIRNAME: dir}
        dims = (attrs.FREQNAME, attrs.DIRNAME)
    darr = xr.DataArray(data=amp.transpose(), coords=coords, dims=dims)

    # Energy density spectrum
    darr = darr ** 2 / (darr.spec.dfarr * darr.spec.dd * 2)

    # Define output dataset
    dset = darr.to_dataset(name=attrs.SPECNAME)
    if nd > 1:
        dset = dset.sortby(attrs.DIRNAME)
    dset["tp"] = tp
    set_spec_attributes(dset)
    return dset
