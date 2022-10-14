"""Read XWaves MAT files"""
import datetime
import xarray as xr
from scipy.io import loadmat

from wavespectra.specdataset import SpecDataset
from wavespectra.core.attributes import attrs, set_spec_attributes
from wavespectra.core.utils import R2D

def read_xwaves(filename):
    """Read Spectra from XWaves MAT format.

    Args:
        - filename (str): File to read.

    Returns:
        - dset (SpecDataset): spectra dataset object read from netcdf file.

    """
    # Load and construct dataset
    data = loadmat(filename)
    time = [datetime.datetime(*row) for row in data["td"]]
    freq = data["fd"].ravel()
    dir = data["thetad"].ravel()
    dset = xr.DataArray(
        data=data["spec2d"] / R2D,
        coords={"time": time, attrs.FREQNAME: freq, attrs.DIRNAME: dir},
        dims=("time", attrs.FREQNAME, attrs.DIRNAME),
        name="efth",
    ).to_dataset()

    # Assign metadata
    header = data["__header__"]
    if isinstance(header, bytes):
        header = header.decode("utf-8")

    version = data["__version__"]
    if isinstance(version, bytes):
        version = version.decode("utf-8")

    dset.attrs = {"header": header, "version": version}

    # Setting standard attributes
    set_spec_attributes(dset)

    return dset