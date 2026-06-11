"""Read Funwave WK_NEW_DATA2D Wave Maker files"""

from xarray.backends import BackendEntrypoint
import numpy as np
import xarray as xr

from wavespectra import SpecArray
from wavespectra.input import read_ascii_or_binary
from wavespectra.core.attributes import attrs, set_spec_attributes


def read_funwave_new(filename_or_obj):
    """Read Spectra in Funwave WK_NEW_DATA2D wavemaker format.

    Args:
        - filename_or_obj (str, filelike): Funwave WaveCompFile to read.

    Returns:
        - dset (SpecDataset): spectra dataset object read from funwave file.

    Note:
        - Format description: https://fengyanshi.github.io/build/html/wavemaker_coherence.html.
        - The file describes individual wave components, the gridded spectrum is
          reconstructed by summing component energies :math:`a^2/2` falling in each
          frequency-direction bin defined by the unique frequencies and directions.
        - Directions converted from Cartesian (0E, CCW, to) to wavespectra (0N, CW, from).
        - A 1D :math:`E(f)` spectrum is returned if all directions are equal.
        - Phases are ignored if present.

    """
    data = read_ascii_or_binary(filename_or_obj, mode="r")

    # Remove any empty rows
    data = [row for row in data if row != "\n"]

    # Number of wave components
    nc = int(data.pop(0).split()[0])

    # Tp
    tp = float(data.pop(0).split()[0])

    # Wave component blocks
    values = [float(row.split()[0]) for row in data]
    freq = np.array(values[:nc])
    dir = np.array(values[nc : 2 * nc])
    amp = np.array(values[2 * nc : 3 * nc])

    # Bin component energies onto the spectral grid
    freqs = np.unique(freq)
    dirs = np.unique(dir)
    ifreq = np.searchsorted(freqs, freq)
    idir = np.searchsorted(dirs, dir)
    energy = np.zeros((freqs.size, dirs.size))
    np.add.at(energy, (ifreq, idir), amp**2 / 2)

    if dirs.size == 1:
        coords = {attrs.FREQNAME: freqs}
        dims = (attrs.FREQNAME,)
        energy = energy.squeeze(axis=1)
    else:
        # Convert dir from cartesian to wavespectra convention
        dirs = (270 - dirs) % 360

        # Turn zero dir into 360 for continuity
        i0 = np.where(dirs == 0)[0]
        if i0.size > 0:
            dirs[i0] = 360.0

        coords = {attrs.FREQNAME: freqs, attrs.DIRNAME: dirs}
        dims = (attrs.FREQNAME, attrs.DIRNAME)
    darr = xr.DataArray(data=energy, coords=coords, dims=dims)

    # Energy density spectrum
    darr = darr / (darr.spec.df * darr.spec.dd)

    # Define output dataset
    dset = darr.to_dataset(name=attrs.SPECNAME)
    if dirs.size > 1:
        dset = dset.sortby(attrs.DIRNAME)
    dset["tp"] = tp
    set_spec_attributes(dset)
    return dset


class FunwaveNewBackendEntrypoint(BackendEntrypoint):
    """Funwave WK_NEW_DATA2D backend engine."""

    def open_dataset(
        self,
        filename_or_obj,
        *,
        drop_variables=None,
    ):
        return read_funwave_new(filename_or_obj)

    def guess_can_open(self, filename_or_obj):
        return False

    description = (
        "Open Funwave WK_NEW_DATA2D ASCII wave component files as a wavespectra "
        "dataset."
    )

    url = "https://github.com/wavespectra/wavespectra"
