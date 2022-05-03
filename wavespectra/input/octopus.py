"""Read Octopus spectra files."""
import datetime
import numpy as np
import xarray as xr

from wavespectra.specdataset import SpecDataset
from wavespectra.core.attributes import attrs, set_spec_attributes


def read_octopus(filename):
    """Read spectra from Octopus file format.

    Args:
        - filename (str, Path): Octopus file to read.

    Returns:
        - dset (SpecDataset): spectra dataset object.

    """
    with open(filename, "r") as f:

        efths = []
        wspds = []
        wdirs = []
        times = []
        dset = xr.Dataset()

        description = f.readline().rstrip("\n")
        nfreqs = int(f.readline().split(",")[1])
        ndirs = int(f.readline().split(",")[1])
        nrecs = int(f.readline().split(",")[1])

        # Coordinates
        dset["lat"] = xr.DataArray([float(f.readline().split(",")[1])], dims=("site",))
        dset["lon"] = xr.DataArray([float(f.readline().split(",")[1])], dims=("site",))
        dset["site"] = [0]

        dpt = float(f.readline().split(",")[1])

        # Append each record
        for i in range(nrecs):
            for __ in range(2):
                next(f)
            parts = [part.lstrip("'") for part in f.readline().split(",")]

            times.append(datetime.datetime.strptime("".join(parts[0:2]), "%Y%m%d%H%M"))
            wdirs.append(float(parts[3]))
            wspds.append(float(parts[4]))

            # Frequencies
            freqs = [float(f) for f in f.readline().split(",")[1:-1]]
            if len(freqs) != nfreqs:
                raise OSError(f"Invalid frequency row for record {times[-1]}")

            data = np.genfromtxt(
                f,
                delimiter=",",
                dtype="float",
                usecols=np.arange(nfreqs + 1),
                max_rows=ndirs,
                unpack=True,
            )

            # Directions
            dirs = data[0, :]

            # Energy data
            efths.append(data[1:, :])

            for __ in range(2):
                next(f)

    # Output
    ds = xr.DataArray(
        data=efths,
        coords={"time": times, "freq": freqs, "dir": dirs},
        dims=("time", "freq", "dir"),
        name="efth",
    ).to_dataset()
    dset["efth"] = (ds.efth / (ds.spec.dfarr * ds.spec.dd)).expand_dims("site", axis=1)
    dset["wspd"] = xr.DataArray(wspds, dims=("time",)).expand_dims("site", axis=1)
    dset["wdir"] = xr.DataArray(wdirs, dims=("time",)).expand_dims("site", axis=1)

    # Set attributes
    set_spec_attributes(dset)
    dset.attrs.update({"description": description})

    return dset
