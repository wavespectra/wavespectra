"""Read Octopus spectra files.

Superfluous header-data is not read as the statistical properties can be obtained from the read spectral data.

"""
import warnings
import datetime
import numpy as np
import xarray as xr

from wavespectra.specdataset import SpecDataset
from wavespectra.core.attributes import attrs, set_spec_attributes
from wavespectra.core.attributes import attrs
from wavespectra.core.utils import bins_from_frequency_grid


def read_record(f):
    """Reads a record or group of record from the file

    input: f: filepointer
    """

    # skip empty lines
    line = "\n"
    while line == "\n":  # EOF is going to result in ''
        line = f.readline()
        if line == "":  # EOF
            return None

    try:
        meta = line.split(",")[0].rstrip("\n")
        nfreqs = int(f.readline().split(",")[1])
        ndirs = int(f.readline().split(",")[1])
        nrecs = int(f.readline().split(",")[1])
        Latitude = float(f.readline().split(",")[1])
        Longitude = float(f.readline().split(",")[1])
        Depth = float(f.readline().split(",")[1])
    except Exception as E:
        print("Error reading " + str(f))
        raise E

    spectra = []
    wind_speed = []
    wind_direction = []
    timestamps = []
    Hsig_reported = []

    for i in range(nrecs):
        for __ in range(2):
            next(f)
        parts = [part.lstrip("'") for part in f.readline().split(",")]

        import ipdb; ipdb.set_trace()
        timestamp = datetime.datetime.strptime("".join(parts[0:2]), "%Y%m%d%H%M")

        wind_direction.append(float(parts[3]))
        wind_speed.append(float(parts[4]))
        Hsig_reported.append(float(parts[16]))

        parts = f.readline().split(",")
        freqs = [float(f) for f in parts[1:-1]]

        assert len(freqs) == nfreqs, f"invalid frequency row for record {timestamp}"

        # read the data-block
        rows = [f.readline().split(",") for i in range(ndirs)]

        headings = [float(r[0]) for r in rows]

        # the first entry on each row is the heading
        # the last entry on each row is the "anspec"
        # everything in between is the spectra data
        #
        # read the known number of entries to avoid problems with line-ends (trailing ,\n )

        data = [[float(r) for r in row[1 : nfreqs + 1]] for row in rows]

        f.readline()  # fspec line (skip)
        f.readline()  # den line (skip)

        spectra.append(data)
        timestamps.append(timestamp)

    # spectra is the wave spectrum (deg,frequency [m2])
    # this is the energy in each bin
    # this needs to be re-written as the wave-spectral density with units  m2*s / deg
    #
    # the freqs are the bin centers
    # the headings are the bin centers

    dheading = np.diff(headings)
    dheading = np.mod(dheading, 360)
    if len(np.unique(dheading)) > 1:
        import warnings

        warnings.warn(
            f"Non-constant heading grid encountered in file: {dheading}, taking the mean value: {np.mean(dheading)}"
        )

    dh = np.mean(dheading)

    left, right, width, center = bins_from_frequency_grid(freqs)

    bin_area = width * dh  # deg / s

    # E        = time, direction, frequency
    # bin_area =                  frequency
    # right-most dimension align, so broadcasting works as we need it

    E = np.array(spectra, dtype=float) / bin_area[np.newaxis, np.newaxis, :]

    # construct the DataSet
    coords = {"time": timestamps, "freq": freqs, "dir": headings}  # frequency in Hz

    efth = xr.DataArray(
        data=E, coords=coords, dims=("time", "dir", "freq"), name="efth"
    )

    wind_speed = xr.DataArray(
        data=wind_speed, coords=[timestamps], dims=["time"], name=attrs.WSPDNAME
    )
    wind_direction = xr.DataArray(
        data=wind_direction, coords=[timestamps], dims=["time"], name=attrs.WDIRNAME
    )

    rhs = xr.DataArray(
        data=Hsig_reported, coords=[timestamps], dims=["time"], name="Reported_Hs"
    )

    dataset = efth.to_dataset()
    dataset = xr.merge([dataset, wind_speed, wind_direction, rhs])
    # coords= {   attrs.WDIRNAME: wind_direction,
    #             attrs.WSPDNAME: wind_speed },
    # name = 'wind')

    dataset.attrs = {
        "lat": Latitude,
        "lon": Longitude,
        "waterdepth": Depth,
        "forcast_issue": meta,
    }

    return dataset


def read_octopus(filename):
    """Read Spectra from octopus (.oct) or fugro, argoss (.csv) file.

    WARNING: As this format is used by quite some different suppliers, always double-check
    the units of wind-speed and others.

        Args:
            - filename (str, Path): File name to read

        Returns:
            - dset (SpecDataset): spectra dataset object read from file.

    """

    dataset = None

    with open(filename, "r") as f:

        while f:
            record = read_record(f)

            if record is None:  # EOF reached
                break

            if dataset is None:
                dataset = record
            else:
                dataset = xr.merge([dataset, record])

    return dataset
