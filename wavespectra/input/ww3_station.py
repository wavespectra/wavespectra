"""Read directional spectra from WW3 station output file."""
from xarray.backends import BackendEntrypoint
from collections import OrderedDict
import re
import datetime
import numpy as np
import xarray as xr

from wavespectra.core.attributes import attrs, set_spec_attributes
from wavespectra.input import read_ascii_or_binary
from wavespectra.core.utils import D2R, R2D


HEADER_REGEX_STR = (
    r"'WAVEWATCH III SPECTRA'\s*([0-9]{0,2})\s*([0-9]{0,2})\s*([0-9]{0,2})"
)


def extract_direction(dir):
    """Convert direction from ww3 station file to be in the correct
    convention.

    Args:
        - dir (float): raw direction string in radians
    """
    dir = abs(((float(dir) - (2.5 * np.pi)) % (2.0 * np.pi)))
    return ((dir * R2D) + 270) % 360


def read_ww3_station(filename_or_fileglob):
    """Read directional spectra from WW3 station output file.

    Args:
        - filename_or_fileglob (str, filelike): filename or file object to read.

    Returns:
        - dset (SpecDataset): spectra dataset object read from
          WW3 station output file.
    """
    lines = read_ascii_or_binary(filename_or_fileglob, mode="r")

    header = lines.pop(0)
    header_regex = re.compile(HEADER_REGEX_STR)
    try:
        # Parse out the frequency and direction dimensions,
        # ignore the location count dimension for now
        nfreq, ndir, nloc = header_regex.match(header).groups()
        nfreq = int(nfreq)
        ndir = int(ndir)
        nloc = int(nloc)
    except Exception:
        raise ValueError("Could not parse header line of WW3 station file")

    # Read the frequency, direction, and location coordinates
    freqs = []
    while len(freqs) < nfreq:
        freqs.extend(map(float, lines.pop(0).split()))

    dirs = []
    while len(dirs) < ndir:
        dirs.extend(map(extract_direction, lines.pop(0).split()))

    # Parse the spectra for each timestep until the end of the file
    date = []
    loc = []
    lat = []
    lon = []
    depth = []
    wind_speed = []
    wind_dir = []
    current_speed = []
    current_dir = []
    spectra = []

    while True:
        try:
            line = lines.pop(0)
            if not line:
                break
        except IndexError:
            break

        # If there are more than one location to be parsed, we need to parse
        # the points for each location. Otherwise, we can just
        # parse the points for the single location for now.
        d = datetime.datetime.strptime(line.strip(), "%Y%m%d %H%M%S")
        date.append(d)

        line = lines.pop(0)
        first_split = line.strip().split("'")
        loc.append(first_split[1].strip())
        parts = first_split[2].split()
        lat.append(float(parts[0]))
        lon.append(float(parts[1]))
        depth.append(float(parts[2]))
        wind_speed.append(float(parts[3]))
        wind_dir.append(float(parts[4]))
        current_speed.append(float(parts[5]))
        current_dir.append(float(parts[6]))

        spec_count = 0
        while spec_count < nfreq * ndir:
            vals = list(map(float, lines.pop(0).strip().split()))
            spectra.extend(vals)
            spec_count += len(vals)

    spectra = np.array(spectra)
    times = np.unique(date)
    locs = np.unique(loc)
    lats = np.unique(lat)
    lons = np.unique(lon)

    spec_arr = np.array(spectra).reshape(
        len(times), len(lats), len(lons), len(dirs), len(freqs)
    )

    # Convert from m2/rad to m2/deg
    spec_arr *= D2R

    dset = xr.DataArray(
        data=spec_arr.swapaxes(3, 4),
        coords=OrderedDict(
            (
                (attrs.TIMENAME, times),
                (attrs.LATNAME, lats),
                (attrs.LONNAME, lons),
                (attrs.FREQNAME, freqs),
                (attrs.DIRNAME, dirs),
            )
        ),
        dims=(
            attrs.TIMENAME,
            attrs.LATNAME,
            attrs.LONNAME,
            attrs.FREQNAME,
            attrs.DIRNAME,
        ),
        name=attrs.SPECNAME,
    ).to_dataset()

    dset[attrs.WSPDNAME] = xr.DataArray(
        data=np.array(wind_speed).reshape(len(times), len(lats), len(lons)),
        dims=[attrs.TIMENAME, attrs.LATNAME, attrs.LONNAME],
    )
    dset[attrs.WDIRNAME] = xr.DataArray(
        data=np.array(wind_dir).reshape(len(times), len(lats), len(lons)),
        dims=[attrs.TIMENAME, attrs.LATNAME, attrs.LONNAME],
    )
    dset[attrs.DEPNAME] = xr.DataArray(
        data=np.array(depth).reshape(len(times), len(lats), len(lons)),
        dims=[attrs.TIMENAME, attrs.LATNAME, attrs.LONNAME],
    )

    dset[attrs.LATNAME] = xr.DataArray(
        data=lats, coords={attrs.SITENAME: locs}, dims=[attrs.SITENAME]
    )
    dset[attrs.LONNAME] = xr.DataArray(
        data=lons, coords={attrs.SITENAME: locs}, dims=[attrs.SITENAME]
    )

    set_spec_attributes(dset)
    dset[attrs.SPECNAME].attrs.update(
        {"_units": "m^{2}.s.degree^{-1}", "_variable_name": "VaDens"}
    )

    return dset


class WW3StationBackendEntrypoint(BackendEntrypoint):
    """WW3 station backend engine."""

    def open_dataset(
        self,
        filename_or_obj,
        *,
        drop_variables=None,
    ):
        return read_ww3_station(filename_or_obj)

    def guess_can_open(self, filename_or_obj):
        return False

    description = "Open WW3 station spectra files as a wavespectra dataset."

    url = "https://github.com/wavespectra/wavespectra"
