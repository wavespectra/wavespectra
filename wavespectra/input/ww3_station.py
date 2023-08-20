import re
import datetime
import xarray as xr

from wavespectra.core.attributes import attrs


HEADER_REGEX_STR = r"'WAVEWATCH III SPECTRA'\s*([0-9]{0,2})\s*([0-9]{0,2})\s*([0-9]{0,2})"


def read_ww3_station(fileobj):
    """Read directional spectra from WW3 station output file.
    Args:
        - fileobj (file-like): file to read.

    Returns:
        - dset (SpecDataset): spectra dataset object read from WW3 station output file.
    """
    header = fileobj.readline()
    header_regex = re.compile(HEADER_REGEX_STR)
    try: 
        # Parse out the frequency and direction dimensions, ignore the location count dimension for now
        nfreq, ndir, nloc = header_regex.match(header).groups()
        nfreq = int(nfreq)
        ndir = int(ndir)
        nloc = int(nloc)
    except Exception as e:
        raise ValueError("Could not parse header line of WW3 station file.") from e
    
    # Read the frequency, direction, and location coordinates
    freq = []
    while len(freq) < nfreq:
        freq.extend(map(float, fileobj.readline().split()))
    
    dir = []
    while len(dir) < ndir:
        dir.extend(map(float, fileobj.readline().split()))

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
        line = fileobj.readline()
        if not line:
            break

        d = datetime.datetime.strptime(line.strip(), "%Y%m%d %H%M%S")
        date.append(d)

        line = fileobj.readline()
        if not line:
            break
        
        first_split = line.split("'")
        loc.append(first_split[0])
        parts = first_split[1].split()
        lat.append(float(parts[0]))
        lon.append(float(parts[1]))
        depth.append(float(parts[2]))
        wind_speed.append(float(parts[3]))
        wind_dir.append(float(parts[4]))
        current_speed.append(float(parts[5]))
        current_dir.append(float(parts[6]))

        spec = []
        while len(spec) < nfreq * ndir:
            spec.extend(map(float, fileobj.readline().split()))

        spectra.append(spec)

    # TODO: Construct the spectra dataset