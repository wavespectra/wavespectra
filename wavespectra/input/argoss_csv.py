import datetime
import numpy as np
import xarray as xr

from wavespectra.core.attributes import attrs

# "wnddir": attrs.WDIRNAME,
# "wnd": attrs.WSPDNAME

def read_record(f):
    """Reads a record or group of record from the file

    input: f: filepointer
    """

    # skip empty lines
    line = '\n'
    while line == '\n':  # EOF is going to result in ''
        line = f.readline()
        if line == '': # EOF
            return None


    meta = line.split(',')[0]
    nfreqs = int(f.readline().split(',')[1])
    ndirs = int(f.readline().split(',')[1])
    nrecs = int(f.readline().split(',')[1])
    Latitude = float(f.readline().split(',')[1])
    Longitude = float(f.readline().split(',')[1])
    Depth = float(f.readline().split(',')[1])

    spectra = []
    wind_speed = []
    wind_direction = []
    timestamps = []
    Hsig_reported = []

    for i in range(nrecs):
        f.readline()  # fist line is empty
        f.readline()  # CCYYMM,DDHHmm,LPoint,WD,WS,ETot,TZ,VMD,ETotSe,TZSe,VMDSe,ETotSw,TZSw,VMDSw,Mo1,Mo2,HSig,DomDr,AngSpr,Tau,,,,,,,,,,,,,,,,,,

        parts = f.readline().split(',')

        parts = [part.lstrip("'") for part in parts]

        CC = int(parts[0][:2])
        YY = int(parts[0][2:4])
        MM = int(parts[0][4:])
        DD = int(parts[1][:2])
        HH = int(parts[1][2:4])
        mm = int(parts[1][4:])

        timestamp = datetime.datetime(year=100 * CC + YY,
                                      month=MM,
                                      day=DD,
                                      hour=HH,
                                      minute=mm)

        wind_direction.append(float(parts[3]))
        wind_speed.append(float(parts[4]))
        Hsig_reported.append(float(parts[16]))

        parts = f.readline().split(',')
        freqs = [float(f) for f in parts[1:-1]]

        assert len(freqs) == nfreqs, f'invalid frequency row for record {timestamp}'



        # read the data-block
        rows = [f.readline().split(',') for i in range(ndirs)]

        headings = [float(r[0]) for r in rows]

        # the first entry on each row is the heading
        # the last entry on each row is the "anspec"
        # everything in between is the spectra data

        data = [[float(r) for r in row[1:-1]] for row in rows]

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
            f'Non-constant heading grid encountered in file {filename}: {dheading}, taking the mean value: {np.mean(dheading)}')

    dh = np.mean(dheading)

    # the frequency grid is exponential
    # to determine the bin-size we need to re-construct the original bin boundaries

    # freq_center_i = c1 * exp(c2 * i)
    #
    # c2 is the n-log of the frequency increase factor
    # take the average to minimize the effect of lack of significant digits
    freqs = np.array(freqs)
    increase_factor = np.mean(freqs[1:] / freqs[:-1])
    c2 = np.log(increase_factor)

    # determine c1 from the highest bin
    c1 = freqs[-1] / np.exp(c2 * nfreqs)

    # check the assumptions
    ifreq = np.arange(1, nfreqs + 1)
    freq_check = c1 * np.exp(c2 * ifreq)

    difference = freqs - freq_check
    max_difference = np.max(np.abs(difference))

    if max_difference > 0.001:  # number of significant decimals in csv file
        raise ValueError("Can accurately fit an exponential curve through the provided frequency bin centers")

    # determine bin edges and bin-widths for frequency bins
    left = c1 * np.exp(c2 * (ifreq - 0.5))
    right = c1 * np.exp(c2 * (ifreq + 0.5))
    df = right - left

    bin_area = df * dh  # deg / s

    # E        = time, direction, frequency
    # bin_area =                  frequency
    # right-most dimension align, so broadcasting works as we need it

    E = np.array(spectra, dtype=float) / bin_area[np.newaxis, np.newaxis, :]

    # construct the DataSet
    coords = {'time': timestamps,
              'freq': freqs,  # frequency in Hz
              'dir': headings}

    efth = xr.DataArray(data=E,
                        coords=coords,
                        dims=('time', 'dir', 'freq'),
                        name='efth')

    wind_speed = xr.DataArray(data = wind_speed, coords=[timestamps], dims = ['time'], name =  attrs.WSPDNAME)
    wind_direction = xr.DataArray(data = wind_direction, coords=[timestamps], dims = ['time'],  name =  attrs.WDIRNAME)

    rhs = xr.DataArray(data=Hsig_reported, coords=[timestamps], dims=['time'], name='Reported_Hs')

    dataset = efth.to_dataset()
    dataset = xr.merge([dataset, wind_speed, wind_direction, rhs])
                        # coords= {   attrs.WDIRNAME: wind_direction,
                        #             attrs.WSPDNAME: wind_speed },
                        # name = 'wind')

    dataset.attrs = {'lat': Latitude,
                     'lon': Longitude,
                     'waterdepth':Depth,
                     'forcast_issue':meta}

    return dataset


def read_argoss_csv(filename):
    """Read Spectra from fugro csv file.

    Args:
        - filename (str, Path): File name to read

    Returns:
        - dset (SpecDataset): spectra dataset object read from file.

    """

    dataset = None

    with open(filename, 'r') as f:

        while f:
            record = read_record(f)

            if record is None:  # EOF reached
                break

            if dataset is None:
                dataset = record
            else:
                dataset = xr.merge([dataset, record])

    return dataset
