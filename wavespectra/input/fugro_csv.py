import datetime
import numpy as np
import xarray as xr

from wavespectra.specdataset import SpecDataset
from wavespectra.core.attributes import attrs, set_spec_attributes
from wavespectra.core.utils import interp_spec

def read_fugro_csv(filename):
    """Read Spectra from fugro csv file.

    Args:
        - filename (str, Path): File name to read

    Returns:
        - dset (SpecDataset): spectra dataset object read from file.

    """

    with open(filename, 'r') as f:

        meta = f.readline().split(',')[0]
        nfreqs = int(f.readline().split(',')[1])
        ndirs = int(f.readline().split(',')[1])
        nrecs = int(f.readline().split(',')[1])
        Latitude = float(f.readline().split(',')[1])
        Longitude = float(f.readline().split(',')[1])
        Depth = float(f.readline().split(',')[1])

        spectra = []
        timestamps = []

        for i in range(nrecs):
            f.readline() # fist line is empty
            f.readline() # CCYYMM,DDHHmm,LPoint,WD,WS,ETot,TZ,VMD,ETotSe,TZSe,VMDSe,ETotSw,TZSw,VMDSw,Mo1,Mo2,HSig,DomDr,AngSpr,Tau,,,,,,,,,,,,,,,,,,

            parts = f.readline().split(',')

            parts = [part.lstrip("'") for part in parts]

            CC = int(parts[0][:2])
            YY = int(parts[0][2:4])
            MM = int(parts[0][4:])
            DD = int(parts[1][:2])
            HH = int(parts[1][2:4])
            mm = int(parts[1][4:])

            timestamp = datetime.datetime(year = 100*CC + YY,
                                          month = MM,
                                          day = DD,
                                          hour=HH,
                                          minute=mm)

            parts = f.readline().split(',')
            freqs = [float(f) for f in parts[1:-1]]

            assert len(freqs)==nfreqs, f'invalid frequency row for record {timestamp}'

            # read the data-block
            rows = [f.readline().split(',') for i in range(ndirs)]

            headings = [float(r[0]) for r in rows]

            # the first entry on each row is the heading
            # the last entry on each row is the "anspec"
            # everything in between is the spectra data

            data = [[float(r) for r in row[1:-1]] for row in rows]

            f.readline() # fspec line (skip)
            f.readline() # den line (skip)

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
        warnings.warn(f'Non-constant heading grid encountered in file {filename}: {dheading}, taking the mean value: {np.mean(dheading)}')


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
    ifreq = np.arange(1,nfreqs+1)
    freq_check = c1 * np.exp(c2 * ifreq)

    difference = freqs - freq_check
    max_difference = np.max(np.abs(difference))

    if max_difference > 0.001:  # number of significant decimals in csv file
        raise ValueError("Can accurately fit an exponential curve through the provided frequency bin centers")

    # determine bin edges and bin-widths for frequency bins
    left = c1 * np.exp(c2 * (ifreq-0.5))
    right = c1 * np.exp(c2 * (ifreq + 0.5))
    df = right - left

    bin_area = df * dh # deg / s

    # E        = time, direction, frequency
    # bin_area =                  frequency
    # right-most dimension align, so broadcasting works as we need it

    E = np.array(spectra, dtype=float) / bin_area[np.newaxis, np.newaxis, :]

    # construct the DataSet
    coords = {'time': timestamps,
              'freq': freqs,     # frequency in Hz
              'dir': headings}

    efth = xr.DataArray(data=E,
                        coords=coords,
                        dims=('time', 'dir', 'freq'),
                        name='efth')

    return efth.to_dataset(name='efth')





# Some tests if the file is run as main
if __name__ == '__main__':

    # Filename relative to the current "workdir"
    filename = r'./tests/sample_files/FUGRO_example.csv'

    dset = read_fugro_csv(filename)

    print(dset.spec.tp())



    import matplotlib.pyplot as plt

    t1 = dset.isel(time=0).spec.oned()

    x = np.array([0.0345,0.038,0.0418,0.0459,0.0505,0.0556,0.0612,0.0673,0.074,0.0814,0.0895,0.0985,0.1083,0.1192,0.1311,0.1442,0.1586,0.1745,0.1919,0.2111,0.2323,0.2555,0.281,0.3091,0.34,0.374,0.4114,0.4526,0.4979,0.5476,0.6024,0.6626,0.7289,0.8018,0.882,0.9702])
    y = [0,0,0,0,0,0,0,0,0.00084,0.00493,0.01974,0.04101,0.05516,0.09985,0.21993,0.39064,0.56939,0.67944,0.60426,0.41623,0.27249,0.19116,0.13955,0.0999,0.06835,0.04573,0.03075,0.0207,0.01392,0.00922,0.00601,0.00416,0.00275,0.0018,0.0012,0.00079]

    plt.plot(1/x, y,label = 'target')

    # plt.plot(1/t1['freq'],dset.isel(time=0).spec.efth)

    plt.plot(1/t1['freq'], t1, label = 'actual')
    plt.legend()

    plt.show()

    ds = dset.isel(time=0)
    ds.spec.plot.contourf(cmap="GnBu")
    plt.show()

    hs = dset.spec.hs()
    hs.plot()
    plt.grid()
    plt.show()

    dset.spec.tp().plot()
    plt.grid()
    plt.show()

    # ds = dset.spec.oned().rename({"freq": "period"})

    # ds = ds.assign_coords({"period": 1 / ds.period})

    # ds.period.attrs.update({"standard_name": "sea_surface_wave_period", "units": "s"})
    ds=dset.spec.oned()

    ds.plot.contourf(x="time", y="freq", vmax=1, cmap='GnBu')
    plt.show()


