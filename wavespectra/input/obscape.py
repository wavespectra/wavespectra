"""WaveSpectra reader for Obscape CSV files (as downloaded from portal).

Buoy data: https://obscape.com/site/
Note: for raw data use the python file that can be downloaded from the obs site.

Revision history:
----------------
2024-03-11 : First version



"""

from datetime import datetime
from logging import warn
from pathlib import Path
import numpy as np
import xarray as xr

from wavespectra.core.attributes import set_spec_attributes


def get_timestamp(stem):


    try:
        year = int(stem[:4])
        month = int(stem[4:6])
        day = int(stem[6:8])
        hour = int(stem[9:11])
        minute = int(stem[11:13])
        second = int(stem[13:15])
    except ValueError:
        warn(
            f'Filename does not contain a valid UTC timestamp, expect the filename to start with yyyymmdd_hhmmss but got: {stem}')
        return None

    # make a python datetime object
    utc = datetime(year, month, day, hour, minute, second)

    return utc


def read_obscape_file(filename : str or Path):
    """
    Read an Obscape file.
    
    - metadata is marked with a # symbol
    - time is in filename, UTC
    
    all other lines are a CSV file
    
    """
    filename = Path(filename)
    assert filename.exists()

    # get the timestamps from the filename
    # filename is like 20240214_000000_wavebuoy_xxx_spec2D

    utc = get_timestamp(filename.stem)


    info = []

    with open(filename, 'r') as f:
        lines = f.readlines()

        metadata = dict()

        for line in lines:
            line = line.strip()
            if line.startswith('#'):
                if '=' in line:
                    key, value = line.split('=')
                    metadata[key[1:].strip()] = value.strip()
                else:
                    info.append(line)


    # read the csv data using numpy, skipping lines that start with #
    data = np.genfromtxt(filename, delimiter=',', comments='#')

    # get frequencies
    freq = metadata['Rows [Hz]']
    freq = freq.split(',')
    freq = [float(f) for f in freq]

    dir = metadata['Columns [deg]']

    # split on , or space
    dir = dir.replace(' ',',').split(',')

    # check if it makes sense
    dd = float(dir[1]) - float(dir[0])

    dirs = np.arange(0, 360, dd)

    # check dims

    if data.shape[0] != len(freq):
        warn(f'Frequency dimension mismatch: {data.shape[0]} != {len(freq)}')

    if data.shape[1] != len(dirs):
        warn(f'Direction dimension mismatch: {data.shape[1]} != {len(dirs)}')

    metadata['info'] = info

    # return as a dict
    R = dict()
    R['freq'] = freq
    R['dir'] = dirs
    R['data'] = data
    R['utc'] = utc
    R['metadata'] = metadata

    return R

def get_obs_files(directory, start_date = None, end_date = None):
    """Return all the .csv files in the directory that have a timestamp
    greater than or equal to start_date and less than or equal to end_date.
    Timestamps are extracted from the filename which are expected to be
    in the format yyyymmdd_hhmmss.....csv

    start_date and end_date are datetime objects

    """

    directory = Path(directory)

    assert directory.exists()

    files = directory.glob('*.csv')
    R = []
    for file in files:

        if start_date is not None:
            timestamp = get_timestamp(file.stem)
            if start_date is not None:
                if timestamp < start_date:
                    continue
            if end_date is not None:
                if timestamp > end_date:
                    continue

            R.append(file)

    return R

def read_obscape(directory, start_date, end_date):
    """Read all the files in the directory that have a timestamp
    greater than or equal to start_date and less than or equal to end_date.
    Timestamps are extracted from the filename which are expected to be
    in the format yyyymmdd_hhmmss.....csv

    start_date and end_date are datetime objects
    use None for start_date or end_date to not filter on that date

    """

    # step 1: get the files

    files = get_obs_files(directory, start_date, end_date)

    # step 2: get the data

    R = []
    for file in files:
        R.append(read_obscape_file(file))

    # step 3: construct the data
    #
    #

    metadata = R[0]['metadata']  # use the first read spectrum
    dirs = R[0]['dir']
    freqs = R[0]['freq']

    efth = [d['data'] for d in R]
    times = [d['utc'] for d in R]

    ds = xr.DataArray(
        data=efth,
        coords={"time": times, "freq": freqs, "dir": dirs},
        dims=("time", "freq", "dir"),
        name="efth",
    ).to_dataset()

    ds = ds.sortby('time', ascending=True)

    for key in ['Station name', 'Device type', 'Device serial', 'Latitude [deg]', 'Longitude [deg]', 'Magnetic declination (corrected) [deg]', 'Directions', 'info']:
        try:
            ds.attrs[key] = metadata[key]
        except KeyError:
            pass

    # Set attributes
    set_spec_attributes(ds)


    # add site dimension
    ds['site'] = [0]

    return ds
