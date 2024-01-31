"""SWAN ASCII output plugin."""
import numpy as np
from wavespectra.core.attributes import attrs
from wavespectra.core.swan import SwanSpecFile


def to_swan(
    self,
    filename,
    append=False,
    id="Created by wavespectra",
    ntime=None,
    lons=None,
    lats=None,
    compresslevel=6,
):
    """Write spectra in SWAN ASCII format.

    Args:
        - filename (str): str, name for output SWAN ASCII file.
        - append (bool): if True append to existing filename.
        - id (str): used for header in output file.
        - ntime (int, None): number of times to load into memory before dumping output
          file if full dataset does not fit into memory, choose None to load all times.
        - lons: (np.array, None): longitudes to use for each site, if None use.
        - lats: (np.array, None): latitudes to use for each site, if None use.
        - compresslevel (int): compression level for gzip compression (1-9).
    Note:
        - Output files are gzipped if filename ends with `.gz`.
        - lons/lats parameters may be prescribed to set site locations if lon/lat are
          not variables in the dataset (their sizes must match the number of sites).
        - If lons/lats are not specified and the dataset does not have lon/lat coords,
          all coordinates default to zero.
        - Extra dimensions other than time, site, lon, lat, freq, dim not yet
          supported.
        - Only 2D spectra E(f,d) are currently supported.
        - ntime=None optimises speed as the dataset is loaded into memory however the
          dataset may not fit into memory in which case a smaller number of times may
          be prescribed.

    """
    # If grid reshape into site, otherwise ensure there is site dim to iterate over
    dset = self._check_and_stack_dims()
    ntime = min(ntime or dset.time.size, dset.time.size)

    # Handle datasets with missing lon/lat
    for arr, name in zip((lons, lats), (attrs.LONNAME, attrs.LATNAME)):
        if name not in dset.data_vars:
            if arr is None:
                arr = np.zeros(dset.site.size)
            elif len(arr) != dset.site.size:
                raise ValueError(f"{name} must have same size as site dimension")
            dset[name] = (("site",), arr)

    # Ensure time dimension exists
    is_time = attrs.TIMENAME in dset[attrs.SPECNAME].dims
    if not is_time:
        dset = dset.expand_dims({attrs.TIMENAME: [None]})
        times = dset[attrs.TIMENAME].values
    else:
        times = dset[attrs.TIMENAME].to_index().to_pydatetime()
        times = [f"{t:%Y%m%d.%H%M%S}" for t in times]

    # Keeping only supported dimensions
    dims_to_keep = {attrs.TIMENAME, attrs.SITENAME, attrs.FREQNAME, attrs.DIRNAME}
    dset = dset.drop_dims(set(dset.dims) - dims_to_keep)

    # Ensure correct shape
    dset = dset.transpose(attrs.TIMENAME, attrs.SITENAME, attrs.FREQNAME, attrs.DIRNAME)

    # Instantiate swan object
    sfile = SwanSpecFile(
        filename,
        freqs=dset.freq,
        dirs=dset.dir,
        time=is_time,
        x=dset[attrs.LONNAME].values,
        y=dset[attrs.LATNAME].values,
        append=append,
        id=id,
        compresslevel=compresslevel,
    )

    # Dump each timestep
    i0 = 0
    i1 = ntime
    while i1 <= dset.time.size or i0 < dset.time.size:
        ds = dset.isel(time=slice(i0, i1))
        part_times = times[i0:i1]
        i0 = i1
        i1 += ntime
        specarray = ds[attrs.SPECNAME].values
        for itime, time in enumerate(part_times):
            darrout = specarray[itime]
            sfile.write_spectra(darrout, time=time)

    sfile.close()
