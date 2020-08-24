"""SWAN ASCII output plugin."""
from wavespectra.core.attributes import attrs
from wavespectra.core.swan import SwanSpecFile
from wavespectra.core.misc import to_datetime


def to_swan(
    self,
    filename,
    append=False,
    id="Created by wavespectra",
):
    """Write spectra in SWAN ASCII format.

    Args:
        - filename (str): str, name for output SWAN ASCII file.
        - append (bool): if True append to existing filename.
        - id (str): used for header in output file.

    Note:
        - Only datasets with lat/lon coordinates are currently supported.
        - Extra dimensions other than time, site, lon, lat, freq, dim not yet
          supported.
        - Only 2D spectra E(f,d) are currently supported.

    """
    # If grid reshape into site, otherwise ensure there is site dim to iterate over
    dset = self._check_and_stack_dims()

    # Ensure time dimension exists
    is_time = attrs.TIMENAME in dset[attrs.SPECNAME].dims
    if not is_time:
        dset = dset.expand_dims({attrs.TIMENAME: [None]})
        times = dset[attrs.TIMENAME].values
    else:
        times = dset[attrs.TIMENAME].to_index().to_pydatetime()

    # Ensure correct shape
    dset = dset.transpose(attrs.TIMENAME, attrs.SITENAME, attrs.FREQNAME, attrs.DIRNAME)

    # Loading for efficiency
    specarray = dset[attrs.SPECNAME].values

    # Instantiate swan object
    try:
        x = dset.lon.values
        y = dset.lat.values
    except AttributeError as err:
        raise NotImplementedError(
            "lon-lat variables are required to write SWAN spectra file"
        ) from err
    sfile = SwanSpecFile(
        filename,
        freqs=dset.freq,
        dirs=dset.dir,
        time=is_time,
        x=x,
        y=y,
        append=append,
        id=id,
    )

    # Dump each timestep
    for itime, time in enumerate(times):
        darrout = specarray[itime]
        sfile.write_spectra(darrout, time=time)

    sfile.close()
