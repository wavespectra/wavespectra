"""Generic netCDF output plugin."""

import dask

from wavespectra.core.attributes import attrs


def to_netcdf(
    self,
    filename,
    specname=attrs.SPECNAME,
    ncformat="NETCDF4",
    compress=True,
    packed=True,
    time_encoding={"units": "seconds since 1970-01-01"},
):
    """Write spectra in netCDF format using wavespectra conventions.

    Args:
        - filename (str): name of output netcdf file.
        - specname (str): name of spectra variable in dataset.
        - ncformat (str): netcdf format for output, see options in native
          to_netcdf method.
        - compress (bool): if True output is compressed, has no effect for
          NETCDF3.
        - packed (bool): Pack spectra as int32 dtype and 1e-5 scale_factor.
        - time_encoding (dict): force standard time units in output files.

    """
    other = self.copy(deep=True)
    encoding = {}
    if compress:
        for ncvar in other.data_vars:
            encoding.update({ncvar: {"zlib": True}})
    if packed:
        encoding[attrs.SPECNAME].update(
            {"scale_factor": 1e-5, "dtype": "int32", "_FillValue": -32768}
        )
    if attrs.TIMENAME in other:
        other.time.encoding.update(time_encoding)
    # Force the single-threaded dask scheduler for the write: computing a lazy
    # (dask-backed) dataset while writing it to netCDF can intermittently
    # deadlock on the netCDF4/HDF5 global lock under the default threaded
    # scheduler. Serialising the write avoids the cross-thread lock contention.
    with dask.config.set(scheduler="single-threaded"):
        other.to_netcdf(filename, format=ncformat, encoding=encoding)
