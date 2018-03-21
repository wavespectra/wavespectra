"""Generic netCDF output plugin."""
from wavespectra.core.attributes import attrs

def to_netcdf(self,
              filename,
              specname=attrs.SPECNAME,
              ncformat='NETCDF4_CLASSIC',
              compress=True,
              time_encoding={'units': 'days since 1900-01-01'}):
    """Preset parameters before calling xarray's native to_netcdf method.

    Args:
        - specname (str): name of spectra variable in dataset.
        - ncformat (str): netcdf format for output, see options in native
          to_netcdf method.
        - compress (bool): if True output is compressed, has no effect for
          NETCDF3.
        - time_encoding (dict): force standard time units in output files.

    """
    other = self.copy(deep=True)
    encoding = {}
    if compress:
        for ncvar in other.data_vars:
            encoding.update({ncvar: {'zlib': True}})
    if attrs.TIMENAME in other:
        other.time.encoding.update(time_encoding)
    other.to_netcdf(filename, format=ncformat, encoding=encoding)