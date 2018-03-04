"""Wrapper around the xarray dataset."""
import types
import os
import re
import sys
import xarray as xr

from spectra.attributes import attrs
from spectra.specarray import SpecArray

here = os.path.dirname(os.path.abspath(__file__))

class Plugin(type):
    """Add all the export functions at class creation time."""
    def __new__(cls, name, bases, dct):
        modules = [__import__('spectra.output.{}'.format(os.path.splitext(fname)[0]), fromlist=['*'])
                    for fname in os.listdir(os.path.join(here, 'output')) if fname.endswith('.py')]
        for module in modules:
            for name in dir(module):
                function = getattr(module, name)
                if isinstance(function, types.FunctionType):
                    dct[function.__name__] = function
        return type.__new__(cls, name, bases, dct)

@xr.register_dataset_accessor('spec')
class SpecDataset(object):
    """Wrapper around the xarray dataset.
    
    Plugin functions defined in spectra/output/<module>
    are attached as methods in this accessor class.

    """
    __metaclass__ = Plugin

    def __init__(self, xarray_dset):
        self.dset = xarray_dset
        
    def __repr__(self):
        return re.sub(r'<.+>', '<{}>'.format(self.__class__.__name__),
                      str(self.dset))

    def __getattr__(self, fn):
        if fn in dir(SpecArray) and (fn[0] != '_'):
            return getattr(self.dset['efth'].spec, fn)
        else:
            return getattr(self.dset, fn)

    def _check_and_stack_dims(self,
                              supported_dims=[attrs.TIMENAME, attrs.SITENAME, attrs.LATNAME,
                                              attrs.LONNAME, attrs.FREQNAME, attrs.DIRNAME]):
        """Ensure dimensions are suitable for dumping in some ascii formats.

        Args:
            supported_dims (list): dimensions that are supported by the dumping method

        Returns:
            Dataset object with site dimension and with no grid dimensions

        Note:
            grid is converted to site dimension which can be iterated over
            site is defined if not in dataset and not a grid
            spectral coordinates are checked to ensure they are supported for dumping
        """
        dset = self.dset.load().copy(deep=True)

        unsupported_dims = set(dset[attrs.SPECNAME].dims) - set(supported_dims)
        if unsupported_dims:
            raise NotImplementedError('Dimensions {} are not supported by {} method'.format(
                unsupported_dims, sys._getframe().f_back.f_code.co_name))

        # If grid reshape into site, if neither define fake site dimension
        if set(('lon','lat')).issubset(dset.dims):
            dset = dset.stack(site=('lat','lon'))
        elif 'site' not in dset.dims:
            dset = dset.expand_dims('site')

        return dset

if __name__ == '__main__':
    from readspec import read_swan
    # ds = read_swan('/source/pyspectra/tests/manus.spec')
    # ds.spec.to_octopus('/tmp/test.oct')
    # ds.spec.to_swan('/tmp/test.swn')
    # ds.spec.to_netcdf('/tmp/test.nc')