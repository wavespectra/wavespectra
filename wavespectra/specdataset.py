"""Wrapper around the xarray dataset."""
import types
import os
import re
import sys
import xarray as xr

from wavespectra.core.attributes import attrs
from wavespectra.specarray import SpecArray

here = os.path.dirname(os.path.abspath(__file__))

class Plugin(type):
    """Add all the export functions at class creation time."""

    def __new__(cls, name, bases, dct):
        modules = [__import__('wavespectra.output.{}'.format(os.path.splitext(fname)[0]), fromlist=['*'])
                    for fname in os.listdir(os.path.join(here, 'output')) if fname.endswith('.py')]
        for module in modules:
            for module_attr in dir(module):
                function = getattr(module, module_attr)
                if isinstance(function, types.FunctionType) and module_attr.startswith('to_'):
                    dct[function.__name__] = function
        return type.__new__(cls, name, bases, dct)

@xr.register_dataset_accessor('spec')
class SpecDataset(object):
    """Wrapper around the xarray dataset.
    
    Plugin functions defined in wavespectra/output/<module>
    are attached as methods in this accessor class.

    """
    __metaclass__ = Plugin

    def __init__(self, xarray_dset):
        self.dset = xarray_dset
        self._wrapper()
        self.supported_dims = [attrs.TIMENAME, attrs.SITENAME, attrs.LATNAME,
                               attrs.LONNAME, attrs.FREQNAME, attrs.DIRNAME]

    def __getattr__(self, attr):
        return getattr(self.dset, attr)

    def __repr__(self):
        return re.sub(r'<.+>', '<{}>'.format(self.__class__.__name__),
                      str(self.dset))

    def _wrapper(self):
        """Wraper around SpecArray methods.

        Allows calling public SpecArray methods from SpecDataset.
        For example:
            self.spec.hs() becomes equivalent to self.efth.spec.hs()

        """
        for method_name in dir(self.dset[attrs.SPECNAME].spec):
            if not method_name.startswith('_'):
                method = getattr(self.dset[attrs.SPECNAME].spec, method_name)
                setattr(self, method_name, method)

    def _check_and_stack_dims(self):
        """Ensure dimensions are suitable for dumping in some ascii formats.

        Returns:
            Dataset object with site dimension and with no grid dimensions

        Note:
            Grid is converted to site dimension which can be iterated over
            Site is defined if not in dataset and not a grid
            Dimensions are checked to ensure they are supported for dumping
        """
        dset = self.dset.load().copy(deep=True)

        unsupported_dims = set(dset[attrs.SPECNAME].dims) - set(self.supported_dims)
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
    from wavespectra.input.swan import read_swan
    here = os.path.dirname(os.path.abspath(__file__))
    ds = read_swan(os.path.join(here, '../tests/swanfile.spec'))
    # ds.spec.to_octopus('/tmp/test.oct')
    # ds.spec.to_swan('/tmp/test.swn')
    # ds.spec.to_netcdf('/tmp/test.nc')