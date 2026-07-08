"""Wrapper around the xarray dataset."""

import functools
import types
import os
import re
import sys
import warnings
import xarray as xr

from wavespectra.core.attributes import attrs
from wavespectra.core.options import OPTIONS, DATASET_TRANSFORMS
from wavespectra.core.select import sel_idw, sel_nearest, sel_bbox
from wavespectra.core.utils import dataset_from_transform
from wavespectra.partition.partition import Partition
from wavespectra.specarray import SpecArray

FUTURE_DATASET_TRANSFORMS = (
    "`{name}` called from the Dataset accessor currently returns a DataArray "
    "but will return a Dataset preserving the non-spectral variables in "
    "wavespectra 5.0. Opt in to the future behaviour with "
    "`wavespectra.set_options(dataset_transforms=True)`, or call it from the "
    "DataArray accessor, e.g. `dset.efth.spec.{name}()`, to retain the "
    "current behaviour and silence this warning."
)

here = os.path.dirname(os.path.abspath(__file__))


class Plugin(type):
    """Add all the export functions at class creation time."""

    def __new__(cls, name, bases, dct):
        modules = [
            __import__(
                f"wavespectra.output.{os.path.splitext(fname)[0]}",
                fromlist=["*"],
            )
            for fname in os.listdir(os.path.join(here, "output"))
            if fname.endswith(".py")
        ]
        for module in modules:
            for module_attr in dir(module):
                function = getattr(module, module_attr)
                if isinstance(function, types.FunctionType) and module_attr.startswith(
                    "to_"
                ):
                    dct[function.__name__] = function
        return type.__new__(cls, name, bases, dct)


@xr.register_dataset_accessor("spec")
class SpecDataset(metaclass=Plugin):
    """Extends xarray's Dataset to deal with wave spectra datasets.

    Plugin functions defined in wavespectra/output/<module>
    are attached as methods in this accessor class.

    Note:
        - When the `dataset_transforms` option is set with
          `wavespectra.set_options(dataset_transforms=True)`, methods that
          transform the spectral variable such as `interp`, `smooth`, `split`
          or `oned` return a Dataset in this accessor with the non-spectral
          variables from the underlying dataset preserved, rather than the
          bare spectral DataArray returned by the SpecArray accessor. This
          will become the default behaviour in wavespectra 5.0.

    """

    _spectral_transforms = (
        "interp",
        "interp_like",
        "oned",
        "rotate",
        "scale_by_hs",
        "smooth",
        "split",
        "to_energy",
    )

    def __init__(self, xarray_dset):
        self.dset = xarray_dset
        self._wrapper()
        self.supported_dims = [
            attrs.TIMENAME,
            attrs.SITENAME,
            attrs.LATNAME,
            attrs.LONNAME,
            attrs.FREQNAME,
            attrs.DIRNAME,
        ]

    def __getattr__(self, attr):
        return getattr(self.dset, attr)

    def __repr__(self):
        return re.sub(r"<.+>", f"<{self.__class__.__name__}>", str(self.dset))

    @property
    def partition(self):
        """Partition interface defined from the full dataset.

        Wind speed, wind direction and depth variables in the dataset are
        used as default arguments by the partitioning methods that require
        them, and partitioned output is returned as a Dataset preserving the
        non-spectral variables.

        """
        return Partition(self.dset)

    def _wrapper(self):
        """Wraper around SpecArray methods.

        Allows calling public SpecArray methods from SpecDataset.
        For example:
            self.spec.hs() becomes equivalent to self.efth.spec.hs()

        Methods that transform the spectral variable are wrapped so they
        return a Dataset that preserves the non-spectral variables.

        """
        for method_name in dir(self.dset[attrs.SPECNAME].spec):
            if method_name.startswith("_") or method_name == "partition":
                continue
            method = getattr(self.dset[attrs.SPECNAME].spec, method_name)
            if method_name in self._spectral_transforms:
                method = self._dataset_output(method)
            setattr(self, method_name, method)

    def _dataset_output(self, method):
        """Wrap spectral transform method to return a Dataset.

        The Dataset output only takes place if the `dataset_transforms`
        option is set, otherwise a FutureWarning is emitted and the bare
        spectral DataArray is returned.

        """

        @functools.wraps(method)
        def wrapped(*args, **kwargs):
            dsout = method(*args, **kwargs)
            if OPTIONS[DATASET_TRANSFORMS]:
                return dataset_from_transform(dsout, self.dset)
            warnings.warn(
                FUTURE_DATASET_TRANSFORMS.format(name=method.__name__),
                FutureWarning,
                stacklevel=2,
            )
            return dsout

        return wrapped

    def _check_and_stack_dims(self):
        """Ensure dimensions are suitable for dumping in some ascii formats.

        Returns:
            Dataset object with site dimension and with no grid dimensions.

        Note:
            Grid is converted to site dimension which can be iterated over.
            Site is defined if not in dataset and not a grid.
            Dimensions are checked to ensure they are supported for dumping.

        """
        dset = self.dset.copy(deep=True)

        unsupported_dims = set(dset[attrs.SPECNAME].dims) - set(self.supported_dims)
        if unsupported_dims:
            raise NotImplementedError(
                f"Dimensions {unsupported_dims} are not supported by "
                f"{sys._getframe().f_back.f_code.co_name} method"
            )

        # If grid reshape into site, if neither define fake site dimension
        if set((attrs.LONNAME, attrs.LATNAME)).issubset(dset.dims):
            dset = dset.stack(site=(attrs.LATNAME, attrs.LONNAME), create_index=False)
        elif attrs.SITENAME not in dset.dims:
            dset = dset.expand_dims(attrs.SITENAME)

        # Ensure lon/lat are not coordinates
        if attrs.LONNAME in dset.coords:
            dset = dset.reset_coords(attrs.LONNAME)
        if attrs.LATNAME in dset.coords:
            dset = dset.reset_coords(attrs.LATNAME)

        # Ensure site dim in lon/lat
        for coord in [attrs.LONNAME, attrs.LATNAME]:
            if coord in dset.data_vars and attrs.SITENAME not in dset[coord].dims:
                dset[coord] = dset[coord].expand_dims(attrs.SITENAME)

        # Ensure times comes first
        if attrs.TIMENAME in dset.dims:
            dset = dset.transpose(attrs.TIMENAME, attrs.SITENAME, ...)

        return dset

    def sel(
        self,
        lons,
        lats,
        method="idw",
        tolerance=2.0,
        **kwargs,
    ):
        """Select stations near or at locations defined by (lons, lats) vector.

        Args:
            - lons (list): Longitude values of locations to select.
            - lats (list): Latitude values of locations to select.
            - method (str): Method to use for inexact matches:
                * idw: Inverse distance weighting selection.
                * nearest: Nearest site selection.
                * bbox: Sites inside bbox [min(lons), min(lats)], [max(lons), max(lats)].
                * None: Only exact matches.
            - tolerance (float): Maximum distance between locations and original stations
              for inexact matches.
            - kwargs: Extra keywargs to pass to the respective sel function
              (i.e., `sel_nearest`, `sel_idw`).

        Return:
            - dset (SpecDataset): Stations Dataset selected at locations defined by zip(lons, lats).

        Note:
            - `tolerance` behaves differently with methods 'idw' and 'nearest'. In 'idw'
              sites with no neighbours within `tolerance` are masked whereas in
              'nearest' an exception is raised.

        """
        funcs = {
            "idw": sel_idw,
            "bbox": sel_bbox,
            "nearest": sel_nearest,
            None: sel_nearest,
        }
        try:
            func = funcs[method]
        except KeyError:
            raise ValueError(
                f"Method '{method}' not supported, valid ones are {list(funcs.keys())}"
            )
        if method is None:
            kwargs.update({"exact": True})
        dsout = func(
            dset=self.dset,
            lons=lons,
            lats=lats,
            tolerance=tolerance,
            **kwargs,
        )
        return dsout
