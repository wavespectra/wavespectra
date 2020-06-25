"""Construct the SpecArray accessor from existing dataset from known format."""
import logging

from wavespectra.input.ww3 import from_ww3
from wavespectra.input.wwm import from_wwm
from wavespectra.input.ncswan import from_ncswan
from wavespectra.specdataset import SpecDataset

logger = logging.getLogger(__name__)


def read_dataset(dset):
    """Format and attach SpecArray accessor to an existing xarray dataset.

    Convenience function to define the SpecArray accessor for a dataset rather than a
        file. The function guesses the original file format based on variable names.

    Args:
        dset (xr.Dataset): Spectra dataset with dimensions, coordinates and data_vars
            consistent any supported file format (currently WW3, SWAN and WWM).

    """
    vars_wavespectra = {"freq", "dir", "site", "efth", "lon", "lat"}
    vars_ww3 = {"frequency", "direction", "station", "efth", "longitude", "latitude"}
    vars_wwm = {"nfreq", "ndir", "nbstation", "AC", "lon", "lat"}
    vars_ncswan = {
        "frequency",
        "direction",
        "points",
        "density",
        "longitude",
        "latitude",
    }

    vars_dset = set(dset.variables.keys()).union(dset.dims)
    if not vars_wavespectra - vars_dset:
        logger.debug("Dataset already in wavespectra convention")
        return dset
    if not vars_ww3 - vars_dset:
        logger.debug("Dataset identified as ww3")
        func = from_ww3
    elif not vars_ncswan - vars_dset:
        logger.debug("Dataset identified as ncswan")
        func = from_ncswan
    elif not vars_wwm - vars_dset:
        logger.debug("Dataset identified as wwm")
        func = from_wwm
    else:
        raise ValueError(
            f"Cannot identify appropriate reader from dataset variables: {vars_dset}"
        )
    return func(dset)
