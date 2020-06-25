"""Access functions to read spectra from data files.

The following structure is expected:
    - Reading functions for each data file type defined in specific modules
    - Modules named as {datatype}.py, e.g. swan.py
    - Functions named as read_{dataname}, e.g. read_swan

All functions defined with these conventions will be dynamically
imported at the module level

"""
import xarray as xr
from fsspec import get_mapper

from wavespectra.core.attributes import attrs


to_keep = {
    attrs.SPECNAME,
    attrs.WSPDNAME,
    attrs.WDIRNAME,
    attrs.DEPNAME,
    attrs.LONNAME,
    attrs.LATNAME,
}


def chunks_dict(chunks, mapping):
    """Get the chunks dict either from keys or values in mapping.

    Handy function to allow specifying chunks using dim names from either original
        dataset or from wavespectra.

    """
    if not mapping:
        mapping = {key: key for key in chunks.keys()}
    _chunks = {}
    for key, val in chunks.items():
        if key in mapping.keys():
            dim = key
        elif key in mapping.values():
            dim = list(mapping.keys())[list(mapping.values()).index(key)]
        else:
            raise KeyError(
                f"Dim '{key}' not in chunks, supported dims are: "
                f" {list(mapping.keys()) + list(mapping.values())}"
            )
        _chunks.update({dim: val})
    return _chunks


def open_netcdf_or_zarr(filename_or_fileglob, file_format, mapping={}, chunks={}):
    """Read spectra dataset in either netcdf or zarr format.

    Args:
        - filename_or_fileglob (str): filename or fileglob specifying multiple
          files to read.
        - file_format (str): format of file to open, one of `netcdf` or `zarr`.
        - mapping (dict): coordinates mapping from original dataset to wavespectra.
        - chunks (dict): chunk sizes for dimensions in dataset. By default
          dataset is loaded using single chunk for all dimensions (see
          xr.open_mfdataset documentation).

    Returns:
        - dset (Dataset): spectra dataset object read from ww3 file.

    """
    # Allow chunking using wavespctra names
    _chunks = chunks_dict(chunks, mapping)
    if file_format == "netcdf":
        dset = xr.open_mfdataset(
            filename_or_fileglob, chunks=_chunks, combine="by_coords"
        )
    elif file_format == "zarr":
        fsmap = get_mapper(filename_or_fileglob)
        dset = xr.open_zarr(fsmap, consolidated=True, chunks=_chunks)
    else:
        raise ValueError("file_format must be one of ('netcdf', 'zarr')")
    return dset
