"""Access functions to read spectra from data files.

The following structure is expected:
    - Reading functions for each data file type defined in specific modules
    - Modules named as {datatype}.py, e.g. swan.py
    - Functions named as read_{dataname}, e.g. read_swan

All functions defined with these conventions will be dynamically
imported at the module level

"""
import xarray as xr

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


def open_netcdf(filename_or_fileglob, chunks):
    """Open netcdf file.

    Netcdf files are attempted to be open with xr.open_mfdataset first to handle either
    single or multiple files. However, this function does not support file-like
    objects, so if an error is raised, we assume a file-like object is passed and open
    it directly with xr.open_dataset.

    Args:
        - filename_or_fileglob (str, list, fileobj): filename, fileglob specifying
          multiple files, or fileobj to read.
        - chunks (dict): chunk sizes for dimensions in dataset.

    Returns:
        - dset (Dataset): spectra dataset object read from netcdf file.

    """
    try:
        dset = xr.open_mfdataset(
            filename_or_fileglob, chunks=chunks, combine="by_coords"
        )
    except (ValueError, IndexError):
        dset = xr.open_dataset(filename_or_fileglob, chunks=chunks)
    return dset


def open_netcdf_or_zarr(filename_or_fileglob, file_format, mapping={}, chunks={}):
    """Read spectra dataset in either netcdf or zarr format.

    Args:
        - filename_or_fileglob (str, list, fileobj): filename, fileglob specifying
          multiple files, or fileobj to read.
        - file_format (str): format of file to open, one of `netcdf` or `zarr`.
        - mapping (dict): coordinates mapping from original dataset to wavespectra.
        - chunks (dict): chunk sizes for dimensions in dataset. By default
          dataset is loaded using single chunk for all dimensions (see
          xr.open_mfdataset documentation).

    Returns:
        - dset (Dataset): spectra dataset object read from netcdf or zarr file.

    """
    # Allow chunking using wavespectra names
    _chunks = chunks_dict(chunks, mapping)
    if file_format == "netcdf":
        dset = open_netcdf(filename_or_fileglob, _chunks)
    elif file_format == "zarr":
        dset = xr.open_zarr(filename_or_fileglob, consolidated=True, chunks=_chunks)
    else:
        raise ValueError("file_format must be one of ('netcdf', 'zarr')")
    return dset


def read_ascii_or_binary(file, mode="r"):
    """Read content from ascii or binary file.

    Args:
        - file (str, fileobj): Name of file or file object to read.
        - mode (str): Mode to open file, one of 'r', 'rb'.

    Returns:
        - data (list): List of lines read from file.

    """
    try:
        data = file.readlines()
    except AttributeError:
        if mode not in ["r", "rb"]:
            raise ValueError("mode must be one of 'r', 'rb'")
        with open(file, mode) as stream:
            data = stream.readlines()
    return data
