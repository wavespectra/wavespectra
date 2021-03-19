import numpy as np
import xarray as xr

from wavespectra.core.attributes import attrs, set_spec_attributes


def spread(dp_matrix, dspr_matrix, dirs):
    """Cosine 2s spreading function.

    Args:
        dp_matrix: wave directions
        dspr_matrix: wave directional spreads
        dirs: direction coordinates
    Returns:
        G1: normalized spreading
    Note:
        Function defined such that \int{G1 d\theta}=1*

    """
    th1 = 0.5 * np.deg2rad(np.array(dirs).reshape((1, -1)))
    th2 = 0.5 * np.deg2rad(dp_matrix)
    a = abs(
        np.cos(th1) * np.cos(th2) + np.sin(th1) * np.sin(th2)
    )  # cos(a-b) = cos(a)cos(b)+sin(a)sin(b)
    G1 = a ** (2.0 * dspr_matrix)  # cos((dirs-dp)/2) ** (2*dspr)
    G1 /= np.expand_dims(G1.sum(axis=-1) * abs(dirs[1] - dirs[0]), axis=-1)
    return G1


def arrange_inputs(*args):
    """Check all inputs are same shape and add frequency and direction dims."""
    argout = []
    shape0 = np.array(args[0]).shape
    for arg in args:
        argm = np.array(arg)
        if argm.shape == () and shape0 != ():  # Broadcast scalar across matrix
            argm = arg * np.ones(shape0)
        elif argm.shape != shape0:
            raise Exception("Input shapes must be the same")
        argout.append(argm[..., np.newaxis, np.newaxis])
    return argout


def make_dataset(spec, freqs, dirs, coordinates=[]):
    """Package spectral matrix to xarray.

    Args:
        spec:
        freqs:
        dirs:
        coordinates:

    Returns:
        dset: SpecDset object

    """
    coords = tuple(coordinates) + ((attrs.FREQNAME, freqs), (attrs.DIRNAME, dirs))
    dimensions = tuple([c[0] for c in coords])
    dset = xr.DataArray(
        data=spec, coords=coords, dims=dimensions, name=attrs.SPECNAME
    ).to_dataset()
    set_spec_attributes(dset)
    return dset


def check_coordinates(param, coordinates):
    """Check coordinates are consistent with parameter.

    Args:
        param:
        coordinates:

    """
    pshape = np.array(param).shape
    if len(pshape) != len(coordinates):
        raise Exception("Incorrect number of coordinates for parameter")
    for idim, dim in enumerate(pshape):
        if dim != len(coordinates[idim][1]):
            raise Exception(
                "Dimension of coordinate %s at position %d does not match parameter"
                % (coordinates[idim][0], dim)
            )
