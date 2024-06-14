"""Wave spectra reconstruction.

References:

- Bouws, E., Gunther, H., Rosenthal, W., Vincent, C. L. (1985), Similarity of the
  wind wave spectrum in finite depth water 1. Spectral form. Journal of
  Geophysical Research, 90 (C1): 975-986.

- Bunney, C., Saulter, A., Palmer, T. (2014), Reconstruction of complex 2D
  wave spectra for rapid deployment of nearshore wave models. From Sea to
  Shore—Meeting the Challenges of the Sea (Coasts, Marine Structures and
  Breakwaters 2013), W. Allsop and K. Burgess, Eds., 1050-1059.

- Cartwright, D. E. (1963), The use of directional spectra in studying the output of
  a wave recorder on a moving ship. Ocean Wave Spectra, 203-18, Prentice-Hall, NJ.

- Hasselmann, K. et al. (1973), Measurements of wind-wave growth and swell decay
  during the joint North Sea Wave Project (JONSWAP). Deutsche hydrographische
  Zeitschrift, Erganzungshefte Reihe A 12: 8-95.

- Pierson, W. J., Moskowitz, L. (1964), A proposed spectral form for fully developed
  wind seas based on the similarity theory of Kitaigorodski. Journal of
  geophysical research 69: 5181-5190.

"""
import xarray as xr
import wavespectra
from wavespectra.core.attributes import attrs, set_spec_attributes
from wavespectra.core.utils import load_function
from wavespectra.construct import frequency, direction


# Wave stats to use with reconstruction
STATS = [
    "hs",
    "tp",
    "fp",
    "dm",
    "dpm",
    "dspr",
    "dpspr",
    "tm01",
    "tm02",
    "gamma",
    "alpha",
]


def construct_partition(
    freq_name="jonswap", dir_name="cartwright", freq_kwargs={}, dir_kwargs={}
):
    """Fit frequency-direction E(f, d) parametric spectrum for a partition.

    Args:
        - freq_name (str): Name of a valid spectral fit function, e.g. `jonswap`.
        - dir_name (str): Name of a valid directional spreak function, e.g. `cartwright`.
        - freq_kwargs (dict): Kwargs to run the `freq_name` spectral fit function.
        - dir_kwargs (dict): Kwargs to run the `dir_name` directional spread function.

    Returns:
        - efth (SpecArray): Two-dimensional, frequency-direction spectrum E(f, d) (m2/Hz/deg).

    Note:
        - Function `freq_name` must be available in main wavespectra package.
        - Function `dir_name` must be available in wavespectra.directional subpackage.
        - Missing values in output spectrum are filled with zeros.

    """
    # Import spectral fit and spreading functions
    freq_func = load_function("wavespectra.construct.frequency", freq_name)
    dir_func = load_function("wavespectra.construct.direction", dir_name)

    # frequency spectrum
    efth1d = freq_func(**freq_kwargs)

    # Directional spreading
    spread = dir_func(**dir_kwargs)

    # Frequency-direction spectrum
    dset = efth1d * spread
    set_spec_attributes(dset)

    return dset.fillna(0.0)


def partition_and_reconstruct(
    dset,
    parts=4,
    freq_name="jonswap",
    dir_name="cartwright",
    partition_method="ptm3",
    method_combine="max",
    use_defaults=["alpha"],
):
    """Partition and reconstruct existing spectra to evaluate.

    Args:
        - dset (SpecDataset): Spectra object to partition and reconstruct.
        - parts (int): Number of partitions to use in reconstruction.
        - freq_name (str, list): Name of a valid fit function, e.g. `jonswap`, or a
          list of names with len=`parts` to define one fit function for each partition.
        - dir_name (str, list): Name of a valid directional spread function, e.g.
          `cartwright`, or a list of names with len=`parts` to define one directional
          spread function for each partition.
        - partition_method (str): Partitioning method, either `ptm1`, `ptm2` or `ptm3`.
        - method_combine (str): Method to combine partitions when reconstructing.
        - use_defaults (list): List of default parameters to use in the construct
          shape functions, i.e., those that are not calculated from the input spectra.

    Returns:
        - dsout (SpecArray): Reconstructed spectra with same coordinates as dset.

    Note:
        - The `ptm3` partitioning method is used by default in which wind sea and swell
          systems are not distiguished or merged together.
        - If `freq_name` or `dir_name` are str, the functions specified by these
          arguments are applied to all sea and swell partitions.

    """
    # Parameter checking
    if isinstance(freq_name, str):
        freq_name = (parts) * [freq_name]
    if isinstance(dir_name, str):
        dir_name = (parts) * [dir_name]
    for name in [freq_name, dir_name]:
        if len(name) != parts:
            raise ValueError(
                f"Len of '{name}' must correspond to the "
                f"number of wave systems '{parts}'"
            )
    if partition_method not in ["ptm1", "ptm2", "ptm3"]:
        raise ValueError(
            f"Invalid partition method '{partition_method}'. "
            f"Choose between 'ptm1', 'ptm2' and 'ptm3'."
        )

    coords = {attrs.FREQNAME: dset[attrs.FREQNAME], attrs.DIRNAME: dset[attrs.DIRNAME]}

    # Partitioning
    if partition_method == "ptm3":
        kw = {"parts": parts}
    else:
        for v in ["wspd", "wdir", "dpt"]:
            if v not in dset.data_vars:
                raise ValueError(f"Missing variable '{v}' in dset for partitioning.")
        kw = dict(wspd=dset.wspd, wdir=dset.wdir, dpt=dset.dpt, swells=parts - 1)
    dspart = getattr(dset.spec.partition, partition_method)(**kw)

    # Calculating parameters
    stats = list(set(STATS) - set(use_defaults))
    dparam = dspart.spec.stats(stats)
    dparam["fm"] = 1 / dparam.tm01

    # Reconstruct partitions
    reconstructed = []
    for ipart, part in enumerate(dspart.part):
        # Turn partitioned parameters for current partition into functions kwargs
        kw = {**coords, **{k: v for k, v in dparam.sel(part=[part]).data_vars.items()}}

        # Reconstruct current partition
        reconstructed.append(
            construct_partition(
                freq_name=freq_name[ipart],
                dir_name=dir_name[ipart],
                freq_kwargs=kw,
                dir_kwargs=kw,
            )
        )

    # Combine partitions
    reconstructed = getattr(xr.concat(reconstructed, attrs.PARTNAME), method_combine)(
        attrs.PARTNAME
    )
    reconstructed = reconstructed.to_dataset(name=attrs.SPECNAME)
    set_spec_attributes(reconstructed)

    # Add back winds and depth
    reconstructed["wspd"] = dset.wspd
    reconstructed["wdir"] = dset.wdir
    reconstructed["dpt"] = dset.dpt

    # Set some attributes
    reconstructed.attrs = {
        "title": "Spectra Reconstruction",
        "source": "wavespectra <https://github.com/wavespectra/wavespectra>",
        "partitions": parts,
        "spectral_shapes": freq_name,
        "directional_spread": dir_name,
    }
    return reconstructed
