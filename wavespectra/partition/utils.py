import numpy as np

from wavespectra.core.npstats import tps_gufunc, dpm_gufunc


def mom1(spectrum, dir, theta=90.0):
    """First directional moment.

    Args:
        - theta (float): angle offset.

    Returns:
        - msin (DataArray): Sin component of the 1st directional moment.
        - mcos (DataArray): Cosine component of the 1st directional moment.

    """
    dd = dir[1] - dir[0]
    cp = np.cos(np.radians(180 + theta - dir))
    sp = np.sin(np.radians(180 + theta - dir))
    msin = (dd * spectrum * sp).sum(axis=1)
    mcos = (dd * spectrum * cp).sum(axis=1)
    return msin, mcos


def combine_partitions(partitions, freq, dir, keep, combine_excluded=True):
    """Combine least energetic partitions according to distance between peaks.

    Args:
        - partitions (list): Partitions sorted in descending order by Hs.
        - freq (1darray): Frequency array.
        - dir (1darray): Direction array.
        - keep (int): Number of swells to keep.
        - combine_excluded (bool): If True, allow combining two small partitions that
          will both be subsequently combined onto another, if False allow only
          combining onto one of the partitions that are going to be kept in the output.

    Returns:
        - combined_partitions (list): List of combined partitions.

    """
    # Peak coordinates
    tp = []
    dpm = []
    for spectrum in partitions:
        spec1d = spectrum.sum(axis=1).astype("float64")
        ipeak = np.argmax(spec1d).astype("int64")
        if (ipeak == 0) or (ipeak == spec1d.size):
            tp.append(np.nan)
            dpm.append(np.nan)
        else:
            tp.append(tps_gufunc(ipeak, spec1d, freq.astype("float32")))
            dpm.append(dpm_gufunc(ipeak, *mom1(spectrum, dir)))

    # Indices of non-null partitions
    iswell = np.where(~np.isnan(tp))[0]

    # Avoid error if all partitions are null
    if iswell.size == 0:
        return partitions[:keep + 1]

    # Drop null partitions
    i0 = iswell[0]
    i1 = iswell[-1] + 1
    partitions = partitions[i0:i1]
    tp = tp[i0:i1]
    dpm = dpm[i0:i1]

    # Recursively merge small partitions until we only have those defined by `swells`
    while i1 > keep:
        # Distances between current swell peak and all other peaks
        dx = tp[-1] - tp[:-1]
        dy = (np.array(dpm[-1]) % 360) - (np.array(dpm[:-1]) % 360)
        dy = np.minimum(dy, 360 - dy)
        dist = np.sqrt(dx ** 2 + dy ** 2)

        # Define if merging onto closest partition from all or from the kept ones
        if not combine_excluded:
            dist = dist[:keep]

        # merge onto selected partition
        ipart = np.argmin(dist)
        # print(f"Merge partition {i1} onto partition {ipart}")
        partitions[ipart] += partitions[-1]

        # Drop last index
        for l in [partitions, tp, dpm]:
            l.pop(-1)

        # Update counter
        i1 -= 1

    return partitions