import logging
import numpy as np

from wavespectra.core.npstats import tps_gufunc, dpm_gufunc, hs
from wavespectra.core.utils import D2R


logger = logging.getLogger(__name__)


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


def frequency_resolution(freq):
    """Frequency resolution.

    Args:
        - freq (1darray): Frequency array (Hz).

    Returns:
        - df (1darray): Frequency resolution with same size as freq.

    """
    if freq.size > 1:
        fact = np.hstack((1.0, np.full(freq.size - 2, 0.5), 1.0))
        ldif = np.hstack((0.0, np.diff(freq)))
        rdif = np.hstack((np.diff(freq), 0.0))
        return fact * (ldif + rdif)
    else:
        return np.array(1.0,)


def spread_hp01(partitions, freq, dir):
    """Spread parameter of Hanson and Phillips (2001).

    Args:
        - partitions (list): List of spectra partitions (m2/Hz/deg).
        - freq (1darray): Frequency array (Hz).
        - dir (1darray): Direction array (deg).

    Returns:
        - spread (list): Spread parameter for each partition.

    """
    npart, nfreq, ndir = np.array(partitions).shape
    dd = abs(float(dir[1] - dir[0]))

    # Frequency resolution broadcast into spectrum shape
    DF = np.tile(frequency_resolution(freq), (ndir, 1)).T

    # Frequency and direction parameters broadcast into spectrum shape
    F = np.tile(freq, (ndir, 1)).T
    F2 = F ** 2
    D = D2R * np.tile(dir, (nfreq, 1))
    COSD = np.cos(D)
    SIND = np.sin(D)
    COS2D = np.cos(D) ** 2
    SIN2D = np.sin(D) ** 2

    # Calculate spread for each partition
    sf2 = np.zeros(npart)
    for ipart, spectrum in enumerate(partitions):
        e = (spectrum * DF * dd).sum()
        fx = (1 / e) * (spectrum * F * COSD * DF * dd).sum()
        fy = (1 / e) * (spectrum * F * SIND * DF * dd).sum()
        f2x = (1 / e) * (spectrum * F2 * COS2D * DF * dd).sum()
        f2y = (1 / e) * (spectrum * F2 * SIN2D * DF * dd).sum()
        sf2[ipart] = f2x - fx ** 2 + f2y - fy ** 2
    return sf2


def combine_partitions_hp01(partitions, freq, dir, keep, k=0.4, hs_threshold=0.2):
    """Combine swell partitions according Hanson and Phillips (2001).

    Args:
        - partitions (list): Partitions sorted in descending order by Hs.
        - freq (1darray): Frequency array.
        - dir (1darray): Direction array.
        - keep (int): Number of swells to keep.
        - k (float): Spread factor in Hanson and Phillips (2001), eq 9.
        - hs_threshold (float): Maximum Hs of individual partitions, any components
          that fall below this value is merged onto closest partition.

    Returns:
        - combined_partitions (list): List of combined partitions.

    Merging criteria:
        - hs < hs_threshold, merge with nerest neighbour.

    TODO:
        - When merging based on hs_threshold, do we update Hs after each merging?
        - Update spread parameter after each merging?
        - Do we consider frequency touching / direction limit to merge based on Hs threshold?

    """
    def partition_stats(spectrum, freq, dir):
        spec1d = spectrum.sum(axis=1).astype("float64")
        ipeak = np.argmax(spec1d).astype("int64")
        h = hs(spectrum, freq, dir)
        f = 1 / tps_gufunc(ipeak, spec1d, freq.astype("float32"))
        d = dpm_gufunc(ipeak, *mom1(spectrum, dir))
        return h, f, d

    def is_contiguous(part0, part1):
        """Check if 1d partitions overlap in frequency space."""
        spec1d_0 = part0.sum(axis=1)
        spec1d_1 = part1.sum(axis=1)
        left0, right0 = np.nonzero(spec1d_0)[0][[0, -1]]
        left1, right1 = np.nonzero(spec1d_1)[0][[0, -1]]
        return right0 >= left1 and right1 >= left0

    # Partition stats
    npart = len(partitions)
    hmo = np.zeros(npart)
    fp = np.zeros(npart)
    dpm = np.zeros(npart)
    for ipart, spectrum in enumerate(partitions):
        hmo[ipart], fp[ipart], dpm[ipart] = partition_stats(spectrum, freq, dir)

    # Spread parameter
    sf2 = spread_hp01(partitions, freq, dir)

    # Peak distance parameters
    fpx = fp * np.cos(D2R * dpm)
    fpy = fp * np.sin(D2R * dpm)

    # Recursively merge partitions satisfying HP01 criterion
    merged_partitions = partitions.copy()
    merged = True
    while merged:
        # Leave while loop if no merge done after iterating over all left partitions
        merged = 0
        logger.info("Entering loop to merge partitions")
        for ind in reversed(range(1, len(merged_partitions))):
            # Distances between current and all other peaks
            fpx1 = fpx[ind]
            fpy1 = fpy[ind]
            fpx2 = fpx[:ind]
            fpy2 = fpy[:ind]
            df2 = (fpx1 - fpx2) ** 2 + (fpy1 - fpy2) ** 2
            isort = df2.argsort()

            imerge = None
            # Criterion 1: merge with nearest neighbour if Hs is smaller than threshold
            if hmo[ind] < hs_threshold:
                print(f"{ind}: merged partition < hs_threshold onto nearest neighbour")
                imerge = isort[0]
            else:
                # Criterion 2: merge with closest contiguous partition whose dist <= k * spread
                for ipart in isort:
                    dist = df2[ipart]
                    spread = max(sf2[ind], sf2[ipart])
                    touch = is_contiguous(merged_partitions[ind], merged_partitions[ipart])
                    if touch and dist <= (k * spread):
                        print(f"{ind}: merged contiguous partition with distance < spread")
                        imerge = ipart
                        break
            # Combine partitions if any of the above criteria is satisfied
            if imerge is not None:
                merged += 1
                merged_partitions[imerge] += merged_partitions[ind]
                merged_partitions[ind] *= 0
                # Update stats for merged partition
                spectrum = merged_partitions[imerge]
                hmo[imerge], fp[imerge], dpm[imerge] = partition_stats(spectrum, freq, dir)
                sf2[imerge] = spread_hp01([spectrum], freq, dir)[0]
                fpx[imerge] = fp[imerge] * np.cos(D2R * dpm[imerge])
                fpy[imerge] = fp[imerge] * np.sin(D2R * dpm[imerge])
            else:
                print(f"{ind}: not merged")

        # Remove null partitions
        merged_partitions = [spectrum for spectrum in merged_partitions if spectrum.sum() > 0]

    return merged_partitions


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