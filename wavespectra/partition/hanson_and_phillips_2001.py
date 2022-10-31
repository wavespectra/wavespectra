import logging
import numpy as np

from wavespectra.core.npstats import tps_gufunc, dpm_gufunc, dp_gufunc, dm_numpy, hs_numpy, mom1_numpy
from wavespectra.core.utils import D2R, angle


logger = logging.getLogger(__name__)


def _partition_stats(spectrum, freq, dir):
    """Wave stats from partition."""
    spec1d = spectrum.sum(axis=1).astype("float64")
    ifpeak = np.argmax(spec1d).astype("int64")
    hs = hs_numpy(spectrum, freq, dir)
    if (ifpeak == 0) or (ifpeak == freq.size - 1):
        fp = dpm = dm = dp = np.nan
    else:
        fp = 1 / tps_gufunc(ifpeak, spec1d, freq.astype("float32"))
        dpm = dpm_gufunc(ifpeak, *mom1_numpy(spectrum, dir))
        dm = dm_numpy(spectrum, dir)
        idpeak = np.argmax((spectrum * _frequency_resolution(freq, dir.size)).sum(axis=0))
        dp = dp_gufunc(idpeak.astype("int64"), dir.astype("float32"))
    return hs, fp, dpm, dm, dp


def _is_contiguous(part0, part1):
    """Check if 1d partitions overlap in frequency space."""
    spec1d_0 = part0.sum(axis=1)
    spec1d_1 = part1.sum(axis=1)
    bounds0 = np.nonzero(spec1d_0)[0]
    bounds1 = np.nonzero(spec1d_1)[0]
    if bounds0.size == 0 or bounds1.size == 0:
        # If any partition is null they won't touch
        return False
    left0, right0 = bounds0[[0, -1]]
    left1, right1 = bounds1[[0, -1]]
    left1, right1 = np.nonzero(spec1d_1)[0][[0, -1]]
    return right0 >= left1 and right1 >= left0


def _frequency_resolution(freq, ndir=None):
    """Frequency resolution.

    Args:
        - freq (1darray): Frequency array (Hz).
        - ndir (int): Number of directions if broadcasting output onto 2darray.

    Returns:
        - df (1darray): Frequency resolution with same size as freq.

    """
    fact = np.hstack((1.0, np.full(freq.size - 2, 0.5), 1.0))
    ldif = np.hstack((0.0, np.diff(freq)))
    rdif = np.hstack((np.diff(freq), 0.0))
    df = fact * (ldif + rdif)
    if ndir is not None:
        df = np.tile(freq, (ndir, 1)).T
    return df


def _plot_partitions(partitions, hs, fp, dp, show=False):
    import matplotlib.pyplot as plt
    # vmin = np.log10(min([spectrum.min() for spectrum in partitions]))
    # vmax = np.log10(max([spectrum.max() for spectrum in partitions]))
    # nplots = len(partitions)
    ncol = 6 # min(5, nplots)
    nrow = 10 #int(np.ceil(nplots / ncol))
    fig = plt.figure(figsize=(12, 15))
    iplot = 1
    for spectrum, h, f, d in zip(partitions, hs, fp, dp):
        if h > hs_threshold:
            alpha = 1
        else:
            alpha = 0.5
        # if imerge == iplot - 1
        ax = fig.add_subplot(nrow, ncol, iplot)
        p = ax.pcolormesh(dir, freq, np.log10(spectrum), cmap="inferno", vmin=-5, vmax=-2, alpha=alpha)
        ax.plot(d, f, "o", markerfacecolor="w", markeredgecolor="k")
        # plt.colorbar(p)
        ax.set_xticklabels("")
        ax.set_yticklabels("")
        iplot += 1
        ax.set_title(f"Hs={float(h):0.2f}m, Dp={float(d):0.0f}deg", fontsize=8)
    if show:
        plt.show()


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
    DF = _frequency_resolution(freq, ndir=ndir)

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


def combine_partitions_hp01(partitions, freq, dir, swells=None, k=0.5, angle_max=30, hs_min=0.2, combine_extra_swells=True):
    """Combine swell partitions according Hanson and Phillips (2001).

    Args:
        - partitions (list): Partitions sorted in descending order by Hs.
        - freq (1darray): Frequency array.
        - dir (1darray): Direction array.
        - swells (int): Number of swells to keep after auto-merging is performed.
        - k (float): Spread factor in Hanson and Phillips (2001)'s eq 9.
        - hs_min (float): Minimum Hs of individual partitions, any components
          that fall below this value is merged onto closest partition.
        - angle_max (float): Maximum relative angle for combining partitions.
        - combine_extra_swells (bool): Combine extra swells with nearest neighbours if
          if more than the number defined by `swells` remain after auto-merging.

    Returns:
        - combined_partitions (list): List of combined partitions.

    Merging criteria:
        - hs < hs_min, merge with nerest neighbour.

    TODO:
        - When merging based on hs_min, do we update Hs after each merging?
        - Update spread parameter after each merging?
        - Do we consider frequency touching / direction limit to merge based on Hs threshold?
        - Use Dm instead of Dpm to test for angle distance.

    """
    #TODO: Remove below
    plot = False

    # Partition stats
    npart = len(partitions)
    hs = np.zeros(npart)
    fp = np.zeros(npart)
    dpm = np.zeros(npart)
    dm = np.zeros(npart)
    dp = np.zeros(npart)
    merged_partitions = []
    for ipart, spectrum in enumerate(partitions):
        hsi, fpi, dpmi, dmi, dpi =  _partition_stats(spectrum, freq, dir)
        if not np.isnan(fpi):
            hs[ipart], fp[ipart], dpm[ipart], dm[ipart], dp[ipart] = hsi, fpi, dpmi, dmi, dpi
            merged_partitions.append(spectrum)
        else:
            logger.debug(f"Ignoring partition {ipart} with hs={hsi}")
    merged_partitions = np.array(merged_partitions)

    # Spread parameter
    sf2 = spread_hp01(merged_partitions, freq, dir)

    # Peak distance parameters
    fpx = fp * np.cos(D2R * dp)
    fpy = fp * np.sin(D2R * dp)

    if plot:
        _plot_partitions(merged_partitions, hs, fp, dp)

    # Recursively merge partitions satisfying HP01 criteria
    merged = True
    while merged:
        # Leave while loop if this remains zero
        merged = 0

        logger.debug("Entering loop to merge partitions")
        for ind in reversed(range(1, len(merged_partitions))):
            imerge = None

            # Skip null partitions
            if hs[ind] == 0:
                logger.debug(f"Skipping null partition: {ind}")
                continue

            # Distances between current and all other peaks
            df2 = (fpx[ind] - fpx[:ind]) ** 2 + (fpy[ind] - fpy[:ind]) ** 2
            isort = df2.argsort()

            # Combine with nearest partition that satisfy all of the following:
            #  - Contiguous in frequency space
            #  - Relative angle under threshold
            #  - Small relative distance between peaks (dist <= k * spread)
            #  - For small partitions, the distance and angle tests are ignored
            is_small_hs = hs[ind] < hs_min
            for ipart in isort:
                dist = df2[ipart]
                spread = max(sf2[ind], sf2[ipart])
                is_close = dist <= (k * spread)
                is_small_angle = angle(dm[ind], dm[ipart]) <= angle_max
                is_touch = _is_contiguous(merged_partitions[ind], merged_partitions[ipart])
                is_touch = True
                if (is_touch and is_small_angle and is_close) or (is_touch and is_small_hs):
                    logger.debug(f"{ind}: merged contiguous partition")
                    imerge = ipart
                    break
            if imerge is not None:
                # Combining
                merged += 1
                merged_partitions[imerge] += merged_partitions[ind]
                merged_partitions[ind] *= 0
                # Update stats of combined partition
                hs[imerge], fp[imerge], dpm[imerge], dm[imerge], dp[imerge] = _partition_stats(
                    merged_partitions[imerge], freq, dir
                )
                sf2[imerge] = spread_hp01([merged_partitions[imerge]], freq, dir)[0]
                fpx[imerge] = fp[imerge] * np.cos(D2R * dp[imerge])
                fpy[imerge] = fp[imerge] * np.sin(D2R * dp[imerge])
                # Update stats of removed partition
                hs[ind], fp[ind], dpm[ind], dm[ind], dp[ind] = 0, np.nan, np.nan, np.nan, np.nan
                sf2[ind] = 0 # This ensures zeroed partition won't be merged onto again
            else:
                logger.debug(f"{ind}: not merged")

        if plot:
            _plot_partitions(merged_partitions, hs, fp, dp, show=True)

    # Sort remaining merged partitions by descending order of Hs
    isort = np.argsort(-hs)
    merged_partitions = merged_partitions[isort]
    hs = hs[isort]
    fpx = fpx[isort]
    fpy = fpy[isort]

    # Remove null partitions
    ikeep = hs > 0
    merged_partitions = merged_partitions[ikeep]
    hs = hs[ikeep]
    fpx = fpx[ikeep]
    fpy = fpy[ikeep]

    # If the number of swell is specified, merge with closest partitions
    if swells is not None:
        if combine_extra_swells:
            while merged_partitions.shape[0] > swells:
                df2 = (fpx[-1] - fpx[:-1]) ** 2 + (fpy[-1] - fpy[:-1]) ** 2
                imerge = df2.argmin()
                logger.info(imerge)
                merged_partitions[imerge] += merged_partitions[-1]
                merged_partitions = merged_partitions[:-1]
                hs = hs[:-1]
                fpx = fpx[:-1]
                fpy = fpy[:-1]
        else:
            merged_partitions = merged_partitions[:swells]
            hs = hs[:swells]

    # Sort output one last time
    isort = np.argsort(-hs)
    merged_partitions = merged_partitions[isort]

    return list(merged_partitions)
