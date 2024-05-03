import logging
import numpy as np

from wavespectra.core import npstats
from wavespectra.core.utils import D2R, angle


logger = logging.getLogger(__name__)


def _partition_stats(spectrum, freq, dir):
    """Wave stats from partition."""
    spec1d = spectrum.sum(axis=1).astype("float64")
    ifpeak = np.argmax(spec1d).astype("int64")
    hs = npstats.hs(spectrum, freq, dir)
    if (ifpeak == 0) or (ifpeak == freq.size - 1):
        fp = dpm = dm = dp = np.nan
    else:
        fp = 1 / npstats.tps(ifpeak, spec1d, freq.astype("float32"))
        dpm = npstats.dpm(ifpeak, *npstats.mom1(spectrum, dir))
        dm = npstats.dm(spectrum, dir)
        idpeak = np.argmax(
            (spectrum * _frequency_resolution(freq, dir.size)).sum(axis=0)
        )
        dp = npstats.dp(idpeak.astype("int64"), dir.astype("float32"))
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
    df = np.gradient(freq)
    if ndir is not None:
        df = np.tile(freq, (ndir, 1)).T
    return df


def _plot_partitions(partitions, freq, hs, fp, dp, hs_threshold, show=False):
    import matplotlib.pyplot as plt

    # vmin = np.log10(min([spectrum.min() for spectrum in partitions]))
    # vmax = np.log10(max([spectrum.max() for spectrum in partitions]))
    # nplots = len(partitions)
    ncol = 6  # min(5, nplots)
    nrow = 10  # int(np.ceil(nplots / ncol))
    fig = plt.figure(figsize=(12, 15))
    iplot = 1
    for spectrum, h, f, d in zip(partitions, hs, fp, dp):
        if h > hs_threshold:
            alpha = 1
        else:
            alpha = 0.5
        # if imerge == iplot - 1
        ax = fig.add_subplot(nrow, ncol, iplot)
        ax.pcolormesh(
            dir, freq, np.log10(spectrum), cmap="inferno", vmin=-5, vmax=-2, alpha=alpha
        )
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
    F2 = F**2
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
        sf2[ipart] = f2x - fx**2 + f2y - fy**2
    return sf2


def _combine_last(parts, index, freq, dir, hs, fp, dpm, dm, dp, sf2, fpx, fpy):
    """Combine, update and reorder parts and stats.

    Args:
        - parts (3darray):"""
    # Combine parts
    parts[index] += parts[-1]
    # Update stats
    hs[index], fp[index], dpm[index], dm[index], dp[index] = _partition_stats(
        parts[index], freq, dir
    )
    sf2[index] = spread_hp01([parts[index]], freq, dir)[0]
    fpx[index] = fp[index] * np.cos(D2R * dp[index])
    fpy[index] = fp[index] * np.sin(D2R * dp[index])
    # Remove merged last index
    parts = parts[:-1]
    hs = hs[:-1]
    fp = fp[:-1]
    dpm = dpm[:-1]
    dm = dm[:-1]
    dp = dp[:-1]
    sf2 = sf2[:-1]
    fpx = fpx[:-1]
    fpy = fpy[:-1]
    # Reorder if Hs order changes
    isort = np.argsort(-hs)
    if len(set(np.diff(isort))) != 1:
        parts = parts[isort]
        hs = hs[isort]
        fp = fp[isort]
        dpm = dpm[isort]
        dm = dm[isort]
        dp = dp[isort]
        sf2 = sf2[isort]
        fpx = fpx[isort]
        fpy = fpy[isort]
    return parts, hs, fp, dpm, dm, dp, sf2, fpx, fpy


def combine_partitions_hp01(
    partitions,
    freq,
    dir,
    swells=None,
    k=0.5,
    angle_max=30,
    hs_min=0.2,
    combine_extra_swells=True,
):
    """Combine swell partitions according Hanson and Phillips (2001).

    Args:
        - partitions (list): Partitions sorted in descending order by Hs.
        - freq (1darray): Frequency array.
        - dir (1darray): Direction array.
        - swells (int): Number of swells to keep after auto-merging is performed.
        - k (float): Spread factor in Hanson and Phillips (2001)'s eq 9.
        - angle_max (float): Maximum relative angle for combining partitions.
        - hs_min (float): Minimum Hs of individual partitions, any components
          that fall below this value is merged onto closest partition.
        - combine_extra_swells (bool): Combine extra swells with nearest neighbours if
          more than the number defined by `swells` remain after auto-merging.

    Returns:
        - combined_partitions (list): List of combined partitions.

    Criteria for merging any 2 partitions:
        - Integrated partitions E(f) must be contiguous in frequency.
        - Mean directions are separated by less than `angle_max`.
        - Polar distance between partitions is small compared to their spread.
        - Partitions < `hs_min` are always combined with closest neighbours.

    """
    # Partition stats
    hs = []
    fp = []
    dpm = []
    dm = []
    dp = []
    merged_partitions = []
    for ipart, spectrum in enumerate(partitions):
        hsi, fpi, dpmi, dmi, dpi = _partition_stats(spectrum, freq, dir)
        if not np.isnan(fpi):
            hs.append(hsi)
            fp.append(fpi)
            dpm.append(dpmi)
            dm.append(dmi)
            dp.append(dpi)
            merged_partitions.append(spectrum)
        else:
            logger.debug(f"Ignoring partition {ipart} with hs={hsi}")
    merged_partitions = np.array(merged_partitions)
    hs = np.array(hs)
    fp = np.array(fp)
    dpm = np.array(dpm)
    dm = np.array(dm)
    dp = np.array(dp)

    # Spread parameter
    sf2 = spread_hp01(merged_partitions, freq, dir)

    # Peak distance parameters
    fpx = fp * np.cos(D2R * dp)
    fpy = fp * np.sin(D2R * dp)

    # Recursively merge partitions satisfying HP01 criteria
    logger.debug("Entering while loop to merge partitions")
    merged = True
    while merged:
        merged = False

        # Distances between last and all other peaks
        df2 = (fpx[-1] - fpx[:-1]) ** 2 + (fpy[-1] - fpy[:-1]) ** 2

        # Iterate through nearest neighbours until combining criteria are met
        for inext in df2.argsort():
            # Only proceed if partitions are contiguous in frequency space
            if _is_contiguous(merged_partitions[-1], merged_partitions[inext]):
                # Only proceed if angle between partitions is small enough
                if angle(dm[-1], dm[inext]) <= angle_max:
                    dist = df2[inext]
                    spread = max(sf2[-1], sf2[inext])
                    # Only proceed if distance between peaks is small enough
                    if dist <= (k * spread):
                        logger.debug(
                            f"Partitions {-1} and {inext} fullfill "
                            "combining criteria and will be merged"
                        )
                        merged = True
                        break
        # Combine small partitions regardless of angle and distance criteria
        if not merged and df2.size > 0 and (hs[-1] < hs_min):
            inext = df2.argsort()[0]
            logger.debug(
                f"Partitions {-1} and {inext} do not fullfill all combining criteria "
                "but Hs is smaller than threshold so they will be merged"
            )
            merged = True

        # Merge partitions and update all stats
        if merged:
            merged_partitions, hs, fp, dpm, dm, dp, sf2, fpx, fpy = _combine_last(
                merged_partitions, inext, freq, dir, hs, fp, dpm, dm, dp, sf2, fpx, fpy
            )

    # Merge extra swells If `swell` is specified and `combine_extra_swells` is True
    if swells is not None:
        if combine_extra_swells:
            while merged_partitions.shape[0] > swells:
                df2 = (fpx[-1] - fpx[:-1]) ** 2 + (fpy[-1] - fpy[:-1]) ** 2
                inext = df2.argmin()
                logger.debug(inext)
                merged_partitions[inext] += merged_partitions[-1]
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
