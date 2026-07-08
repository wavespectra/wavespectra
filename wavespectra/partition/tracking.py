import numpy as np
from scipy.constants import pi, g
import xarray as xr


def dfp_wsea(wspd: float, fp: float, dt: float, scaling: float = 1.0) -> float:
    """Rate of change of the wind-sea peak wave frequency.
    Based on fetch-limited relationships, (Ewans & Kibblewhite, 1986).

    Ewans, K.C., and A.C. Kibblewhite (1988).
    Spectral characteristics of ocean waves undergoing generation in New Zealand waters.
    9th Australian Fluid Mechanics Conference, Auckland, 8-12 December, 1986.

    Parameters
    ----------
    wspd: float
        Wind speed (m/s)
    fp: float
        Peak wave frequency (Hz)
    dt: float
        Time duration (s)
    scaling: float
        Scaling parameter

    """

    tmp = 15.8 * (g / wspd) ** 0.57

    t0 = (fp / tmp) ** (-1 / 0.43)
    return scaling * tmp * (t0 + dt) ** (-0.43) - fp


def dfp_swell(dt: float, distance: float = 1e6) -> float:
    """Rate of change of the swell peak wave frequency.
    Based on the swell dispersion relationship derived by Snodgrass et al (1966).

    Snodgrass, F. E., G. W. Groves, K. F. Hasselmann, G. R. Miller, and W. H. Munk (1966).
    Propagation of Ocean Swell across the Pacific.
    Philosophical Transactions of the Royal Society of London.
    Series A, Mathematical and Physical Sciences, Vol. 259, No. 1103 (May 5, 1966), pp. 431-497

    Parameters
    ----------
    dt: float
        Time difference (s)
    distance: float
        Distance to the swell source (m)

    """
    return dt * g / (4 * pi * distance)


def match_consecutive_partitions(fp, dpm, dfp_max, dfp_min, ddpm_max):
    """
    Match partitions of consecutive spectra based on evolution of peak frequency
    and peak direction.

    Candidate pairs within the thresholds are assigned in ascending order of
    their normalised distance in frequency-direction space so that the closest
    pairs are always matched first, independently of partition ordering.

    Parameters
    ----------
    fp: np.ndarray
        Array containing the peak wave frequency for all partitions and
        the two consecutive time steps. Shape (npartitions, 2).
    dpm: np.ndarray
        Array containing the mean peak wave direction for all partitions and
        the two consecutive time steps. Shape (npartitions, 2).
    dfp_max: np.ndarray
        Maximum (most positive) delta fp for a match onto each partition of the
        previous time step, shape (npartitions,).
    dfp_min: np.ndarray
        Minimum (most negative) delta fp for a match onto each partition of the
        previous time step, shape (npartitions,).
    ddpm_max: np.ndarray
        Maximum absolute delta dpm for a match onto each partition of the
        previous time step, shape (npartitions,).

    Returns
    -------
    matches: np.ndarray
        Array of matches between partitions of consecutive spectra.
        The value in the nth position contains the partition number in
        the previous time step that matches the partition number n in the
        current time step.
        -999 is for nans and -888 is for partitions that have no match.

    """

    # Initialise matches to -999
    matches = np.ones_like(fp[:, 0], dtype="int32") * -999

    # Calculation distance between partitions of the two consecutive time steps
    # Calculate angle difference between current and previous partition
    ddpm = np.abs(
        (
            (
                np.repeat(dpm[:, 1].reshape((-1, 1)), dpm.shape[0], axis=1)
                - np.repeat(dpm[:, 0].reshape((1, -1)), dpm.shape[0], axis=0)
            )
            + 180
        )
        % 360
        - 180
    )

    # Calculate peak frequency difference between current and previous partition
    dfp = np.repeat(fp[:, 1].reshape((-1, 1)), fp.shape[0], axis=1) - np.repeat(
        fp[:, 0].reshape((1, -1)), fp.shape[0], axis=0
    )

    # For all partition matches which are within the thresholds calculate the
    # distance between the two partitions as the sum of normalised dfp and ddpm,
    # the thresholds are aligned with the previous partitions (columns)
    with np.errstate(invalid="ignore"):
        within = (ddpm < ddpm_max) & (dfp < dfp_max) & (dfp > dfp_min)
        partition_distance = np.where(
            within,
            np.abs(dfp) / np.maximum(dfp_max, np.abs(dfp_min)) + ddpm / ddpm_max,
            np.inf,
        )

    # Partitions with no energy at each time step cannot be matched
    partition_distance[np.isnan(fp[:, 1]), :] = np.inf
    partition_distance[:, np.isnan(fp[:, 0])] = np.inf

    # Mark all energetic partitions in the current time step as unmatched
    matches[~np.isnan(fp[:, 1])] = -888

    # Assign candidate pairs globally in ascending order of distance so that
    # the closest pairs are matched first
    icurr, iprev = np.nonzero(partition_distance < np.inf)
    order = np.argsort(partition_distance[icurr, iprev], kind="stable")
    for ic, ip in zip(icurr[order], iprev[order]):
        if matches[ic] == -888 and ip not in matches:
            matches[ic] = ip

    return matches


def np_track_partitions(
    times,
    fp,
    dpm,
    wspd=None,
    ddpm_sea_max=30,
    ddpm_swell_max=20,
    dfp_sea_scaling=1,
    dfp_swell_source_distance=1e6,
    nsea=1,
):
    """
    Track partitions at a site in a series of consecutive spectra based on
    the evolution of peak frequency and peak direction.
    Partitions are matched with the closest partition in the frequency-direction
    space of the previous time step for which the difference in direction with less
    than ddpm_max and the difference in peak frequency is less than dfp_max.
    ddpm_max differs for sea and swell partitions and is set manually.
    dfp_max also differs for sea and swell partitions. In the case of sea partitions
    it is a function of wind speed and is set to the rate of change of the wind-sea peak
    wave frequency estimated from fetch-limited relationships, (Ewans & Kibblewhite, 1986).
    In the case of swell partitions it is set to the rate of change of the swell peak wave
    frequency based on the swell dispersion relationship derived by Snodgrass et al (1966)
    assuming the distance to the source is 1e6 m.

    Parameters
    ----------
    times: np.ndarray
        Array containing the time stamps of the spectra statistics.
    fp: np.ndarray
        Array containing the peak wave frequency with shape (npart, ntime).
    dpm: np.ndarray
        Array containing the peak wave direction with shape (npart, ntime).
    wspd: np.ndarray
        Wind speed at 10 metres (m/s), required if nsea > 0.
    ddpm_sea_max: float
        Maximum delta dpm for sea partitions.
        Default is 30 degrees.
    ddpm_swell_max: float
        Maximum delta dpm for swell partitions.
        Default is 20 degrees.
    dfp_sea_scaling: float
        Scaling parameter for the rate of change of the wind-sea peak wave frequency.
        Default is 1.
    dfp_swell_source_distance: float
        Distance to the swell source (m) for the rate of change of the swell peak wave
        frequency. Default is 1e6 m.
    nsea: int
        Number of leading partitions that represent wind seas, e.g., 1 for
        partitions from the ptm1 and hp01 methods, 2 for the ptm2 method and 0
        for the ptm3 method whose partitions are not classified. Wind-sea
        matching thresholds are applied to the first nsea partitions and swell
        thresholds to the remaining ones.

    Returns
    -------
    track_ids: np.ndarray
        Array containing the track ids for each partition and each time step.
        -999 is for nans.
    ntracks: int
        Number of wave systems tracked.

    Notes
    -----
    The time step is evaluated for each pair of consecutive spectra so records
    with gaps or irregular sampling use matching thresholds consistent with the
    actual time elapsed.

    """
    if np.any(np.diff(times) <= np.timedelta64(0, "s")):
        raise ValueError("times must be strictly increasing to track partitions")
    nsea = min(nsea, fp.shape[0])
    if nsea > 0 and wspd is None:
        raise ValueError("wspd is required to track wind sea partitions (nsea > 0)")

    npart = fp.shape[0]
    ddpm_max = np.array([ddpm_sea_max] * nsea + [ddpm_swell_max] * (npart - nsea))

    # Calculate local matches (match partitions between consecutive time steps)
    # -999 is for nans and -888 is for partitions that have no match
    matches = [np.ones((npart, 1), dtype="int32") * -999]
    for it in range(1, times.shape[0]):
        dt = float((times[it] - times[it - 1]) / np.timedelta64(1, "s"))
        dfp_swell_max = dfp_swell(dt=dt, distance=dfp_swell_source_distance)
        # The delta fp thresholds are asymmetric for wind seas: fp is expected
        # to decrease under growing seas at the fetch-limited rate, and to only
        # increase slowly under decaying seas at the swell dispersion rate
        dfp_min = np.full(npart, -dfp_swell_max)
        for ip in range(nsea):
            dfp_min[ip] = dfp_wsea(
                wspd=wspd[it - 1], fp=fp[ip, it - 1], dt=dt, scaling=dfp_sea_scaling
            )
        dfp_max = np.full(npart, dfp_swell_max)
        matches.append(
            match_consecutive_partitions(
                fp=fp[:, it - 1 : it + 1],
                dpm=dpm[:, it - 1 : it + 1],
                dfp_max=dfp_max,
                dfp_min=dfp_min,
                ddpm_max=ddpm_max,
            ).reshape((-1, 1))
        )
    track_ids = np.hstack(matches)

    # Turn the local matches into global track ids
    track_id = 0  # Tracks are numbered from 0

    # Number the partitions in the first time step
    for ip, vfp in enumerate(fp[:, 0]):
        if ~np.isnan(vfp):
            track_ids[ip, 0] = track_id
            track_id += 1

    # Propagate the track ids through time
    for it in range(1, times.size):
        for ip in range(fp.shape[0]):
            if track_ids[ip, it] == -888:
                track_ids[ip, it] = track_id
                track_id += 1
            elif track_ids[ip, it] != -999:
                track_ids[ip, it] = track_ids[track_ids[ip, it], it - 1]

    return track_ids, track_id


def track_partitions(
    stats,
    wspd=None,
    ddpm_sea_max=30,
    ddpm_swell_max=20,
    dfp_sea_scaling=1,
    dfp_swell_source_distance=1e6,
    nsea=1,
):
    """
    Track partitions in a series of consecutive spectra based on
    the evolution of peak frequency and peak direction.
    Partitions are matched with the closest partition in the frequency-direction
    space of the previous time step for which the difference in direction with less
    than ddpm_max and the difference in peak frequency is less than dfp_max.
    ddpm_max differs for sea and swell partitions and is set manually.
    dfp_max also differs for sea and swell partitions. In the case of sea partitions
    it is a function of wind speed and is set to the rate of change of the wind-sea peak
    wave frequency estimated from fetch-limited relationships, (Ewans & Kibblewhite, 1986).
    In the case of swell partitions it is set to the rate of change of the swell peak wave
    frequency based on the swell dispersion relationship derived by Snodgrass et al (1966)
    assuming the distance to the source is 1e6 m.

    Parameters
    ----------
    stats: xr.Dataset
        Statistics of the spectral partitions. Requires fp and dpm.
    wspd: xr.DataArray
        Wind speed (m/s), required if nsea > 0. Time/site should match that
        of stats.
    ddpm_sea_max: float
        Maximum delta dpm for sea partitions.
        Default is 30 degrees.
    ddpm_swell_max: float
        Maximum delta dpm for swell partitions.
        Default is 20 degrees.
    dfp_sea_scaling: float
        Scaling parameter for the rate of change of the wind-sea peak wave frequency.
        Default is 1.
    dfp_swell_source_distance: float
        Distance to the swell source (m) for the rate of change of the swell peak wave
        frequency. Default is 1e6 m.
    nsea: int
        Number of leading partitions that represent wind seas, e.g., 1 for
        partitions from the ptm1 and hp01 methods, 2 for the ptm2 method and 0
        for the ptm3 method whose partitions are not classified. Wind-sea
        matching thresholds are applied to the first nsea partitions and swell
        thresholds to the remaining ones.

    Returns
    -------
    tracks: xr.Dataset
        Dataset with the `track_id` of each partition at each time step and the
        number of wave systems tracked `ntracks`.

    """
    if nsea > 0 and wspd is None:
        raise ValueError("wspd is required to track wind sea partitions (nsea > 0)")
    if wspd is None:
        # Placeholder so the wind can be broadcast by apply_ufunc, not used
        wspd = xr.full_like(stats["fp"].isel(part=0, drop=True), np.nan)

    # Track partitions
    track_id, ntracks = xr.apply_ufunc(
        np_track_partitions,
        stats.time,
        stats.fp,
        stats.dpm,
        wspd,
        ddpm_sea_max,
        ddpm_swell_max,
        dfp_sea_scaling,
        dfp_swell_source_distance,
        nsea,
        input_core_dims=[
            ["time"],
            ["part", "time"],
            ["part", "time"],
            ["time"],
            [],
            [],
            [],
            [],
            [],
        ],
        output_core_dims=[["part", "time"], []],
        vectorize=True,
        dask="parallelized",
        output_dtypes=[np.int32, "int"],
        dask_gufunc_kwargs={"allow_rechunk": True},
    )

    # Finalise output
    tracks = track_id.to_dataset(name="track_id")
    tracks["ntracks"] = ntracks

    # Add metadata
    tracks.track_id.attrs = {
        "long_name": "Wave system track ID",
        "units": "1",
        "description": (
            "ID of the tracked wave system each partition belongs to, "
            "-999 for null partitions"
        ),
        "_FillValue": -999,
    }
    tracks.ntracks.attrs = {
        "long_name": "Number of wave systems",
        "units": "1",
        "description": "Number of wave systems tracked over the time series",
    }

    return tracks
