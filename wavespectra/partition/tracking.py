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
    wspd: xr.DataArray
        Wind speed (m/s)
    t: float
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
    dt: xr.DataArray
        Time difference
    distance: xr.DataArray
        Distance to the swell source (m)

    """
    return dt * g / (4 * pi * distance)


def match_consecutive_partitions(
    fp, dpm, dfp_sea_max, dfp_swell_max, ddpm_sea_max, ddpm_swell_max
):
    """
    Match partitions of consecutive spectra based on evolution of peak frequency
    and peak direction.

    Parameters
    ----------
    fp: np.ndarray
        Array containing the peak wave frequency for all partitions and
        the two consecutive time steps. Shape (npartitions, 2).
    dpm: np.ndarray
        Array containing the mean peak wave direction for all partitions and
        the two consecutive time steps. Shape (npartitions, 2).
    dfp_sea_max: float
        Maximum delta fp for sea partitions.
    dfp_swell_max: float
        Maximum delta fp for swell partitions.
    ddpm_sea_max: float
        Maximum delta dpm for sea partitions.
    ddpm_swell_max: float
        Maximum delta dpm for swell partitions.

    Returns
    -------
    matches: np.ndarray
        Array of matches between partitions of consecutive spectra.
        Array contains value in nth position contains the partition number in
        the previous time step that matches the partition number n in the
        current time step.
        -999 is for nans and -888 is for partitions that have no match.

    """

    # Initialise matches to -999
    matches = np.ones_like(fp[:, 0], dtype="int16") * -999

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

    # Pick angle threshold based on partition type
    ddpm_max = np.array([ddpm_sea_max] + [ddpm_swell_max] * (dpm.shape[0] - 1))
    # If the partition was a sea partition we expect a negative delta
    # hence we use the swell partition maximum delta as a maximum threshold
    dfp_max = np.array([dfp_swell_max] * fp.shape[0])
    # Minimum threshold is based on the sea/swell partition maximum delta
    # depending on the partition type
    dfp_min = np.array([dfp_sea_max] + [-dfp_swell_max] * (fp.shape[0] - 1))

    # For all partition matches which are within the thresholds
    # calculate the distance between the two partitions the sum
    # of normalized dfp and ddpm
    partition_distance = np.where(
        np.logical_and(ddpm < ddpm_max, np.logical_and(dfp < dfp_max, dfp > dfp_min)),
        (np.abs(dfp) / np.maximum(dfp_max, np.abs(dfp_min)) + ddpm / ddpm_max),
        999,
    )

    # Those are all the partitions in the previous time step that have energy
    available = [
        ip for ip, fp_is_not_nan in enumerate(~np.isnan(fp[:, 0])) if fp_is_not_nan
    ]

    # Loop over all partitions in the current time step
    for ip_curr, fp_curr in enumerate(fp[:, 1]):
        if ~np.isnan(fp_curr):
            # Find all possible matches for the current partition sorted by increasing distance
            part_matches = sorted(
                [
                    (ip_prev, d)
                    for ip_prev, d in enumerate(partition_distance[ip_curr, :])
                    if d != 999 and ip_prev in available
                ],
                key=lambda x: x[-1],
            )

            # If no match found, create new partition to track
            if len(part_matches) == 0:
                matches[ip_curr] = -888
            else:  # If match found mark it and remove it from the available list
                matches[ip_curr] = part_matches[0][0]
                available.remove(part_matches[0][0])

    return matches


def np_track_partitions(
    times,
    fp,
    dpm,
    wspd,
    ddpm_sea_max=30,
    ddpm_swell_max=20,
    dfp_sea_scaling=1,
    dfp_swell_source_distance=1e6,
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
        Array containing the peak wave frequency.
    dpm: np.ndarray
        Array containing the mean peak wave direction.
    wspd: np.ndarray
        Wind speed at 10 metres (m/s).
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

    Returns
    -------
    part_ids: np.ndarray
        Array containing the partition ids for each partition and each time step.
        -999 is for nans.
    n_parts: int
        Number of partitions tracked.

    """

    # Assuming that the time step is constant across the dataset
    # calculate dt in seconds
    dt = float(np.diff(times[:2]) / np.timedelta64(1, "s"))

    # Calculate maximum delta fp for sea partitions
    # as it is a function of wind speed this is a data array
    dfp_sea_max = dfp_wsea(wspd=wspd, fp=fp[0, :], dt=dt, scaling=dfp_sea_scaling)

    # Calculate maximum delta fp for swell partitions
    # it is a scalar
    dfp_swell_max = dfp_swell(dt=dt, distance=dfp_swell_source_distance)

    # Calculate local matches (match partitions between consecutive time steps)
    # -999 is for nans and -888 is for partitions that have no match
    # This has more potential to be parallelised but the way the xarray dataset
    # rolling window work makes it non trivial
    part_ids = np.hstack(
        [np.ones((fp.shape[0], 1), dtype="int16") * -999]
        + [
            match_consecutive_partitions(
                fp=fp[:, it - 1 : it + 1],
                dpm=dpm[:, it - 1 : it + 1],
                dfp_sea_max=dfp_sea_max[it - 1],
                dfp_swell_max=dfp_swell_max,
                ddpm_sea_max=ddpm_sea_max,
                ddpm_swell_max=ddpm_swell_max,
            ).reshape((-1, 1))
            for it in range(1, times.shape[0])
        ]
    )

    # Turn the local matches into global matches
    part_id = 0  # Partitions are numbered from 0

    # Number the partitions in the first time step
    for ip, vfp in enumerate(fp[:, 0]):
        if ~np.isnan(vfp):
            part_ids[ip, 0] = part_id
            part_id += 1

    # Propagate the partition ids through time
    for it in range(1, times.size):
        for ip in range(fp.shape[0]):
            if part_ids[ip, it] == -888:
                part_ids[ip, it] = part_id
                part_id += 1
            elif part_ids[ip, it] != -999:
                part_ids[ip, it] = part_ids[part_ids[ip, it], it - 1]

    return part_ids, part_id


def track_partitions(
    stats,
    wspd,
    ddpm_sea_max=30,
    ddpm_swell_max=20,
    dfp_sea_scaling=1,
    dfp_swell_source_distance=1e6,
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
        Wind speed (m/s). Time/site should match that of stats.
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

    Returns
    -------
    part_ids: xr.Dataset
        Dataset containing the partition ids for each partition and each time step.

    """

    # Track partitions
    part_ids, npart_id = xr.apply_ufunc(
        np_track_partitions,
        stats.time,
        stats.fp,
        stats.dpm,
        wspd,
        ddpm_sea_max,
        ddpm_swell_max,
        dfp_sea_scaling,
        dfp_swell_source_distance,
        input_core_dims=[
            ["time"],
            ["part", "time"],
            ["part", "time"],
            ["time"],
            [],
            [],
            [],
            [],
        ],
        output_core_dims=[["part", "time"], []],
        vectorize=True,
        dask="parallelized",
        output_dtypes=[np.int16, "int"],
        dask_gufunc_kwargs={"allow_rechunk": True},
    )

    # Finalise output
    part_ids = part_ids.to_dataset(name="part_id")
    part_ids["npart_id"] = npart_id

    # Add metadata
    part_ids.part_id.attrs = {
        "long_name": "Partition ID",
        "units": "1",
        "description": "ID of tracked partition",
        "_FillValue": -999,
    }
    part_ids.npart_id.attrs = {
        "long_name": "Number of partitions",
        "units": "1",
        "description": "Number of partitions tracked for a given non-spectral dim",
    }

    return part_ids
