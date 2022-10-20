"""Partitioning interface."""
import numpy as np
import xarray as xr

from wavespectra.specpart import specpart
from wavespectra.core.utils import set_spec_attributes, regrid_spec, smooth_spec, check_same_coordinates, D2R, celerity
from wavespectra.core.attributes import attrs
from wavespectra.core.npstats import hs, dpm_gufunc, tps_gufunc
from wavespectra.partition.utils import combine_partitions


class Partition:
    """Spectra partition methods.

    References:
        - Hanson and Phillips (2001), Automated Analysis of Ocean Surface Directional Wave Spectra,
          Journal of Atmospheric and Oceanic Technology, 18, 277-293.
        - Hanson et al. (2009), Pacific hindcast performance of three numerical
          wave models, JTECH 26.8, 1614-1633.
        - Portilla et al. (2009), Spectral Partitioning and Identification of Wind Sea and Swell,
          Journal of Atmospheric and Oceanic Technology, 107-122.
        - Tracy et al. (2007), Wind Sea and Swell Delineation for Numerical Wave Modeling,
          JCOMM Tech. Rep. 41, WMO/TDNo, 1442, Paper P12.
        - Vincent et al. (1991) Watersheds in digital spaces: an efficient algorithm
          based on immersion simulations, IEEE Transactions on Pattern Analysis and
          Machine Intelligence, Vol. 13, No. 6, June 1991, p. 583-598.
        - WW3 wave model documentation, https://github.com/NOAA-EMC/WW3.

    """
    def _set_metadata(self, dsout):
        """Define metadata attributes in output."""
        dsout.name = "efth"
        dsout["part"] = np.arange(dsout.part.size)
        set_spec_attributes(dsout)
        dsout.attrs = attrs.ATTRS[attrs.SPECNAME]
        return dsout

    def ptm1(
        self,
        dset,
        wspd,
        wdir,
        dpt,
        agefac=1.7,
        wscut=0.3333,
        swells=3,
        combine=False,
        smooth=False,
        window=3,
    ):
        """PTM1 spectra partitioning.

        Args:
            - dset (xr.DataArray): Spectra in wavespectra convention.
            - wspd (xr.DataArray): Wind speed DataArray.
            - wdir (xr.DataArray): Wind direction DataArray.
            - dpt (xr.DataArray): Depth DataArray.
            - swells (int): Number of swell partitions to compute.
            - agefac (float): Age factor.
            - wscut (float): Wind sea fraction cutoff.
            - combine (bool): Combine adjacent swell partitions based on the distances
              between spectral peaks (Hanson at al., 2001).
            - smooth (bool): Compute watershed boundaries from smoothed spectra
              as described in Portilla et al., 2009.
            - window (int): Size of running window for smoothing spectra when smooth==True.

        Returns:
            - dspart (xr.Dataset): Partitioned spectra with extra `part` dimension
              where the 0th index are the wind sea and remaining indices are the swells
              sorted by descending order of Hs.

        In PTM1, topographic partitions for which the percentage of wind-sea energy exceeds a 
        defined fraction are aggregated and assigned to the wind-sea component (e.g., the first
        partition). The remaining partitions are assigned as swell components in order of 
        decreasing wave height.

        References:
            - Hanson and Phillips (2001).
            - Hanson et al. (2009).
            - Portilla et al. (2009).
            - Tracy et al. (2007).
            - Vincent et al. (1991).
            - WW3 documentation (https://github.com/NOAA-EMC/WW3).

        TODO: Test if more efficient calculating windmask outside ufunc.

        """
        # Sort out inputs
        check_same_coordinates(wspd, wdir, dpt)
        if isinstance(dset, xr.Dataset):
            dset = dset[attrs.SPECNAME]
        if smooth:
            dset_smooth = smooth_spec(dset, window)
        else:
            dset_smooth = dset

        # Partitioning full spectra
        dsout = xr.apply_ufunc(
            np_ptm1,
            dset,
            dset_smooth,
            dset.freq,
            dset.dir,
            wspd,
            wdir,
            dpt,
            agefac,
            wscut,
            swells,
            combine,
            input_core_dims=[["freq", "dir"], ["freq", "dir"], ["freq"], ["dir"], [], [], [], [], [], [], []],
            output_core_dims=[["part", "freq", "dir"]],
            vectorize=True,
            dask="parallelized",
            output_dtypes=["float32"],
            dask_gufunc_kwargs={"allow_rechunk": True, "output_sizes": {"part": swells + 1}},
        )

        # Finalise output
        dsout = self._set_metadata(dsout)
        parts_description = {
            "part0": "wind sea", "part2-n": "swells in descending order of hs",
        }
        dsout.attrs.update(parts_description)

        return dsout.transpose("part", ...)

    def ptm2(
        self,
        dset,
        wspd,
        wdir,
        dpt,
        agefac=1.7,
        wscut=0.3333,
        swells=3,
        combine=False,
        smooth=False,
        window=3,
    ):
        """Watershed partitioning with secondary wind-sea assigned from individual spectral bins.

        Args:
            - dset (xr.DataArray): Spectra in wavespectra convention.
            - wspd (xr.DataArray): Wind speed DataArray.
            - wdir (xr.DataArray): Wind direction DataArray.
            - dpt (xr.DataArray): Depth DataArray.
            - swells (int): Number of swell partitions to compute.
            - agefac (float): Age factor.
            - wscut (float): Wind sea fraction cutoff.
            - combine (bool): Combine adjacent swell partitions based on the distances
              between spectral peaks (Hanson at al., 2001).
            - smooth (bool): Compute watershed boundaries from smoothed spectra
              as described in Portilla et al., 2009.
            - window (int): Size of running window for smoothing spectra when smooth==True.

        Returns:
            - dspart (xr.Dataset): Partitioned spectra with extra `part` dimension
              where the 0th and 1st indices are the primary and secondary wind seas
              and remaining indices are the swells sorted by descending order of Hs.

        PTM2 works in a very similar way to PTM1, by first identifying a primary wind-sea component,
        which is assigned as the first partition, then a number of swell (or secondary wind-sea) 
        partitions are identified, as follows. A set of secondary spectral partitions is established 
        using the topographic method, each partition is checked in turn, with any of their spectral 
        bins influenced by the wind (based on a wave age criterion) being removed and assigned as 
        separate, secondary wind-sea partitions. The latter are by default combined into a single 
        partition, but may remain separate if the namelist parameter FLC is set to ".False.". Swell 
        and secondary wind-sea partitions are then ordered by decreasing wave height. Operational 
        forecasts made at the Met Office suggests that when PTM2 is run with the default single wind-sea 
        partition, this provides a smoother spatial transition between partitions and a more direct link 
        between the primary wind-sea component and wind speed than PTM1. Using the default method, the 
        fraction of wind-sea for all partitions except the primary wind-sea partition should be close to 0.0.

        References:
            - Hanson and Phillips (2001).
            - Hanson et al. (2009).
            - Portilla et al. (2009).
            - Tracy et al. (2007).
            - Vincent et al. (1991).
            - WW3 documentation (https://github.com/NOAA-EMC/WW3).

        """
        # Sort out inputs
        check_same_coordinates(wspd, wdir, dpt)
        if isinstance(dset, xr.Dataset):
            dset = dset[attrs.SPECNAME]
        if smooth:
            dset_smooth = smooth_spec(dset, window)
        else:
            dset_smooth = dset

        # Partitioning full spectra
        dsout = xr.apply_ufunc(
            np_ptm2,
            dset,
            dset_smooth,
            dset.freq,
            dset.dir,
            wspd,
            wdir,
            dpt,
            agefac,
            wscut,
            swells,
            combine,
            input_core_dims=[["freq", "dir"], ["freq", "dir"], ["freq"], ["dir"], [], [], [], [], [], [], []],
            output_core_dims=[["part", "freq", "dir"]],
            vectorize=True,
            dask="parallelized",
            output_dtypes=["float32"],
            dask_gufunc_kwargs={"allow_rechunk": True, "output_sizes": {"part": swells + 2}},
        )

        # Finalise output
        dsout = self._set_metadata(dsout)
        parts_description = {
            "part0": "primary wind sea",
            "part1": "secondary wind sea",
            "part2-n": "swells in descending order of hs",
        }
        dsout.attrs.update(parts_description)

        return dsout.transpose("part", ...)

    def ptm3(self, dset, parts=3, combine=False, smooth=False, window=3):
        """Watershed partitioning with no wind-sea or swell classification

        Args:
            - dset (xr.DataArray): Spectra in Wavespectra convention.
            - parts (int): Number of partitions to keep.
            - combine (bool): Combine adjacent swell partitions based on the distances
              between spectral peaks (Hanson at al., 2001).
            - smooth (bool): Compute watershed boundaries from smoothed spectra
              as described in Portilla et al., 2009.
            - window (int): Size of running window for smoothing spectra when smooth==True.

        Returns:
            - dspart (xr.Dataset): Partitioned spectra with extra `part` dimension
              defining watershed partitions sorted by descending order of Hs.

        PTM3 does not classify the topographic partitions into wind-sea or swell - it simply orders them
        by wave height. This approach is useful for producing data for spectral reconstruction applications
        using a limited number of partitions, where the  classification of the partition as wind-sea or
        swell is less important than the proportion of overall spectral energy each partition represents.

        References:
            - Hanson and Phillips (2001).
            - Hanson et al. (2009).
            - Portilla et al. (2009).
            - Tracy et al. (2007).
            - Vincent et al. (1991).
            - WW3 documentation (https://github.com/NOAA-EMC/WW3).

        """
        # Sort out inputs
        if isinstance(dset, xr.Dataset):
            dset = dset[attrs.SPECNAME]
        if smooth:
            dset_smooth = smooth_spec(dset, window)
        else:
            dset_smooth = dset

        # Partitioning full spectra
        dsout = xr.apply_ufunc(
            np_ptm3,
            dset,
            dset_smooth,
            dset.freq,
            dset.dir,
            parts,
            combine,
            input_core_dims=[["freq", "dir"], ["freq", "dir"], ["freq"], ["dir"], [], []],
            output_core_dims=[["part", "freq", "dir"]],
            vectorize=True,
            dask="parallelized",
            output_dtypes=["float32"],
            dask_gufunc_kwargs={"allow_rechunk": True, "output_sizes": {"part": parts}},
        )

        # Finalise output
        dsout = self._set_metadata(dsout)
        dsout.attrs.update({"part0-n": "partitions in descending order of hs"})

        return dsout.transpose("part", ...)

    def ptm4(self, dset, wspd, wdir, dpt, agefac=1.7):
        """WAM partitioning of sea and swell based on wave age criterion..

        Args:
            - dset (xr.DataArray): Spectra in wavespectra convention.
            - wspd (xr.DataArray): Wind speed DataArray.
            - wdir (xr.DataArray): Wind direction DataArray.
            - dpt (xr.DataArray): Depth DataArray.
            - agefac (float): Age factor.

        Returns:
            - dspart (xr.Dataset): Partitioned spectra with extra `part` dimension
              where the 0th index is the wind sea and the 1st index is the swell.

        PTM4 uses the wave age criterion derived from the local wind speed to split the spectrum in
        to a wind-sea and single swell partition. In this case  waves with a celerity greater
        than the directional component of the local wind speed are considered to be
        freely propogating swell (i.e. unforced by the wind). This is similar to the
        method commonly used to generate wind-sea and swell from the WAM model.

        References:
            - WW3 documentation (https://github.com/NOAA-EMC/WW3).

        """
        if isinstance(dset, xr.Dataset):
            dset = dset[attrs.SPECNAME]

        dsout = dset.sortby("dir").sortby("freq")

        wind_speed_component = agefac * wspd * np.cos(D2R * (dsout.dir - wdir))
        wave_celerity = celerity(dsout.freq, dpt)
        windseamask = wave_celerity <= wind_speed_component

        # Masking wind sea and swell regions
        sea = dsout.where(windseamask)
        swell = dsout.where(~windseamask)

        # Combining into part index
        dsout = xr.concat([sea, swell], dim="part")

        # Finalise output
        dsout = self._set_metadata(dsout)
        dsout.attrs.update( {"part0": "wind sea", "part1": "swell"})

        return dsout.fillna(0.)

    def ptm5(self, dset, fcut, interpolate=True):
        """SWAN partitioning of sea and swell based on user-defined threshold.

        Args:
            - dset (xr.DataArray): Spectra in wavespectra convention.
            - fcut (float): Frequency cutoff (Hz).
            - interpolate (bool): Interpolate spectra at fcut if it is not an exact
              frequency in the dset.

        Returns:
            - dspart (xr.Dataset): Partitioned spectra with extra `part` dimension
              where the 0th index is the wind sea and the 1st index is the swell.

        PTM5 splits spectra into wind sea and swell based on a user defined static cutoff.

        Note:
            - Spectra are interpolated at `fcut` if this frequency is not in `dset`
              and `interpolate` is set to True which implies the frequency coordinates
              will be different between the input and the output.

        References:
            - WW3 documentation (https://github.com/NOAA-EMC/WW3).

        """
        if isinstance(dset, xr.Dataset):
            dset = dset[attrs.SPECNAME]

        dsout = dset.sortby("dir").sortby("freq")

        # Include cuttof if not in coordinates
        if interpolate:
            freqs = sorted(set(dset.freq.values).union([fcut]))     
            if len(freqs) > dset.freq.size:
                dsout = regrid_spec(dset, freq=freqs)

        # Zero data outside the domain of each partition
        hf = dsout.where((dsout.freq >= fcut))
        lf = dsout.where((dsout.freq <= fcut))

        # Combining into part index
        dsout = xr.concat([hf, lf], dim="part")

         # Finalise output
        dsout = self._set_metadata(dsout)
        dsout.attrs.update( {"part0": "sea", "part1": "swell"})

        return dsout.fillna(0.)


def np_ptm1(
    spectrum,
    spectrum_smooth,
    freq,
    dir,
    wspd,
    wdir,
    dpt,
    agefac=1.7,
    wscut=0.3333,
    swells=None,
    combine=False
):
    """PTM1 spectra partitioning on numpy arrays.

    Args:
        - spectrum (2darray): Wave spectrum array with shape (nf, nd).
        - spectrum_smooth (2darray): Smoothed wave spectrum array with shape (nf, nd).
        - freq (1darray): Wave frequency array with shape (nf).
        - dir (1darray): Wave direction array with shape (nd).
        - wspd (float): Wind speed.
        - wdir (float): Wind direction.
        - dpt (float): Water depth.
        - agefac (float): Age factor.
        - wscut (float): Wind sea fraction cutoff.
        - swells (int): Number of swell partitions to compute, all detected by default.
        - combine (bool): Combine less energitic partitions onto one of the keeping
          ones according to shortest distance between spectral peaks.

    Returns:
        - specpart (3darray): Wave spectrum partitions sorted in decreasing order of Hs
          with shape (np, nf, nd).

    Note:
        - The smooth spectrum `spectrum_smooth` is used to define the watershed
          boundaries which are applied to the original spectrum.
        - The `combine` option ensures spectral variance is conserved but
          could yields multiple peaks into single partitions.

    """
    # Use smooth spectrum to define morphological boundaries
    watershed_map = specpart.partition(spectrum_smooth)
    nparts = watershed_map.max()

    # Wind sea mask
    up = np.tile(agefac * wspd * np.cos(D2R * (dir - wdir)), (freq.size, 1))
    windseamask = up > np.tile(celerity(freq, dpt)[:, np.newaxis], (1, dir.size))

    # Assign partitioned arrays from raw spectrum and morphological boundaries
    wsea_partition = np.zeros_like(spectrum)
    swell_partitions = [np.zeros_like(spectrum) for n in range(nparts)]
    for ipart in range(nparts):
        part = np.where(watershed_map == ipart + 1, spectrum, 0.0) # start at 1
        wsfrac = part[windseamask].sum() / part.sum()
        if wsfrac > wscut:
            wsea_partition += part
        else:
            swell_partitions[ipart] += part

    # Sort swells by Hs
    isort = np.argsort([-hs(swell, freq, dir) for swell in swell_partitions])
    swell_partitions = [swell for _, swell in sorted(zip(isort, swell_partitions))]

    # Dealing with the number of swells
    if swells is None:
        # Exclude null swell partitions if the number of output swells is undefined
        swell_partitions = [swell for swell in swell_partitions if swell.sum() > 0]
    else:
        if nparts > swells and combine:
            # Combine extra swell partitions into main ones
            swell_partitions = combine_partitions(swell_partitions, freq, dir, swells)
        elif nparts > swells and not combine:
            # Discard extra partitions
            swell_partitions = swell_partitions[:swells]
        elif nparts < swells:
            # Extend partitions list with null spectra
            n = swells - len(swell_partitions)
            for i in range(n):
                swell_partitions.append(np.zeros_like(spectrum))

    return np.array([wsea_partition] + swell_partitions)


def np_ptm2(
    spectrum,
    spectrum_smooth,
    freq,
    dir,
    wspd,
    wdir,
    dpt,
    agefac=1.7,
    wscut=0.3333,
    swells=None,
    combine=False,
):
    """PTM2 spectra partitioning on numpy arrays.

    Args:
        - spectrum (2darray): Wave spectrum array with shape (nf, nd).
        - spectrum_smooth (2darray): Smoothed wave spectrum array with shape (nf, nd).
        - freq (1darray): Wave frequency array with shape (nf).
        - dir (1darray): Wave direction array with shape (nd).
        - wspd (float): Wind speed.
        - wdir (float): Wind direction.
        - dpt (float): Water depth.
        - agefac (float): Age factor.
        - wscut (float): Wind sea fraction cutoff.
        - swells (int): Number of swell partitions to compute, all detected by default.
        - combine (bool): Combine less energitic partitions onto one of the keeping
          ones according to shortest distance between spectral peaks.

    Returns:
        - specpart (3darray): Wave spectrum partitions sorted in decreasing order of Hs
          with shape (np, nf, nd) with np indices 0 and 1 reserved for primary and
          secondary wind sea partitions and remaining ones for ordered swells.

    Note:
        - The smooth spectrum `spectrum_smooth` is used to define the watershed
          boundaries which are applied to the original spectrum.
        - The `combine` option ensures spectral variance is conserved but
          could yields multiple peaks into single partitions.
        - The option in WW3 to leave secondary wind seas as separate partitions is not
          available as it makes it harder to distinguish them from swells in the output.

    """
    # Use smooth spectrum to define morphological boundaries
    watershed_map = specpart.partition(spectrum_smooth)
    nparts = watershed_map.max()

    # Wind sea mask
    up = np.tile(agefac * wspd * np.cos(D2R * (dir - wdir)), (freq.size, 1))
    windseamask = up > np.tile(celerity(freq, dpt)[:, np.newaxis], (1, dir.size))

    # Assign partitioned arrays from raw spectrum and morphological boundaries
    wsea_primary_partition = np.zeros_like(spectrum)
    wsea_secondary_partition = np.zeros_like(spectrum)
    swell_partitions = [np.zeros_like(spectrum) for n in range(nparts)]
    for ipart in range(nparts):
        part = np.where(watershed_map == ipart + 1, spectrum, 0.0) # start at 1
        wsfrac = part[windseamask].sum() / part.sum()
        if wsfrac > wscut:
            wsea_primary_partition += part
        else:
            wsea_secondary_partition += np.where(windseamask, part, 0.0)
            swell_partitions[ipart] += np.where(windseamask, 0.0, part)

    # Sort swells by Hs
    isort = np.argsort([-hs(swell, freq, dir) for swell in swell_partitions])
    swell_partitions = [swell for _, swell in sorted(zip(isort, swell_partitions))]

    # Dealing with the number of swells
    if swells is None:
        # Exclude null swell partitions if the number of output swells is undefined
        swell_partitions = [swell for swell in swell_partitions if swell.sum() > 0]
    else:
        if nparts > swells and combine:
            # Combine extra swell partitions into main ones
            swell_partitions = combine_partitions(swell_partitions, freq, dir, swells)
        elif nparts > swells and not combine:
            # Discard extra partitions
            swell_partitions = swell_partitions[:swells]
        elif nparts < swells:
            # Extend partitions list with null spectra
            n = swells - len(swell_partitions)
            for i in range(n):
                swell_partitions.append(np.zeros_like(spectrum))

    wsea_partitions = [wsea_primary_partition, wsea_secondary_partition]
    return np.array(wsea_partitions + swell_partitions)


def np_ptm3(spectrum, spectrum_smooth, freq, dir, parts=None, combine=False):
    """PTM3 spectra partitioning on numpy arrays.

    Args:
        - spectrum (2darray): Wave spectrum array with shape (nf, nd).
        - spectrum_smooth (2darray): Smoothed wave spectrum array with shape (nf, nd).
        - freq (1darray): Wave frequency array with shape (nf).
        - dir (1darray): Wave direction array with shape (nd).
        - parts (int): Number of partitions to compute, all detected by default.
        - combine (bool): Combine less energitic partitions onto one of the keeping
          ones according to shortest distance between spectral peaks.

    Returns:
        - specpart (3darray): Wave spectrum partitions sorted in decreasing order of Hs
          with shape (np, nf, nd).

    Note:
        - The smooth spectrum `spectrum_smooth` is used to define the watershed
          boundaries which are applied to the original spectrum.
        - The `combine` option ensures spectral variance is conserved but
          could yields multiple peaks into single partitions.

    """
    # Use smooth spectrum to define morphological boundaries
    watershed_map = specpart.partition(spectrum_smooth)
    nparts = watershed_map.max()

    # Assign partitioned arrays from raw spectrum and morphological boundaries
    partitions = []
    for npart in range(1, nparts + 1):
        partitions.append(np.where(watershed_map == npart, spectrum, 0.0))

    # Sort partitions by Hs
    hs_partitions = [hs(partition, freq, dir) for partition in partitions]
    partitions = [p for _, p in sorted(zip(hs_partitions, partitions), reverse=True)]

    if parts is not None:
        if nparts > parts and combine:
            # Combine extra partitions into main ones
            partitions = combine_partitions(partitions, freq, dir, parts)
        elif nparts > parts and not combine:
            # Discard extra partitions
            partitions = partitions[:parts]
        elif nparts < parts:
            # Extend partitions list with zero arrays
            template = np.zeros_like(spectrum)
            n = parts - len(partitions)
            for i in range(n):
                partitions.append(template)

    return np.array(partitions)
