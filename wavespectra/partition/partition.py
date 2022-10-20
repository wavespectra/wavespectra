"""Partitioning interface."""
import numpy as np
import xarray as xr

from wavespectra.core.utils import set_spec_attributes, regrid_spec, smooth_spec, check_same_coordinates
from wavespectra.core.attributes import attrs
from wavespectra.partition.watershed import np_ptm1, np_ptm3


class Partition:

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
            - dset (xr.DataArray, xr.Dataset): Spectra DataArray or Dataset in wavespectra convention.
            - wspd (xr.DataArray): Wind speed DataArray.
            - wdir (xr.DataArray): Wind direction DataArray.
            - dpt (xr.DataArray): Depth DataArray.
            - swells (int): Number of swell partitions to compute.
            - agefac (float): Age factor.
            - wscut (float): Wind sea fraction cutoff.
            - combine (bool): Combine less energitic partitions onto one of the keeping
              ones according to shortest distance between spectral peaks.
            - smooth (bool): compute watershed boundaries over smoothed spectra.
            - window (int): Size of running window for smoothing spectra when smooth==True.

        Returns:
            - dspart (xr.Dataset): Partitioned spectra dataset with extra dimension.

        In PTM1, topographic partitions for which the percentage of wind-sea energy exceeds a 
        defined fraction are aggregated and assigned to the wind-sea component (e.g., the first
        partition). The remaining partitions are assigned as swell components in order of 
        decreasing wave height.

        References:
            - Hanson, Jeffrey L., et al. "Pacific hindcast performance of three
              numerical wave models." JTECH 26.8 (2009): 1614-1633.

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
        dsout.name = "efth"
        dsout["part"] = np.arange(swells + 1)
        dsout.part.attrs = {"standard_name": "spectral_partition_number", "units": ""}

        return dsout.transpose("part", ...)

    def ptm2(self):
        """Watershed partitioning with secondary wind-sea assigned from individual spectral bins.

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

        """
        pass

    def ptm3(self, dset, parts=3, combine=False, smooth=False, window=3):
        """Watershed partitioning with no wind-sea or swell classification

        Args:
            - dset (Dataset, DataArray): Spectra in Wavespectra convention.
            - parts (int): Number of partitions to keep.
            - combine (bool): Combine all extra partitions onto one of the keeping
              ones based on shortest distance between spectral peaks.
            - smooth (bool): compute watershed boundaries over smoothed spectra.
            - window (int): Size of running window for smoothing spectra when smooth==True.

        PTM3 does not classify the topographic partitions into wind-sea or swell - it simply orders them
        by wave height. This approach is useful for producing data for spectral reconstruction applications
        using a limited number of partitions, where the  classification of the partition as wind-sea or
        swell is less important than the proportion of overall spectral energy each partition represents.

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
        dsout.name = "efth"
        dsout["part"] = np.arange(parts)
        set_spec_attributes(dsout)

        return dsout.transpose("part", ...)

    def ptm4(self):
        """WAM partitioning of sea and swell based on wave age creterion.

        PTM4 uses the wave age criterion derived from the local wind speed to split the spectrum in
        to a wind-sea and single swell partition. In this case  waves with a celerity greater
        than the directional component of the local wind speed are considered to be
        freely propogating swell (i.e. unforced by the wind). This is similar to the
        method commonly used to generate wind-sea and swell from the WAM model.

        """
        pass

    def ptm5(self, efth, fcut, interpolate=True):
        """SWAN partitioning of sea and swell based on user-defined threshold.

        Args:
            - efth (DataArray): Spectra DataArray in Wavespectra convention.
            - fcut (float): Frequency cutoff (Hz).
            - interpolate (bool): Interpolate spectra at fcut if it is not an exact
              frequency in the efth.

        PTM5 splits spectra into wind sea and swell based on a user defined static cutoff.

        Note:
            - Spectra are interpolated at `fcut` if not in freq and `interpolate` is True.

        """
        dsout = efth.sortby("dir").sortby("freq")

        # Include cuttof if not in coordinates
        if interpolate:
            freqs = sorted(set(efth.freq.values).union([fcut]))     
            if len(freqs) > efth.freq.size:
                dsout = regrid_spec(efth, freq=freqs)

        # Zero data outside the domain of each partition
        hf = dsout.where((dsout.freq >= fcut))
        lf = dsout.where((dsout.freq <= fcut))

        # Combining into part index
        dsout = xr.concat([hf, lf], dim="part")
        set_spec_attributes(dsout)

        return dsout.fillna(0.)
