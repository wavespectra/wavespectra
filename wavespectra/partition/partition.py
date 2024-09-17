"""Partitioning interface."""
from itertools import combinations
import numpy as np
import xarray as xr

from wavespectra.partition import specpart
from wavespectra.core.utils import (
    set_spec_attributes,
    regrid_spec,
    smooth_spec,
    check_same_coordinates,
    D2R,
    celerity,
    is_overlap,
    waveage,
)
from wavespectra.core.attributes import attrs
from wavespectra.core import npstats
from wavespectra.partition.hanson_and_phillips_2001 import combine_partitions_hp01
from wavespectra.partition.tracking import track_partitions


DEFAULTS = {
    "ihmax": 100,
    "angle_max": 30,
    "hs_min": 0.2,
    "k": 0.5,
    "swells": 3,
    "smooth": False,
    "window": 3,
    "wscut": 0.3333,
    "agefac": 1.7,
}


class Partition:
    """Spectra partition methods.

    Methods:
        - ptm1: In PTM1, topographic partitions for which the percentage of wind-sea energy exceeds a
          defined fraction are aggregated and assigned to the wind-sea component (e.g., the first
          partition). The remaining partitions are assigned as swell components in order of
          decreasing wave height.
        - ptm2: PTM2 works in a similar way to PTM1 by identifying a primary wind sea (assigned as
          partition 0) and one or more swell components. In this method however all the swell
          partitions are checked for the influence of wind-sea with energy within spectral bins
          within the wind-sea range (as defined by a wave age criterion) removed and combined
          into a secondary wind-sea partition (assigned as partition 1). The remaining swell
          partitions are then assigned in order of decreasing wave height from partition 2 onwards.
          This implies PTM2 has an extra partition compared to PTM1.
        - ptm3: PTM3 does not classify the topographic partitions into wind-sea or swell - it simply orders them
          by wave height. This approach is useful for producing data for spectral reconstruction applications
          using a limited number of partitions, where the classification of the partition as wind-sea or
          swell is less important than the proportion of overall spectral energy each partition represents.
        - ptm4: PTM4 uses the wave age criterion derived from the local wind speed to split the spectrum in
          to a wind-sea and single swell partition. In this case  waves with a celerity greater
          than the directional component of the local wind speed are considered to be
          freely propogating swell (i.e. unforced by the wind). This is similar to the
          method commonly used to generate wind-sea and swell from the WAM model.
        - ptm5: PTM5 splits spectra into wind sea and swell based on a user defined static cutoff.
        - hp01: HP01 partitions the spectra and merges wind-sea components as in the PTM1 method, then it merges
          adjacent swells following the criteria outlined in Hanson and Phillips (2001) and Hanson et al. (2009).
        - bbox: BBOX partitions the spectra based on user-defined bounding boxes in frequency-direction space.
        - ptm1_track: Partition spectra using the PTM1 method and track the partitions using the evolution of
          peak frequency and peak direction in time.

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

    def __init__(self, dset):
        if isinstance(dset, xr.DataArray):
            self.dset = dset
        elif isinstance(dset, xr.Dataset):
            self.dset = dset[attrs.SPECNAME]
        else:
            raise ValueError("dset needs to be either SpecArray or SpecDataset")

    def _set_metadata(self, dsout):
        """Define metadata attributes in output."""
        dsout.name = "efth"
        dsout["part"] = np.arange(dsout.part.size)
        set_spec_attributes(dsout)
        dsout.attrs = attrs.ATTRS[attrs.SPECNAME]
        return dsout

    def ptm1(
        self,
        wspd,
        wdir,
        dpt,
        agefac=DEFAULTS["agefac"],
        wscut=DEFAULTS["wscut"],
        swells=DEFAULTS["swells"],
        smooth=DEFAULTS["smooth"],
        freq_window=DEFAULTS["window"],
        dir_window=DEFAULTS["window"],
        ihmax=DEFAULTS["ihmax"],
    ):
        """PTM1 watershed partitioning.

        In PTM1, topographic partitions for which the percentage of wind-sea energy
        exceeds a defined fraction are aggregated and assigned to the wind-sea
        component (e.g., the first partition). The remaining partitions are assigned as
        swell components in order of decreasing wave height.

        Args:
            - wspd (xr.DataArray): Wind speed DataArray.
            - wdir (xr.DataArray): Wind direction DataArray.
            - dpt (xr.DataArray): Depth DataArray.
            - swells (int): Number of swell partitions to compute.
            - agefac (float): Age factor.
            - wscut (float): Wind sea fraction cutoff.
            - smooth (bool): Compute watershed boundaries from smoothed spectra
              as described in Portilla et al., 2009.
            - freq_window (int): Size of running window for smoothing spectra.
            - dir_window (int): Size of running window for smoothing spectra.
            - ihmax (int): Number of discrete spectral levels in WW3 Watershed code.

        Returns:
            - dspart (xr.Dataset): Partitioned spectra with extra `part` dimension
              where the 0th index are the wind sea and remaining indices are the swells
              sorted by descending order of Hs.

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
        if smooth:
            dset_smooth = smooth_spec(self.dset, freq_window, dir_window)
        else:
            dset_smooth = self.dset

        # Partitioning full spectra
        dsout = xr.apply_ufunc(
            np_ptm1,
            self.dset,
            dset_smooth,
            self.dset.freq,
            self.dset.dir,
            wspd,
            wdir,
            dpt,
            agefac,
            wscut,
            swells,
            ihmax,
            input_core_dims=[
                ["freq", "dir"],
                ["freq", "dir"],
                ["freq"],
                ["dir"],
                [],
                [],
                [],
                [],
                [],
                [],
                [],
            ],
            output_core_dims=[["part", "freq", "dir"]],
            vectorize=True,
            dask="parallelized",
            output_dtypes=["float32"],
            dask_gufunc_kwargs={
                "allow_rechunk": True,
                "output_sizes": {"part": swells + 1},
            },
        )

        # Finalise output
        dsout = self._set_metadata(dsout)
        parts_description = {
            "part0": "wind sea",
            "part1-n": "swells in descending order of hs",
        }
        dsout.attrs.update(parts_description)

        return dsout.transpose("part", ...)

    def ptm2(
        self,
        wspd,
        wdir,
        dpt,
        agefac=DEFAULTS["agefac"],
        wscut=DEFAULTS["wscut"],
        swells=DEFAULTS["swells"],
        smooth=DEFAULTS["smooth"],
        freq_window=DEFAULTS["window"],
        dir_window=DEFAULTS["window"],
        ihmax=DEFAULTS["ihmax"],
    ):
        """PTM2 watershed partitioning with secondary wind-sea.

        PTM2 works in a similar way to PTM1 by identifying a primary wind sea (assigned
        to partition 0) and one or more swell components. In this method however all
        the swell partitions are checked for the influence of wind-sea with energy
        within spectral bins within the wind-sea range (as defined by a wave age
        criterion) removed and combined into a secondary wind-sea partition (assigned
        to partition 1). The remaining swell partitions are then assigned in order of
        decreasing wave height from partition 2 onwards. This implies PTM2 has an extra
        partition compared to PTM1.

        Args:
            - wspd (xr.DataArray): Wind speed DataArray.
            - wdir (xr.DataArray): Wind direction DataArray.
            - dpt (xr.DataArray): Depth DataArray.
            - swells (int): Number of swell partitions to compute.
            - agefac (float): Age factor.
            - wscut (float): Wind sea fraction cutoff.
            - smooth (bool): Compute watershed boundaries from smoothed spectra
              as described in Portilla et al., 2009.
            - freq_window (int): Size of running window along `freq` for smoothing spectra.
            - dir_window (int): Size of running window along `dir` for smoothing spectra.
            - ihmax (int): Number of discrete spectral levels in WW3 Watershed code.

        Returns:
            - dspart (xr.Dataset): Partitioned spectra with extra `part` dimension
              where the 0th and 1st indices are the primary and secondary wind seas
              and remaining indices are the swells sorted by descending order of Hs.

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
        if smooth:
            dset_smooth = smooth_spec(self.dset, freq_window, dir_window)
        else:
            dset_smooth = self.dset

        # Partitioning full spectra
        dsout = xr.apply_ufunc(
            np_ptm2,
            self.dset,
            dset_smooth,
            self.dset.freq,
            self.dset.dir,
            wspd,
            wdir,
            dpt,
            agefac,
            wscut,
            swells,
            ihmax,
            input_core_dims=[
                ["freq", "dir"],
                ["freq", "dir"],
                ["freq"],
                ["dir"],
                [],
                [],
                [],
                [],
                [],
                [],
                [],
            ],
            output_core_dims=[["part", "freq", "dir"]],
            vectorize=True,
            dask="parallelized",
            output_dtypes=["float32"],
            dask_gufunc_kwargs={
                "allow_rechunk": True,
                "output_sizes": {"part": swells + 2},
            },
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

    def ptm3(
        self,
        parts=DEFAULTS["swells"],
        smooth=DEFAULTS["smooth"],
        freq_window=DEFAULTS["window"],
        dir_window=DEFAULTS["window"],
        ihmax=DEFAULTS["ihmax"],
    ):
        """PTM3 watershed partitioning with no wind-sea or swell classification.

        PTM3 does not classify the topographic partitions into wind-sea or swell - it
        simply orders them by wave height. This approach is useful for producing data
        for spectral reconstruction applications using a limited number of partitions,
        where the classification of the partition as wind-sea or swell is less
        important than the proportion of overall spectral energy each partition
        represents. In addition, this method does not require wind and water depth
        information and can be used with any spectral dataset.

        Args:
            - parts (int): Number of partitions to keep.
            - smooth (bool): Compute watershed boundaries from smoothed spectra
              as described in Portilla et al., 2009.
            - freq_window (int): Size of running window along `freq` for smoothing spectra.
            - dir_window (int): Size of running window along `dir` for smoothing spectra.
            - ihmax (int): Number of discrete spectral levels in WW3 Watershed code.

        Returns:
            - dspart (xr.Dataset): Partitioned spectra with extra `part` dimension
              defining watershed partitions sorted by descending order of Hs.

        References:
            - Portilla et al. (2009).
            - Tracy et al. (2007).
            - Vincent et al. (1991).
            - WW3 documentation (https://github.com/NOAA-EMC/WW3).

        """
        # Sort out inputs
        if smooth:
            dset_smooth = smooth_spec(self.dset, freq_window, dir_window)
        else:
            dset_smooth = self.dset

        # Partitioning full spectra
        dsout = xr.apply_ufunc(
            np_ptm3,
            self.dset,
            dset_smooth,
            self.dset.freq,
            self.dset.dir,
            parts,
            ihmax,
            input_core_dims=[
                ["freq", "dir"],
                ["freq", "dir"],
                ["freq"],
                ["dir"],
                [],
                [],
            ],
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

    def ptm4(self, wspd, wdir, dpt, agefac=DEFAULTS["agefac"]):
        """PTM4 WAM partitioning of sea and swell based on wave age criterion..

        PTM4 uses the wave age criterion derived from the local wind speed to split the
        spectrum into a wind-sea and single swell partition. In this case waves with a
        celerity greater than the directional component of the local wind speed are
        considered to be freely propogating swell (i.e. unforced by the wind). This is
        similar to the method commonly used to generate wind-sea and swell from the WAM
        model.

        Args:
            - wspd (xr.DataArray): Wind speed DataArray.
            - wdir (xr.DataArray): Wind direction DataArray.
            - dpt (xr.DataArray): Depth DataArray.
            - agefac (float): Age factor.

        Returns:
            - dspart (xr.Dataset): Partitioned spectra with extra `part` dimension
              where the 0th index is the wind sea and the 1st index is the swell.

        References:
            - WW3 documentation (https://github.com/NOAA-EMC/WW3).

        """
        dsout = self.dset.sortby("dir").sortby("freq")

        # Masking wind sea and swell regions
        windseamask = waveage(dsout.freq, dsout.dir, wspd, wdir, dpt, agefac)
        sea = dsout.where(windseamask)
        swell = dsout.where(~windseamask)

        # Combining into part index
        dsout = xr.concat([sea, swell], dim="part")

        # Finalise output
        dsout = self._set_metadata(dsout)
        dsout.attrs.update({"part0": "wind sea", "part1": "swell"})

        return dsout.fillna(0.0)

    def ptm5(self, fcut, interpolate=True):
        """PTM5 SWAN partitioning of sea and swell based on user-defined threshold.

        PTM5 splits spectra into wind sea and swell based on a user defined static
        cutoff. This method differs from :meth:`~wavespectra.specarray.SpecArray.split`
        in that here the output partitioned spectra dataset has an extra `part`
        dimension and the sea and swell partitions have zero-values outside the defined
        frequency ranges. Conversely, :meth:`~wavespectra.specarray.SpecArray.split`
        returns a single partition with frequencies truncated to the defined ranges.
        Notice there could be slight differences when integrating the partitions
        generated by these two methods since in PTM5 there will be an "area" at one of
        the frequency adges adjacent to the zero-values.

        Args:
            - fcut (float): Frequency cutoff (Hz).
            - interpolate (bool): Interpolate spectra at fcut if it is not an exact
              frequency in the dset.

        Returns:
            - dspart (xr.Dataset): Partitioned spectra with extra `part` dimension
              where the 0th index is the wind sea and the 1st index is the swell.

        Note:
            - Spectra are interpolated at `fcut` if this frequency is not in `dset`
              and `interpolate` is set to True which implies the frequency coordinates
              will be different between the input and the output.

        References:
            - WW3 documentation (https://github.com/NOAA-EMC/WW3).

        """
        dsout = self.dset.sortby("dir").sortby("freq")

        # Include cuttof if not in coordinates
        if interpolate:
            freqs = sorted(set(self.dset.freq.values).union([fcut]))
            if len(freqs) > self.dset.freq.size:
                dsout = regrid_spec(self.dset, freq=freqs)

        # Zero data outside the domain of each partition
        hf = dsout.where((dsout.freq >= fcut))
        lf = dsout.where((dsout.freq <= fcut))

        # Combining into part index
        dsout = xr.concat([hf, lf], dim="part")

        # Finalise output
        dsout = self._set_metadata(dsout)
        dsout.attrs.update({"part0": "sea", "part1": "swell"})

        return dsout.fillna(0.0)

    def hp01(
        self,
        wspd=None,
        wdir=None,
        dpt=None,
        agefac=DEFAULTS["agefac"],
        wscut=DEFAULTS["wscut"],
        swells=DEFAULTS["swells"],
        smooth=DEFAULTS["smooth"],
        freq_window=DEFAULTS["window"],
        dir_window=DEFAULTS["window"],
        k=DEFAULTS["k"],
        angle_max=DEFAULTS["angle_max"],
        hs_min=DEFAULTS["hs_min"],
        ihmax=DEFAULTS["ihmax"],
        combine_extra_swells=True,
        wstype=0,
    ):
        """Hanson and Phillips 2001 spectra partitioning and swell merging.

        HP01 partitions the spectra and merges wind-sea components as in the PTM1
        method, then it merges adjacent swells following the criteria outlined in
        Hanson and Phillips (2001) and Hanson et al. (2009). This method can be useful
        when partitioning measured wave spectra which are typically noisy and may
        contain small, non-physical partitions. Please notice this method is still
        under development and may not work as expected.

        Args:
            - wspd (xr.DataArray): Wind speed DataArray.
            - wdir (xr.DataArray): Wind direction DataArray.
            - dpt (xr.DataArray): Depth DataArray.
            - swells (int): Number of swell partitions to compute.
            - agefac (float): Age factor.
            - wscut (float): Wind sea fraction cutoff.
            - smooth (bool): Compute watershed boundaries from smoothed spectra
              as described in Portilla et al., 2009.
            - freq_window (int): Size of running window along `freq` for smoothing spectra.
            - dir_window (int): Size of running window along `dir` for smoothing spectra.
            - k (float): Spread factor in Hanson and Phillips (2001)'s eq 9.
            - angle_max (float): Maximum relative angle for combining partitions.
            - hs_min (float): Minimum Hs of swell partitions, smaller ones are always
              combined with closest ones regardless of other criteria being satisfied.
            - ihmax (int): Number of discrete spectral levels in WW3 Watershed code.
            - combine_extra_swells (bool): If True and the number of swells partitions
              from the Hanson and Phillips 2001 algorithm is greater than the requested
              number given in the `swells` argument, merge each extra swell with its
              closest neighbour until the requested number is achieved. If False, extra
              swells are ignored from the output.
            - wstype (int):
                - 0: Wind partition defined by all full partitions whose wind sea
                  fraction exceed wscut.
                - 1: Wind partition defined only by spectral bins falling within
                  the wave age parabola, from all partitions available.
                - 2: Combination of [0] and [1], wind partition defined by all full
                  partitions whose wind sea fraction exceed wscut plus bins from all
                  other swell partitions falling witing wave age parabola.

        Returns:
            - dspart (xr.Dataset): Partitioned spectra with extra `part` dimension
              where the 0th index are the wind sea and remaining indices are the swells
              sorted by descending order of Hs.

        Note:
            - If wspd, wdir or dpt are not provided no wind sea classification is done.

        References:
            - Hanson and Phillips (2001).
            - Hanson et al. (2009).
            - Portilla et al. (2009).
            - Tracy et al. (2007).
            - Vincent et al. (1991).
            - WW3 documentation (https://github.com/NOAA-EMC/WW3).

        """
        # Temporary: allow choosing how wind sea partition is defined
        if wstype == 0:
            func = np_hp01
        elif wstype == 1:
            func = np_hp01_wseabins
        elif wstype == 2:
            func = np_hp01_wseafrac_wseabins
        else:
            raise ValueError(f"wstype must be 0, 1 or 2, got {wstype}")

        check_same_coordinates(wspd, wdir, dpt)
        # Smooth spectra for defining watershed boundaries
        if smooth:
            dset_smooth = smooth_spec(self.dset, freq_window, dir_window)
        else:
            dset_smooth = self.dset
        # Wind sea mask
        if wspd is None or wdir is None or dpt is None:
            windseamask = xr.zeros_like(self.dset).astype(bool)
        else:
            windseamask = waveage(
                self.dset.freq,
                self.dset.dir,
                wspd,
                wdir,
                dpt,
                agefac,
            )

        # Partitioning full spectra
        dsout = xr.apply_ufunc(
            func,
            self.dset,
            dset_smooth,
            windseamask,
            self.dset.freq,
            self.dset.dir,
            wscut,
            swells,
            k,
            angle_max,
            hs_min,
            ihmax,
            combine_extra_swells,
            input_core_dims=[
                ["freq", "dir"],
                ["freq", "dir"],
                ["freq", "dir"],
                ["freq"],
                ["dir"],
                [],
                [],
                [],
                [],
                [],
                [],
                [],
            ],
            output_core_dims=[["part", "freq", "dir"]],
            vectorize=True,
            dask="parallelized",
            output_dtypes=["float32"],
            dask_gufunc_kwargs={
                "allow_rechunk": True,
                "output_sizes": {"part": swells + 1},
            },
        )

        # Finalise output
        dsout = self._set_metadata(dsout)
        parts_description = {
            "part0": "wind sea",
            "part1-n": "merged swells in descending order of hs",
        }
        dsout.attrs.update(parts_description)

        return dsout.transpose("part", ...)

    def bbox(self, bboxes):
        """Partition based on user-defined bounding boxes in frequency-direction space.

        BBOX partitions the spectra based on user-defined bounding boxes in
        frequency-direction space.

        Args:
            - bboxes (list(dict)): List of dictionaries with keys `fmin`, `fmax`,
              `dmin` and `dmax` specifying the boundaries of each bounding box.

        Returns:
            - dspart (xr.Dataset): Partitioned spectra with extra `part` dimension
              with indices ordered as in the list of dictionaries.

        Note:
            - Non-specified bounds in each bbox dict are defined from the bounds of the
              freq / dir bounds in the spectrum, e.g., `fmin=min(freq)`.
            - Bounding boxes must not overlap.
            - Last part index is defined by spectral bins not covered by any bboxes.

        TODO: Deal with directions crossing 360.

        """
        ds = self.dset.sortby("dir").sortby("freq")

        # Chec inputs and define rectangles
        rectangles = []
        for bbox in bboxes:
            fmin = bbox.get("fmin", float(ds.freq.min())) or float(ds.freq.min())
            fmax = bbox.get("fmax", float(ds.freq.max())) or float(ds.freq.max())
            dmin = bbox.get("dmin", float(ds.dir.min())) or float(ds.dir.min())
            dmax = bbox.get("dmax", float(ds.dir.min())) or float(ds.dir.max())

            if fmin >= fmax:
                raise ValueError(f"fmin {fmin} Hz >= fmax {fmax} Hz")

            rectangles.append([fmin, dmin, fmax, dmax])

        # Ensure there is no overlapping among bboxes
        for rect1, rect2 in combinations(rectangles, 2):
            if is_overlap(rect1, rect2):
                l1, b1, r1, t1 = rect1
                l2, b2, r2, t2 = rect2
                raise ValueError(
                    f"bboxes [fmin={l1:g}, dmin={b1:g}, fmax={r1:g}, dmax={t1:g}] and "
                    f"[fmin={l2:g}, dmin={b2:g}, fmax={r2:g}, dmax={t2:g}] overlap"
                )

        # Define partitions
        partitions = []
        masks = False
        for rect in rectangles:
            fmin, dmin, fmax, dmax = rect
            mask = (
                (ds.freq >= fmin)
                & (ds.freq <= fmax)
                & (ds.dir >= dmin)
                & (ds.dir <= dmax)
            )
            partitions.append(ds.where(mask))
            masks = masks | mask
        # Last partition
        partitions.append(ds.where(~masks))

        # Combining into part index
        dsout = xr.concat(partitions, dim="part")

        # Finalise output
        dsout = self._set_metadata(dsout)
        for ind, rect in enumerate(rectangles):
            fmin, dmin, fmax, dmax = rect
            dsout.attrs.update(
                {f"part{ind}": f"fmin={fmin}, fmax={fmax}, dmin={dmin}, dmax={dmax}"}
            )
        dsout.attrs.update({f"part{ind + 1}": "complement"})

        return dsout.fillna(0.0)

    def ptm1_track(
        self,
        wspd,
        wdir,
        dpt,
        agefac=DEFAULTS["agefac"],
        wscut=DEFAULTS["wscut"],
        swells=DEFAULTS["swells"],
        smooth=DEFAULTS["smooth"],
        freq_window=DEFAULTS["window"],
        dir_window=DEFAULTS["window"],
        ihmax=DEFAULTS["ihmax"],
        ddpm_sea_max=30,
        ddpm_swell_max=20,
        dfp_sea_scaling=1,
        dfp_swell_source_distance=1e6,
    ):
        """Partition and combine spectra from the same wave system over time.

        Partition spectra using the PTM1 method and track the partitions
        using the evolution of peak frequency and peak direction. Partitions are
        matched with the closest partition in the frequency-direction space of the
        previous time step for which the difference in direction with less than
        ddpm_max and the difference in peak frequency is less than dfp_max. ddpm_max
        differs for sea and swell partitions and is set manually. dfp_max also differs
        for sea and swell partitions. In the case of sea partitions it is a function of
        wind speed and is set to the rate of change of the wind-sea peak wave frequency
        estimated from fetch-limited relationships, (Ewans & Kibblewhite, 1986). In the
        case of swell partitions it is set to the rate of change of the swell peak wave
        frequency based on the swell dispersion relationship derived by
        Snodgrass et al (1966) assuming the distance to the source is 1e6 m.

        Args:
            - wspd (xr.DataArray): Wind speed DataArray.
            - wdir (xr.DataArray): Wind direction DataArray.
            - dpt (xr.DataArray): Depth DataArray.
            - swells (int): Number of swell partitions to compute.
            - agefac (float): Age factor.
            - wscut (float): Wind sea fraction cutoff.
            - smooth (bool): Compute watershed boundaries from smoothed spectra
              as described in Portilla et al., 2009.
            - freq_window (int): Size of running window along `freq` for smoothing spectra.
            - dir_window (int): Size of running window along `dir` for smoothing spectra.
            - ihmax (int): Number of discrete spectral levels in WW3 Watershed code.
            - ddpm_sea_max (float): Maximum peak direction difference for wind sea partition.
              Default is 30 degrees.
            - ddpm_swell_max (float): Maximum peak direction difference for swell partitions.
              Default is 20 degrees.
            - dfp_sea_scaling (float): Scaling factor for maximum peak frequency difference
              for wind sea partition. Default is 1.
            - dfp_swell_source_distance (float): Distance to source for swell peak frequency
              difference. Default is 1e6 m.

        Returns:
            - dspart (xr.Dataset): Partitioned spectra with extra `part` dimension
              where the 0th index are the wind sea and remaining indices are the swells
              sorted by descending order of Hs. Plus array `part_id` containing an integer
              representing the partition id for each time step and partition. Plus array
              `npart_id` containing the existing partition ids.

        """

        # Do the partitioning
        dspart = self.ptm1(
            wspd=wspd,
            wdir=wdir,
            dpt=dpt,
            agefac=agefac,
            wscut=wscut,
            swells=swells,
            smooth=smooth,
            freq_window=freq_window,
            dir_window=dir_window,
            ihmax=ihmax,
        )

        # Calculate peak frequency and peak direction
        stats = dspart.spec.stats(["fp", "dpm"])

        # Track partitions
        part_ids = track_partitions(
            stats,
            wspd=wspd,
            ddpm_sea_max=ddpm_sea_max,
            ddpm_swell_max=ddpm_swell_max,
            dfp_sea_scaling=dfp_sea_scaling,
            dfp_swell_source_distance=dfp_swell_source_distance,
        )

        # Add partition ids to partition data and return
        return xr.merge([dspart, part_ids])


def np_ptm1(
    spectrum,
    spectrum_smooth,
    freq,
    dir,
    wspd,
    wdir,
    dpt,
    agefac=DEFAULTS["agefac"],
    wscut=DEFAULTS["wscut"],
    swells=DEFAULTS["swells"],
    ihmax=DEFAULTS["ihmax"],
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
        - swells (int): Number of swell partitions to compute, all detected if None.
        - ihmax (int): Number of discrete spectral levels in WW3 Watershed code.

    Returns:
        - specpart (3darray): Wave spectrum partitions sorted in decreasing order of Hs
          with shape (np, nf, nd).

    Note:
        - The smooth spectrum `spectrum_smooth` is used to define the watershed
          boundaries which are applied to the original spectrum.

    """
    # Use smooth spectrum to define morphological boundaries
    watershed_map = specpart.partition(spectrum_smooth.astype(np.float32), ihmax)
    nparts = watershed_map.max()

    # Wind sea mask
    up = np.tile(agefac * wspd * np.cos(D2R * (dir - wdir)), (freq.size, 1))
    windseamask = up > np.tile(celerity(freq, dpt)[:, np.newaxis], (1, dir.size))

    # Assign partitioned arrays from raw spectrum and morphological boundaries
    wsea_partition = np.zeros_like(spectrum)
    swell_partitions = [np.zeros_like(spectrum) for n in range(nparts)]
    for ipart in range(nparts):
        part = np.where(watershed_map == ipart + 1, spectrum, 0.0)  # start at 1
        wsfrac = part[windseamask].sum() / part.sum()
        if wsfrac > wscut:
            wsea_partition += part
        else:
            swell_partitions[ipart] += part

    # Sort swells by Hs
    isort = np.argsort([-npstats.hs(swell, freq, dir) for swell in swell_partitions])
    swell_partitions = list(np.array(swell_partitions)[isort])

    # Dealing with the number of swells
    if swells is None:
        # Exclude null swell partitions if the number of output swells is undefined
        swell_partitions = [swell for swell in swell_partitions if swell.sum() > 0]
    else:
        if nparts > swells:
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
    agefac=DEFAULTS["agefac"],
    wscut=DEFAULTS["wscut"],
    swells=DEFAULTS["swells"],
    ihmax=DEFAULTS["ihmax"],
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
        - swells (int): Number of swell partitions to compute, all detected if None.
        - ihmax (int): Number of discrete spectral levels in WW3 Watershed code.

    Returns:
        - specpart (3darray): Wave spectrum partitions sorted in decreasing order of Hs
          with shape (np, nf, nd) with np indices 0 and 1 reserved for primary and
          secondary wind sea partitions and remaining ones for ordered swells.

    Note:
        - The smooth spectrum `spectrum_smooth` is used to define the watershed
          boundaries which are applied to the original spectrum.
        - The option in WW3 to leave secondary wind seas as separate partitions is not
          available as it makes it harder to distinguish them from swells in the output.

    """
    # Use smooth spectrum to define morphological boundaries
    watershed_map = specpart.partition(spectrum_smooth.astype(np.float32), ihmax)
    nparts = watershed_map.max()

    # Wind sea mask
    up = np.tile(agefac * wspd * np.cos(D2R * (dir - wdir)), (freq.size, 1))
    windseamask = up > np.tile(celerity(freq, dpt)[:, np.newaxis], (1, dir.size))

    # Assign partitioned arrays from raw spectrum and morphological boundaries
    wsea_primary_partition = np.zeros_like(spectrum)
    wsea_secondary_partition = np.zeros_like(spectrum)
    swell_partitions = [np.zeros_like(spectrum) for n in range(nparts)]
    for ipart in range(nparts):
        part = np.where(watershed_map == ipart + 1, spectrum, 0.0)  # start at 1
        wsfrac = part[windseamask].sum() / part.sum()
        if wsfrac > wscut:
            wsea_primary_partition += part
        else:
            wsea_secondary_partition += np.where(windseamask, part, 0.0)
            swell_partitions[ipart] += np.where(windseamask, 0.0, part)

    # Sort swells by Hs
    isort = np.argsort([-npstats.hs(swell, freq, dir) for swell in swell_partitions])
    swell_partitions = list(np.array(swell_partitions)[isort])

    # Dealing with the number of swells
    if swells is None:
        # Exclude null swell partitions if the number of output swells is undefined
        swell_partitions = [swell for swell in swell_partitions if swell.sum() > 0]
    else:
        if nparts > swells:
            # Discard extra partitions
            swell_partitions = swell_partitions[:swells]
        elif nparts < swells:
            # Extend partitions list with null spectra
            n = swells - len(swell_partitions)
            for i in range(n):
                swell_partitions.append(np.zeros_like(spectrum))

    wsea_partitions = [wsea_primary_partition, wsea_secondary_partition]
    return np.array(wsea_partitions + swell_partitions)


def np_ptm3(
    spectrum,
    spectrum_smooth,
    freq,
    dir,
    parts=DEFAULTS["swells"],
    ihmax=DEFAULTS["ihmax"],
):
    """PTM3 spectra partitioning on numpy arrays.

    Args:
        - spectrum (2darray): Wave spectrum array with shape (nf, nd).
        - spectrum_smooth (2darray): Smoothed wave spectrum array with shape (nf, nd).
        - freq (1darray): Wave frequency array with shape (nf).
        - dir (1darray): Wave direction array with shape (nd).
        - parts (int): Number of partitions to compute, all detected by default.
        - ihmax (int): Number of discrete spectral levels in WW3 Watershed code.

    Returns:
        - specpart (3darray): Wave spectrum partitions sorted in decreasing order of Hs
          with shape (np, nf, nd).

    Note:
        - The smooth spectrum `spectrum_smooth` is used to define the watershed
          boundaries which are applied to the original spectrum.

    """
    # Use smooth spectrum to define morphological boundaries
    watershed_map = specpart.partition(spectrum_smooth.astype(np.float32), ihmax)
    nparts = watershed_map.max()

    # Assign partitioned arrays from raw spectrum and morphological boundaries
    partitions = []
    for npart in range(1, nparts + 1):
        partitions.append(np.where(watershed_map == npart, spectrum, 0.0))

    # Sort partitions by Hs
    isort = np.argsort([-npstats.hs(swell, freq, dir) for swell in partitions])
    partitions = list(np.array(partitions)[isort])

    if parts is not None:
        if nparts > parts:
            # Discard extra partitions
            partitions = partitions[:parts]
        elif nparts < parts:
            # Extend partitions list with zero arrays
            template = np.zeros_like(spectrum)
            n = parts - len(partitions)
            for i in range(n):
                partitions.append(template)

    return np.array(partitions)


def np_hp01(
    spectrum,
    spectrum_smooth,
    windseamask,
    freq,
    dir,
    wscut=DEFAULTS["wscut"],
    swells=DEFAULTS["swells"],
    k=DEFAULTS["k"],
    angle_max=DEFAULTS["angle_max"],
    hs_min=DEFAULTS["hs_min"],
    ihmax=DEFAULTS["ihmax"],
    combine_extra_swells=True,
):
    """Hanson and Phillips 2001 spectra partitioning on numpy arrays.

    Args:
        - spectrum (2darray): Wave spectrum array with shape (nf, nd).
        - spectrum_smooth (2darray): Smoothed wave spectrum array with shape (nf, nd).
        - windseamask (2darray): Wind-sea mask array with shape (nf, nd).
        - freq (1darray): Wave frequency array with shape (nf).
        - dir (1darray): Wave direction array with shape (nd).
        - wscut (float): Wind sea fraction cutoff.
        - swells (int): Number of swell partitions to compute, all detected by default.
        - k (float): Spread factor in Hanson and Phillips (2001)'s eq 9.
        - angle_max (float): Maximum relative angle for combining partitions.
        - hs_min (float): Minimum Hs of swell partitions, smaller partitions are always
          combined with closest ones regardless of minimum distance and angle criteria.
        - combine_extra_swells (bool): If True and the number of swells partitions
          from the Hanson and Phillips 2001 algorithm is greater than the requested
          number given in the `swells` argument, merge each extra swell with its
          closest neighbour until the requested number is achieved. If False, extra
          swells are ignored from the output.

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
    watershed_map = specpart.partition(spectrum_smooth.astype(np.float32), ihmax)
    nparts = watershed_map.max()

    # Assign partitioned arrays from raw spectrum and morphological boundaries
    wsea_partition = np.zeros_like(spectrum)
    swell_partitions = []
    for ipart in range(nparts):
        part = np.where(watershed_map == ipart + 1, spectrum, 0.0)  # start at 1
        wsfrac = part[windseamask].sum() / part.sum()
        if wsfrac > wscut:
            wsea_partition += part
        else:
            swell_partitions.append(part)
    # Convert to numpy array for easy sorting
    swell_partitions = np.array(swell_partitions)

    # Sort swells by Hs
    hs_swells_neg = [-npstats.hs(swell, freq, dir) for swell in swell_partitions]
    isort = np.argsort(hs_swells_neg)
    swell_partitions = list(swell_partitions[isort])

    # Combine extra swell partitions
    if len(swell_partitions) > 1:
        swell_partitions = combine_partitions_hp01(
            partitions=swell_partitions,
            freq=freq,
            dir=dir,
            swells=swells,
            k=k,
            angle_max=angle_max,
            hs_min=hs_min,
            combine_extra_swells=combine_extra_swells,
        )

    # Extend list to ensure the correct number of partitions is returned
    nswells = len(swell_partitions)
    if swells is not None and nswells < swells:
        nullspec = np.zeros_like(spectrum)
        n = swells - nswells
        for i in range(n):
            swell_partitions.append(nullspec)

    return np.array([wsea_partition] + swell_partitions)


def np_hp01_wseabins(
    spectrum,
    spectrum_smooth,
    windseamask,
    freq,
    dir,
    wscut=DEFAULTS["wscut"],
    swells=DEFAULTS["swells"],
    k=DEFAULTS["k"],
    angle_max=DEFAULTS["angle_max"],
    hs_min=DEFAULTS["hs_min"],
    ihmax=DEFAULTS["ihmax"],
    combine_extra_swells=True,
):
    """Hanson and Phillips 2001 spectra partitioning on numpy arrays.

    Modified from np_hp01 to set all wsea bins from all partitions as part0.

    Args:
        - spectrum (2darray): Wave spectrum array with shape (nf, nd).
        - spectrum_smooth (2darray): Smoothed wave spectrum array with shape (nf, nd).
        - windseamask (2darray): Wind-sea mask array with shape (nf, nd).
        - freq (1darray): Wave frequency array with shape (nf).
        - dir (1darray): Wave direction array with shape (nd).
        - wscut (float): Wind sea fraction cutoff.
        - swells (int): Number of swell partitions to compute, all detected by default.
        - k (float): Spread factor in Hanson and Phillips (2001)'s eq 9.
        - angle_max (float): Maximum relative angle for combining partitions.
        - hs_min (float): Minimum Hs of swell partitions, smaller partitions are always
          combined with closest ones regardless of minimum distance and angle criteria.
        - combine_extra_swells (bool): If True and the number of swells partitions
          from the Hanson and Phillips 2001 algorithm is greater than the requested
          number given in the `swells` argument, merge each extra swell with its
          closest neighbour until the requested number is achieved. If False, extra
          swells are ignored from the output.

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
    watershed_map = specpart.partition(spectrum_smooth.astype(np.float32), ihmax)
    nparts = watershed_map.max()

    # Assign partitioned arrays from raw spectrum and morphological boundaries
    wsea_partition = np.zeros_like(spectrum)
    swell_partitions = []
    for ipart in range(nparts):
        part = np.where(watershed_map == ipart + 1, spectrum, 0.0)  # start at 1
        # wsfrac = part[windseamask].sum() / part.sum()
        wsea_partition += np.where(windseamask, part, 0.0)
        swell_partitions.append(np.where(windseamask, 0.0, part))
        # if wsfrac > wscut:
        #     wsea_partition += part
        # else:
        #     swell_partitions.append(part)
    # Convert to numpy array for easy sorting
    swell_partitions = np.array(swell_partitions)

    # Sort swells by Hs
    hs_swells_neg = [-npstats.hs(swell, freq, dir) for swell in swell_partitions]
    isort = np.argsort(hs_swells_neg)
    swell_partitions = list(swell_partitions[isort])

    # Combine extra swell partitions
    if len(swell_partitions) > 1:
        swell_partitions = combine_partitions_hp01(
            partitions=swell_partitions,
            freq=freq,
            dir=dir,
            swells=swells,
            k=k,
            angle_max=angle_max,
            hs_min=hs_min,
            combine_extra_swells=combine_extra_swells,
        )

    # Extend list to ensure the correct number of partitions is returned
    nswells = len(swell_partitions)
    if swells is not None and nswells < swells:
        nullspec = np.zeros_like(spectrum)
        n = swells - nswells
        for i in range(n):
            swell_partitions.append(nullspec)

    return np.array([wsea_partition] + swell_partitions)


def np_hp01_wseafrac_wseabins(
    spectrum,
    spectrum_smooth,
    windseamask,
    freq,
    dir,
    wscut=DEFAULTS["wscut"],
    swells=DEFAULTS["swells"],
    k=DEFAULTS["k"],
    angle_max=DEFAULTS["angle_max"],
    hs_min=DEFAULTS["hs_min"],
    ihmax=DEFAULTS["ihmax"],
    combine_extra_swells=True,
):
    """Hanson and Phillips 2001 spectra partitioning on numpy arrays.

    Modified from np_hp01 to set as wsea partitions exceeding the wscut condition
    plus all wsea bins from all partition.

    Args:
        - spectrum (2darray): Wave spectrum array with shape (nf, nd).
        - spectrum_smooth (2darray): Smoothed wave spectrum array with shape (nf, nd).
        - windseamask (2darray): Wind-sea mask array with shape (nf, nd).
        - freq (1darray): Wave frequency array with shape (nf).
        - dir (1darray): Wave direction array with shape (nd).
        - wscut (float): Wind sea fraction cutoff.
        - swells (int): Number of swell partitions to compute, all detected by default.
        - k (float): Spread factor in Hanson and Phillips (2001)'s eq 9.
        - angle_max (float): Maximum relative angle for combining partitions.
        - hs_min (float): Minimum Hs of swell partitions, smaller partitions are always
          combined with closest ones regardless of minimum distance and angle criteria.
        - combine_extra_swells (bool): If True and the number of swells partitions
          from the Hanson and Phillips 2001 algorithm is greater than the requested
          number given in the `swells` argument, merge each extra swell with its
          closest neighbour until the requested number is achieved. If False, extra
          swells are ignored from the output.

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
    watershed_map = specpart.partition(spectrum_smooth.astype(np.float32), ihmax)
    nparts = watershed_map.max()

    # Assign partitioned arrays from raw spectrum and morphological boundaries
    wsea_partition = np.zeros_like(spectrum)
    swell_partitions = []
    for ipart in range(nparts):
        part = np.where(watershed_map == ipart + 1, spectrum, 0.0)  # start at 1
        wsfrac = part[windseamask].sum() / part.sum()
        # wsea_partition += np.where(windseamask, part, 0.0)
        # swell_partitions.append(np.where(windseamask, 0.0, part))
        if wsfrac > wscut:
            wsea_partition += part
        else:
            wsea_partition += np.where(windseamask, part, 0.0)
            swell_partitions.append(np.where(windseamask, 0.0, part))
    # Convert to numpy array for easy sorting
    swell_partitions = np.array(swell_partitions)

    # Sort swells by Hs
    hs_swells_neg = [-npstats.hs(swell, freq, dir) for swell in swell_partitions]
    isort = np.argsort(hs_swells_neg)
    swell_partitions = list(swell_partitions[isort])

    # Combine extra swell partitions
    if len(swell_partitions) > 1:
        swell_partitions = combine_partitions_hp01(
            partitions=swell_partitions,
            freq=freq,
            dir=dir,
            swells=swells,
            k=k,
            angle_max=angle_max,
            hs_min=hs_min,
            combine_extra_swells=combine_extra_swells,
        )

    # Extend list to ensure the correct number of partitions is returned
    nswells = len(swell_partitions)
    if swells is not None and nswells < swells:
        nullspec = np.zeros_like(spectrum)
        n = swells - nswells
        for i in range(n):
            swell_partitions.append(nullspec)

    return np.array([wsea_partition] + swell_partitions)
