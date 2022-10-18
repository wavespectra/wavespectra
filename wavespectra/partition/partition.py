"""Partitioning interface."""
import xarray as xr

from wavespectra.core.utils import set_spec_attributes, regrid_spec


class Partition:

    def ptm1(self):
        """Watershed partitioning with wind-sea defined from wind-sea fraction. 

        In PTM1, topographic partitions for which the percentage of wind-sea energy exceeds a 
        defined fraction are aggregated and assigned to the wind-sea component (e.g., the first
        partition). The remaining partitions are assigned as swell components in order of 
        decreasing wave height.

        """
        pass

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

    def ptm3(self):
        """Watershed partitioning with no wind-sea or swell classification

        PTM3 does not classify the topographic partitions into wind-sea or swell - it simply orders
        them by wave height. This approach is useful for producing data for spectral reconstruction 
        applications using a limited number of partitions (e.g. \cite{pro:BSP13}), where the 
        classification of the partition as wind-sea or swell is less important than the proportion
        of overall spectral energy each partition represents.

        """
        pass

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
            - Spectra are interpolated at `fcut` / `dcut` if they are not in freq.

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
