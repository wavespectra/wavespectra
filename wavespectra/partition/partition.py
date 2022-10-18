


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

    def ptm5(self, efth, fmin=None, fmax=None, dmin=None, dmax=None):
        """SWAN partitioning of sea and swell based on user-defined thresholds.

        Args:
            - efth (DataArray): Spectra DataArray in Wavespectra convention.
            - fmin (float): lowest frequency to split spectra, by default the lowest.
            - fmax (float): highest frequency to split spectra, by default the highest.
            - dmin (float): lowest direction to split spectra at, by default min(dir).
            - dmax (float): highest direction to split spectra at, by default max(dir).

        PTM5 splits spectra into wind sea and swell based on a user defined static cutoff.

        Note:
            - Spectra are interpolated at `fmin` / `fmax` if they are not in freq.

        """
        if fmax is not None and fmin is not None and fmax <= fmin:
            raise ValueError("fmax needs to be greater than fmin")
        if dmax is not None and dmin is not None and dmax <= dmin:
            raise ValueError("dmax needs to be greater than dmin")

        # Slice frequencies
        other = efth.sel(freq=slice(fmin, fmax))

        # Slice directions
        if attrs.DIRNAME in other.dims and (dmin or dmax):
            other = efth.sortby([attrs.DIRNAME]).sel(dir=slice(dmin, dmax))

        # Interpolate at fmin
        if fmin is not None and (other.freq.min() > fmin) and (self.freq.min() <= fmin):
            other = xr.concat([self._interp_freq(fmin), other], dim=attrs.FREQNAME)

        # Interpolate at fmax
        if fmax is not None and (other.freq.max() < fmax) and (self.freq.max() >= fmax):
            other = xr.concat([other, self._interp_freq(fmax)], dim=attrs.FREQNAME)

        other.freq.attrs = efth.freq.attrs
        other.dir.attrs = self._obj.dir.attrs

        if rechunk:
            other = other.chunk({attrs.FREQNAME: None, attrs.DIRNAME: None})

        return other
