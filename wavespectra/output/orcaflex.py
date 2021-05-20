"""Orcaflex output plugin - using orcaflex API."""
import numpy as np


def to_orcaflex(self, model, minEnergy=1e-6):
    """Writes the spectrum to an Orcaflex model

    Uses the orcaflex API (OrcFxAPI) to set the wave-data of the provided orcaflex model.

    The axis system conversion used is:
    - Orcaflex global X  = Towards East
    - Orcaflex global Y  = Towards North

    This function creates a wave-train for each of the directions in this spectrum using a user-defined spectrum.

    Calculation of wave-components in orcaflex is computationally expensive. To save computational time:
    1. Use the minEnergy parameter of this function to define a treshold for the amount of energy in a wave-train.
    2. In orcaflex itself: limit the amount of wave-components
    3. Before exporting: regrid the spectrum to a lower amount of directions.

    Orcaflex theory:
    - https://www.orcina.com/webhelp/OrcaFlex/Content/html/Wavetheory.htm
    - https://www.orcina.com/webhelp/OrcaFlex/Content/html/Directionconventions.htm

    Example:
        >>> from OrcFxAPI import *
        >>> from wavespectra import read_triaxys
        >>> m = Model()
        >>> spectrum = read_triaxys("triaxys.DIRSPEC")).isel(time=0)  # get only the fist spectrum in time
        >>> spectrum.spec.to_orcaflex(m)


    Args:
        - model : orcaflex model (OrcFxAPI.model instance)
        - minEnergy [1e-6] : threshold for minimum sum of energy in a direction before it is exported

    Note:
        - an Orcaflex license is required to work with the orcaflex API.
        - Only 2D spectra E(f,d) are currently supported.
        - The DataArray should contain only a single spectrum. Hint: first_spetrum = spectra.isel(time=0)

    """

    dirs = np.array(self.dir.values)
    freqs = np.array(self.freq.values)

    ddir = self.dd

    # verify what all coordinates other than dir and freq are one
    if not np.prod(self.efth.shape) == len(dirs) * len(freqs):
        raise ValueError(
            "The DataArray should contain only a single spectrum.\nHint: first_spetrum = spectra.isel(time=0)"
        )

    nTrains = 0
    env = model.environment  # alias

    for dir in dirs:
        e = self.efth.sel(dict(dir=dir)).values.flatten()

        E = ddir * e

        if np.sum(E) <= minEnergy:
            continue

        nTrains += 1

        env.NumberOfWaveTrains = nTrains
        env.SelectedWaveTrainIndex = nTrains - 1  # zero-based = f'Wave{nTrains}'
        env.WaveDirection = np.mod(
            90 - dir + 180, 360
        )  # convert from coming from to going to and from compass to ofx
        env.WaveType = "User Defined Spectrum"
        env.WaveNumberOfSpectralDirections = 1

        # interior points in the spectrum with zero energy are not allowed by orcaflex
        iFirst = np.where(E > 0)[0][0]
        iLast = np.where(E > 0)[0][-1]

        for i in range(iFirst, iLast):
            if E[i] < 1e-10:
                E[i] = 1e-10

        if iFirst > 0:
            iFirst -= 1
        if iLast < len(E) - 2:
            iLast += 1

        env.WaveNumberOfUserSpectralPoints = len(E[iFirst:iLast])
        env.WaveSpectrumS = E[iFirst:iLast]
        env.WaveSpectrumFrequency = freqs[iFirst:iLast]

        env.WaveType = 'Airy'  #Temporary set the wave-type to Airy. This is to avoid re-calcultion of
                               # the spectral properties each time the next train is set (can slow-down
                               # considerably when using many sprectral components
                               # !thank you people at orcina for your help solving this!


    # When all data is set, restore all trains to 'user-defined'. The data that we set earlier
    # will still be there.
    for env.SelectedWaveTrainIndex in range(nTrains):
        env.WaveType = 'User Defined Spectrum'

    if nTrains == 0:
        raise ValueError(
            "No data exported, no directions with more than the minimum amount of energy"
        )
