"""Orcaflex output plugin - using orcaflex API."""
import numpy as np

def to_orcaflex(self, model, minEnergy = 1e-6):
    """Writes the first spectrum to and orcaflex model

    Args:
        - model : orcaflex model (OrcFxAPI.model instance)

    """

    dirs = np.array(self.dir.values)
    freqs = np.array(self.freq.values)

    data = np.array(self.efth.values)

    if len(data.shape) > 2:
        data = data[:,:,0]

    ddir = self.dd

    nTrains = 0
    env = model.environment  #alias

    for dir in dirs:
        i_dir = np.where(dirs == dir)
        e = data[:,i_dir].flatten()
        E = ddir * e

        if np.sum(E) <= minEnergy:
            continue

        nTrains+=1

        env.NumberOfWaveTrains = nTrains
        env.SelectedWaveTrain = f'Wave{nTrains}'
        env.WaveDirection = np.mod(90-dir + 180, 360) # convert from coming from to going to and from compass to ofx
        env.WaveType = 'User Defined Spectrum'

        # interior points in the spectrum with zero energy are not allowed by orcaflex
        iFirst = np.where(E>0)[0][0]
        iLast = np.where(E>0)[0][-1]

        for i in range(iFirst, iLast):
            if E[i]<1e-10:
                E[i] = 1e-10

        if iFirst > 0:
            iFirst -=1
        if iLast < len(E)-2:
            iLast += 1

        env.WaveNumberOfUserSpectralPoints = len(E[iFirst:iLast])
        env.WaveSpectrumS = E[iFirst:iLast]
        env.WaveSpectrumFrequency = freqs[iFirst:iLast]  # convert o rad/s



