import numpy as np
from wavespectra.construct.helpers import (spread, check_coordinates,
    arrange_inputs, make_dataset)

def jonswap(tp, dp, alpha, gamma=3.3, dspr=20, freqs=np.arange(0.04,1.0,0.02),
            dirs=np.arange(0,360,10), coordinates=[], sumpart=True):
    """Constructs JONSWAP spectra from peak period and peak direction."""
    check_coordinates(tp,coordinates)

    #Arrange inputs
    tp_m, dp_m, alpha_m, gamma_m, dspr_m = arrange_inputs(tp, dp, alpha, gamma, dspr)
    f = freqs.reshape((-1,1))

    #Calculate JONSWAP
    fp = 1.0 / np.array(tp_m)
    sig = np.where(f<=fp, 0.07, 0.09)
    r = np.exp(-(f-fp)**2. / (2 * sig**2 * fp**2))
    S = 0.0617 * alpha_m * f**(-5) * np.exp(-1.25*(f/fp)**(-4)) * gamma_m**r

    #Apply spreading
    G1 = spread(dp_m, dspr_m, dirs)
    spec = S * G1

    # sum partitions
    if sumpart:
        idimpart = [i for i, t in enumerate(coordinates) if t[0]=='part']
        if idimpart:
            spec = np.sum(spec, axis=idimpart[0])
            coordinates.pop(idimpart[0])

    return make_dataset(spec, freqs, dirs, coordinates)