import numpy as np
from wavespectra.construct.helpers import (spread, check_coordinates,
    arrange_inputs, make_dataset)

gamma = lambda x: np.sqrt(2.*np.pi/x) * ((x/np.exp(1.)) * np.sqrt(x*np.sinh(1./x)))**x

def ochihubble(hs, tp, L, dp, dspr, freqs = np.arange(0.02,1.0,0.02),
               dirs=np.arange(0,360,10), coordinates=[("part", [0,1])], sumpart=True):
    """OchiHubble construct function."""
    check_coordinates(hs, coordinates)

    #Arrange inputs
    hs_m, tp_m, l_m, dp_m, dspr_m = arrange_inputs(hs, tp, L, dp, dspr)
    w = 2 * np.pi * freqs.reshape((-1, 1))

    #Create 1D spectra
    w0 = 2 * np.pi / tp_m
    B = np.maximum(l_m, 0.01) + 0.25
    A = 0.5 * np.pi * hs_m**2 * ((B*w0**4)**l_m/gamma(l_m))
    a = np.minimum((w0/w)**4, 100.)
    S = A * np.exp(-B*a) / (np.power(w, 4.*B))

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