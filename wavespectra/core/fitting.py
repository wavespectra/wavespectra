import numpy as np
from scipy import optimize

from wavespectra.fit.jonswap import np_jonswap


PARAM_JONSWAP_START = [0.1, 12.0, 1.0, 0.1, 5.0, 1.0]


def fit_jonswap_spectra(ef, freq, hs, tp, gamma, param_start=PARAM_JONSWAP_START):
    """Wrapper to return only spectrum from _fit_jonswap to run as ufunc."""
    param_active = [True, True, True, hs, tp, gamma]
    return _fit_jonswap(ef, freq, param_active, param_start=param_start)[0]


def fit_jonswap_params(ef, freq, hs, tp, gamma, param_start=PARAM_JONSWAP_START):
    """Wrapper to return only parameters from _fit_jonswap to run as ufunc."""
    param_active = [True, True, True, hs, tp, gamma]
    return _fit_jonswap(ef, freq, param_active, param_start=param_start)[1][-1]


def _fit_jonswap(ef, freq, param_active, param_start=PARAM_JONSWAP_START):
    """Fit Jonswap for sea and swell partitions.

    Args:
        - ds (DataArray): Spectrum to fit in wavespectra conventions.
        - ef (1darray): Frequency spectrum array E(f) (m2/Hz).
        - freq (1darray): Frequency array (Hz).
        - param_active (list): Jonswap fixed parameters for the sea and swell parts
          [hs1, tp1, gamma1, hs2, tp2, gamma2] or `True` if fitting them.
        - param_start (list): Jonswap start parameters for the sea and swell parts
          [hs1, tp1, gamma1, hs2, tp2, gamma2].

    Returns:
        - dsout (DataArray): Fitted Jonswap spectrum.
        - params (list): Fitted Jonswap parameters for the sea and swell parts
          [hs1, tp1, gamma1, hs2, tp2, gamma2].

    """
    def func(x0, ef, freq, params):
        """Function with error to mimimise in the Jonswap estimate.

        Args:
            - x0 (list): Start values for the Jonswap parameters to minimise for
              the sea and swell parts [hs1, tp1, gamma1, hs2, tp2, gamma2].
            - ef (1darray): Frequency spectra `E(f)` (m2/Hz).
            - freq (1darray): Frequency array (Hz).
            - params (list): Jonswap fixed parameters for the sea and swell parts
              [hs1, tp1, gamma1, hs2, tp2, gamma2] or `True` if fitting them.

        Returns:
            - err (list): Errors for each Jonswap parameter.

        """
        # Map only active arguments: params - True or FixedValue  # [Hs1, Tp1, gamma1, Hs2, Tp2, gamma2]
        for i, v in enumerate(params):
            if v != True:
                x0[i] = params[i]

        # Make a new spectra
        hs1, tp1, gamma1 = x0[:3]
        hs2, tp2, gamma2 = x0[3:]
        ef1 = np_jonswap(freq=freq, hs=hs1, fp=1/tp1, gamma=gamma1)
        ef2 = np_jonswap(freq=freq, hs=hs2, fp=1/tp2, gamma=gamma2)

        # Calculate error
        err = np.sqrt(np.sum(np.power(ef1 + ef2 - ef, 2)))

        if x0[0] < 0:
            err += 10
        if x0[3] < 0:
            err += 10
        if x0[1] < 3:
            err += 10
        if x0[4] < 3:
            err += 10
        if x0[2] < 0.6:
            err += 10
        if x0[5] < 0.6:
            err += 10

        return err

    # Run the fitting routine
    res = optimize.minimize(
        fun=func,
        x0=param_start,
        args=(ef, freq, param_active),
        tol=1e-4,
        method="Nelder-Mead",
        options={"adaptive": True}
    )
    for i,v in enumerate(param_active):
        if v != True:
            res.x[i] = param_active[i]

    hs1, tp1, gamma1 = res.x[:3]
    hs2, tp2, gamma2 = res.x[3:]
    ef1 = np_jonswap(freq=freq, hs=hs1, fp=1/tp1, gamma=gamma1)
    ef2 = np_jonswap(freq=freq, hs=hs2, fp=1/tp2, gamma=gamma2)

    ef_out = ef1 + ef2

    return ef_out, res.x


# def fit_jonswap(ds, param_start=PARAM_JONSWAP_START):
#     """Nonlinear fit Jonswap spectra.

#     Args:
#         - ds (DataArray): Spectrum to fit in wavespectra conventions.
#         - param_start (list): Jonswap start parameters for the sea and swell parts
#         [hs1, tp1, gamma1, hs2, tp2, gamma2].

#     Returns:
#         - dsout (DataArray): Fitted Jonswap spectrum.
#         - param (list): Fitted Jonswap parameters for the sea and swell parts
#         [hs1, tp1, gamma1, hs2, tp2, gamma2].

#     """
#     # Fit sea & swell separately
#     fcut = 0.1
#     dswell = ds.spec.split(fmax=fcut)
#     dsea = ds.spec.split(fmin=fcut)

#     sSwell, pSwell = _fit_jonswap(dswell, [True, True, True, 0, 3, 2], param_start)
#     sSea, pSea = _fit_jonswap(dsea, [pSwell[0], pSwell[1], pSwell[2], True, True, True], param_start)

#     # Fit both together Hs, gamma
#     param_start = [pSwell[0], pSwell[1], pSwell[2], pSea[3], pSea[4], pSea[5]]
#     dsout, parm = _fit_jonswap(ds, [True, True, True, True, True, True], param_start)

#     # Rescale final combined spectrum
#     dsout = dsout * (ds.spec.hs() / dsout.spec.hs())**2

#     return dsout, parm


if __name__ == "__main__":
    from wavespectra import fit_jonswap as make_jonswap

    hs = 1.0
    fp = 1 / 12.0
    gamma = 1.1
    sigma_a = 0.07
    sigma_b = 0.09

    f = np.arange(0.01, 0.51, 0.01)
    ds = make_jonswap(freq=f, fp=fp, gamma=gamma, sigma_a=sigma_a, sigma_b=sigma_b, hs=hs)

    # s, p = fit_jonswap_spectrum_auto()

    # dsout, params = fit_jonswap(ds)
    dsout, params = fit_jonswap(ds.values, ds.freq.values, [True, True, True, hs, (1/fp)-1, 1.0], param_start=PARAM_JONSWAP_START)

    print([p for p in params])

    import ipdb; ipdb.set_trace()
