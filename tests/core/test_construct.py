import sys
import os
import logging
import pytest
import time
import numpy as np
from numpy.testing import assert_array_almost_equal

plot = False
if plot:
    import matplotlib.pyplot as plt

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

# from wavespectra.construct import jonswap, ochihubble


# def integrate_2d_hs(freqs, dirs, S):
#     """Numerical integration of 2D spectrum."""
#     sum1 = np.trapz(S, freqs, axis=0)
#     sum2 = np.trapz(sum1, dirs)
#     return 4 * sum2 ** 0.5


# class TestJonswap(object):
#     def hs(self, tp, alpha, gamma=3.3, df=0.02):
#         """Calculate 1D JONSWAP."""
#         f = np.arange(df, 1.0, df)
#         fp = 1.0 / tp
#         sig = np.where(f <= fp, 0.07, 0.09)
#         r = np.exp(-(f - fp) ** 2.0 / (2 * sig ** 2 * fp ** 2))
#         S = (
#             0.0617
#             * np.array(alpha)
#             * f ** (-5)
#             * np.exp(-1.25 * (f / fp) ** (-4))
#             * gamma ** r
#         )
#         return 4 * (S.sum() * df) ** 0.5


#     def test_jonswap_scalar(self):
#         dset = jonswap(tp=10, dp=90, alpha=0.01)
#         hs = integrate_2d_hs(dset["freq"], dset["dir"], dset["efth"])
#         assert hs == pytest.approx(self.hs(10, 0.01), rel=1e-3)
#         if plot:
#             plt.pcolormesh(dset["freq"], dset["dir"], dset["efth"].T)
#             plt.show()


#     def test_jonswap_series(self):
#         tp = [10, 5]
#         dp = [90, 180]
#         dspr = [25, 40]
#         dset = jonswap(tp, dp, alpha=0.01, dspr=dspr, coordinates=[("time", [0, 0])])
#         for i, spec in enumerate(dset["efth"]):
#             hs = integrate_2d_hs(dset["freq"], dset["dir"], spec)
#             assert hs == pytest.approx(self.hs(tp[i], 0.01), rel=1e-3)
#         if plot:
#             plt.pcolormesh(dset["freq"], dset["dir"], dset["efth"][0].T)
#             plt.show()
#             plt.pcolormesh(dset["freq"], dset["dir"], dset["efth"][1].T)
#             plt.show()


# class TestOchiHubble(object):
#     def gamma(self, val):
#         return (
#             np.sqrt(2.0 * np.pi / val)
#             * ((val / np.exp(1.0)) * np.sqrt(val * np.sinh(1.0 / val))) ** val
#         )

#     def hs(self, hs, tp, l, df=0.02):
#         """Calculate 1D OH."""
#         w = 2 * np.pi * np.arange(df, 1.0, df)
#         S = np.zeros((len(w)))
#         for i, H in enumerate(hs):
#             # Create 1D spectrum
#             w0 = 2 * np.pi / tp[i]
#             B = np.maximum(l[i], 0.01) + 0.25
#             A = 0.5 * np.pi * H ** 2 * ((B * w0 ** 4) ** l[i] / self.gamma(l[i]))
#             a = np.minimum((w0 / w) ** 4, 100.0)
#             S += A * np.exp(-B * a) / (np.power(w, 4.0 * B))
#         return 4 * (S.sum() * df) ** 0.5

#     def test_oh_scalar(self):
#         """Test single set of parameters."""
#         hs_list = [1.0, 1.0]
#         tp_list = [10, 5]
#         L_list = [1, 3]
#         dp_list = [90, 180]
#         dspr_list = [25, 40]
#         dset = ochihubble(hs=hs_list, tp=tp_list, L=L_list, dp=dp_list, dspr=dspr_list)
#         hs = integrate_2d_hs(dset["freq"], dset["dir"], dset["efth"])
#         assert hs == pytest.approx(self.hs(hs_list, tp_list, L_list), rel=1e-3)
#         if plot:
#             plt.pcolormesh(dset["freq"], dset["dir"], dset["efth"].T)
#             plt.show()
