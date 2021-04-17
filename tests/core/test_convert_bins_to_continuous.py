from wavespectra.construct import ochihubble
from wavespectra.core.utils import bins_from_frequency_grid
import matplotlib.pyplot as plt
import numpy as np

freqs = np.sort(1.5 / 1.2 ** np.arange(0, 25))

demo = ochihubble(
    hs=[1, 1.2],
    tp=[3, 20],
    dp=[180, 180],
    L=[1, 1],
    freqs=freqs,
    dspr=[0, 0],
).spec.oned()

# plot the input spectrum using bins

# plot as bins
left, right, width, center = bins_from_frequency_grid(freqs)
for l, r, d in zip(left, right, demo.data):
    plt.plot([l, l, r, r], [0, d, d, 0],lw=1)

demo.spec.from_bins_to_continuous()

plt.plot(demo.freq, demo.data,'k.-')

plt.xlim([0, 0.12])
plt.show()



"""

# Just some shape, please ignore the code - we just need something that looks like a wave-spectrum
edges = np.sort(1.5 / 1.2 ** np.arange(0, 25))
left = edges[:-1]
right = edges[1:]
widths = right - left
datapoints = 0.5 * (left + right)

bin_values = ochihubble(
    hs=[1, 1.2],
    tp=[3, 20],
    dp=[180, 180],
    L=[1, 1],
    freqs=0.5 * (right + left),
    dspr=[0, 0],
).spec.oned()

# what we have
# left
# right
# width
# bin_value

for l, r, d in zip(left, right, bin_values):
    plt.plot([l, l, r, r], [0, d, d, 0])

plt.plot(datapoints, bin_values,'kx')
m0_bins = np.sum(widths * bin_values).values

plt.title(f"Input bins\n$m_0$ = {m0_bins:.2f}")

plt.show()

# Convert to continuous function
# Calculate the values at the internal edges

edge_internal = edges[1:-1]
f_left = datapoints[:-1]
f_right = datapoints[1:]
s_left = bin_values[:-1].values
s_right = bin_values[1:].values

dfl = edge_internal - f_left
dfr = f_right - edge_internal

s_edge_internal = (dfl * s_left + dfr * s_right) / (dfr + dfl)

# values at the outer edges are zero
s_edge = np.array([0, *s_edge_internal, 0], dtype=float)

plt.plot(edges, s_edge, 'r.')
# plt.show()

# move the data-points such that the energy per bins stays constant
#
# Energy in an original bin = bin_value * widths
# Energy in the continuous spectrum in the same area as the bin
#
# the spectral density at the edges is s_edge
# the spectral density at the locations of the original datapoint can now be calculated
#
# bin_value * width =
# 0.5 * (s_left_edge + s_datapoint) * (datapoint - left_edge) +
# 0.5 * (s_right_edge + s_datapoint) * (right_edge - datapoint) +

e0 = bin_values.values * widths
d_left = datapoints - left
d_right = right - datapoints

s_left = s_edge[:-1]
s_right = s_edge[1:]

# e0 =
# 0.5 * (s_left_edge + s_datapoint) * d_left +
# 0.5 * (s_right_edge + s_datapoint) * d_right
# =
# 0.5*s_left_edge * d_left + 0.5*s_datapoint * d_left + 0.5 * s_right*edge*d_right + 0.5 * s_datapoint * d_right
#
# 0.5 * s_datapoint ( d_left + d_right ) = e0 - 0.5 * s_left_edge * d_left - 0.5 * s_right_edge * d_right

s_datapoint = (e0 - 0.5 * s_left * d_left - 0.5 * s_right * d_right) / (0.5 * (d_left + d_right))

plt.plot(datapoints, s_datapoint, 'g*')


# we may have gotten datapoints below zero
# there are ways to solve this by introducing additional frequency points
# but that is undesirable because when dealing with multiple spectra or 2d spectra we really want
# to keep the frequency axis consistent.
#
# So set all values below zero to zero
# and then
# Scale the whole spectrum such that the original energy is maintained

s_datapoint = np.maximum(s_datapoint, 0)  # note, not max!

c_freqs = np.append(edges, datapoints)
s = np.append(s_edge, s_datapoint)
inds = np.argsort(c_freqs)

c_freqs = c_freqs[inds]
s = s[inds]

plt.plot(c_freqs, s, label = 'continuous spectrum before scaling')

m0_cont = -np.trapz(c_freqs, s)
scale = m0_bins / m0_cont

s *= scale

plt.plot(c_freqs, s, label = 'continuous spectrum after scaling')

plt.show()



"""
