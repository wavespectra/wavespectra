from wavespectra.construct import ochihubble
import matplotlib.pyplot as plt
import numpy as np


# Just some shape, please ignore the code - we just need something that looks like a wave-spectrum
edges = np.sort(1.5 / 1.2 ** np.arange(0, 25))
left = edges[:-1]
right = edges[1:]
widths = right - left
datapoints = 0.5 * (left + right)

# bin_values = ochihubble(
#     hs=[1, 1.2],
#     tp=[3, 20],
#     dp=[180, 180],
#     L=[1, 1],
#     freqs=0.5 * (right + left),
#     dspr=[0, 0],
# ).spec.oned()

ochihubble(
    hs=[1, 1.2],
    tp=[3, 20],
    dp=[180, 180],
    L=[1, 1],
    freqs=0.5 * (right + left),
    dspr=[0, 0],
).spec.oned().spec.from_bins_to_continuous()


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

# plt.show()

# Convert to continuous function
# Calculate the values at the internal edges


update_rate = 0.5   # new iteration = update_rate *guess + (1-update_rate) * actual_values
s_datapoint = bin_values.values # initial
last_max_update = 0

for i in range(100):

    edge_internal = edges[1:-1]
    f_left = datapoints[:-1]
    f_right = datapoints[1:]
    s_left = s_datapoint[:-1]
    s_right = s_datapoint[1:]

    dfl = edge_internal - f_left
    dfr = f_right - edge_internal

    # s_edge_internal = (dfl * s_left + dfr * s_right) / (dfr + dfl)
    s_edge_internal = s_left + dfl * (s_right - s_left) / (dfr + dfl)

    # values at the outer edges are zero
    s_edge = np.array([0, *s_edge_internal, 0], dtype=float)

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

    new_estimate = (e0 - 0.5 * s_left * d_left - 0.5 * s_right * d_right) / (0.5 * (d_left + d_right))

    # we may have gotten datapoints below zero
    new_estimate = np.maximum(new_estimate, 0)  # note, not max!

    i_maxchange = np.argmax(np.abs(new_estimate - s_datapoint))
    max_update = (new_estimate - s_datapoint)[i_maxchange] # signed

    change_in_update = max_update - last_max_update
    last_max_update = max_update
    print(f'change in update: {change_in_update}')

    if abs(change_in_update)<1e-5:
        break

    print(f'it {i} , max update {max_update} as point {i_maxchange}')

    s_datapoint = update_rate * new_estimate + (1 - update_rate) * s_datapoint



    # plt.plot(datapoints, s_datapoint, 'g*')



    # scale to original m0
    m0_cont = -np.trapz(datapoints, s_datapoint)
    scale = m0_bins / m0_cont

    s_datapoint *= scale

    if True: # (i % 5 == 0):
        c_freqs = np.append(edges, datapoints)
        s = np.append(s_edge, s_datapoint)
        inds = np.argsort(c_freqs)

        c_freqs = c_freqs[inds]
        s = s[inds]

        plt.plot(c_freqs, s, label=f'iteration {i} ', linewidth = 0.5)

        plt.text(datapoints[5],s_datapoint[5], str(i))


plt.plot(datapoints, s_datapoint, 'k-', label = 'Final')
plt.legend()
plt.show()



