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


# plot as bins
left, right, width, center = bins_from_frequency_grid(demo.freq)
for l, r, d in zip(left, right, demo.data):
    plt.plot([l, l, r, r], [0, d, d, 0],lw=1)

d1d = demo.spec.from_bins_to_continuous()

plt.plot(d1d.freq, d1d.data,'k.-')

plt.xlim([0, 0.12])
plt.show()
