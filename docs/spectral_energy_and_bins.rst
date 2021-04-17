.. image:: _static/wavespectra_logo.png
    :width: 150 px
    :align: right

==========================
Wave spectral definitions
==========================

Some things look trivial but are not. Here comes trouble, brace yourself.

Wave-spectra are continuous by nature. There is no physical reson for discontinuities in the spectral shape (thank you visocity).
This means that the spectral shape s(frequency, dir) will be smooth.

To store a spectral shape it can be discretized into a finite number of points. The true specrtral shape can then be re-constructed by fitting a curve* through these datapoints.

This approach straight forward when constructing wave spectra from an emperical formulation such as ochi-hubble or jonswap.

When measuring waves you don't measure a spectrum. You measure a wave-train: A sequence of water-elevations over time. A spectrum can be constructed by filtering. windowing and FFT.
The result of this analysis is a finite number of frequency bins and the amount of energy in that bin. This results in another way to store spectral data: bins and the amount of energy per bin.


TLDR;
=======
There are two major ways of storing spectral data:

1. As points on a continuous curve
2. As energy bins defined by the center of the bin and the amount of energy in that bin.

Both stored data-formats look identical

.. image:: _static/cont_vs_bins1.png
    :width: 150 px
    :align: right

If the size of the bins is constant then the these two can for all practical purposes be converted to eachother by multiplcation or division by the bin size.

If....





