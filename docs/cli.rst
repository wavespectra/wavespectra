.. image:: _static/wavespectra_logo.png
    :width: 150 px
    :align: right

===
CLI
===

Wavespectra has a command line interface to access some of the functionality of the
library. For example, to convert a SWAN ASCII spectra file into the wavespectra
NetCDF format:

.. code-block:: console

    $ wavespectra convert format spectra.spec swan spectra.nc netcdf

or to write a NetCDF file with integrated parameters calculated from a WAVEWATCH III
spectra file:

.. code-block:: console

    $ wavespectra convert stats ww3file.nc ww3 stats.nc -p hs -p tp -p dpm

Top level commands
------------------

.. command-output:: wavespectra --help


Convert
-------

.. command-output:: wavespectra convert --help

Convert file format

.. command-output:: wavespectra convert format --help

Convert to integrated spectral parameters

.. command-output:: wavespectra convert stats --help


Reconstruct
-----------

.. command-output:: wavespectra reconstruct --help

Partition and reconstruct spectra from file

.. command-output:: wavespectra reconstruct spectra --help
