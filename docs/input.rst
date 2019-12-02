=====
Input
=====

.. py:module:: wavespectra.input

Functions to read wave spectra from file into
:py:class:`~wavespectra.specdataset.SpecDataset`.

The input functions allow abstracting away the format the wave spectra data are
stored on disk and loading them into a standard SpecDataset object. The methods
for calculating integrated spectral parameters and writing spectra as different
file formats become available from the ``spec`` namespece.

Reading functions are defined in modules within
:py:mod:`wavespectra.input` subpackage. The functions are imported at the main
module level and can be accessed for instance as:

.. code:: python

    from wavespectra import read_swan

    dset = read_swan('my_swan_file')

The following convention is expected for defining reading functions:

- Funcions for different file types are defined in different modules within
  :py:mod:`wavespectra.input` subpackage.
- Modules are named as `filetype`.py, e.g., ``swan.py``.
- Functions are named as read_`filetype`, e.g., ``read_swan``.

Input functions can also be defined without following this convention. However
they are not accessible from the main module level and need to be imported from
their full module path, for instance:

.. code:: python

    from wavespectra.input.swan import read_hotswan

    dset = read_hotswan('my_swan_hotfiles')


Module level reader functions
-----------------------------

The following input functions are currently available from main module level (e.g., `from wavespectra import read_ww3`):

WW3
~~~

.. autofunction:: wavespectra.read_ww3
   :noindex:

NCSWAN
~~~~~~

.. autofunction:: wavespectra.read_ncswan
   :noindex:

WWM
~~~~~~

.. autofunction:: wavespectra.read_wwm
   :noindex:

WW3-MSL
~~~~~~~

.. autofunction:: wavespectra.read_ww3_msl
   :noindex:

NETCDF
~~~~~~

.. autofunction:: wavespectra.read_netcdf
   :noindex:

DATASET
~~~~~~~

.. autofunction:: wavespectra.read_dataset
   :noindex:

SWAN
~~~~

.. autofunction:: wavespectra.read_swan
   :noindex:

TRIAXYS
~~~~~~~

.. autofunction:: wavespectra.read_triaxys
   :noindex:

SPOTTER
~~~~~~~

.. autofunction:: wavespectra.read_spotter
   :noindex:

OCTOPUS
~~~~~~~

.. autofunction:: wavespectra.read_octopus
   :noindex:

Other functions
~~~~~~~~~~~~~~~
These functions are not accessible from the main module level and need to be
imported from their full module path:

.. autofunction:: wavespectra.input.swan.read_hotswan
   :noindex:

.. autofunction:: wavespectra.input.swan.read_swans
   :noindex:

.. autofunction:: wavespectra.input.swan.read_swanow
   :noindex:
