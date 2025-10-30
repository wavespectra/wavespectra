===========
wavespectra
===========

.. image:: https://deepwiki.com/badge.svg
   :target: https://deepwiki.com/wavespectra/wavespectra
   :alt: Ask DeepWiki

.. image:: https://zenodo.org/badge/205463939.svg
   :target: https://zenodo.org/badge/latestdoi/205463939
   :alt: DOI

.. image:: https://img.shields.io/github/actions/workflow/status/wavespectra/wavespectra/python-publish.yml
   :target: https://github.com/wavespectra/wavespectra/actions
   :alt: Build Status

.. image:: https://coveralls.io/repos/github/wavespectra/wavespectra/badge.svg?branch=master
   :target: https://coveralls.io/github/wavespectra/wavespectra?branch=master
   :alt: Coverage

.. image:: https://readthedocs.org/projects/wavespectra/badge/?version=latest
   :target: https://wavespectra.readthedocs.io/en/latest/
   :alt: Documentation

.. image:: https://img.shields.io/pypi/v/wavespectra.svg
   :target: https://pypi.org/project/wavespectra/
   :alt: PyPI

.. image:: https://img.shields.io/pypi/dm/wavespectra
   :target: https://pypistats.org/packages/wavespectra
   :alt: Downloads

.. image:: https://anaconda.org/conda-forge/wavespectra/badges/version.svg
   :target: https://anaconda.org/conda-forge/wavespectra
   :alt: Conda

.. image:: https://img.shields.io/pypi/pyversions/wavespectra
   :target: https://pypi.org/project/wavespectra/
   :alt: Python

**Python library for ocean wave spectral data analysis and processing**

Wavespectra is a powerful, open-source Python library built on top of `xarray`_ for working with ocean wave spectral data. It provides comprehensive tools for reading, analysing, manipulating, and visualising wave spectra from various sources including numerical models and buoy observations.

.. _xarray: https://xarray.pydata.org/

Key Features
============

- **Unified Data Model**: Built on xarray with standardised conventions for wave spectral data
- **Extensive I/O Support**: Read/write 15+ formats including WW3, SWAN, ERA5, NDBC, and more
- **Rich Analysis Tools**: 60+ methods for wave parameter calculation and spectral transformations
- **Spectral Partitioning**: Separate wind sea and swell using multiple algorithms (PTM1-5, watershed, wave age)
- **Spectral Construction**: Create synthetic spectra using parametric forms (JONSWAP, TMA, Gaussian, Pierson-Moskowitz)
- **Flexible Visualisation**: Polar spectral plots with matplotlib integration
- **High Performance**: Leverages dask for efficient processing of large datasets
- **Extensible**: Plugin architecture for custom readers and analysis methods

Quick Start
===========

Installation
------------

Install from PyPI:

.. code-block:: console

   # Basic installation
   $ pip install wavespectra

   # Full installation with all optional dependencies
   $ pip install wavespectra[extra]

Or from conda-forge:

.. code-block:: console

   $ conda install -c conda-forge wavespectra

Basic Usage
-----------

.. code-block:: python

   import xarray as xr
   from wavespectra import read_swan

   # Read wave spectra from various formats
   dset = read_swan("spectra.swn")  # SWAN format
   # dset = xr.open_dataset("era5.nc", engine="era5")  # ERA5 reanalysis
   # dset = xr.open_dataset("ww3.nc", engine="ww3")    # WAVEWATCH III

   # Calculate wave parameters
   hs = dset.spec.hs()          # Significant wave height
   tp = dset.spec.tp()          # Peak period
   dm = dset.spec.dm()          # Mean direction
   dspr = dset.spec.dspr()      # Directional spreading

   # Multiple parameters at once
   stats = dset.spec.stats(["hs", "tp", "dm", "dspr"])

   # Spectral transformations
   spectrum_1d = dset.spec.oned()                    # Convert to 1D
   subset = dset.spec.split(fmin=0.05, fmax=0.5)     # Frequency subset
   rotated = dset.spec.rotate(angle=15)              # Rotate directions
   interpolated = dset.spec.interp(freq=new_freq)    # Interpolate

   # Visualisation
   dset.spec.plot(kind="contourf", figsize=(8, 6))   # Polar plot

Working with Different Data Sources
-----------------------------------

.. code-block:: python

   # Numerical model outputs
   ww3_data = xr.open_dataset("ww3_output.nc", engine="ww3")
   swan_data = read_swan("swan_output.swn")
   era5_data = xr.open_dataset("era5_waves.nc", engine="era5")

   # Buoy observations
   ndbc_data = xr.open_dataset("ndbc_data.nc", engine="ndbc")
   triaxys_data = xr.open_dataset("triaxys.nc", engine="triaxys")

   # All use the same analysis interface
   for dataset in [ww3_data, swan_data, era5_data]:
       hs = dataset.spec.hs()
       tp = dataset.spec.tp()

Advanced Analysis
-----------------

Spectral Partitioning
~~~~~~~~~~~~~~~~~~~~~

Separate spectra into wind sea and swell components using various methods:

.. code-block:: python

   # PTM1: Watershed partitioning with wind sea identification
   partitions = dset.spec.partition.ptm1(
       wspd=dset.wspd, wdir=dset.wdir, dpt=dset.dpt, swells=2
   )
   
   # PTM3: Simple ordering by wave height (no wind/depth needed)
   partitions = dset.spec.partition.ptm3(parts=3)
   
   # PTM4: Wave age criterion to separate wind sea from swell
   partitions = dset.spec.partition.ptm4(
       wspd=dset.wspd, wdir=dset.wdir, dpt=dset.dpt, agefac=1.7
   )
   
   # PTM1_TRACK: Track partitions from unique wave systems over time
   # Useful for following the evolution of individual swell events
   partitions = dset.spec.partition.ptm1_track(
       wspd=dset.wspd, wdir=dset.wdir, dpt=dset.dpt, swells=2
   )

Spectral Construction
~~~~~~~~~~~~~~~~~~~~~

Create synthetic spectra from parametric forms:

.. code-block:: python

   from wavespectra.construct.frequency import jonswap, tma, gaussian
   from wavespectra.construct.direction import cartwright
   from wavespectra.construct import construct_partition
   
   # Create JONSWAP spectrum for developing seas
   freq = np.arange(0.03, 0.4, 0.01)
   spectrum = jonswap(freq=freq, hs=2.5, fp=0.1, gamma=3.3)
   
   # Create TMA spectrum for finite depth
   spectrum_shallow = tma(freq=freq, hs=2.0, fp=0.1, dep=15)
   
   # Create 2D spectrum by combining frequency and directional components
   dir = np.arange(0, 360, 10)
   spectrum_2d = jonswap(freq=freq, hs=2.5, fp=0.1) * cartwright(dir=dir, dm=270, dspr=30)
   
   # Or use construct_partition for a complete 2D spectrum
   spectrum_2d = construct_partition(
       freq_name="jonswap",
       dir_name="cartwright",
       freq_kwargs={"freq": freq, "hs": 2.5, "fp": 0.1, "gamma": 3.3},
       dir_kwargs={"dir": dir, "dm": 270, "dspr": 30}
   )

Spectral Fitting
~~~~~~~~~~~~~~~~

.. code-block:: python

   # Fit parametric forms to existing spectra
   jonswap_params = dset.spec.fit_jonswap()          # Fit JONSWAP spectrum

Wave Physics
~~~~~~~~~~~~

.. code-block:: python

   # Calculate wave physics parameters
   celerity = dset.spec.celerity(depth=50)           # Wave speed
   wavelength = dset.spec.wavelen(depth=50)          # Wavelength
   stokes_drift = dset.spec.uss()                    # Stokes drift


Data Requirements
=================

Wavespectra expects xarray objects with specific coordinate and variable naming:

**Required coordinates:**

- ``freq``: Wave frequency in Hz
- ``dir``: Wave direction in degrees (for 2D spectra)

**Required variables:**

- ``efth``: Wave energy density in m²/Hz/degree (2D) or m²/Hz (1D)

**Optional variables:**

- ``wspd``: Wind speed in m/s
- ``wdir``: Wind direction in degrees
- ``dpt``: Water depth in metres

Supported Formats
=================

Input and Output Formats
------------------------

- **Wave Models**: WAVEWATCH III, SWAN, WWM, FUNWAVE, OrcaFlex
- **Reanalysis**: ERA5, ERA-Interim, ECMWF
- **Observations**: NDBC, TRIAXYS, Spotter, Octopus, AWAC
- **Generic**: NetCDF, JSON, CSV

Documentation
=============

Full documentation is available at `wavespectra.readthedocs.io`_

- `Installation Guide`_
- `Quick Start Tutorial`_
- `Spectral Construction`_
- `API Reference`_
- `Example Gallery`_

.. _wavespectra.readthedocs.io: https://wavespectra.readthedocs.io/en/latest/
.. _Installation Guide: https://wavespectra.readthedocs.io/en/latest/install.html
.. _Quick Start Tutorial: https://wavespectra.readthedocs.io/en/latest/quickstart.html
.. _Spectral Construction: https://wavespectra.readthedocs.io/en/latest/construction.html
.. _API Reference: https://wavespectra.readthedocs.io/en/latest/api.html
.. _Example Gallery: https://wavespectra.readthedocs.io/en/latest/gallery.html

Development
===========

Contributing
------------

We welcome contributions! Please see our `Contributing Guide`_ for details.

.. _Contributing Guide: https://wavespectra.readthedocs.io/en/latest/contributing.html

Development Installation
------------------------

.. code-block:: console

   $ git clone https://github.com/wavespectra/wavespectra.git
   $ cd wavespectra
   $ pip install -e .[extra,test,docs]

Running Tests
-------------

.. code-block:: console

   $ pytest tests

Building Documentation
----------------------

.. code-block:: console

   $ make docs

Citation
========

If you use wavespectra in your research, please cite:

.. code-block:: bibtex

   @software{wavespectra,
     author = {Guedes, Rafael and Durrant, Tom and de Bruin, Ruben and Perez, Jorge and Iannucci, Matthew and Delaux, Sebastien and Harrington, John and others},
     title = {wavespectra: Python library for ocean wave spectral data},
     url = {https://github.com/wavespectra/wavespectra},
     doi = {10.5281/zenodo.15238968}
   }

Licence
=======

This project is licenced under the MIT Licence - see the `LICENSE`_ file for details.

.. _LICENSE: LICENSE.txt

Support
=======

- **Documentation**: `wavespectra.readthedocs.io`_
- **Issues**: `GitHub Issues`_
- **Discussions**: `GitHub Discussions`_

.. _GitHub Issues: https://github.com/wavespectra/wavespectra/issues
.. _GitHub Discussions: https://github.com/wavespectra/wavespectra/discussions
