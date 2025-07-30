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

Wavespectra is a powerful, open-source Python library built on top of `xarray`_ for working with ocean wave spectral data [1](#1-0) . It provides comprehensive tools for reading, analysing, manipulating, and visualising wave spectra from various sources including numerical models and buoy observations.

.. _xarray: https://xarray.pydata.org/

Key Features
============

- **Unified Data Model**: Built on xarray with standardised conventions for wave spectral data
- **Extensive I/O Support**: Read/write 15+ formats including WW3, SWAN, ERA5, NDBC, and more
- **Rich Analysis Tools**: 60+ methods for wave parameter calculation, spectral partitioning, construction, and transformations
- **Flexible Visualisation**: Polar plots, time series, and spatial maps with matplotlib integration
- **High Performance**: Leverages dask for efficient processing of large datasets
- **Extensible**: Plugin architecture for custom readers and analysis methods

Quick Start
===========

Installation
------------

Install from PyPI [2](#1-1) :

.. code-block:: console

   # Basic installation
   $ pip install wavespectra

   # Full installation with all optional dependencies
   $ pip install wavespectra[extra]

Or from conda-forge [3](#1-2) :

.. code-block:: console

   $ conda install -c conda-forge wavespectra

Basic Usage
-----------

.. code-block:: python

   import xarray as xr
   import numpy as np
   from wavespectra import read_swan
   from wavespectra.specarray import SpecArray
   from wavespectra.specdataset import SpecDataset

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

.. code-block:: python

   # Spectral partitioning
   partitions = dset.spec.partition.ptm1(wspd=10, wdir=270, parts=3)

   # Wave physics calculations
   celerity = dset.spec.celerity(depth=50)           # Wave speed
   wavelength = dset.spec.wavelen(depth=50)          # Wavelength
   stokes_drift = dset.spec.uss()                    # Stokes drift

   # Spectral fitting
   jonswap_params = dset.spec.fit_jonswap()          # Fit JONSWAP spectrum

Data Requirements
=================

Wavespectra expects xarray objects with specific coordinate and variable naming [4](#1-3) :

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

Input and output Formats
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
- `API Reference`_
- `Example Gallery`_

.. _wavespectra.readthedocs.io: https://wavespectra.readthedocs.io/en/latest/
.. _Installation Guide: https://wavespectra.readthedocs.io/en/latest/install.html
.. _Quick Start Tutorial: https://wavespectra.readthedocs.io/en/latest/quickstart.html
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
------------- [5](#1-4) 

.. code-block:: console

   $ pytest tests

Building Documentation
---------------------- [6](#1-5) 

.. code-block:: console

   $ make docs

Citation
========

If you use wavespectra in your research, please cite:

.. code-block:: bibtex


   @software{wavespectra,
     author = {Guedes, Rafael and Durrant, Tom and de Bruin, Ruben and Perez, Jorge and Iannucci, Matthew and seboceanum and Harrington, John and others},
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

History
=======

Wavespectra was originally developed at `Metocean Solutions`_ and was open-sourced in April 2018 [7](#1-6) . The project transitioned to community development in July 2019 under the `wavespectra GitHub organisation`_.

.. _Metocean Solutions: https://www.metocean.co.nz/
.. _wavespectra GitHub organisation: https://github.com/wavespectra

