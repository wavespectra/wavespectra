=======
History
=======

Wavespectra is an open source project that was started at MetOcean Solutions and open
sourced in April 2018. In July 2019 it was moved into the wavespectra github open
source organisation and transitioned into a fully community developed project. This
changelog covers the release history since v3.0 when wavespectra was open-sourced.


********
Releases
********


3.7.2 (2021-01-12)
~~~~~~~~~~~~~~~~~~

New Features
------------
* Handle ndbc spectra files with no minutes column (`PR25 <https://github.com/wavespectra/wavespectra/pull/25>`_).
* Writers `to_swan`_ and `to_octopus`_ now deal with extra non-supported dimensions.

Internal Changes
----------------
* Stop fixing pandas and xarray versions.
* Remove attrdict dependency.
* Define `_FillValue` in `to_netcdf`_.

Bug Fixes
---------
* Fix bug in sel with `"nearest"` option.
* Ensure last time chunk is written in `to_swan`_ when the dataset time size is not divisible by ntime (`GH20 <https://github.com/wavespectra/wavespectra/issues/24>`_).


.. _`to_netcdf`: https://github.com/wavespectra/wavespectra/blob/master/wavespectra/output/netcdf.py


3.7.1 (2020-08-26)
~~~~~~~~~~~~~~~~~~

Internal Changes
----------------
* Optimise `to_swan`_ (over 100x improvements when writing very large spectra).
* Optimise `to_octopus`_ (over 10x improvements when writing very large spectra).
* Allow loading time chunks when writing swan and octopus files.

.. _`to_swan`: https://github.com/wavespectra/wavespectra/blob/master/wavespectra/output/swan.py
.. _`to_octopus`: https://github.com/wavespectra/wavespectra/blob/master/wavespectra/output/octopus.py


3.7.0 (2020-07-16)
~~~~~~~~~~~~~~~~~~

New Features
------------
* New json reader and writer (`PR21 <https://github.com/wavespectra/wavespectra/pull/21>`_).

Internal Changes
----------------
* Raise exception when trying to compute directional methods on 1d, frequency spectra.


3.6.5 (2020-07-10)
~~~~~~~~~~~~~~~~~~

Bug Fixes
---------
* Fix bug in sel methods.


3.6.4 (2020-06-29)
~~~~~~~~~~~~~~~~~~

Bug Fixes
---------
* Ensure yml config is shipped with distribution.


3.6.3 (2020-06-28)
~~~~~~~~~~~~~~~~~~

Internal Changes
----------------
* Increase time resolution in netcdf outptu from to_netcdf.


3.6.2 (2020-06-28)
~~~~~~~~~~~~~~~~~~

Internal Changes
----------------
* Make netcdf packing work for datasets in zarr format.


3.6.1 (2020-06-28)
~~~~~~~~~~~~~~~~~~

Internal Changes
----------------
* Packing output netcdf files as int32 dtype by default.


3.6.0 (2020-06-27)
~~~~~~~~~~~~~~~~~~

New Features
------------
* New method to construct spectra from NDBC buoy data (`PR17 <https://github.com/wavespectra/wavespectra/pull/17>`_).
* New method to output spectra in native WW3 format.

Bug Fixes
---------
* Fix bug with selecting circular longitudes in different conventions (`GH20 <https://github.com/wavespectra/wavespectra/issues/20>`_).
* Ensure directions in coming-from convention in read_era5 (`PR18 <https://github.com/wavespectra/wavespectra/pull/18>`_).
* Fix radian convertions in read_era5 (`PR19 <https://github.com/wavespectra/wavespectra/pull/19>`_).
* Fix coordinate values assignment errors with xarray>=0.15.1 (`GH16 <https://github.com/wavespectra/wavespectra/issues/16>`_).
* Ensure coordinates attributes are kept with certain readers.

deprecation
-----------
* Deprecated legacy `read_ww3_msl` reader.
* Deprecated `read_dictionary` in favour of using xarray's `to_dict`_ and `from_dict`_ methods.

.. _`to_dict`: http://xarray.pydata.org/en/stable/generated/xarray.DataArray.to_dict.html
.. _`from_dict`: http://xarray.pydata.org/en/stable/generated/xarray.DataArray.from_dict.html


Internal Changes
----------------
* Remove curly brackets from units.
* Remove original variable attributes from files hidden with underscores (`_units` and `_variable_name`).
* Remove xarray version limitation to <0.15.0.


3.5.3 (2020-04-14)
~~~~~~~~~~~~~~~~~~
Fix xarray version until breaking changes with 0.15.1 are taken care of.

Bug Fixes
---------
* Avoid index duplication when merging datasets in to_octopus function.

Internal Changes
----------------
* Fix xarray at 0.15.0 for now as 0.15.1 introduces many breaking changes.


3.5.2 (2020-03-09)
~~~~~~~~~~~~~~~~~~

New Features
------------
* New method `read_era5`_ to read spectra in ERA5 format by `John Harrington`_.
* New method `read_wavespectra`_ to read files already in wavespectra convention.

.. _`read_era5`: https://github.com/wavespectra/wavespectra/blob/master/wavespectra/input/era5.py
.. _`read_wavespectra`: https://github.com/wavespectra/wavespectra/blob/master/wavespectra/input/wavespectra.py
.. _`John Harrington`: https://github.com/JohnCHarrington


3.5.1 (2019-12-12)
~~~~~~~~~~~~~~~~~~

Bug Fixes
---------
* Import accessors within try block in __init__.py so install won't break.

Internal Changes
----------------
* Implemented coveralls.
* Added some more tests.


3.5.0 (2019-12-09)
~~~~~~~~~~~~~~~~~~
**The first PyPI release from new** `wavespectra`_ **github organisation.**

Breaking Changes
----------------
* Drop support for Python 2.
* Drop support for Python < 3.6.

New Features
------------
* Add method in SpecDataset accessor to plot polar wave spectra, api borrowed from `xarray`_.
* New `sel` method in SpecDataset accessor to select sites using different methods.
* Support for `zarr`_ wave spectra datasets from either local or remote sources.
* New `read_spotter` function to read spectra from Spotter file format, currently only reading as 1D.
* Add `read_dataset` function to convert existing dataset from unknown file into SpecDataset.
* Python Notebooks split into a new `notebooks`_ repository within the `wavespectra`_ organisation.
* New branch `pure-python`_ with fortran watershed algorithm replaced by python. This code is ~3x slower
  than the fortran one but it is easier to install particularly if the system does not have fortran
  compiler. We will make an effort to keep this branch in sync with Master.
* Redefined autodocs.

.. _`pure-python`: https://github.com/wavespectra/wavespectra/tree/pure-python

Bug Fixes
---------
* Consolidate history to link to github commits from all contributors.
* Fix error in `partition` with dask array not supportting item assignment.
* Fix docs building, currently working from `pure-python` branch due to gfortran dependency.

Internal Changes
----------------
* Decouple file reading from accessor definition in input functions so existing datasets can be converted.
* Compute method `_twod` lazily.
* Replace drop calls to fix deprecation warnings.
* Consolidate changelog in history file.
* Building with travis and tox.
* Adopt `black`_ code formatting.
* Set up flake8.


3.4.0 (2019-03-28)
~~~~~~~~~~~~~~~~~~
**The last PyPI release from old metocean github organisation.**

New Features
------------
* Add support to Python 3.


3.3.1 (2019-03-19)
~~~~~~~~~~~~~~~~~~

New Features
------------
* Support SWAN Cartesian locations.
* Support energy unit in SWAN ASCII spectra.


3.3.0 (2019-02-21)
~~~~~~~~~~~~~~~~~~

New Features
------------
* Add `dircap_270` option in `read_swan`.

Bug Fixes
---------
* Ensure lazy computations in `swe` method.

Internal Changes
----------------
* Remove `inplace` calls that will deprecate in xarray.


3.2.5 (2019-01-25)
~~~~~~~~~~~~~~~~~~

Bug Fixes
---------
* Ensure datasets are loaded lazily in `read_swan` and `read_wwm`.


3.2.4 (2019-01-23)
~~~~~~~~~~~~~~~~~~

Bug Fixes
---------
* Fix tp-smooth bug caused by float32 dtype.


3.2.3 (2019-01-08)
~~~~~~~~~~~~~~~~~~

New Features
------------
* Function `read_triaxys` to read spectra from TRIAXYS file format.

Bug Fixes
---------
* Fix bug with frequency and energy units in `read_wwm`.


3.2.2 (2018-12-04)
~~~~~~~~~~~~~~~~~~

Bug Fixes
---------
* Ensure dataset from swan netcdf has site coordinate.


3.2.1 (2018-11-14)
~~~~~~~~~~~~~~~~~~

New Features
------------
* Function `read_wwm` to read spectra from WWM model format.

Bug Fixes
---------
* Convert direction to degree in `read_ncswan`.


3.2.0 (2018-11-04)
~~~~~~~~~~~~~~~~~~

New Features
------------
* Function `read_ncswan` to read spectra from SWAN netcdf model format.

Bug Fixes
---------
* Ensure lazy computation in `uv_to_spddir`.

Internal changes
----------------
* Unify library PyPI release versions. 


3.1.4 (2018-08-29)
~~~~~~~~~~~~~~~~~~

Bug Fixes
---------
* Fix bug in `read_swans` when handling swan bnd files with `ntimes` argument.


3.1.3 (2018-07-27)
~~~~~~~~~~~~~~~~~~

Changes
-------
* Use 10m convention in default wind standard names.


3.1.2 (2018-07-05)
~~~~~~~~~~~~~~~~~~

Changes
-------
* Adjust default standard name for `dm`.

Bug Fixes
---------
* Fix renaming option in `stats` method.


3.1.1 (2018-05-17)
~~~~~~~~~~~~~~~~~~

Bug Fixes
---------

New Features
------------
* Allow choosing maximum number of partitions in `partition` method.


3.1.0 (2018-05-09)
~~~~~~~~~~~~~~~~~~

New Features
------------
* Function to read spectra in cf-json formatting.

Bug Fixes
---------
* Fix but in `read_swan` when files have no timestamp.


3.0.2 (2018-05-03)
~~~~~~~~~~~~~~~~~~

Bug Fixes
---------
* Ensure data is not loaded into memory in `read_ww3`.


3.0.1 (2018-04-28)
~~~~~~~~~~~~~~~~~~

New Features
------------
* Sphinx autodoc.
* Method `read_dictionary` to define SpecDataset from python dictionary.
* Set pytest as the testing framework and add several new testings.
* Add notebooks.

Bug Fixes
---------
* Get rid of left over `freq` coordinate in `hs` method.
* Fix calculation in `_peak` method.
* Stop misleading warning in `tp` method.
* Fix to `hs` method.

Internal Changes
----------------
* Replace obsolete sort method by `xarray`_'s sortby.
* Falster calculation in `tp`.
* Improvements to SpecDataset wrapper.


3.0 (2018-03-05)
~~~~~~~~~~~~~~~~~~
**This major release marks the migration from the predecessor** `pyspectra` **library,
as well as the open-sourcing of wavespectra and first PyPI release.**

New Features
------------
* Library restructured with plugins input / output modules .
* New `_peak` method to return the true peak instead of the maxima.
* Making reading functions available at module level.

Bug Fixes
---------
* Ensure slicing won't break due to precision (xarray bug).

Internal Changes
----------------
* Rename package.



.. _`MetOcean Solutions`: https://www.metocean.co.nz/
.. _`metocean`: https://github.com/metocean/wavespectra
.. _`wavespectra`: https://github.com/wavespectra
.. _`notebooks`: https://github.com/wavespectra/notebooks
.. _`xarray`: https://xarray.pydata.org/en/latest/
.. _`black`: https://black.readthedocs.io/en/stable/
.. _`zarr`: https://zarr.readthedocs.io/en/stable/