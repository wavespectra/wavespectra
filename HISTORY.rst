=======
History
=======

Wavespectra is an open source project initially developed while the lead developers
worked at `MetOcean Solutions`_. The code was open sourced in April 2018 and was moved
in July 2019 into the `wavespectra`_ github open source organisation.


********
Releases
********


3.5.0 (YYYY-MM-DD)
~~~~~~~~~~~~~~~~~~
The first PyPI release from new `wavespectra`_ github organisation.

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

Bug Fixes
---------
* Consolidate history to link to github commits from all contributors.
* Fix error in `partition` with dask array not supportting item assignment

Internal Changes
----------------
* Decouple file reading from accessor definition in input functions so existing datasets can be converted.
* Building with travis and tox.
* Adopt `black`_ code formatting.
* Consolidate changelog in history file.


3.4.0 (2019-03-28)
~~~~~~~~~~~~~~~~~~
The last PyPI release from old metocean github organisation.

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

Bug Fixes
---------
* Ensure lazy computations in `swe` method.

New Features
------------
* Add `dircap_270` option in `read_swan`.

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

Bug Fixes
---------
* Fix bug with frequency and energy units in `read_wwm`.

New Features
------------
* Function `read_triaxys` to read spectra from TRIAXYS file format.


3.2.2 (2018-12-04)
~~~~~~~~~~~~~~~~~~

Bug Fixes
---------
* Ensure dataset from swan netcdf has site coordinate.


3.2.1 (2018-11-14)
~~~~~~~~~~~~~~~~~~

Bug Fixes
---------
* Convert direction to degree in `read_ncswan`.

New Features
------------
* Function `read_wwm` to read spectra from WWM model format.


3.2.0 (2018-11-04)
~~~~~~~~~~~~~~~~~~

Bug Fixes
---------
* Ensure lazy computation in `uuv_to_spddir`.

New Features
------------
* Function `read_ncswan` to read spectra from SWAN netcdf model format.


3.1.4 (2018-08-29)
~~~~~~~~~~~~~~~~~~

Bug Fixes
---------
* Fix bug in `read_swans` when handling swan bnd files with `ntimes` argument.




.. _`MetOcean Solutions`: https://www.metocean.co.nz/
.. _`metocean`: https://github.com/metocean/wavespectra
.. _`wavespectra`: https://github.com/wavespectra
.. _`xarray`: https://xarray.pydata.org/en/latest/
.. _`black`: https://black.readthedocs.io/en/stable/
.. _`zarr`: https://zarr.readthedocs.io/en/stable/