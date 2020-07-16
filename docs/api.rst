.. currentmodule:: wavespectra

=================
API documentation
=================

General description of modules, objects and functions in wavespectra.


Wavespectra accessors
---------------------

.. autosummary::
   :nosignatures:
   :toctree: generated/

   SpecArray
   SpecDataset

\* The two accessors are attached to the respective xarray objects via the `spec` namespace.


SpecArray
---------

All methods in :py:class:`SpecArray` accessor are also available from  :py:class:`SpecDataset`.

**Integrated spectral parameters**

.. autosummary::
   :nosignatures:
   :toctree: generated/

   SpecArray.hs
   SpecArray.hmax
   SpecArray.tp
   SpecArray.tm01
   SpecArray.tm02
   SpecArray.dpm
   SpecArray.dp
   SpecArray.dm
   SpecArray.dspr
   SpecArray.dpw
   SpecArray.swe
   SpecArray.sw
   SpecArray.momf
   SpecArray.momd

**Spectral partitioning**

.. autosummary::
   :nosignatures:
   :toctree: generated/

   SpecArray.split
   SpecArray.partition

**Other methods**

.. autosummary::
   :nosignatures:
   :toctree: generated/

   SpecArray.wavelen
   SpecArray.celerity
   SpecArray.stats
   SpecArray.plot
   SpecArray.scale_by_hs
   SpecArray.oned
   SpecArray.to_energy


SpecDataset
-----------

.. autosummary::
   :nosignatures:
   :toctree: generated/

   SpecDataset.sel

Output methods described in the `Output functions`_ section: 
:py:attr:`~SpecDataset.to_swan`
:py:attr:`~SpecDataset.to_netcdf`
:py:attr:`~SpecDataset.to_octopus`
:py:attr:`~SpecDataset.to_ww3`
:py:attr:`~SpecDataset.to_json`


Input functions
---------------

**NetCDF-based input functions**

.. autosummary::
   :nosignatures:
   :toctree: generated/

   read_ww3
   read_ncswan
   read_wwm
   read_netcdf
   read_era5

\* These functions also support Zarr files


**Other input functions**

.. autosummary::
   :nosignatures:
   :toctree: generated/

   read_swan
   read_triaxys
   read_spotter
   read_octopus
   read_dataset
   read_ndbc
   read_json

**Convenience SWAN ASCII input functions**

.. autosummary::
   :nosignatures:
   :toctree: generated/

   input.swan.read_swans
   input.swan.read_hotswan
   input.swan.read_swanow


.. _`Output functions`:

Output functions
----------------

.. autosummary::
   :nosignatures:
   :toctree: generated/

   SpecDataset.to_swan
   SpecDataset.to_netcdf
   SpecDataset.to_octopus
   SpecDataset.to_ww3
   SpecDataset.to_json


Spectral reconstruction
-----------------------

Spectral reconstruction functionality is under development. There are functions
available to fit parametric spectrum shapes from wave partitions but the construct
api is not properly established.

.. autosummary::
   :nosignatures:
   :toctree: generated/

   construct.jonswap
   construct.ochihubble
   construct.helpers.spread
   construct.helpers.arrange_inputs
   construct.helpers.make_dataset
   construct.helpers.check_coordinates


Internal core functions and objects
-----------------------------------

**attrubutes module**

.. autosummary::
   :nosignatures:
   :toctree: generated/

   core.attributes.set_spec_attributes
   core.attributes.attrs

**select module**

.. autosummary::
   :nosignatures:
   :toctree: generated/

   core.select.sel_nearest
   core.select.sel_bbox
   core.select.sel_idw
   core.select.Coordinates.distance
   core.select.Coordinates.nearer
   core.select.Coordinates.nearest

**misc module**

.. autosummary::
   :nosignatures:
   :toctree: generated/

   core.misc.dnum_to_datetime
   core.misc.to_nautical
   core.misc.unique_times
   core.misc.to_datetime
   core.misc.spddir_to_uv
   core.misc.uv_to_spddir
   core.misc.interp_spec
   core.misc.flatten_list

**timer module**

.. autosummary::
   :nosignatures:
   :toctree: generated/

   core.timer.Timer

**swan module**

.. autosummary::
   :nosignatures:
   :toctree: generated/

   core.swan.read_tab
   core.swan.SwanSpecFile
   core.swan._dateparse