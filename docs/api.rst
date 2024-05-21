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
   SpecArray.hrms
   SpecArray.hmax
   SpecArray.fp
   SpecArray.tp
   SpecArray.tm01
   SpecArray.tm02
   SpecArray.dpm
   SpecArray.dp
   SpecArray.dm
   SpecArray.dspr
   SpecArray.dpspr
   SpecArray.swe
   SpecArray.sw
   SpecArray.gw
   SpecArray.gamma
   SpecArray.alpha
   SpecArray.goda

**Spectral partitioning**

.. autosummary::
   :nosignatures:
   :toctree: generated/

   SpecArray.split
   SpecArray.partition
   partition.partition.Partition.ptm1
   partition.partition.Partition.ptm2
   partition.partition.Partition.ptm3
   partition.partition.Partition.ptm4
   partition.partition.Partition.ptm5
   partition.partition.Partition.hp01
   partition.partition.Partition.bbox
   partition.partition.Partition.ptm1_track


**Other methods**

.. autosummary::
   :nosignatures:
   :toctree: generated/

   SpecArray.momf
   SpecArray.momd
   SpecArray.wavelen
   SpecArray.celerity
   SpecArray.stats
   SpecArray.plot
   SpecArray.scale_by_hs
   SpecArray.oned
   SpecArray.to_energy
   SpecArray.interp
   SpecArray.interp_like


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
   read_ndbc

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
   read_ndbc_ascii
   read_json
   read_funwave
   read_xwaves
   read_dataset
   read_wavespectra
   read_ww3_station

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
   SpecDataset.to_funwave
   SpecDataset.to_orcaflex


Construct
---------

.. autosummary::
   :nosignatures:
   :toctree: generated/

   construct.frequency.pierson_moskowitz
   construct.frequency.jonswap
   construct.frequency.tma
   construct.frequency.gaussian
   construct.direction.cartwright
   construct.direction.asymmetric
   construct.construct_partition
   construct.partition_and_reconstruct


Internal core functions and objects
-----------------------------------

**Partition subpackage**

.. autosummary::
   :nosignatures:
   :toctree: generated/

   partition.partition.Partition
   partition.partition.Partition.ptm1
   partition.partition.np_ptm1
   partition.partition.np_ptm2
   partition.partition.np_ptm3
   partition.partition.np_hp01
   partition.tracking.dfp_wsea
   partition.tracking.dfp_swell
   partition.tracking.match_consecutive_partitions
   partition.tracking.np_track_partitions
   partition.tracking.track_partitions
   partition.hanson_and_phillips_2001._partition_stats
   partition.hanson_and_phillips_2001._is_contiguous
   partition.hanson_and_phillips_2001._frequency_resolution
   partition.hanson_and_phillips_2001._plot_partitions
   partition.hanson_and_phillips_2001.spread_hp01
   partition.hanson_and_phillips_2001._combine_last
   partition.hanson_and_phillips_2001.combine_partitions_hp01

**attributes module**

.. autosummary::
   :nosignatures:
   :toctree: generated/

   core.attributes.set_spec_attributes
   core.attributes.attrs

**npstats**

.. autosummary::
   :nosignatures:
   :toctree: generated/

   core.npstats.hs
   core.npstats.dpm
   core.npstats.dp
   core.npstats.dm
   core.npstats.tps
   core.npstats.tp
   core.npstats.dpspr
   core.npstats.mom1
   core.npstats.jonswap
   core.npstats.gaussian

**xrstats**

.. autosummary::
   :nosignatures:
   :toctree: generated/

   core.xrstats.peak_wave_direction
   core.xrstats.mean_direction_at_peak_wave_period
   core.xrstats.peak_wave_period
   core.xrstats.peak_directional_spread

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

**utils module**

.. autosummary::
   :nosignatures:
   :toctree: generated/

   core.utils.wavelen
   core.utils.wavenuma
   core.utils.celerity
   core.utils.to_nautical
   core.utils.unique_indices
   core.utils.unique_times
   core.utils.spddir_to_uv
   core.utils.uv_to_spddir
   core.utils.interp_spec
   core.utils.flatten_list
   core.utils.regrid_spec

**swan module**

.. autosummary::
   :nosignatures:
   :toctree: generated/

   core.swan.read_tab
   core.swan.SwanSpecFile
