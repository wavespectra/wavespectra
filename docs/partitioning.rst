.. image:: _static/wavespectra_logo.png
    :width: 150 px
    :align: right

============
Partitioning
============

Spectral partitioning splits the wave spectrum into components such as wind sea and
swells, so integrated parameters can be calculated for each individual wave system.
Most methods in wavespectra are based on the watershed algorithm of
`Hanson et al. (2009)`_ implemented in spectral wave models such as WW3 and SWAN.

The partitioning methods are available from the ``spec.partition`` namespace (in
version 4 this namespace replaced the previous ``partition`` method, see
:doc:`migration`):

- :meth:`~wavespectra.partition.partition.Partition.ptm1`
- :meth:`~wavespectra.partition.partition.Partition.track`
- :meth:`~wavespectra.partition.partition.Partition.ptm2`
- :meth:`~wavespectra.partition.partition.Partition.ptm3`
- :meth:`~wavespectra.partition.partition.Partition.ptm4`
- :meth:`~wavespectra.partition.partition.Partition.ptm5`
- :meth:`~wavespectra.partition.partition.Partition.hp01`
- :meth:`~wavespectra.partition.partition.Partition.bbox`

The `PTM` methods are named after the convention in the `WAVEWATCHIII`_ spectral wave
model from which they were derived (`TRACK` runs one of the partitioning methods and
tracks the partitions in time).

The `HP01` method implements the combining of nearby swell partitions (in spectral
space) described in `Hanson and Phillips (2001)`_ and `Hanson et al. (2009)`_. Adjacent
partitions are combined when the saddle point between them is high relative to the
smaller of the two peaks, or when their peaks are close relative to the spectral
spread of either partition and their mean directions agree. An exact number of swell
partitions can be requested, in which case the least separated partitions are further
combined until the requested number is reached.

The `BBOX` method is a custom method to split the energy
density inside and outside a defined bounding box in spectral space.

.. list-table::
   :header-rows: 1
   :widths: 14 46 20 20

   * - Method
     - Description
     - Classifies wind sea / swell
     - Requires wind and depth
   * - ``ptm1``
     - Watershed with all wind-sea partitions combined into partition 0
     - yes
     - yes
   * - ``track``
     - Any of ``ptm1``, ``ptm2``, ``ptm3`` or ``hp01``, with the partitions
       tracked in time into wave systems
     - as per method
     - as per method
   * - ``ptm2``
     - As ``ptm1``, with a secondary wind sea split from the swell partitions
     - yes
     - yes
   * - ``ptm3``
     - Plain watershed partitions, no wind-sea classification
     - no
     - no
   * - ``ptm4``
     - Wave-age split into one wind sea and one swell, no watershed
     - yes
     - yes
   * - ``ptm5``
     - Frequency-threshold split, no watershed
     - no
     - no
   * - ``hp01``
     - Watershed with combining of swells from the same wave system, supports
       prescribing an exact number of output partitions
     - yes
     - optional
   * - ``bbox``
     - Split inside/outside a bounding box in spectral space
     - no
     - no

Some parameters are shared by several methods:

* ``agefac``: Wave age factor used in the wind-sea criterion; spectral bins whose
  celerity is smaller than ``agefac`` times the local wind speed component are
  considered under wind forcing.
* ``wscut``: Wind-sea fraction cutoff; watershed partitions whose wind-forced energy
  fraction exceeds this value are classified as wind sea.
* ``ihmax``: Number of discrete levels used to bin the spectra in the watershed
  algorithm.
* ``swells`` (``parts`` in ``ptm3``): Number of partition slots in the output
  ``part`` dimension; smaller partitions are dropped (or combined in ``hp01``) and
  missing slots are null-padded. Setting it to None sizes the output from the
  largest number of partitions detected across all spectra, at the cost of an extra
  pass over the data.


.. ipython:: python
    :okwarning:

    import numpy as np
    import xarray as xr
    import matplotlib.pyplot as plt
    import cmocean
    from wavespectra import read_ww3, read_wwm


PTM1
____

The PTM1 method corresponds to the deprecated `spec.partition()` method from Wavespectra
version 3. In PTM1, topographic partitions for which the percentage of wind-sea energy
exceeds a defined fraction are aggregated and assigned to the wind-sea component (e.g.,
the first partition). The remaining partitions are assigned as swell components in
order of decreasing wave height.

.. ipython:: python
    :okwarning:

    dset = read_wwm("_static/wwmfile.nc")
    dspart = dset.spec.partition.ptm1(
        dset.wspd,
        dset.wdir,
        dset.dpt,
        swells=2,
    )
    dspart.isel(time=0, site=0, drop=True).spec.plot(col="part");

    @savefig partitioning_ptm1.png
    plt.draw()

Smoothing the spectra before partitioning can help to avoid spurious partitions as
suggested by `Portilla et al. (2009)`_.

.. ipython:: python
    :okwarning:

    dset = read_wwm("_static/wwmfile.nc")
    dspart = dset.spec.partition.ptm1(
        dset.wspd,
        dset.wdir,
        dset.dpt,
        swells=2,
        smooth=True,
    )
    dspart.isel(time=0, site=0, drop=True).spec.plot(col="part");

    @savefig partitioning_ptm1_smooth.png
    plt.draw()


Some watershed parameters are exposed to the user for tuning the partitioning algorithm:

.. ipython:: python
    :okwarning:

    dset = read_wwm("_static/wwmfile.nc")
    dspart = dset.spec.partition.ptm1(
        dset.wspd,
        dset.wdir,
        dset.dpt,
        swells=2,
        agefac=1.5,
        wscut=0.5,
        ihmax=200,
    )
    dspart.isel(time=0, site=0, drop=True).spec.plot(col="part");

    @savefig partitioning_ptm1_tuning.png
    plt.draw()


PTM2
____

PTM2 works in a similar way to PTM1 by identifying a primary wind sea (assigned to
partition 0) and one or more swell components. In this method, however, all swell
partitions are checked for wind-sea influence: energy in spectral bins within the
wind-sea range (defined by a wave age criterion) is removed and combined
into a secondary wind-sea partition (assigned to partition 1). The remaining swell
partitions are then assigned in order of decreasing wave height from partition 2 onwards.
This implies PTM2 has an extra partition compared to PTM1.

.. ipython:: python
    :okwarning:

    dset = read_wwm("_static/wwmfile.nc")
    dspart = dset.spec.partition.ptm2(
        dset.wspd,
        dset.wdir,
        dset.dpt,
        swells=2,
    )
    dspart.isel(time=0, site=0, drop=True).spec.plot(col="part");

    @savefig partitioning_ptm2.png
    plt.draw()


PTM3
____
PTM3 does not classify the topographic partitions into wind-sea or swell - it simply
orders them by wave height. This approach is useful for producing data for spectral
reconstruction applications using a limited number of partitions, where the
classification of the partition as wind-sea or swell is less important than the
proportion of overall spectral energy each partition represents. In addition, this method
does not require wind and water depth information and can be used with any spectral
dataset.

.. ipython:: python
    :okwarning:

    dset = read_wwm("_static/wwmfile.nc")
    dspart = dset.spec.partition.ptm3(parts=3)
    dspart.isel(time=0, site=0, drop=True).spec.plot(col="part");

    @savefig partitioning_ptm3.png
    plt.draw()


PTM4
____
PTM4 uses the wave age criterion derived from the local wind speed to split the spectrum
into wind-sea and a single swell partition. In this case waves with a celerity greater
than the directional component of the local wind speed are considered to be freely
propagating swell (i.e. unforced by the wind). This is similar to the method commonly
used to generate wind-sea and swell from the WAM model.

.. ipython:: python
    :okwarning:

    dset = read_wwm("_static/wwmfile.nc")
    dspart = dset.spec.partition.ptm4(
        dset.wspd,
        dset.wdir,
        dset.dpt,
        agefac=1.7,
    )
    dspart.isel(time=0, site=0, drop=True).spec.plot(col="part");

    @savefig partitioning_ptm4.png
    plt.draw()

The wind sea region used to partition the spectra in PTM4 can be calculated
from the :func:`~wavespectra.core.utils.waveage` method:

.. ipython:: python
    :okwarning:

    from wavespectra.core.utils import waveage
    ds = read_ww3("_static/ww3file.nc").sortby("dir").isel(site=0, drop=True)
    windmask = waveage(ds.freq, ds.dir, ds.wspd, ds.wdir, ds.dpt, 1.7)
    f = windmask.fillna(1.0).spec.plot(col="time", col_wrap=3);
    for ind, ax in enumerate(f.axs.flat):
        wdir = float(ds.wdir.isel(time=ind).values)
        ax.set_title(f"wdir={wdir:0.0f} deg")

    @savefig partitioning_windmask.png
    plt.draw()


PTM5
____
PTM5 splits spectra into wind sea and swell based on a user defined static cutoff. This
method differs from :meth:`~wavespectra.specarray.SpecArray.split` in that here the
output partitioned spectra dataset has an extra `part` dimension and the sea and swell
partitions have zero-values outside the defined frequency ranges. Conversely, the
:meth:`~wavespectra.specarray.SpecArray.split` method returns a single partition with
frequencies truncated to the defined ranges. Notice there could be slight differences
when integrating the partitions generated by these two methods since in PTM5 there will
be an "area" at one of the frequency edges adjacent to the zero-values.

.. ipython:: python
    :okwarning:

    dset = read_wwm("_static/wwmfile.nc")
    dspart = dset.spec.partition.ptm5(fcut=0.1)
    dspart.isel(time=0, site=0, drop=True).spec.plot(col="part");

    @savefig partitioning_ptm5.png
    plt.draw()


BBOX
____

BBOX partitions the spectra based on user-defined bounding boxes in frequency-direction
space.

.. ipython:: python
    :okwarning:

    dset = read_wwm("_static/wwmfile.nc")
    bbox = dict(fmin=0.05, fmax=0.1, dmin=30, dmax=120)
    dspart = dset.spec.partition.bbox(bboxes=[bbox])
    dspart.isel(time=0, site=0, drop=True).spec.plot(col="part");

    @savefig partitioning_bbox.png
    plt.draw()


HP01
____

HP01 partitions the spectra and merges wind-sea components as in the PTM1 method, then
it combines adjacent swells belonging to the same wave system following the criteria
outlined in `Hanson and Phillips (2001)`_ and `Hanson et al. (2009)`_. This method is
particularly useful when partitioning measured wave spectra, which are typically noisy
and tend to be over-segmented by the watershed algorithm, and to prescribe an exact
number of output partitions.

Two adjacent swell partitions are combined when their mean directions agree within
`angle_max` (30 degrees by default, the optimum found by Hanson et al., 2009) and
either of the following criteria is met:

- **Minimum between peaks**: the spectral density at the saddle point between the two
  partitions exceeds a fraction `zeta` of the smaller of the two peak densities.
- **Peak separation**: the distance between the two peaks in cartesian frequency space
  :math:`(f_x, f_y) = (f\cos\theta, f\sin\theta)` is smaller than a fraction `kappa`
  of the spectral spread of either partition (eqs 6-9 in Hanson and Phillips, 2001).

Partition adjacency and saddle points are evaluated on the shared boundaries of the
watershed partitions, statistics are recomputed after every merge and the candidate
pairs satisfying the criteria most strongly are always combined first, so results do
not depend on the ordering of the partitions. Partitions smaller than `hs_min` (or
below the optional noise threshold defined by `noise_a` and `noise_b`, eq 10 in Hanson
and Phillips, 2001) are merged onto their most connected neighbours so that spectral
variance is conserved.

The `swells` argument prescribes the exact number of swell partitions returned: if
more systems remain after the combining criteria are exhausted, the least separated
ones are further combined until the requested number is achieved (or the smallest ones
are excluded if `combine_extra_swells` is False). Setting `swells=None` instead sizes
the output from the number of swell systems detected across all spectra, at the cost
of an extra pass over the data.

The example below shows the partitioning of model spectra which aren't noisy, the
result is the same as the PTM1 method.

.. ipython:: python
    :okwarning:

    dset = read_wwm("_static/wwmfile.nc")
    dspart = dset.spec.partition.hp01(
        dset.wspd,
        dset.wdir,
        dset.dpt,
        swells=2,
    )
    dspart.isel(time=0, site=0, drop=True).spec.plot(col="part");

    @savefig partitioning_hp01.png
    plt.draw()


TRACK
_____
TRACK partitions the spectra with any of the `PTM1`, `PTM2`, `PTM3` or `HP01` methods
and tracks the partitions in time using the evolution of peak frequency and peak
direction. Wind sea partitions are matched with wind-sea thresholds based on
fetch-limited growth rates and swell partitions with thresholds based on the swell
dispersion rate. The method returns the partitioned dataset with two extra data
variables: `track_id`, identifying the wave system each partition belongs to at each
time step, and `ntracks`, the number of wave systems tracked:

.. ipython:: python
    :okwarning:

    dset = read_ww3("_static/ww3file.nc").isel(site=0, drop=True)
    dspart = dset.spec.partition.track(
        wspd=dset.wspd,
        wdir=dset.wdir,
        dpt=dset.dpt,
        method="ptm1",
    )
    # Add some spectral parameters to visualise
    dspart = xr.merge([dspart, dspart.spec.stats(["hs", "tp", "dpm"])])
    dspart

The `track_id` variable identifies all unique wave systems over the time range of the
input spectra dataset and can be used to combine these systems to yield consistent
time series. The `ptm4`, `ptm5` and `bbox` methods define partitions as fixed spectral
regions whose identity is already continuous in time, so they are not available for
tracking. The `ptm3` method has no wind sea classification, all partitions are matched
with the swell thresholds and wind inputs are not required.

Compare the original partitions with no tracking:

.. ipython:: python
    :okwarning:

    fig, axes = plt.subplots(3, 1, figsize=(10, 10))

    # Iterate over each original partition
    for part in dspart.part.values:
        pstats = dspart.sel(part=part)
        # Plot stats for this wave system
        for ax, var in zip(axes, ["hs", "tp", "dpm"]):
            ax.plot(pstats.time, pstats[var], ".-", label=f"Partition {part}")
            ax.set_ylabel(var)
    ax.legend();

    @savefig partitioning_nontracked.png
    plt.draw()

Against the tracked partitions:

.. ipython:: python
    :okwarning:

    fig, axes = plt.subplots(3, 1, figsize=(10, 10))

    # Iterate over each unique wave system
    for track_id in range(int(dspart.ntracks)):
        ind = np.where(dspart.track_id.values.flatten() == track_id)[0]
        pstats = dspart.stack(tpart=("part", "time")).isel(tpart=ind).sortby("time")
        # Plot stats for this wave system
        for ax, var in zip(axes, ["hs", "tp", "dpm"]):
            ax.plot(pstats.time, pstats[var], ".-", label=f"System {track_id}")
            ax.set_ylabel(var)
    ax.legend()

    @savefig partitioning_tracked.png
    plt.draw()


.. _`WAVEWATCHIII`: https://github.com/NOAA-EMC/WW3
.. _`Hanson and Phillips (2001)`: https://journals.ametsoc.org/view/journals/atot/18/2/1520-0426_2001_018_0277_aaoosd_2_0_co_2.xml
.. _`Hanson et al. (2009)`: https://journals.ametsoc.org/view/journals/atot/26/8/2009jtecho650_1.xml
.. _`Portilla et al. (2009)`: https://journals.ametsoc.org/view/journals/atot/26/1/2008jtecho609_1.xml
