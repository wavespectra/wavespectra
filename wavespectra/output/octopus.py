"""OCTOPUS output plugin."""

import gzip
import numpy as np
import xarray as xr

from wavespectra.core.attributes import attrs


def to_octopus(
    self,
    filename,
    site_id="spec",
    fcut=0.125,
    missing_val=-99999,
    ntime=None,
    lons=None,
    lats=None,
    compresslevel=6,
):
    """Save spectra in Octopus format.

    Args:
        - filename (str): name for output OCTOPUS file.
        - site_id (str): used to construct LPoint header.
        - fcut (float): frequency for splitting spectra.
        - missing_value (int): missing value in output file.
        - ntime (int, None): number of times to load into memory before dumping output
          file if full dataset does not fit into memory, choose None to load all times.
        - lons: (np.array, None): longitudes to use for each site if not in dataset.
        - lats: (np.array, None): latitudes to use for each site if not in dataset.
        - compresslevel (int): compression level for gzip compression (1-9).

    Note:
        - Output files are gzipped if filename ends with `.gz`.
        - lons/lats parameters may be prescribed to set site locations if lon/lat are
          not variables in the dataset (their sizes must match the number of sites).
        - If lons/lats are not specified and the dataset does not have lon/lat coords,
          all coordinates default to zero.
        - multiple locations dumped at same file with one location header per site.
        - 1D spectra not supported.
        - ntime=None optimises speed as the dataset is loaded into memory however the
          dataset may not fit into memory in which case a smaller number of times may
          be prescribed.

    """
    assert attrs.TIMENAME in self.dims, "Octopus output requires time dimension"

    # If grid reshape into site, otherwise ensure there is site dim to iterate over
    dset_stacked = self._check_and_stack_dims()
    ntime = min(ntime or dset_stacked.time.size, dset_stacked.time.size)

    # Handle datasets with missing lon/lat
    for arr, name in zip((lons, lats), (attrs.LONNAME, attrs.LATNAME)):
        if name not in dset_stacked.data_vars:
            if arr is None:
                arr = np.zeros(dset_stacked.site.size)
            elif len(arr) != dset_stacked.site.size:
                raise ValueError(f"{name} must have same size as site dimension")
            dset_stacked[name] = (("site",), arr)

    # Writing format definitions
    fmt = ",".join(len(self.freq) * ["{:8.7f}"]) + ","
    fmt2 = "{:0.0f}," + fmt + "{:8.7f}"
    fmt2 = fmt2.replace("{", "").replace("}", "").replace(":", "%").split(",")
    fmt2[-1] += ","

    # Open file for writing
    if filename.endswith(".gz"):
        fid = gzip.open(filename, "wt", compresslevel=compresslevel)
    else:
        fid = open(filename, "wt")

    # Load up to ntime times at a time to optimise memory and speed
    i0 = 0
    i1 = ntime
    while i1 <= dset_stacked.time.size:
        dset = dset_stacked.isel(time=slice(i0, i1)).load()
        i0 = i1
        i1 += ntime

        # Time arrays
        times = dset.time.to_index().to_pydatetime()
        if len(times) > 1:
            dt = (times[1] - times[0]).total_seconds() / 3600.0
        else:
            dt = 0.0
        tstart = f"{times[0]:%d-%b-%Y %H:%M:%S}"
        ym = [f"{time:%Y%m}" for time in times]
        dhm = [f"'{time:%d%H%M}" for time in times]
        times = [f"{time:%Y%m%d_%Hz}" for time in times]

        # Assign for speed
        freqs = dset.freq.values
        nfreq = dset.freq.size
        ndir = dset.dir.size
        ntime = dset.time.size

        # Parameters
        stats = ["hs", "tm01", "dm"]
        dset = xr.merge(
            [
                dset,
                dset.spec.stats(stats + ["dpm", "dspr"]),
                dset.spec.stats(stats, names=[s + "_swell" for s in stats], fmax=fcut),
                dset.spec.stats(stats, names=[s + "_sea" for s in stats], fmin=fcut),
                dset.spec.momf(mom=1).rename("momf1"),
                dset.spec.momf(mom=2).rename("momf2"),
                dset.spec.momd(mom=0)[0].rename("momd"),
                dset.spec.to_energy(),
                (dset.efth.spec.df * dset.spec.momd(mom=0)[0])
                .transpose(attrs.TIMENAME, attrs.SITENAME, attrs.FREQNAME)
                .rename("fSpec"),
            ],
            join="left",
        )
        dset = dset.drop_vars("efth")
        dset = dset.sortby("dir").fillna(missing_val)

        if attrs.WDIRNAME not in dset:
            dset[attrs.WDIRNAME] = 0 * dset["hs"] + missing_val
        if attrs.WSPDNAME not in dset:
            dset[attrs.WSPDNAME] = 0 * dset["hs"] + missing_val
        if attrs.DEPNAME not in dset:
            dset[attrs.DEPNAME] = 0 * dset["hs"] + missing_val

        # Keeping only supported dimensions
        dims_to_keep = {attrs.TIMENAME, attrs.SITENAME, attrs.FREQNAME, attrs.DIRNAME}
        dset = dset.drop_dims(set(dset.dims) - dims_to_keep)

        # Put everything in dict because it is a lot faster to slice
        dset_dict = {v: dset[v].values for v in dset.data_vars}
        data_vars = list(dset_dict.keys() - (attrs.LONNAME, attrs.LATNAME))

        # Extend energy array with directions values and sums along frequencies
        right = dset["energy"].sum(dim="freq").expand_dims({"freq": [10]})
        left = right * 0 + dset.dir
        dset_dict["energy"] = (
            xr.concat(
                (left.assign_coords({"freq": [-10]}), dset["energy"], right), dim="freq"
            )
            .transpose(attrs.TIMENAME, attrs.SITENAME, attrs.DIRNAME, attrs.FREQNAME)
            .values
        )

        lons = np.atleast_1d(dset_dict[attrs.LONNAME])
        lats = np.atleast_1d(dset_dict[attrs.LATNAME])

        # Looping over each site
        for isite in range(dset.site.size):
            lon = lons[isite]
            lat = lats[isite]
            dsite = {v: dset_dict[v][:, isite] for v in data_vars}

            # General header
            fid.write(f"Forecast valid for {tstart}\n")
            fid.write(f"nfreqs,{nfreq:d}\n")
            fid.write(f"ndir,{ndir:d}\n")
            fid.write(f"nrecs,{ntime:d}\n")
            fid.write(f"Latitude,{lat:0.6f}\n")
            fid.write(f"Longitude,{lon:0.6f}\n")
            fid.write(f"Depth,{dsite[attrs.DEPNAME][0]:0.2f}\n\n")

            # Dump each timestep
            for itime, time in enumerate(times):
                ds = {v: dsite[v][itime] for v in data_vars}

                # Timestamp header
                lp = f"{site_id}_{time}"
                fid.write(
                    "CCYYMM,DDHHmm,LPoint,WD,WS,ETot,TZ,VMD,ETotSe,TZSe,VMDSe,ETotSw,"
                    "TZSw,VMDSw,Mo1,Mo2,HSig,DomDr,AngSpr,Tau\n"
                )

                # Header and parameters
                fid.write(
                    "{},{},{},{:0.0f},{:.2f},{:.4f},{:.2f},{:.1f},{:.4f},"
                    "{:.2f},{:.1f},{:.4f},{:.2f},{:.1f},{:.5f},{:.5f},{:.4f},{:0.0f},"
                    "{:0.0f},{:0.0f}\n".format(
                        ym[itime],
                        dhm[itime],
                        lp,
                        ds[attrs.WDIRNAME],
                        ds[attrs.WSPDNAME],
                        0.25 * ds["hs"] ** 2,
                        ds["tm01"],
                        ds["dm"],
                        0.25 * ds["hs_sea"] ** 2,
                        ds["tm01_sea"],
                        ds["dm_sea"],
                        0.25 * ds["hs_swell"] ** 2,
                        ds["tm01_swell"],
                        ds["dm_swell"],
                        ds["momf1"],
                        ds["momf2"],
                        ds["hs"],
                        ds["dpm"],
                        ds["dspr"],
                        itime * dt,
                    )
                )

                # Spectra
                specdump = ""
                fid.write(("freq," + fmt + "anspec\n").format(*freqs))
                np.savetxt(fid, ds["energy"], fmt=fmt2, delimiter=",")
                fid.write(("fSpec," + fmt + "\n").format(*ds["fSpec"]))
                fid.write(("den," + fmt + "\n\n").format(*ds["momd"]))

    fid.close()
