"""OCTOPUS output plugin."""
import numpy as np
import xarray as xr

from wavespectra.core.attributes import attrs


def to_octopus(self, filename, site_id="spec", fcut=0.125, missing_val=-99999):
    """Save spectra in Octopus format.

    Args:
        - filename (str): name for output OCTOPUS file.
        - site_id (str): used to construct LPoint header.
        - fcut (float): frequency for splitting spectra.
        - missing_value (int): missing value in output file.

    Note:
        - dataset needs to have lon/lat/time coordinates.
        - dataset with multiple locations dumped at same file with one location
          header per site.
        - 1D spectra not supported.

    """
    assert attrs.TIMENAME in self.dims, "Octopus output requires time dimension"

    # If grid reshape into site, otherwise ensure there is site dim to iterate over
    dset = self._check_and_stack_dims()

    # Writing format definitions
    fmt = ",".join(len(self.freq) * ["{:6.5f}"]) + ","
    fmt2 = "{:0.0f}," + fmt + "{:6.5f}"
    fmt2 = fmt2.replace("{", "").replace("}", "").replace(":", "%").split(",")
    fmt2[-1] += ","

    # Loading for efficiency
    dset = dset.load()
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
    dirs = dset.dir.values
    nfreq = dset.freq.size
    ndir = dset.dir.size
    ntime = dset.time.size

    # Parameters
    stats = ["hs", "tm01", "dm"]
    dset = (
        xr.merge(
            [
                dset,
                dset.spec.stats(stats + ["dpm", "dspr"]),
                dset.spec.stats(stats, names=[s + "_swell" for s in stats], fmax=fcut),
                dset.spec.stats(stats, names=[s + "_sea" for s in stats], fmin=fcut),
                dset.spec.momf(mom=1).sum(dim="dir").rename("momf1"),
                dset.spec.momf(mom=2).sum(dim="dir").rename("momf2"),
                dset.spec.momd(mom=0)[0].rename("momd"),
                dset.spec.to_energy(),
                (dset.efth.spec.dfarr * dset.spec.momd(mom=0)[0]).transpose(
                    attrs.TIMENAME, attrs.SITENAME, attrs.FREQNAME
                    ).rename("fSpec"),
            ],
            join="left"
        )
    )
    dset = dset.drop("efth")
    dset = dset.sortby("dir").fillna(missing_val)

    if attrs.WDIRNAME not in dset:
        dset[attrs.WDIRNAME] = 0 * dset["hs"] + missing_val
    if attrs.WSPDNAME not in dset:
        dset[attrs.WSPDNAME] = 0 * dset["hs"] + missing_val
    if attrs.DEPNAME not in dset:
        dset[attrs.DEPNAME] = 0 * dset["hs"] + missing_val

    # Put everything in dict because it is a lot faster to slice
    dset_dict = {v: dset[v].values for v in dset.data_vars}
    data_vars = list(dset_dict.keys() - ("lon", "lat"))

    # Extend energy array with directions values and sums along frequencies
    right = dset["energy"].sum(dim="freq").expand_dims({"freq": [10]})
    left = right * 0 + dset.dir
    dset_dict["energy"] = xr.concat(
        (left.assign_coords({"freq": [-10]}), dset["energy"], right), dim="freq"
    ).transpose(attrs.TIMENAME, attrs.SITENAME, attrs.DIRNAME, attrs.FREQNAME).values

    try:
        lons = dset_dict["lon"]
        lats = dset_dict["lat"]
    except AttributeError as err:
        raise NotImplementedError(
            "lon-lat variables are required to write Octopus spectra file"
        ) from err

    # Open output file
    with open(filename, "w") as f:

        # Looping over each site
        for isite in range(dset.site.size):
            # dsite = dset.isel(site=[isite])
            lon = lons[isite]
            lat = lats[isite]
            dsite = {v: dset_dict[v][:, isite] for v in data_vars}

            # General header
            f.write(f"Forecast valid for {tstart}\n")
            f.write(f"nfreqs,{nfreq:d}\n")
            f.write(f"ndir,{ndir:d}\n")
            f.write(f"nrecs,{ntime:d}\n")
            f.write(f"Latitude,{lat:0.6f}\n")
            f.write(f"Longitude,{lon:0.6f}\n")
            f.write(f"Depth,{dsite[attrs.DEPNAME][0]:0.2f}\n\n")

            # Dump each timestep
            for itime, time in enumerate(times):
                # ds = dsite.isel(time=i)
                ds = {v: dsite[v][itime] for v in data_vars}

                # Timestamp header
                lp = f"{site_id}_{time}"
                f.write(
                    "CCYYMM,DDHHmm,LPoint,WD,WS,ETot,TZ,VMD,ETotSe,TZSe,VMDSe,ETotSw,"
                    "TZSw,VMDSw,Mo1,Mo2,HSig,DomDr,AngSpr,Tau\n"
                )

                # Header and parameters
                f.write(
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
                # for idir, direc in enumerate(dirs):
                #     row = ds["energy"][idir]
                #     specdump += f"{direc:0.0f},"
                #     specdump += fmt.format(*row)
                #     specdump += f"{row.sum():6.5f},\n"
                f.write(("freq," + fmt + "anspec\n").format(*freqs))
                np.savetxt(f, ds["energy"], fmt=fmt2, delimiter=",")
                # f.write(specdump)
                f.write(("fSpec," + fmt + "\n").format(*ds["fSpec"]))
                f.write(("den," + fmt + "\n\n").format(*ds["momd"]))
