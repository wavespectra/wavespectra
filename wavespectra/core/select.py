"""Interpolate stations."""
import numpy as np
import xarray as xr
import logging

from wavespectra.core.misc import unique_times
from wavespectra.core.attributes import attrs, set_spec_attributes


logger = logging.getLogger(__name__)


def distance(lons, lats, lon, lat):
    """Distances between each station in (lons, lats) and site (lon, lat).

    Args:
        lons (array): Longitudes of stations to search.
        lats (array): Latitudes of stations to search.
        lon (float): Longitude of of station to locate from lons.
        lat (float): Latitude of of station to locate from lats.

    Returns:
        List of distances between each station and site.

    """
    return np.sqrt((lons - (lon % 360.0)) ** 2 + (lats - lat) ** 2)


def nearer(lons, lats, lon, lat, tolerance=np.inf, max_sites=None):
    """Nearer stations in (lons, lats) to site (lon, lat).

    Args:
        lons (array): Longitudes of stations to search.
        lats (array): Latitudes of stations to search.
        lon (float): Longitude of of station to locate from lons.
        lat (float): Latitude of of station to locate from lats.
        tolerance (float): Maximum distance for scanning neighbours.
        max_sites (int): Maximum number of neighbours.

    Returns:
        Indices and distances of up to `max_sites` neighbour stations not farther from
            `tolerance`, ordered from closer to farthest station.

    """
    dist = distance(lons, lats, lon, lat)
    closest_ids = np.argsort(dist)
    closest_dist = dist[closest_ids]
    keep_ids = closest_ids[closest_dist <= tolerance][:max_sites]
    return keep_ids, dist[keep_ids]


def nearest(lons, lats, lon, lat):
    """Nearest station in (lons, lats) to site (lon, lat).

        Args:
            lons (array): Longitudes of stations to search.
            lats (array): Latitudes of stations to search.
            lon (float): Longitude of of station to locate from lons.
            lat (float): Latitude of of station to locate from lats.

        Returns:
            Index and distance of closest station.

        """
    dist = distance(lons, lats, lon, lat)
    closest_id = dist.argmin()
    closest_dist = dist[closest_id]
    return closest_id, closest_dist


def sel_nearest(
    dset,
    lons,
    lats,
    tolerance=2.0,
    unique=False,
    exact=False,
    dset_lons=None,
    dset_lats=None,
):
    """Select sites from nearest distance.

    Args:
        dset (Dataset): Stations SpecDataset to select from.
        lons (array): Longitude of sites to interpolate spectra at.
        lats (array): Latitude of sites to interpolate spectra at.
        tolerance (float): Maximum distance to use site for interpolation.
        unique (bool): Only returns unique sites in case of repeated inexact matches.
        exact (bool): Require exact matches.
        dset_lons (array): Longitude of stations in dset.
        dset_lats (array): Latitude of stations in dset.

    Returns:
        Selected SpecDataset at locations defined by (lons, lats).

    Note:
        Args `dset_lons`, `dset_lats` are not required but can improve performance when
            `dset` is chunked with site=1 (expensive to access station coordinates) and
            improve precision if projected coordinates are provided at high latitudes.

    """
    assert len(lons) == len(lats), "`lons` and `lats` must be the same size."
    if (
        attrs.LONNAME in dset.dims
        or attrs.LATNAME in dset.dims
        or attrs.SITENAME not in dset.dims
    ):
        raise NotImplementedError("sel_nearest only implemented for stations dataset.")

    # Providing station coordinates could be a lot more efficient for chunked datasets
    if dset_lons is None:
        dset_lons = dset[attrs.LONNAME].values
    if dset_lats is None:
        dset_lats = dset[attrs.LATNAME].values

    station_ids = []
    for lon, lat in zip(lons, lats):
        closest_id, closest_dist = nearest(dset_lons, dset_lats, lon, lat)
        if closest_dist > tolerance:
            raise AssertionError(
                "Nearest site from (lat={}, lon={}) is {:g} deg away but tolerance is {:g} deg.".format(
                    lat, lon, closest_dist, tolerance
                )
            )
        if exact and closest_dist > 0:
            raise AssertionError(
                "Exact match required but no site at (lat={}, lon={}), nearest site is {} deg away.".format(
                    lat, lon, closest_dist
                )
            )
        station_ids.append(closest_id)
    if unique:
        station_ids = list(set(station_ids))

    dsout = dset.isel(**{attrs.SITENAME: station_ids})
    dsout[attrs.SITENAME].values = np.arange(len(station_ids))
    return dsout


def sel_idw(
    dset, lons, lats, tolerance=2.0, max_sites=4, dset_lons=None, dset_lats=None
):
    """Select sites from inverse distance weighting.

    Args:
        dset (Dataset): Stations SpecDataset to interpolate from.
        lons (array): Longitude of sites to interpolate spectra at.
        lats (array): Latitude of sites to interpolate spectra at.
        tolerance (float): Maximum distance to use site for interpolation.
        max_sites (int): Maximum number of neighbour sites to use for interpolation.
        dset_lons (array): Longitude of stations in dset.
        dset_lats (array): Latitude of stations in dset.

    Returns:
        Selected SpecDataset at locations defined by (lons, lats).

    Note:
        Args `dset_lons`, `dset_lats` are not required but can improve performance when
            `dset` is chunked with site=1 (expensive to access station coordinates) and
            improve precision if projected coordinates are provided at high latitudes.

    """
    assert len(lons) == len(lats), "`lons` and `lats` must be the same size."
    if (
        attrs.LONNAME in dset.dims
        or attrs.LATNAME in dset.dims
        or attrs.SITENAME not in dset.dims
    ):
        raise NotImplementedError("sel_idw only implemented for stations dataset.")

    # Providing station coordinates could be a lot more efficient for chunked datasets
    if dset_lons is None:
        dset_lons = dset[attrs.LONNAME].values
    if dset_lats is None:
        dset_lats = dset[attrs.LATNAME].values

    mask = dset.isel(site=0, drop=True) * np.nan
    dsout = []
    for lon, lat in zip(lons, lats):
        closest_ids, closest_dist = nearer(
            dset_lons, dset_lats, lon, lat, tolerance, max_sites
        )
        if len(closest_ids) == 0:
            logger.debug(
                "No stations within {} deg of site (lat={}, lon={}), this site will be masked.".format(
                    tolerance, lat, lon
                )
            )
        # Collect ids and factors of neighbours
        indices = []
        factors = []
        for ind, dist in zip(closest_ids, closest_dist):
            indices.append(ind)
            if dist == 0:
                factors.append(1.0)
                break
            factors.append(1.0 / dist)
        # Mask it if no neighbour is found
        if len(indices) == 0:
            dsout.append(mask)
        else:
            # Inverse distance weighting
            sumfac = float(1.0 / sum(factors))
            ind = indices.pop(0)
            fac = factors.pop(0)
            weighted = float(fac) * dset.isel(site=ind, drop=True)
            for ind, fac in zip(indices, factors):
                weighted += float(fac) * dset.isel(site=ind, drop=True)
            if len(indices) > 0:
                weighted *= sumfac
            dsout.append(weighted)

    # Concat sites into dataset
    dsout = xr.concat(dsout, dim=attrs.SITENAME).transpose(*dset[attrs.SPECNAME].dims)

    # Redefining coordinates and variables
    dsout[attrs.SITENAME].values = np.arange(len(lons))
    dsout[attrs.LONNAME] = ((attrs.SITENAME), lons)
    dsout[attrs.LATNAME] = ((attrs.SITENAME), lats)
    dsout.attrs = dset.attrs
    set_spec_attributes(dsout)

    return dsout


if __name__ == "__main__":

    import matplotlib.pyplot as plt
    from wavespectra import read_ww3

    filename = "../../tests/sample_files/spec20170101T00_spec.nc"
    dset = read_ww3(filename, chunks={"site": None})
    dset_lons = dset.lon.values
    dset_lats = dset.lat.values
    dset = read_ww3(filename, chunks={"site": 1})

    lons = [283.5, 284, 284.4974365234375, 285, 285.49993896484375, 100]
    lats = [
        -53.500091552734375,
        -53.500091552734375,
        -53.500091552734375,
        -53.500091552734375,
        -53.500091552734375,
        30,
    ]
    print("IDW")
    ds1 = sel_idw(
        dset,
        lons,
        lats,
        tolerance=2.0,
        max_sites=4,
        dset_lons=dset_lons,
        dset_lats=dset_lats,
    ).load()

    print("Nearest")
    ds2 = sel_nearest(
        dset,
        lons,
        lats,
        tolerance=2.0,
        unique=False,
        exact=False,
        dset_lons=dset_lons,
        dset_lats=dset_lats,
    ).load()

    for ds in [ds1, ds2]:
        ds = ds.isel(time=0)
        ds.spec.plot.contourf(
            col="site", vmin=-5.6, vmax=-0.8, levels=np.arange(-5.6, 0, 0.8)
        )
        print("hs", ds.spec.hs().values)
        print("tp", ds.spec.tp().values)
        print("dpm", ds.spec.dpm().values)
        print("dspr", ds.spec.dspr().values)
        plt.show()
