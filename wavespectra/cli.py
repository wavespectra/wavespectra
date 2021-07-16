"""Console script for wavespectra."""
import os
import click
import yaml
import xarray as xr
from dask.diagnostics.progress import ProgressBar

from wavespectra.core.utils import load_function
from wavespectra.core.attributes import attrs
from wavespectra.construct import construct_partition


SNAME = attrs.SPECNAME
FNAME = attrs.FREQNAME
DNAME = attrs.DIRNAME
PNAME = attrs.PARTNAME
STATS = [
    "hs",
    "tp",
    "fp",
    "dm",
    "dpm",
    "dspr",
    # "dpspr",
    "tm01",
    "tm02",
    "gamma",
    "alpha",
]


@click.group()
def main(args=None):
    pass


@main.group()
def reconstruct():
    pass


@reconstruct.command()
@click.argument("infile")
@click.argument("outfile")
@click.option("-f", "--fit_name", default="fit_jonswap", help="Fit function", show_default=True)
@click.option("-d", "--dir_name", default="cartwright", help="Spread function", show_default=True)
@click.option("-m", "--method_combine", default="max", help="Method to combine partitions", show_default=True)
@click.option("-s", "--swells", type=int, default=6, help="Swell partitions to keep", show_default=True)
@click.option("-r", "--reader", default="read_ww3", help="Spectra file reader", show_default=True)
@click.option("-c", "--chunks", default="{}", help="chunks dictionary to chunk dataset", show_default=True)
def spectra(infile, outfile, fit_name, dir_name, method_combine, swells, reader, chunks):
    """Partition and reconstruct spectra from file."""

    if os.path.realpath(infile) == os.path.realpath(outfile):
        raise ValueError("INFILE and OUTFILE must be different to avoid overwriting.")

    # Read spectra to reconstruct
    reader = load_function("wavespectra", f"{reader}")
    chunks = yaml.load(chunks, Loader=yaml.Loader)
    dset = reader(infile).chunk(chunks)

    # Partitioning
    dspart = dset.spec.partition(dset.wspd, dset.wdir, dset.dpt, swells=swells)

    # Calculating parameters
    dsparam = dspart.spec.stats(STATS)

    # Turn DataArrays into Kwargs for sea partition
    kw_sea = {k: v for k, v in dsparam.isel(part=[0]).data_vars.items()}
    kw_sea.update({FNAME: dset[FNAME], DNAME: dset[DNAME]})

    # Turn DataArrays into Kwargs for swell partition
    kw_sw = {k: v for k, v in dsparam.isel(part=slice(1, None)).data_vars.items()}
    kw_sw.update({FNAME: dset[FNAME], DNAME: dset[DNAME]})

    # Reconstruct partitions
    sea = construct_partition(fit_name, dir_name, fit_kwargs=kw_sea, dir_kwargs=kw_sea)
    swell = construct_partition(fit_name, dir_name, fit_kwargs=kw_sw, dir_kwargs=kw_sw)

    # Combine partitions
    reconstructed = getattr(xr.concat([sea, swell], PNAME), method_combine)(PNAME)

    # Save to file
    with ProgressBar():
        reconstructed.to_dataset(name=SNAME).spec.to_netcdf(outfile)

    click.echo(f"Reconstructed file created: {outfile}")
