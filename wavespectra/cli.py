"""Console script for wavespectra."""
import os
import click
import yaml
import xarray as xr
from dask.diagnostics.progress import ProgressBar

from wavespectra.core.utils import load_function
from wavespectra.core.attributes import attrs
from wavespectra.construct import partition_and_reconstruct, STATS


@click.group()
def main(args=None):
    pass


@main.group()
def reconstruct():
    pass


@reconstruct.command()
@click.argument("infile")
@click.argument("outfile")
@click.option(
    "-f", "--fit_name", default="fit_jonswap", help="Fit function", show_default=True
)
@click.option(
    "-d", "--dir_name", default="cartwright", help="Spread function", show_default=True
)
@click.option(
    "-m",
    "--method_combine",
    default="max",
    help="Method to combine partitions",
    show_default=True,
)
@click.option(
    "-p",
    "--parts",
    type=int,
    default=4,
    help="Number of partitions to keep",
    show_default=True,
)
@click.option(
    "-r", "--reader", default="read_ww3", help="Spectra file reader", show_default=True
)
@click.option(
    "-c",
    "--chunks",
    default="{}",
    help="chunks dictionary to chunk dataset",
    show_default=True,
)
def spectra(infile, outfile, fit_name, dir_name, method_combine, parts, reader, chunks):
    """Partition and reconstruct spectra from file."""

    if os.path.realpath(infile) == os.path.realpath(outfile):
        raise ValueError("INFILE and OUTFILE must be different to avoid overwriting.")

    # Read spectra to reconstruct
    reader = load_function("wavespectra", f"{reader}")
    chunks = yaml.load(chunks, Loader=yaml.Loader)
    dset = reader(infile).chunk(chunks)

    # Sorting out input arguments
    if "," in fit_name:
        fit_name = fit_name.split(",")
    if "," in dir_name:
        dir_name = dir_name.split(",")

    # Run
    reconstructed = partition_and_reconstruct(
        dset,
        parts=parts,
        fit_name=fit_name,
        dir_name=dir_name,
        method_combine=method_combine,
    )

    # Save to file
    with ProgressBar():
        reconstructed.spec.to_netcdf(outfile)

    click.echo(f"Reconstructed file created: {outfile}")
