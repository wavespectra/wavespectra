"""Console script for wavespectra."""
import sys
import click

from wavespectra.core.utils import load_function


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
@click.argument("config")
@click.option("-s", "--swells", type=int, default=6, help="Swell partitions to keep")
@click.option("-f", "--fmt", default="ww3", help="File format to define reader")
def spectra(infile, outfile, config, swells, fmt):
    """Reconstruct existing spectra from file."""

    # Read spectra to reconstruct
    reader = load_function("wavespectra", f"read_{fmt}")
    dset = reader(infile)

    # Partitioning
    dspart = dset.spec.partition(dset.wspd, dset.wdir, dset.dpt, swells=swells)

    # Calculating parameters
    dsparam = dspart.spec.stats(STATS)

    click.echo(dsparam)
    