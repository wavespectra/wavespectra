"""Console script for wavespectra."""
from pathlib import Path
import click
import yaml
import xarray as xr
from dask.diagnostics.progress import ProgressBar

from wavespectra.core.utils import load_function
from wavespectra.construct import partition_and_reconstruct
from wavespectra.core.utils import load_function


def file_reader(function):
    """Arguments and options for reading files to share among commands."""
    function = click.argument("engine")(function)
    function = click.argument("infile")(function)
    function = click.option(
        "-k",
        "--engine_kwargs",
        type=(str, str),
        nargs=2,
        multiple=True,
        help="kwargs to pass to file reader engine, repeat for multiple key-value pairs",
        show_default=False,
    )(function)
    return function


def parse_kwargs(kwargs):
    """Parse kwargs from command line into a dictionary."""
    return {k: yaml.safe_load(v) for k, v in dict(kwargs).items()}


@click.group()
def main(args=None):
    pass


# =====================================================================================
# Convert commands
# =====================================================================================
@main.group()
def convert():
    pass


@convert.command()
@file_reader
@click.argument("outfile")
@click.argument("fmt")
@click.option("-o", "--overwrite", is_flag=True, help="Overwrite OUTFILE if it exists")
def format(infile, engine, outfile, fmt, engine_kwargs, overwrite):
    """Converts INFILE with format defined by ENGINE to OUTFILE with new format FMT."""

    if Path(infile).absolute() == Path(outfile).absolute() and not overwrite:
        raise ValueError(
            "INFILE and OUTFILE must be different to avoid overwriting if -o is not set"
        )

    # Engine specific kwargs, force as_file for swan ascii if not provided
    kw = parse_kwargs(engine_kwargs)
    if not kw and engine == "swan":
        kw["as_site"] = True

    # File spectra to convert
    dset = xr.open_dataset(infile, engine=engine, **kw)

    # Save to file
    if not hasattr(dset.spec, f"to_{fmt}"):
        raise ValueError(f"Output format '{fmt}' not supported.")
    getattr(dset.spec, f"to_{fmt}")(outfile)

    click.echo(f"Converted file created: {outfile}")


@convert.command()
@file_reader
@click.argument("outfile")
@click.option(
    "-p",
    "--parameters",
    multiple=True,
    default=["hs", "tp", "dpm"],
    help="Integrated parameters to include, repeat for multiple parameters",
    show_default=True,
)
def stats(infile, engine, outfile, parameters, engine_kwargs):
    """Write a new file OUTFILE with integrated parameters from INFILE."""

    if Path(infile).absolute() == Path(outfile).absolute():
        raise ValueError(
            "INFILE and OUTFILE must be different to avoid overwriting if -o is not set"
        )

    # File spectra to convert
    dset = xr.open_dataset(infile, engine=engine, **parse_kwargs(engine_kwargs))

    # Integrated stats
    dsout = dset.spec.stats(parameters)

    # Save to file
    dsout.to_netcdf(outfile)

    click.echo(f"Integrated parameters file created: {outfile}")


# =====================================================================================
# Reconstruct commands
# =====================================================================================
@main.group()
def reconstruct():
    pass


@reconstruct.command()
@file_reader
@click.argument("outfile")
@click.option(
    "-f", "--freq_name", default="jonswap", help="Frequency function", show_default=True
)
@click.option(
    "-d", "--dir_name", default="cartwright", help="Spread function", show_default=True
)
@click.option(
    "-pm",
    "--partition_method",
    default="ptm3",
    help="Partitioning method, either `ptm1`, `ptm2` or `ptm3`",
    show_default=True,
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
    "-c",
    "--chunks",
    type=(str, str),
    nargs=2,
    multiple=True,
    help="Chunks dict to chunk dataset before partitioning, repeat for multiple dims",
    show_default=False,
)
def spectra(
    infile,
    engine,
    outfile,
    engine_kwargs,
    freq_name,
    dir_name,
    partition_method,
    method_combine,
    parts,
    chunks,
):
    """Partition and reconstruct spectra from file."""

    if Path(infile).absolute() == Path(outfile).absolute():
        raise ValueError("INFILE and OUTFILE must be different to avoid overwriting.")

    # File spectra to convert
    dset = xr.open_dataset(infile, engine=engine, **parse_kwargs(engine_kwargs))
    dset = dset.chunk(**parse_kwargs(chunks))

    # Sorting out input arguments
    if "," in freq_name:
        freq_name = freq_name.split(",")
    if "," in dir_name:
        dir_name = dir_name.split(",")

    # Run
    reconstructed = partition_and_reconstruct(
        dset,
        parts=parts,
        freq_name=freq_name,
        dir_name=dir_name,
        partition_method=partition_method,
        method_combine=method_combine,
    )

    # Save to file
    with ProgressBar():
        reconstructed.spec.to_netcdf(outfile)

    click.echo(f"Reconstructed file created: {outfile}")
