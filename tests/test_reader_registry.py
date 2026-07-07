"""Parity tests between reader modules, top-level functions and xarray engines.

These guard against the plugin registries drifting apart:
    - every module in wavespectra/input must expose a read_<module> function;
    - every reader must be importable and attached at the top level;
    - every reader module must define a BackendEntrypoint registered as an
      xarray engine of the same name in pyproject.toml.
"""

from pathlib import Path

import pytest
import xarray as xr
from xarray.backends import BackendEntrypoint

import wavespectra

INPUT_DIR = Path(wavespectra.__file__).parent / "input"
READER_MODULES = sorted(p.stem for p in INPUT_DIR.glob("*.py") if p.stem != "__init__")


def test_reader_modules_found():
    assert len(READER_MODULES) >= 20


@pytest.mark.parametrize("module", READER_MODULES)
def test_reader_importable_at_top_level(module):
    """Every input module exposes read_<module>, attached to wavespectra."""
    func = getattr(wavespectra, f"read_{module}", None)
    assert callable(func), (
        f"wavespectra.read_{module} is missing; the module failed to import "
        "(run `from wavespectra.input import {module}` to see the error)"
    )


@pytest.mark.parametrize("module", READER_MODULES)
def test_reader_registered_as_xarray_engine(module):
    """Every reader module is registered as an xarray backend engine."""
    if module == "dataset":
        pytest.skip("read_dataset formats an existing dataset, not a file")
    engines = xr.backends.list_engines()
    assert module in engines, (
        f"'{module}' is not a registered xarray engine; add it to "
        '[project.entry-points."xarray.backends"] in pyproject.toml'
    )
    assert isinstance(engines[module], BackendEntrypoint)


@pytest.mark.parametrize("module", READER_MODULES)
def test_backend_entrypoint_signature_matches_reader(module):
    """Backend open_dataset kwargs must exist in the reader signature.

    Guards against entrypoints passing arguments the reader does not
    accept (e.g. the DatawellBackendEntrypoint filetype bug).
    """
    import inspect
    from importlib import import_module

    if module == "dataset":
        pytest.skip("read_dataset has no backend entrypoint")

    mod = import_module(f"wavespectra.input.{module}")
    entrypoints = [
        cls
        for name, cls in vars(mod).items()
        if isinstance(cls, type)
        and issubclass(cls, BackendEntrypoint)
        and cls is not BackendEntrypoint
    ]
    assert entrypoints, f"no BackendEntrypoint class in wavespectra.input.{module}"

    reader = getattr(mod, f"read_{module}")
    reader_params = set(inspect.signature(reader).parameters)
    for cls in entrypoints:
        params = inspect.signature(cls.open_dataset).parameters
        kwargs = {
            name
            for name, p in params.items()
            if name not in ("self", "filename_or_obj", "drop_variables")
            and p.kind is not inspect.Parameter.VAR_KEYWORD
        }
        unknown = kwargs - reader_params
        assert not unknown, (
            f"{cls.__name__}.open_dataset passes argument(s) {sorted(unknown)} "
            f"that read_{module} does not accept"
        )
