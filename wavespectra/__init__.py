"""Define module attributes.

- Defining packaging attributes accessed by setup.py
- Making reading functions available at module level

"""
import warnings

try:
    from wavespectra.specdataset import SpecDataset
    from wavespectra.specarray import SpecArray
except ImportError:
    warnings.warn("Cannot import accessors at the main module level")


__version__ = "3.6.1"
__author__ = "Wavespectra Developers"
__contact__ = "r.guedes@oceanum.science"
__url__ = "http://github.com/wavespectra/wavespectra"
__description__ = "Library for ocean wave spectra"
__keywords__ = "wave spectra ocean xarray statistics analysis"


def _import_read_functions(pkgname="input"):
    """Make read functions available at module level.

    Functions are imported here if:
        - they are defined in a module wavespectra.input.{modname}
        - they are named as read_{modname}

    """
    import os
    import glob
    from importlib import import_module

    here = os.path.dirname(os.path.abspath(__file__))
    for filename in glob.glob1(os.path.join(here, pkgname), "*.py"):
        module = os.path.splitext(filename)[0]
        if module == "__init__":
            continue
        func_name = f"read_{module}"
        try:
            globals()[func_name] = getattr(
                import_module(f"wavespectra.{pkgname}.{module}"), func_name
            )
        except Exception as exc:
            print(f"Cannot import reading function {func_name}:\n{exc}")


_import_read_functions()
