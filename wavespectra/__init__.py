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


__version__ = "4.2.0"


def _import_functions(pkgname="input", prefix="read"):
    """Import functions from pkgname with defined prefix at module level.

    Functions are imported here if:
        - they are defined in a module wavespectra.{pkgname}.{name}
        - they are named as {prefix}_{name}

    Example:
        - wavespectra.input.swan.read_swan
        - wavespectra.fit.jonswap.fit_jonswap

    """
    import os
    import glob
    from importlib import import_module

    here = os.path.dirname(os.path.abspath(__file__))
    for filename in glob.glob1(os.path.join(here, pkgname), "*.py"):
        module = os.path.splitext(filename)[0]
        if module == "__init__":
            continue
        func_name = f"{prefix}_{module}"
        try:
            globals()[func_name] = getattr(
                import_module(f"wavespectra.{pkgname}.{module}"), func_name
            )
        except Exception as exc:
            print(f"Cannot import reading function {func_name} because:\n{exc}")


_import_functions(pkgname="input", prefix="read")
