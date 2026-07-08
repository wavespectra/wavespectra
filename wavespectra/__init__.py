"""Define module attributes.

- Defining packaging attributes accessed by setup.py
- Making reading functions available at module level

"""

import warnings

try:
    from wavespectra.specdataset import SpecDataset
    from wavespectra.specarray import SpecArray
except ImportError as exc:
    warnings.warn(f"Cannot import accessors at the main module level:\n{exc}")


__version__ = "4.6.0"


def _import_functions(pkgname="input", prefix="read"):
    """Import functions from pkgname with defined prefix at module level.

    Functions are imported here if:
        - they are defined in a module wavespectra.{pkgname}.{name}
        - they are named as {prefix}_{name}

    Example:
        - wavespectra.input.swan.read_swan

    Import failures (e.g. a missing optional dependency) are reported as
    warnings rather than raised, so the package remains importable; the
    corresponding function is simply not made available.

    """
    from importlib import import_module
    from pathlib import Path

    here = Path(__file__).parent
    for path in sorted((here / pkgname).glob("*.py")):
        module = path.stem
        if module == "__init__":
            continue
        func_name = f"{prefix}_{module}"
        try:
            globals()[func_name] = getattr(
                import_module(f"wavespectra.{pkgname}.{module}"), func_name
            )
        except Exception as exc:
            warnings.warn(
                f"Cannot import function {func_name} from module "
                f"wavespectra.{pkgname}.{module}:\n{exc}"
            )


_import_functions(pkgname="input", prefix="read")
