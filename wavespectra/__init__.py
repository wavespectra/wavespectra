"""Define module attributes.

- Defining packaging attributes accessed by setup.py
- Making reading functions available at module level

"""

__version__ = '0.1.0'
__author__ = 'MetOcean Solutions Ltd'
__contact__ = 'r.guedes@metocean.co.nz'
__url__ = 'http://github.com/metocean/pyspectra'
__description__ = 'Ocean wave spectra tools'
__keywords__ = 'wave spectra xarray statistics analysis'

def _import_read_functions(pkgname='input'):
    """Make read functions available at module level.

    Functions are imported here if:
        - they are defined in a module wavespectra.input.{modname}
        - they are named as read_{modname}

    """
    import os
    import sys
    import glob
    from importlib import import_module

    for filename in glob.glob1(pkgname, '*.py'):
        module = os.path.splitext(filename)[0]
        if module == '__init__':
            continue
        func_name = 'read_{}'.format(module)
        try:
            globals()[func_name] = getattr(
                import_module('wavespectra.{}.{}'.format(pkgname, module)),
                func_name
                )
        except:
            print('Cannot import reading function:'.format(func_name),
                    sys.exc_info()[0])

_import_read_functions()