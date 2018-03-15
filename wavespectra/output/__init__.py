"""Output function plugins to save spectra as data files.

The functions are designed to be dynamically plugged in to
the SpecDataset class as methods.

The following structure is required for defining the plugins:
    - Functions must take (self) as the first arg
    - Function name must start with 'to_', e.g. 'to_swan'

Ideally the plugins must be organised as:
    - Writing functions for each data file type defined in specific modules
    - Modules named as {datatype}.py, e.g. swan.py

All functions defined with these conventions will be dynamically
attached to the SpecDataset class. Attributes defined within that
class scope can be accessed within the functions since they will
be attached to the same scope.

"""