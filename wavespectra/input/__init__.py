"""Access functions to read spectra from data files.

The following structure is expected:
    - Reading functions for each data file type defined in specific modules
    - Modules named as {datatype}.py, e.g. swan.py
    - Functions named as read_{dataname}, e.g. read_swan

All functions defined with these conventions will be dynamically
imported at the module level

"""