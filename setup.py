"""
numpy.distutils.core.setup seems to handle subpackages better than
setuptools.setup
"""
from numpy.distutils.core import setup

setup(name='pyspectra',
      version='1.1.0',
      description='Spectra base class and tools based on DataArray',
      author='MetOcean Solutions Ltd.',
      author_email='r.guedes@metocean.co.nz',
      url='http://www.metocean.co.nz/',
      # packages=['pymsl', 'pymsl.dataio', 'pymsl.core'],
      )

