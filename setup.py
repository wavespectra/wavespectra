# from setuptools import setup
import os
from numpy.distutils.core import setup
from numpy.distutils.misc_util import Configuration

import spectra

install_requires = [
    'xarray>=0.10.0',
    'dask',
    'toolz',
    'cloudpickle',
    'sortedcontainers',
    'scipy',
    'sympy',
    ]

test_requires = [
    'unittest',
    ]

def read(fname):
    return open(os.path.join(os.path.dirname(__file__), fname)).read()

def ext_configuration(parent_package='', top_path=None):
    config = Configuration('', '', '')
    config.add_extension('spectra.specpart', sources=['spectra/specpart/specpart.pyf',
                                                      'spectra/specpart/specpart.f90'])
    return config

k = ext_configuration(top_path='').todict()

if __name__ == '__main__':
    setup(name ='spectra',
          version=spectra.__version__,
          description='Spectra base class and tools based on DataArray',
          long_description=read('README.md'),
          author='MetOcean Solutions Ltd',
          install_requires=install_requires,
          test_require=test_requires,
          author_email='r.guedes@metocean.co.nz',
          url='http://github.com/metocean/pyspectra',
          packages=['spectra','spectra.construct'],
          **k
)
