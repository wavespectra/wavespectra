from numpy.distutils.core import setup
from numpy.distutils.misc_util import Configuration

def ext_configuration(parent_package='',top_path=None):
    config = Configuration('', parent_package, top_path)
    config.add_extension('spectra.specpart', sources=['spectra/specpart/specpart.pyf',
                                                                'spectra/specpart/specpart.f90'])
    return config

k = ext_configuration(top_path='').todict()
k['packages'] = ['spectra', 'spectra.io']

setup(name='pyspectra',
      version='1.1.0',
      description='Spectra base class and tools based on DataArray',
      author='MetOcean Solutions Ltd.',
      author_email='r.guedes@metocean.co.nz',
      url='http://www.metocean.co.nz/',
      **k
      )
