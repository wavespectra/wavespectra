from setuptools import setup

install_requires = [
    'xarray>=0.9',
    'dask',
    'toolz',
    'numpy',
    'cloudpickle',
    ]

test_requires = [
    'unittest',
    ]

def read(fname):
    return open(os.path.join(os.path.dirname(__file__), fname)).read()

if __name__ == '__main__':
    setup(name ='spectra',
          version='0.1',
          description='Spectra base class and tools based on DataArray',
          author='MetOcean Solutions Ltd',
          install_requires=install_requires,
          test_require=test_requires,
          author_email='r.guedes@metocean.co.nz',
          url='http://github.com/metocean/pyspectra',
          packages=['spectra'],
)