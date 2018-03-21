# from setuptools import setup
import os
import setuptools
from codecs import open
from numpy.distutils.core import setup
from numpy.distutils.misc_util import Configuration

import wavespectra

NAME = 'wavespectra'

CLASSIFIERS = [
    'Development Status :: 4 - Beta',
    'Intended Audience :: Science/Research',
    'License :: OSI Approved :: MIT License',
    'Programming Language :: Python :: 2',
    'Programming Language :: Python :: 2.7',
    'Topic :: Scientific/Engineering',
    'Topic :: Scientific/Engineering :: Physics',
    'Topic :: Scientific/Engineering :: Visualization',
]

PROJECT_URLS = {
    'Funding': 'http://www.metocean.co.nz',
    'Say Thanks!': 'http://www.metocean.co.nz',
    'Source': 'https://github.com/metocean/wavespectra',
    'Bug Reports': 'https://github.com/metocean/wavespectra/issues',
}

def _strip_comments(l):
    return l.split('#', 1)[0].strip()

def _pip_requirement(req):
    if req.startswith('-r '):
        _, path = req.split()
        return reqs(*path.split('/'))
    return [req]

def _reqs(*f):
    return [
        _pip_requirement(r) for r in (
            _strip_comments(l) for l in open(
                os.path.join(os.getcwd(), 'requirements', *f)).readlines()
        ) if r]

def reqs(*f):
    """Parse requirement file.

    Returns:
        List[str]: list of requirements specified in the file.

    Example:
        reqs('default.txt')          # requirements/default.txt
        reqs('extras', 'redis.txt')  # requirements/extras/redis.txt

    """
    return [req for subreq in _reqs(*f) for req in subreq]

def install_requires():
    """Get list of requirements required for installation."""
    return reqs('default.txt')

def extras_require():
    """Get map of all extra requirements."""
    return {'extra': reqs('extra.txt')}

def read(fname):
    return open(os.path.join(os.path.dirname(__file__), fname)).read()

def ext_configuration(parent_package='', top_path=None):
    config = Configuration('', '', '')
    config.add_extension('wavespectra.specpart', sources=['wavespectra/specpart/specpart.pyf',
                                                          'wavespectra/specpart/specpart.f90'])
    config.add_data_files('LICENSE.txt',
                          'wavespectra/core/attributes.yml')
    return config

kwargs = ext_configuration(top_path='').todict()

setup(
    name=NAME,
    version=wavespectra.__version__,
    description=wavespectra.__description__,
    long_description=read('README.rst'),
    keywords=wavespectra.__keywords__,
    author=wavespectra.__author__,
    author_email=wavespectra.__contact__,
    url=wavespectra.__url__,
    license='MIT',
    packages=setuptools.find_packages(exclude=['test*']),
    include_package_data=True,
    package_data={'attributes': ['wavespectra/core/attributes.yml']},
    platforms=['any'],
    install_requires=install_requires(),
    extras_require=extras_require(),
    setup_requires=['pytest-runner'],
    tests_require=reqs('test.txt'),
    python_requires=">=2.7, <3",
    classifiers=CLASSIFIERS,
    project_urls=PROJECT_URLS,
    **kwargs
)
