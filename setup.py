from setuptools import setup, Extension
from numpy import get_include as get_numpy_include

setup(
    name="wavespectra",
    ext_modules=[
        Extension(
            name="wavespectra.partition.specpartc",
            sources=[
                "wavespectra/partition/specpart/specpartc_wrap.c",
                "wavespectra/partition/specpart/specpartc.c",
            ],
            include_dirs=[get_numpy_include()],
        )
    ],
)
