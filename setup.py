from setuptools import setup, Extension
from numpy import get_include as get_numpy_include

setup(
    name="wavespectra",
    ext_modules=[
        Extension(
            name="wavespectra.partition.specpart",
            sources=[
                "wavespectra/partition/specpart/specpart_wrap.c",
                "wavespectra/partition/specpart/specpart.c",
            ],
            include_dirs=[get_numpy_include()],
        )
    ],
)
