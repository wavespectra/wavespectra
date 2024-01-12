from numpy.distutils.core import setup, Extension


setup(
    name="wavespectra",
    ext_modules=[
        Extension(
            name='wavespectra.specpart',
            sources=[
                "wavespectra/partition/specpart/specpart.pyf",
                "wavespectra/partition/specpart/specpart.f90",
            ]
        )
    ]
)
