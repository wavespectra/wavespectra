from numpy.distutils.core import setup, Extension


setup(
    name="wavespectra",
    ext_modules=[
        Extension(
            name='wavespectra.specpart',
            sources=[
                "wavespectra/specpart/specpart.pyf",
                "wavespectra/specpart/specpart.f90",
            ]
        )
    ]
)
