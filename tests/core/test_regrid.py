import wavespectra
from wavespectra import read_triaxys
from wavespectra.core.attributes import attrs
import os
from pathlib import Path
import numpy as np

FILES_DIR = Path(__file__).parent.parent / 'sample_files'

def test_regrid_triaxys():

    buoy = read_triaxys(str(FILES_DIR / 'triaxys.DIRSPEC'))

    new_freq = np.linspace(0, 0.5, num=25)
    new_dir = np.arange(0, 360, step=5)

    buoy_regrid = buoy.spec.regrid_spec(new_dir=new_dir, new_freq=new_freq)

    import matplotlib.pyplot as plt

    plt.figure()
    buoy.spec.plot(as_log10=False)

    plt.figure()
    buoy_regrid.spec.plot(as_log10=False)
    plt.show()


    # how to test properly?
    buoy.spec.hs()
    buoy_regrid.spec.hs()


def test_regrid_multiple():
    pass
    # TODO, test with octupus file (after merge)



