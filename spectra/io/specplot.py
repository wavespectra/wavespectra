"""
Plotting functions to support SpecArray object
"""
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib.colors import Normalize

def plot_watershed(darray, outfile=None, cmap='Accent', figsize=(15,10), wireframe=True):
    """
    3D surface plot of wave spectra coloured based on watershed partitions
    - darray :: frequency-direction data array with spectra, must be 2D
    - outfile :: path name for saving output figure file, if provided
    - cmap :: colormap for displaying partitioning map
    - figsize :: tuple defining figure size
    - wireframe :: if True 3d wireframe is overlaid
    """
    assert darray.ndim < 3, ('darray must be 2D freq-dir spectrum object, slice from %s coordinates before calling this function' %
        list(set(darray.dims).difference(['dir', 'freq'])))
    from pymo.core import specpart

    # Setting colourmap based on partitions
    partitions = specpart.specpart.partition(darray)
    nparts = partitions.max()
    vmin, vmax = 1, nparts
    norm = Normalize(vmin, vmax)
    cmap = plt.get_cmap(cmap, partitions.max())
    m = plt.cm.ScalarMappable(norm=norm, cmap=cmap)
    m.set_array([])
    fcolors = m.to_rgba(partitions)

    # plotting
    xx, yy = np.meshgrid(darray.dir, darray.freq)
    fig = plt.figure(figsize=figsize)
    ax = fig.add_subplot(111, projection='3d')
    ax.plot_surface(xx, yy, darray, rstride=1, cstride=1, facecolors=fcolors, vmin=vmin, vmax=vmax, shade=False)
    if wireframe:
        ax.plot_wireframe(xx, yy, darray, color='k', alpha=0.3)
    ax.set_xlabel(r'Direction ($degree$)')
    ax.set_ylabel(r'Frequency ($Hz$)')
    ax.set_zlabel(r'Energy density ($m^2/s^{-1}degree^{-1}$)') #(m2/Hz/deg)')
    ax.set_zlim((0, darray.values.max()))

    # Setting colorbar
    cbar = plt.colorbar(m)
    cbar.ax.get_yaxis().set_ticks([])
    for j, lab in enumerate(['$%i$'%(n) for n in range(nparts)]):
        cbar.ax.text(.5, (2 * j + 1) / float(2*nparts), lab, ha='center', va='center')
    cbar.ax.get_yaxis().labelpad = 20
    cbar.ax.set_ylabel('Partition number', rotation=270)

    if outfile is not None:
        plt.savefig(outfile)

    return fig


if __name__ == '__main__':
    from os.path import join, expanduser
    from spectra.io.swan import read_swan
    home = expanduser("~")

    ds = read_swan('/source/pyspectra/tests/antf0.20170207_06z.bnd.swn')
    darray = ds.efth.isel(time=0, site=0)
    fig = plot_watershed(darray, outfile=join(home, 'Pictures/partitions.png'), wireframe=True)

