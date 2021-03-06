"""Watershed partitioning."""
import numpy as np
import xarray as xr

from wavespectra.core.attributes import attrs
from wavespectra.core.utils import D2R, R2D, celerity
from wavespectra.core.npstats import hs


def ptnghb(nk, nth):
    """Build list of neighbours for each point.

    Args:
        - nk (int): Number of frequencies in spectrum.
        - nth (int): Number of directions in spectrum.

    Returns:
        - neigh (2darray): 9 by nspec array with indices of neighbours for each point.

    """
    nspec = nk * nth
    neigh = np.zeros((9, nspec), dtype="int32")

    for n in range(1, nspec + 1):

        j = int((n - 1) / nk + 1)
        i = int(n - (j - 1) * nk)
        k = int(0)

        # ... point at the left(1)
        if i != 1:
            k = k + 1
            neigh[k - 1, n - 1] = n - 1

        # ... point at the right (2)
        if i != nk: 
            k = k + 1
            neigh[k - 1, n - 1] = n + 1

        # ... point at the bottom(3)
        if j != 1:
            k = k + 1
            neigh[k - 1, n - 1] = n - nk

        # ... add point at bottom_wrap to top
        if j == 1:
            k = k + 1
            neigh[k - 1, n - 1] = nspec - (nk - i)

        # ... point at the top(4)
        if j != nth:
            k = k + 1
            neigh[k - 1, n - 1 ] = n + nk

        # ... add point to top_wrap to bottom
        if j == nth:
            k = k + 1
            neigh[k - 1, n - 1] = n - (nth - 1) * nk

        # ... point at the bottom, left(5)
        if i != 1 and j != 1:
            k = k + 1
            neigh[k - 1, n - 1] = n - nk - 1

        # ... point at the bottom, left with wrap.
        if i != 1 and j == 1:
            k = k + 1
            neigh[k - 1, n - 1] = n - 1 + nk * (nth - 1)

        # ... point at the bottom, right(6)
        if i != nk and j != 1:
            k = k + 1
            neigh[k - 1, n - 1] = n - nk + 1

        # ... point at the bottom, right with wrap
        if i != nk and j == 1:
            k = k + 1
            neigh[k - 1, n - 1] = n + 1 + nk * (nth - 1)

        # ... point at the top, left(7)
        if i != 1 and j != nth:
            k = k + 1
            neigh[k - 1, n - 1] = n + nk - 1

    # ... point at the top, left with wrap
        if i != 1 and j == nth:
            k = k + 1
            neigh[k - 1, n - 1] = n - 1 - nk * (nth - 1)

        # ... point at the top, right(8)
        if i != nk and j != nth:
            k = k + 1
            neigh[k - 1, n - 1] = n + nk + 1

        # ... point at top, right with wrap
        if i!= nk and j == nth:
            k = k + 1
            neigh[k - 1, n - 1] = n + 1 - nk * (nth - 1)

        neigh[8, n - 1] = k

    return neigh


def ptsort(imi, ihmax):
    """Sort discretised image."""
    numv = np.zeros(ihmax, dtype="int32")
    iaddr = np.zeros(ihmax,dtype="int32")
    nspec = len(imi)
    ind = np.zeros(nspec, dtype="int32")
    iorder = np.zeros(nspec, dtype="int32")

    # 1.  occurences per height
    for i in range(nspec):
        numv[imi[i]] = numv[imi[i]] + 1

    # 2.  starting address per height
    iaddr[0] = 0
    for i in range(ihmax-1):
        iaddr[i + 1] = iaddr[i] + numv[i]

    # 3.  order points
    for i in range(nspec):
        iv = imi[i]
        inn = iaddr[iv]
        iorder[i] = inn
        iaddr[iv] = inn + 1

    # 4.  sort points
    for i in range(nspec):
        ind[iorder[i]] = i

    return ind


def specpart(spectrum, ihmax=200):
    """."""

    def fifo_add(iv, iq_end):
        """Add point to fifo queue."""
        iq[iq_end] = iv
        iq_end = iq_end + 1
        if iq_end > nspec - 1:
            iq_end = 0
        return iq_end

    def fifo_first(iq_start):
        """Get point out of queue."""
        iv = iq[iq_start]
        iq_start = iq_start + 1
        if iq_start > nspec - 1:
            iq_start = 0
        return iv, iq_start

    def fifo_empty():
        """Check if queue is empty."""
        if iq_start != iq_end:
            iempty = 0
        else:
            iempty = 1
        return iempty

    # -------------------------------------------------------------------- /
    # 0.  initializations
    nk, nth = spectrum.shape
    neigh = ptnghb(nk, nth)
    neigh[:8, :] = neigh[:8, :] - 1

    nspec = spectrum.size
    zmin = spectrum.min()
    zmax = spectrum.max()
    zp = zmax - spectrum.flatten(order="F")
    mask = -2
    init = -1
    iwshed =  0
    imo = np.full(nspec, init, dtype="int32")
    ic_label = 0
    imd = np.zeros(nspec, dtype="int32")
    ifict_pixel = -101
    iq_start =  0
    iq_end =  0
    zpmax = zp.max()

    iq = np.full(nspec, -30000, dtype="int32")

    fact = (ihmax - 1) / (zmax - zmin)
    imi = np.maximum(1, np.minimum(ihmax, np.round(1 + zp * fact))).astype("int32") - 1

    ind = ptsort(imi, ihmax)

    dummy = 0

    # 1.  loop over levels
    m = 0
    for ih in range(ihmax):
        msave = m
        # 1.a pixels at level ih
        while True:
            ip = ind[m]
            if imi[ip] != ih:
                break
            # flag the point, if it stays flagged, it is a separate minimum.
            imo[ip] = mask
            # consider neighbors. if there is neighbor, set distance and add to queue.
            for i in range(neigh[8, ip]):
                ipp = neigh[i, ip]
                if imo[ipp] > 0 or imo[ipp] == iwshed:
                    imd[ip] = 1
                    iq_end = fifo_add(ip, iq_end)
                    break
            if m + 1 > nspec - 1:
                break
            else:
                m = m + 1

        # 1.b process the queue

        ic_dist = 1

        iq_end = fifo_add(ifict_pixel, iq_end)

        while True:
            ip, iq_start = fifo_first(iq_start)
            # check for end of processing
            if ip == ifict_pixel:
                iempty = fifo_empty()
                if iempty == 1:
                    break
                else:
                    iq_end = fifo_add(ifict_pixel, iq_end)
                    ic_dist = ic_dist + 1
                    ip, iq_start = fifo_first(iq_start)
            # process queue
            for i in range(neigh[8, ip]):
                ipp = neigh[i, ip]
                # check for labeled watersheds or basins
                if imd[ipp] < ic_dist and imo[ipp] >= 0:
                    if imo[ipp] > 0:
                        if (imo[ip] == mask or imo[ip] == iwshed):
                            imo[ip] = imo[ipp]
                        elif imo[ip] != imo[ipp]:
                            imo[ip] = iwshed
                    elif imo[ip] == mask:
                        imo[ip] = iwshed
                elif imo[ipp]  == mask and imd[ipp] == 0:
                    imd[ipp] = ic_dist + 1
                    iq_end = fifo_add(ipp, iq_end)

        # 1.c check for mask values in imo to identify new basins
        m = msave
        while True:
            ip = ind[m]
            if imi[ip] != ih:
                # print(f"imi[ip], ih, iq_end {imi[ip]} {ih} {iq_end}")
                break
            imd[ip] = 0
            if imo[ip] == mask:
                # ... new label for pixel
                ic_label = ic_label + 1
                iq_end = fifo_add(ip, iq_end)
                imo[ip] = ic_label
                # ... and all connected to it ...
                while True:
                    iempty = fifo_empty()
                    if iempty == 1:
                        break
                    ipp, iq_start = fifo_first(iq_start)
                    for i in range(neigh[8, ipp]):
                        ippp = neigh[i, ipp]
                        if imo[ippp] == mask:
                            iq_end = fifo_add(ippp, iq_end)
                            imo[ippp] = ic_label

            if m + 1 > nspec - 1:
                break
            else:
                m = m + 1

    # 2.  find nearest neighbor of 0 watershed points and replace
    #     use original input to check which group to affiliate with 0
    #     soring changes first in imd to assure symetry in adjustment.

    for j in range(5):
        imd = imo
        for jl in range(nspec):
            ipt = -1
            if imo[jl] == 0:
                ep1 = zpmax
                for jn in range(neigh[8, jl]):
                    diff = abs(zp[jl] - zp[neigh[jn,jl]])
                    if diff <= ep1 and imo[neigh[jn,jl]] != 0:
                        ep1 = diff
                        ipt = jn + 1
                print(f"jl jn ipt   {jl}   {jn}   {ipt}")
                if ipt > 0:
                    imd[jl] = imo[neigh[ipt, jl]]
        imo = imd
        if min(imo) > 0:
            print(f"Exiting at j == {j}")
            break

    # npart = ic_label
    return imo.reshape((nk, nth), order="F")



def specpart0(spectrum, ihmax=200):
    """Watershed partitioning.

    Args:
        - spectrum (2darray): Spectrum array E(f, d).
        - ihmax (int): Number of iterations.

    Returns:
        - part_array (2darray): Numbered partitions array with same shape as spectrum.

    """
    nk, nth = spectrum.shape  # ensure this is the correct order
    neigh = ptnghb(nk, nth).T - 1

    nspec = spectrum.size
    zmin = spectrum.min()
    zmax = spectrum.max()
    zp = -spectrum.flatten() + zmax
    fact = (ihmax - 1) / (zmax - zmin)
    imi = np.around(zp * fact).astype(int)
    ind = zp.argsort()

    #  0.  initializations
    imo = -np.ones(nspec, dtype=int)  # mask = -2, init = -1, iwshed =  0
    ic_label = 0
    imd = np.zeros(nspec, dtype=int)
    ifict_pixel = -100
    iq1 = []
    mstart = 0

    # 1.  loop over levels
    for ih in range(ihmax):
        # 1.a pixels at level ih
        for m in range(mstart, nspec):
            ip = ind[m-1]
            if imi[ip] != ih:
                break

            # flag the point, if it stays flagged, it is a separate minimum.
            imo[ip] = -2

            # if there is neighbor, set distance and add to queue.
            if any(imo[neigh[ip]] >= 0):
                imd[ip] = 1
                iq1.append(ip)

        # 1.b process the queue
        ic_dist = 1
        iq1.append(ifict_pixel)

        while True:
            ip = iq1.pop(0)

            # check for end of processing
            if ip == ifict_pixel:
                if not iq1:
                    break

                iq1.append(ifict_pixel)
                ic_dist += 1
                ip = iq1.pop(0)

            # process queue
            for ipp in neigh[ip]:
                # check for labeled watersheds or basins
                if imo[ipp] >= 0 and imd[ipp] < ic_dist:
                    if imo[ipp] > 0:
                        if imo[ip] in [-2, 0]:
                            imo[ip] = imo[ipp]
                        elif imo[ip] != imo[ipp]:
                            imo[ip] = 0
                    elif imo[ip] == -2:
                        imo[ip] = 0
                elif imo[ipp] == -2 and imd[ipp] == 0:
                    imd[ipp] = ic_dist + 1
                    iq1.append(ipp)

        # 1.c check for mask values in imo to identify new basins
        for ip in ind[mstart:m]:
            imd[ip] = 0
            if imo[ip] == -2:
                ic_label += 1  # ... new basin
                iq2 = [ip]
                while iq2:
                    imo[iq2] = ic_label
                    iqset = set([n for i in iq2 for n in neigh[i]])
                    iq2 = [
                        i for i in iqset if imo[i] == -2
                    ]  # ... all masked points connected to it
        mstart = m

    # 2.  find nearest neighbor of 0 watershed points and replace
    #     use original input to check which group to affiliate with 0
    #     storing changes first in imd to assure symetry in adjustment.
    for _ in range(5):
        watershed0 = np.where(imo == 0)[0]
        if not any(watershed0):
            break

        newvals = []
        for jl in watershed0:
            jnl = [j for j in neigh[jl] if imo[j] != 0]
            if any(jnl):
                ipt = abs(zp[jnl] - zp[jl]).argmin()
                newvals.append(imo[jnl[ipt]])
            else:
                newvals.append(0)
        imo[watershed0] = newvals

    part_array = imo.reshape(spectrum.shape)
    return part_array


if __name__ == "__main__":
    from wavespectra import read_wavespectra

    dset = read_wavespectra("/source/consultancy/jogchum/route/route_feb21/p04/spec.nc")
    dsi = dset.isel(time=0, freq=slice(None, 10), dir=slice(None, 9)).load()
    spectrum = dsi.efth.values
    # nk, nth = spectrum.shape
    # ptnghb(nk, nth)
    p = specpart(spectrum)
    print(p)
