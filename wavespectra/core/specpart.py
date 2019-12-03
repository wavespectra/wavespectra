import numpy as np


def ptnghb(nk, nth):
    """Short description.

    Long description if required.

    Args:
        - `nk`: number of frequencies/wavenumbers in spectrum.
        - `nth`: number of directions in spectrum.

    Returns:
        - `neigh`: add description.

    """
    # build list of neighbours for each point
    nspec = nk * nth
    neigh = [[] for _ in range(nspec)]

    for n in range(nspec):
        ith = n % nth
        ik = n // nth

        if ik > 0:  # ... point at the bottom
            neigh[n].append(n - nth)

        if ik < nk - 1:  # ... point at the top
            neigh[n].append(n + nth)

        if ith > 0:  # ... point at the left
            neigh[n].append(n - 1)
        else:  # ... with wrap.
            neigh[n].append(n - 1 + nth)

        if ith < nth - 1:  # ... point at the right
            neigh[n].append(n + 1)
        else:  # ... with wrap.
            neigh[n].append(n + 1 - nth)

        if ik > 0 and ith > 0:  # ... point at the bottom-left
            neigh[n].append(n - nth - 1)
        elif ik > 0 and ith == 0:  # ... with wrap.
            neigh[n].append(n - nth - 1 + nth)

        if ik < nk - 1 and ith > 0:  # ... point at the top-left
            neigh[n].append(n + nth - 1)
        elif ik < nk - 1 and ith == 0:  # ... with wrap
            neigh[n].append(n + nth - 1 + nth)

        if ik > 0 and ith < nth - 1:  # ... point at the bottom-right
            neigh[n].append(n - nth + 1)
        elif ik > 0 and ith == nth - 1:  # ... with wrap
            neigh[n].append(n - nth + 1 - nth)

        if ik < nk - 1 and ith < nth - 1:  # ... point at the top-right
            neigh[n].append(n + nth + 1)
        elif ik < nk - 1 and ith == nth - 1:  # ... with wrap
            neigh[n].append(n + nth + 1 - nth)

    return neigh


def partition(spec, ihmax=200):
    """Return the array with numbered partitions.

    Args:
        - `spec`: 2D spectrum array Ed(y=freq, x=dir).
        - `ihmax`: add description.

    Returns:
        - `part_array`: array with same shape of `spec` with
          the numbered partitions.

    """
    nk, nth = spec.shape  # ensure this is the correct order
    neigh = ptnghb(nk, nth)

    nspec = spec.size
    zmin = spec.min()
    zmax = spec.max()
    zp = -spec.flatten() + zmax
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
            ip = ind[m]
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

    part_array = imo.reshape(spec.shape)
    return part_array
