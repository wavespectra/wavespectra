/* Watershed partitioning of a 2D wave spectrum.
 *
 * C port of the WAVEWATCH III w3partmd watershed algorithm (Vincent et al.
 * immersion flooding), as previously implemented in Fortran in wavespectra
 * < v4.  Verified to produce partition maps identical to the Fortran
 * reference over model output, buoy-like spectra and synthetic cases.
 *
 * The implementation keeps no global state: all work arrays are allocated
 * per call, which makes it reentrant and thread-safe (safe under e.g.
 * dask's threaded scheduler).
 */

#include <math.h>
#include <stdlib.h>
#include "specpart.h"

/* Internal work arrays layout notes:
 *   zp     spectral values copied into "freq fastest" order:
 *          zp[ik + nk * ith], flipped so the spectral peak becomes 0.
 *   imi    discretised (binned) level of each point, 0 .. ihmax-1.
 *   ind    point indices sorted by increasing level (stable counting sort).
 *   imo    output partition label per point (watershed result).
 *   neigh  neigh[9*n .. 9*n+7] neighbour indices of point n,
 *          neigh[9*n+8] the number of neighbours.  The directional axis
 *          wraps (directions are circular); the frequency axis does not.
 */

/* Build the 8-connected neighbour table with directional wrap. */
static void ptnghb(int *neigh, int nk, int nth) {
  int nspec = nk * nth;
  int n, j, i, k;

  for (n = 0; n < nspec; n++) {
    j = n / nk;      /* direction index */
    i = n - j * nk;  /* frequency index */
    k = -1;

    /* left (lower frequency) */
    if (i != 0) neigh[++k + 9 * n] = n - 1;
    /* right (higher frequency) */
    if (i != nk - 1) neigh[++k + 9 * n] = n + 1;
    /* bottom (previous direction), wrapping */
    if (j != 0) neigh[++k + 9 * n] = n - nk;
    else neigh[++k + 9 * n] = nspec - (nk - i);
    /* top (next direction), wrapping */
    if (j != nth - 1) neigh[++k + 9 * n] = n + nk;
    else neigh[++k + 9 * n] = n - (nth - 1) * nk;
    /* bottom-left */
    if (i != 0 && j != 0) neigh[++k + 9 * n] = n - nk - 1;
    if (i != 0 && j == 0) neigh[++k + 9 * n] = n - 1 + nk * (nth - 1);
    /* bottom-right */
    if (i != nk - 1 && j != 0) neigh[++k + 9 * n] = n - nk + 1;
    if (i != nk - 1 && j == 0) neigh[++k + 9 * n] = n + 1 + nk * (nth - 1);
    /* top-left */
    if (i != 0 && j != nth - 1) neigh[++k + 9 * n] = n + nk - 1;
    if (i != 0 && j == nth - 1) neigh[++k + 9 * n] = n - 1 - nk * (nth - 1);
    /* top-right */
    if (i != nk - 1 && j != nth - 1) neigh[++k + 9 * n] = n + nk + 1;
    if (i != nk - 1 && j == nth - 1) neigh[++k + 9 * n] = n + 1 - nk * (nth - 1);

    neigh[8 + 9 * n] = k + 1;
  }
}

/* Stable counting sort of point indices by increasing level.
 * numv/iaddr are scratch buffers of size ihmax; iorder of size nspec. */
static void ptsort(const int *imi, int *ind, int ihmax, int nspec,
                   int *numv, int *iaddr, int *iorder) {
  int i, in, iv;

  for (i = 0; i < ihmax; i++) numv[i] = 0;
  for (i = 0; i < nspec; i++) numv[imi[i]]++;

  iaddr[0] = 0;
  for (i = 0; i < ihmax - 1; i++) iaddr[i + 1] = iaddr[i] + numv[i];

  for (i = 0; i < nspec; i++) {
    iv = imi[i];
    in = iaddr[iv];
    iorder[i] = in;
    iaddr[iv] = in + 1;
  }
  for (i = 0; i < nspec; i++) ind[iorder[i]] = i;
}

/* Circular FIFO queue of capacity nspec. */
static int fifo_add(int *iq, int nspec, int iq_end, int iv) {
  iq[iq_end] = iv;
  if (iq_end > nspec - 2) return 0;
  return iq_end + 1;
}

static int fifo_empty(int iq_start, int iq_end) {
  return iq_start == iq_end;
}

static int fifo_first(const int *iq, int nspec, int *iq_start) {
  int iv = iq[*iq_start];
  *iq_start = *iq_start + 1;
  if (*iq_start > nspec - 1) *iq_start = 0;
  return iv;
}

/* Immersion flooding: label basins level by level.
 * iq and imd are scratch buffers of size nspec.  Returns the number of
 * partitions found. */
static int pt_fld(const int *imi, const int *ind, int *imo, const float *zp,
                  const int *neigh, int nspec, int ihmax, int *iq, int *imd) {
  const int mask = -2, init = -1, iwshed = 0, ifict_pixel = -100;
  int ic_label = 0, m, ih, msave;
  int ip, i, ipp, ic_dist, ippp;
  int jl, jn, ipt, j;
  int iq_start = 0, iq_end = 0;
  float zpmax, ep1, diff;

  for (i = 0; i < nspec; i++) imo[i] = init;
  for (i = 0; i < nspec; i++) imd[i] = 0;

  zpmax = zp[0];
  for (i = 1; i < nspec; i++) zpmax = fmaxf(zpmax, zp[i]);

  /* 1. loop over levels (binned spectral values) */
  m = 0;
  for (ih = 0; ih < ihmax; ih++) {
    msave = m;

    /* 1.a flag pixels at level ih; queue those bordering existing labels */
    for (;;) {
      ip = ind[m];
      if (imi[ip] != ih) break;

      /* flag the point; if it stays flagged it is a separate minimum */
      imo[ip] = mask;

      for (i = 0; i < neigh[8 + 9 * ip]; i++) {
        ipp = neigh[i + 9 * ip];
        if (imo[ipp] > 0 || imo[ipp] == iwshed) {
          imd[ip] = 1;
          iq_end = fifo_add(iq, nspec, iq_end, ip);
          break;
        }
      }

      if (m > nspec - 2) break;
      m++;
    }

    /* 1.b process the queue: extend existing basins into this level */
    ic_dist = 1;
    iq_end = fifo_add(iq, nspec, iq_end, ifict_pixel);
    for (;;) {
      ip = fifo_first(iq, nspec, &iq_start);
      if (ip == ifict_pixel) {
        if (fifo_empty(iq_start, iq_end)) break;
        iq_end = fifo_add(iq, nspec, iq_end, ifict_pixel);
        ic_dist++;
        ip = fifo_first(iq, nspec, &iq_start);
      }
      for (i = 0; i < neigh[8 + 9 * ip]; i++) {
        ipp = neigh[i + 9 * ip];
        if (imd[ipp] < ic_dist && (imo[ipp] > 0 || imo[ipp] == iwshed)) {
          if (imo[ipp] > 0) {
            if (imo[ip] == mask || imo[ip] == iwshed) imo[ip] = imo[ipp];
            else if (imo[ip] != imo[ipp]) imo[ip] = iwshed;
          } else if (imo[ip] == mask) {
            imo[ip] = iwshed;
          }
        } else if (imo[ipp] == mask && imd[ipp] == 0) {
          imd[ipp] = ic_dist + 1;
          iq_end = fifo_add(iq, nspec, iq_end, ipp);
        }
      }
    }

    /* 1.c any pixel still flagged is a new basin (local minimum) */
    m = msave;
    for (;;) {
      ip = ind[m];
      if (imi[ip] != ih) break;
      imd[ip] = 0;

      if (imo[ip] == mask) {
        ic_label++;
        iq_end = fifo_add(iq, nspec, iq_end, ip);
        imo[ip] = ic_label;
        for (;;) {
          if (fifo_empty(iq_start, iq_end)) break;
          ipp = fifo_first(iq, nspec, &iq_start);
          for (i = 0; i < neigh[8 + 9 * ipp]; i++) {
            ippp = neigh[i + 9 * ipp];
            if (imo[ippp] == mask) {
              iq_end = fifo_add(iq, nspec, iq_end, ippp);
              imo[ippp] = ic_label;
            }
          }
        }
      }

      if (m > nspec - 2) break;
      m++;
    }
  }

  /* 2. reassign watershed (0) points to the nearest basin, using the
   * original values; changes staged in imd for symmetry. */
  for (j = 0; j < 5; j++) {
    int minv;
    for (i = 0; i < nspec; i++) imd[i] = imo[i];
    for (jl = 0; jl < nspec; jl++) {
      ipt = -1;
      if (imo[jl] == 0) {
        ep1 = zpmax;
        for (jn = 0; jn < neigh[8 + 9 * jl]; jn++) {
          diff = fabsf(zp[jl] - zp[neigh[jn + 9 * jl]]);
          if (diff <= ep1 && imo[neigh[jn + 9 * jl]] != 0) {
            ep1 = diff;
            ipt = jn;
          }
        }
        if (ipt > -1) imd[jl] = imo[neigh[ipt + 9 * jl]];
      }
    }
    for (i = 0; i < nspec; i++) imo[i] = imd[i];
    minv = imo[0];
    for (i = 1; i < nspec; i++)
      if (imo[i] < minv) minv = imo[i];
    if (minv > 0) break;
  }

  return ic_label;
}

int specpart_partition(const float *spec, int *ipart, int nk, int nth,
                       int ihmax) {
  int nspec = nk * nth;
  double zmin, zmax, fact;
  int i, ik, ith, npart;

  float *zp = NULL;
  int *imi = NULL, *ind = NULL, *imo = NULL, *neigh = NULL;
  int *iq = NULL, *imd = NULL, *numv = NULL, *iaddr = NULL, *iorder = NULL;

  zp = malloc(nspec * sizeof(float));
  imi = malloc(nspec * sizeof(int));
  ind = malloc(nspec * sizeof(int));
  imo = malloc(nspec * sizeof(int));
  neigh = malloc(9 * (size_t)nspec * sizeof(int));
  iq = malloc(nspec * sizeof(int));
  imd = malloc(nspec * sizeof(int));
  numv = malloc(ihmax * sizeof(int));
  iaddr = malloc(ihmax * sizeof(int));
  iorder = malloc(nspec * sizeof(int));
  if (!zp || !imi || !ind || !imo || !neigh || !iq || !imd || !numv ||
      !iaddr || !iorder) {
    npart = -1;
    goto cleanup;
  }

  /* Copy the C-ordered (nk, nth) spectrum into freq-fastest order. */
  for (ith = 0; ith < nth; ith++)
    for (ik = 0; ik < nk; ik++)
      zp[ik + nk * ith] = spec[ik * nth + ith];

  zmin = zp[0];
  zmax = zp[0];
  for (i = 1; i < nspec; i++) {
    zmin = fmin(zmin, zp[i]);
    zmax = fmax(zmax, zp[i]);
  }

  /* A (nearly) flat spectrum has no partitions. */
  if (zmax - zmin < 1e-9) {
    for (i = 0; i < nspec; i++) ipart[i] = 0;
    npart = 0;
    goto cleanup;
  }

  /* Flip so the spectral peak becomes 0 and bin into ihmax levels. */
  for (i = 0; i < nspec; i++) zp[i] = zmax - zp[i];
  fact = (ihmax - 1.0) / (zmax - zmin);
  for (i = 0; i < nspec; i++)
    imi[i] = fmax(0, fmin(ihmax - 1, round(zp[i] * fact)));

  ptsort(imi, ind, ihmax, nspec, numv, iaddr, iorder);
  ptnghb(neigh, nk, nth);
  npart = pt_fld(imi, ind, imo, zp, neigh, nspec, ihmax, iq, imd);

  /* Copy the freq-fastest result back into C order. */
  for (ik = 0; ik < nk; ik++)
    for (ith = 0; ith < nth; ith++)
      ipart[ik * nth + ith] = imo[ik + nk * ith];

cleanup:
  free(zp);
  free(imi);
  free(ind);
  free(imo);
  free(neigh);
  free(iq);
  free(imd);
  free(numv);
  free(iaddr);
  free(iorder);
  return npart;
}
