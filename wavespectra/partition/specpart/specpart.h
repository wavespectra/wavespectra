#ifndef SPECPART_H
#define SPECPART_H

/* Watershed partitioning of a 2D wave spectrum.
 *
 * C port of the WAVEWATCH III w3partmd watershed algorithm (Vincent et al.
 * immersion flooding), as previously implemented in Fortran in wavespectra
 * < v4.  The implementation is self-contained and reentrant: it keeps no
 * global state and is safe to call concurrently from multiple threads.
 *
 * Arguments:
 *   spec   input spectrum, C-ordered (nk, nth), i.e. spec[ik * nth + ith].
 *   ipart  output partition map, C-ordered (nk, nth).  Partition ids start
 *          at 1; 0 is only returned for a flat spectrum (no partitions).
 *   nk     number of frequencies (rows).
 *   nth    number of directions (columns).  The directional axis wraps.
 *   ihmax  number of discrete levels used to bin the spectrum.
 *
 * Returns the number of partitions found (>= 0), or -1 on memory
 * allocation failure.
 */
int specpart_partition(const float *spec, int *ipart, int nk, int nth,
                       int ihmax);

#endif /* SPECPART_H */
