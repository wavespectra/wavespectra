"""Combine swell partitions as per Hanson and Phillips (2001).

The watershed algorithm tends to over-segment noisy spectra, splitting single
wave systems into multiple adjacent partitions. Hanson and Phillips (2001)
(HP01) describe criteria to identify and combine such "mutual" partitions:

* Peak separation (HP01 eq 9): two partitions are combined if the squared
  distance between their spectral peaks in (fx, fy) = (f cos0, f sin0) space
  is small compared to the spread of either partition, df2 <= kappa * spread.
* Minimum between peaks: two partitions are combined if the spectral density
  at the saddle point between them exceeds a fraction `zeta` of the smaller
  of the two peak densities.
* Energy threshold (HP01 eq 10): partitions with total energy below the noise
  floor a / (fp**4 + b) do not represent real wave systems.
* Angle test (Hanson et al. 2009): the mean directions of two partitions must
  lie within `angle_max` degrees for them to be combined, 30 degrees was found
  to yield optimum results in that study.

Adjacency between partitions and the saddle heights between them are measured
on the shared boundaries of the watershed basins (8-connected, periodic in
direction) so only truly adjacent partitions are ever combined. Partition
statistics are recomputed after every merge so newly combined partitions are
themselves eligible for further combining (HP01 section 2a), and candidate
pairs are always merged strongest-first so the outcome does not depend on the
input ordering. Sub-threshold partitions are merged onto their neighbours
rather than discarded so that spectral variance is conserved.

The combining parameters have been tuned differently in previous studies:
zeta = 0.85 and kappa = 1 in Hasselmann et al. (1996), zeta = 0.70 and
kappa = 0.5 in Voorrips et al. (1997), zeta = 0.65-0.75 and kappa = 0.4-0.5
in HP01 (table 1), see Portilla et al. (2009) for a discussion on the
sensitivity to these choices.

References:
    - Hanson and Phillips (2001), Automated analysis of ocean surface
      directional wave spectra, J. Atmos. Oceanic Technol., 18, 277-293.
    - Hanson et al. (2009), Pacific hindcast performance of three numerical
      wave models, J. Atmos. Oceanic Technol., 26, 1614-1633.
    - Hasselmann et al. (1996), An improved algorithm for the retrieval of
      ocean wave spectra from synthetic aperture radar image spectra,
      J. Geophys. Res., 101, 16615-16629.
    - Portilla et al. (2009), Spectral partitioning and identification of
      wind sea and swell, J. Atmos. Oceanic Technol., 26, 107-122.
    - Voorrips et al. (1997), Assimilation of wave spectra from pitch-and-roll
      buoys in a North Sea wave model, J. Geophys. Res., 102, 5829-5849.

"""

import logging

import numpy as np

from wavespectra.core.utils import D2R, angle


logger = logging.getLogger(__name__)

# Default combining parameters from Hanson and Phillips (2001), table 1
# (buoy data used kappa=0.4, zeta=0.65; WAM model data used kappa=0.5, zeta=0.75)
# and the optimum angle threshold from Hanson et al. (2009)
KAPPA = 0.4
ZETA = 0.65
ANGLE_MAX = 30.0


def _label_map(partitions):
    """Label map reconstructed from a list of disjoint partition arrays.

    Args:
        - partitions (3darray): Partitioned spectra with shape (npart, nf, nd),
          non-overlapping in their non-zero support.

    Returns:
        - lmap (2darray): Integer label array with shape (nf, nd) where values
          1..npart identify the partition owning each bin and 0 marks bins
          with no energy in any partition.

    """
    lmap = (np.argmax(partitions != 0, axis=0) + 1).astype(np.intp)
    lmap[~np.any(partitions != 0, axis=0)] = 0
    return lmap


def _adjacency_saddles(lmap, spectrum, npart):
    """Saddle point heights between adjacent partitions.

    Args:
        - lmap (2darray): Label array with shape (nf, nd), 0 = no partition.
        - spectrum (2darray): Spectrum from which saddle heights are measured.
        - npart (int): Number of partitions (labels run from 1 to npart).

    Returns:
        - saddles (2darray): Symmetric array with shape (npart+1, npart+1)
          where saddles[a, b] is the highest point on the boundary between
          partitions a and b, measured as the maximum over all pairs of
          8-connected boundary bins of the smaller of the two bin densities.
          Non-adjacent partitions have saddle 0.

    Note:
        - The direction axis (axis=1) is treated as periodic, consistent with
          the watershed algorithm used to define the partition boundaries.

    """
    saddles = np.zeros((npart + 1, npart + 1))

    def update(l1, l2, s1, s2):
        mask = (l1 != l2) & (l1 > 0) & (l2 > 0)
        if not mask.any():
            return
        a = l1[mask]
        b = l2[mask]
        h = np.minimum(s1[mask], s2[mask])
        np.maximum.at(saddles, (a, b), h)

    ld = np.roll(lmap, 1, axis=1)
    sd = np.roll(spectrum, 1, axis=1)
    # Neighbours along dir (periodic), freq, and both diagonals
    update(lmap, ld, spectrum, sd)
    update(lmap[1:], lmap[:-1], spectrum[1:], spectrum[:-1])
    update(lmap[1:], ld[:-1], spectrum[1:], sd[:-1])
    update(ld[1:], lmap[:-1], sd[1:], spectrum[:-1])

    return np.maximum(saddles, saddles.T)


class _PartitionGraph:
    """Bookkeeping for merging partitions on an adjacency graph.

    Maintains per-partition integrated statistics and pairwise saddle heights,
    updating both as partitions are merged so that newly combined partitions
    are themselves eligible for further combining (HP01, section 2a).

    """

    def __init__(self, partitions, freq, dir):
        self.partitions = [p.astype("float64") for p in partitions]
        self.freq = freq
        self.dir = dir

        nfreq, ndir = partitions[0].shape
        df = np.gradient(freq)
        dd = abs(float(dir[1] - dir[0]))
        self._weights = np.repeat(df[:, None], ndir, axis=1) * dd
        fmat = np.repeat(freq.astype("float64")[:, None], ndir, axis=1)
        dmat = D2R * np.tile(dir.astype("float64"), (nfreq, 1))
        self._fx = fmat * np.cos(dmat)
        self._fy = fmat * np.sin(dmat)
        self._sind = np.sin(dmat)
        self._cosd = np.cos(dmat)

        npart = len(partitions)
        self.stats = {ip: self._calc_stats(self.partitions[ip]) for ip in range(npart)}
        lmap = _label_map(np.array(partitions))
        spectrum = np.sum(partitions, axis=0)
        self._saddles = _adjacency_saddles(lmap, spectrum, npart)[1:, 1:]

    def _calc_stats(self, spectrum):
        """Integrated parameters of a single partition."""
        w = spectrum * self._weights
        e = w.sum()
        if e <= 0:
            return None
        ipeak = np.unravel_index(np.argmax(spectrum), spectrum.shape)
        fx = (w * self._fx).sum() / e
        fy = (w * self._fy).sum() / e
        f2 = (w * (self._fx**2 + self._fy**2)).sum() / e
        dm = (
            np.degrees(np.arctan2((w * self._sind).sum(), (w * self._cosd).sum())) % 360
        )
        return {
            "e": e,
            "peak": spectrum[ipeak],
            "fp": self.freq[ipeak[0]],
            "fpx": self._fx[ipeak],
            "fpy": self._fy[ipeak],
            "dm": dm,
            "spread": f2 - fx**2 - fy**2,
        }

    def saddle(self, ipart, jpart):
        return self._saddles[ipart, jpart]

    def neighbours(self, ipart):
        return [jp for jp in self.stats if self._saddles[ipart, jp] > 0]

    def pairs(self):
        """All pairs of adjacent partitions."""
        labels = sorted(self.stats)
        for i, ipart in enumerate(labels):
            for jpart in labels[i + 1 :]:
                if self._saddles[ipart, jpart] > 0:
                    yield ipart, jpart

    def merge(self, ipart, jpart):
        """Merge partition jpart into partition ipart."""
        self.partitions[ipart] = self.partitions[ipart] + self.partitions[jpart]
        self.stats[ipart] = self._calc_stats(self.partitions[ipart])
        del self.stats[jpart]
        # Contract the adjacency graph, boundaries of the merged partition are
        # the union of the boundaries of its members
        self._saddles[ipart, :] = np.maximum(
            self._saddles[ipart, :], self._saddles[jpart, :]
        )
        self._saddles[:, ipart] = self._saddles[ipart, :]
        self._saddles[ipart, ipart] = 0.0
        self._saddles[jpart, :] = 0.0
        self._saddles[:, jpart] = 0.0

    def peak_distance(self, ipart, jpart):
        """Squared distance between partition peaks in (fx, fy) space."""
        si, sj = self.stats[ipart], self.stats[jpart]
        return (si["fpx"] - sj["fpx"]) ** 2 + (si["fpy"] - sj["fpy"]) ** 2

    def is_mutual(self, ipart, jpart, kappa, zeta, angle_max):
        """Evaluate the HP01 combining criteria for two adjacent partitions.

        Returns a score > 0 quantifying how strongly the pair satisfies either
        combining criterion, or 0 if neither criterion is met.

        """
        si, sj = self.stats[ipart], self.stats[jpart]
        if angle_max is not None and angle(si["dm"], sj["dm"]) > angle_max:
            return 0.0
        # Minimum between peaks: saddle height relative to the smaller peak
        score_saddle = self.saddle(ipart, jpart) / min(si["peak"], sj["peak"])
        if score_saddle < zeta:
            score_saddle = 0.0
        # Peak separation relative to the spread of either partition
        spread = max(si["spread"], sj["spread"])
        score_dist = 0.0
        if spread > 0:
            df2 = self.peak_distance(ipart, jpart)
            if df2 <= kappa * spread:
                score_dist = 1 - df2 / (kappa * spread)
        return max(score_saddle, score_dist)


def combine_partitions_hp01(
    partitions,
    freq,
    dir,
    swells=None,
    kappa=KAPPA,
    zeta=ZETA,
    angle_max=ANGLE_MAX,
    hs_min=0.0,
    noise_a=None,
    noise_b=0.0,
    combine_extra_swells=True,
):
    """Combine swell partitions according to Hanson and Phillips (2001).

    Args:
        - partitions (list): List of 2darray spectra partitions (m2/Hz/deg)
          with non-overlapping support, e.g. from a watershed algorithm.
        - freq (1darray): Frequency array (Hz).
        - dir (1darray): Direction array (deg).
        - swells (int): Exact number of swell partitions to return, extra
          partitions are either combined or dropped depending on
          `combine_extra_swells`. All combined partitions returned if None.
        - kappa (float): Spread factor in the peak separation criterion,
          HP01 eq 9.
        - zeta (float): Peak minimum factor, the fraction of the smaller peak
          density the saddle point between two partitions must exceed for the
          partitions to be combined.
        - angle_max (float): Maximum angle (deg) between partition mean
          directions for combining as per Hanson et al. (2009), disabled
          if None.
        - hs_min (float): Minimum Hs of individual partitions, any partition
          below this value is merged onto its most connected neighbour.
        - noise_a (float): Factor `A` in HP01's noise threshold eq 10,
          e <= A / (fp^4 + B). Disabled if None.
        - noise_b (float): Factor `B` in HP01's noise threshold eq 10.
        - combine_extra_swells (bool): If True and more than `swells`
          partitions remain after combining, merge the extra ones onto their
          closest neighbours; if False drop the smallest ones.

    Returns:
        - combined_partitions (list): List of combined partitions sorted by
          descending total energy.

    Criteria for combining two adjacent partitions:
        - The saddle point between them exceeds `zeta` times the smaller peak, or
        - Their peaks are closer in (fx, fy) space than `kappa` times the
          spread of either partition (and within `angle_max` if specified).

    Partitions failing the `hs_min` / noise thresholds are always merged onto
    their most connected (highest saddle) neighbour so that no energy is lost.

    """
    partitions = np.asarray(partitions)
    if partitions.ndim != 3:
        raise ValueError("partitions must have shape (npart, nfreq, ndir)")

    graph = _PartitionGraph(partitions, freq, dir)

    # Null partitions carry no energy and are dropped upfront
    for ipart in [ip for ip, s in graph.stats.items() if s is None]:
        del graph.stats[ipart]
    if not graph.stats:
        return []

    # Iteratively combine mutual partitions, strongest candidate pair first so
    # the outcome does not depend on input ordering
    while len(graph.stats) > 1:
        pairs = list(graph.pairs())
        if not pairs:
            break
        scores = [graph.is_mutual(ip, jp, kappa, zeta, angle_max) for ip, jp in pairs]
        imax = int(np.argmax(scores))
        if scores[imax] <= 0:
            break
        ipart, jpart = pairs[imax]
        logger.debug(f"Combining mutual partitions {jpart} -> {ipart}")
        graph.merge(ipart, jpart)

    # Merge low-energy partitions onto their most connected neighbour
    def energy_floor(stats):
        floor = (hs_min / 4) ** 2
        if noise_a is not None:
            floor = max(floor, noise_a / (stats["fp"] ** 4 + noise_b))
        return floor

    merged = True
    while merged and len(graph.stats) > 1:
        merged = False
        for ipart in sorted(graph.stats, key=lambda ip: graph.stats[ip]["e"]):
            if graph.stats[ipart]["e"] >= energy_floor(graph.stats[ipart]):
                continue
            neighbours = graph.neighbours(ipart)
            if neighbours:
                jpart = max(neighbours, key=lambda jp: graph.saddle(ipart, jp))
            else:
                others = [jp for jp in graph.stats if jp != ipart]
                jpart = min(others, key=lambda jp: graph.peak_distance(ipart, jp))
            logger.debug(f"Merging low-energy partition {ipart} -> {jpart}")
            graph.merge(jpart, ipart)
            merged = True
            break

    # Reduce to the requested number of swells
    if swells is not None and len(graph.stats) > swells:
        if combine_extra_swells:
            while len(graph.stats) > swells:
                # Merge the least separated pair, preferring adjacent pairs
                pairs = list(graph.pairs())
                if not pairs:
                    labels = sorted(graph.stats)
                    pairs = [
                        (ip, jp)
                        for i, ip in enumerate(labels)
                        for jp in labels[i + 1 :]
                    ]
                dists = [graph.peak_distance(ip, jp) for ip, jp in pairs]
                ipart, jpart = pairs[int(np.argmin(dists))]
                logger.debug(f"Merging extra partition {jpart} -> {ipart}")
                graph.merge(ipart, jpart)
        else:
            for ipart in sorted(graph.stats, key=lambda ip: graph.stats[ip]["e"])[
                : len(graph.stats) - swells
            ]:
                logger.debug(f"Dropping extra partition {ipart}")
                del graph.stats[ipart]

    # Sort by descending energy
    labels = sorted(graph.stats, key=lambda ip: -graph.stats[ip]["e"])
    return [graph.partitions[ipart] for ipart in labels]
