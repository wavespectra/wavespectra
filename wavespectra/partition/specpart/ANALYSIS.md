# specpart: C vs Fortran watershed comparison (GH issue #142)

**Date:** 2026-07-07 · **Scope:** the watershed partitioning kernel
(`wavespectra/partition/specpart/`), C implementation (wavespectra ≥ v4)
versus the original Fortran implementation (wavespectra < v4, itself a port
of WAVEWATCH III `w3partmd`).

## TL;DR

- The C port of the watershed **algorithm is faithful**: with correctly
  presented input, C and Fortran produce **bit-identical partition maps in
  all 1420 comparisons** run (real model output, the issue-142 dataset, and
  synthetic/degenerate spectra, at ihmax = 25/50/100/200).
- Issue #142 was real, but it was **not an algorithmic difference**: the C
  *wrapper* read the input buffer assuming C-contiguous float32 data and
  silently mis-read any transposed/strided input. The f2py wrapper around
  the Fortran code always handled strides correctly — so v3 was right and
  v4 was silently wrong for those inputs.
- The wrapper and kernel have been rewritten: any input dtype/layout is now
  converted at the boundary, the kernel is stateless (thread-safe), and all
  the memory/robustness defects found in review are fixed. Layout
  sensitivity end-to-end is now 0.0 m (was up to 1.59 m on the issue-142
  data).

## Background

wavespectra < v4 shipped the watershed as Fortran (`specpart.f90`, a direct
copy of WW3's `w3partmd`) wrapped with f2py. In v4 it was rewritten in C to
drop the Fortran toolchain dependency. The developer compared results at the
time and found them "very similar", but an exhaustive equivalence check was
never done. In issue #142 a user reported ~10 cm Hs differences in partition
statistics between v3.13.0 and v4.1.1 with `ptm1(..., ihmax=200)`, and
provided a notebook plus a 234-timestep test dataset.

## Method

1. The Fortran source (`specpart.f90` + `specpart.pyf`) was recovered from
   git history (parent of commit `7983308`, which replaced it with C) and
   built with `f2py`/gfortran as a standalone reference module.
2. Both kernels were run on identical inputs and the resulting partition
   maps compared **cell-by-cell** (exact equality):
   - `tests/sample_files/ww3file.nc` — 18 real model spectra (25×24);
   - the issue-142 dataset — 234 spectra (36×36) provided by the reporter;
   - 100 synthetic spectra — multi-system JONSWAP×cartwright constructions,
     random rough fields, hard-quantised fields (to stress exact ties),
     zero blocks, flat and single-peak degenerate cases;
   - each at ihmax = 25, 50, 100 and 200 → **1420 comparisons**.
3. End-to-end attribution on the issue-142 dataset through the public
   `ptm1` API, comparing dataset layouts.

## Findings

### 1. The algorithm port is exact

With input presented as C-contiguous float32 (the layout the C kernel
assumes), **all 1420 partition maps were bit-identical** between C and
Fortran. This includes tie-stressed quantised spectra, so the minor
arithmetic differences between the implementations (the Fortran bins levels
in single precision with `nint(1. + zp*fact)`, the C in double precision
with `round(zp*fact)`) did not produce a single divergent map in practice.

### 2. Root cause of issue #142: the wrapper, not the algorithm

The pre-fix wrapper did:

```c
spec = (float *) PyArray_DATA(specin);   /* no dtype/contiguity checks */
```

Any input that was not already C-contiguous float32 was silently
reinterpreted. The issue-142 dataset is stored with dims
`(time, dir, freq)`; the partition pipeline transposes `efth` to
`(freq, dir)`, which produces an F-ordered view, and
`spectrum.astype(np.float32)` (default `order='K'`) **preserves** that
F-order. The C kernel then read the transposed bytes as if they were
C-ordered — producing structurally wrong partitions (wrong shapes *and*
wrong partition counts).

The f2py wrapper never had this problem: `real dimension(nk,nth)` inputs
are stride-aware-copied by f2py regardless of layout. Hence: **v3 (Fortran)
was correct; v4 (C) was silently wrong for any dataset whose spectra were
not C-contiguous `(..., freq, dir)` in memory.**

Reproduction on the reporter's own dataset (public API, both layouts of the
same data):

| | max abs Hs difference across partitions/times |
|---|---|
| before fix | **1.59 m** (mean ~0.07 m — the reporter's ~10 cm was typical) |
| after fix | 0.00 m |

### 3. Additional defects fixed in the rewrite

Found in code review of the pre-fix extension; all fixed:

| Defect | Where | Consequence |
|---|---|---|
| No dtype/ndim/contiguity validation | `specpart_wrap.c` | silent wrong results (issue #142) |
| Global mutable state (`nspec, mk, mth, neigh, …`) | `specpart.c` | not reentrant; races under threaded dask |
| `neigh` malloc'd twice (in `partinit` and `ptnghb`) | `specpart.c` | memory leak on every re-init |
| `exit(EXIT_FAILURE)` on dimension mismatch | `specpart.c` | would kill the host Python process |
| No `malloc` return checks | both | segfault on OOM |
| Output array created Fortran-ordered (`PyArray_ZEROS(..., 1)`) | wrapper | worked only by pairing with the kernel's internal layout |
| `Py_Initialize()` in module init, placeholder docstrings | wrapper | cosmetic/wrong |

The rewritten kernel keeps **no global state** (all work arrays are
per-call), which makes it reentrant; the wrapper releases the GIL during
computation, so partitioning now parallelises under dask's threaded
scheduler. Input conversion uses `PyArray_FROM_OTF(..., NPY_FLOAT32,
NPY_ARRAY_IN_ARRAY | NPY_ARRAY_FORCECAST)`, matching f2py's permissive
behaviour (any real dtype, any layout).

## Verification of the fix

- Algorithm equivalence sweep re-run on the rewritten kernel:
  **1420/1420 bit-identical** to the Fortran reference.
- All 234 issue-142 spectra × 5 layout/dtype variants (C/F-ordered ×
  float32/float64, plus a strided slice view): **all identical to Fortran**.
- End-to-end `ptm1` on the issue-142 dataset in both layouts: max Hs
  difference **0.0 m**.
- Thread-safety stress: 2000 concurrent `partition` calls on 8 threads,
  all results identical.
- Error paths: 3D input and `ihmax < 2` raise `ValueError`; allocation
  failure raises `MemoryError`; flat spectra return an all-zero map.
- Full wavespectra test suite passes; new regression tests added in
  `tests/test_partition.py::TestSpecpartKernel`.
- Throughput sanity: ~74 µs per 36×36 spectrum at ihmax=200 (single
  thread), comparable to the previous implementation.

## Notes for issue #142 follow-up

The reporter compared v3.13.0 `spec.partition(...)` against v4
`spec.partition.ptm1(...)`. Beyond the wrapper bug (the dominant effect for
their dataset, which is `(time, dir, freq)`-ordered), the v3 pipeline also
differed from v4 `ptm1` in ways that can produce further (legitimate,
documented) differences: v3 applied an additional frequency-inflection
splitting step to watershed partitions, which v4 does not implement. With
the wrapper fixed, kernel-level results are exactly equivalent to v3; any
remaining pipeline-level differences are design changes of the v4 API, not
defects.

## Reproducing this analysis

The Fortran reference can be rebuilt from git history:

```bash
git show 7983308^:wavespectra/partition/specpart/specpart.f90 > specpart.f90
git show 7983308^:wavespectra/partition/specpart/specpart.pyf > specpart.pyf
python -m numpy.f2py -c specpart.pyf specpart.f90 --backend meson
```

and compared against `wavespectra.partition.specpart.partition` on any
spectra (cast inputs with `np.ascontiguousarray(spec, dtype=np.float32)` to
compare kernels independently of the wrapper conversion). The issue-142
test dataset is attached to the GitHub issue
(`debug_notebook.zip` → `test_dset_before_partition.nc`).
