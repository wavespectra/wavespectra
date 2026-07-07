#define PY_SSIZE_T_CLEAN
#include <Python.h>
#include "numpy/arrayobject.h"
#include "specpart.h"

PyDoc_STRVAR(partition_doc,
"partition(spec, ihmax)\n"
"--\n\n"
"Watershed partitioning of a 2D wave spectrum.\n\n"
"C port of the WAVEWATCH III w3partmd watershed algorithm.\n\n"
"Parameters\n"
"----------\n"
"spec : array_like, shape (nfreq, ndir)\n"
"    Wave spectrum. Any real dtype and memory layout is accepted; the\n"
"    input is converted to a C-contiguous float32 array internally.\n"
"ihmax : int\n"
"    Number of discrete levels used to bin the spectrum (>= 2).\n\n"
"Returns\n"
"-------\n"
"ipart : ndarray of int, shape (nfreq, ndir)\n"
"    Partition map. Partition ids start at 1; 0 is only returned for a\n"
"    flat spectrum (no partitions).\n");

static PyObject *specpart_partition_py(PyObject *self, PyObject *args) {
  PyObject *specobj;
  PyArrayObject *specin = NULL, *ipartout = NULL;
  int ihmax, nk, nth, npart;

  if (!PyArg_ParseTuple(args, "Oi", &specobj, &ihmax)) return NULL;

  if (ihmax < 2) {
    PyErr_Format(PyExc_ValueError, "ihmax must be >= 2, got %d", ihmax);
    return NULL;
  }

  /* Convert to an aligned, C-contiguous float32 array. This makes the
   * wrapper robust to any input dtype and memory layout (e.g. transposed
   * or otherwise strided views), matching the behaviour of the f2py
   * wrapper around the original Fortran code. */
  specin = (PyArrayObject *)PyArray_FROM_OTF(
      specobj, NPY_FLOAT32, NPY_ARRAY_IN_ARRAY | NPY_ARRAY_FORCECAST);
  if (specin == NULL) return NULL;

  if (PyArray_NDIM(specin) != 2) {
    PyErr_Format(PyExc_ValueError,
                 "spec must be 2D with shape (nfreq, ndir), got %d dimension(s)",
                 PyArray_NDIM(specin));
    goto fail;
  }

  nk = (int)PyArray_DIM(specin, 0);
  nth = (int)PyArray_DIM(specin, 1);
  if (nk < 1 || nth < 1) {
    PyErr_SetString(PyExc_ValueError, "spec dimensions must be non-empty");
    goto fail;
  }

  ipartout = (PyArrayObject *)PyArray_ZEROS(2, PyArray_DIMS(specin), NPY_INT, 0);
  if (ipartout == NULL) goto fail;

  {
    const float *spec = (const float *)PyArray_DATA(specin);
    int *ipart = (int *)PyArray_DATA(ipartout);
    /* The kernel is reentrant (no global state), so the GIL can be
     * released during the computation. */
    Py_BEGIN_ALLOW_THREADS
    npart = specpart_partition(spec, ipart, nk, nth, ihmax);
    Py_END_ALLOW_THREADS
  }

  if (npart < 0) {
    PyErr_NoMemory();
    goto fail;
  }

  Py_DECREF(specin);
  return PyArray_Return(ipartout);

fail:
  Py_XDECREF(specin);
  Py_XDECREF(ipartout);
  return NULL;
}

static PyMethodDef specpart_methods[] = {
    {"partition", specpart_partition_py, METH_VARARGS, partition_doc},
    {NULL, NULL, 0, NULL}};

PyDoc_STRVAR(module_doc,
"Watershed partitioning of wave spectra (WAVEWATCH III w3partmd port).");

static struct PyModuleDef specpart_definition = {
    PyModuleDef_HEAD_INIT, "specpart", module_doc, -1, specpart_methods};

PyMODINIT_FUNC PyInit_specpart(void) {
  import_array();
  return PyModule_Create(&specpart_definition);
}
