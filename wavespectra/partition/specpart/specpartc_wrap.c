#include <Python.h>
#include "numpy/arrayobject.h"
#include "specpartc.h"

static PyObject * specpartc(PyObject *self, PyObject *args);

/* ==== Set up the methods table ====================== */
static PyMethodDef specpartc_methods[] = {
  {"partition", specpartc, METH_VARARGS, "Description"},
  {NULL, NULL, 0, NULL}
};

static struct PyModuleDef specpartc_definition  = { 
    PyModuleDef_HEAD_INIT,
    "specpartc",
    "A Python module that prints 'hello world' from C code.",
    -1, 
    specpartc_methods
};

PyMODINIT_FUNC PyInit_specpartc(void) {
  Py_Initialize();
  import_array();
  return PyModule_Create(&specpartc_definition);
}

static PyObject * specpartc(PyObject *self, PyObject *args)
{
  PyArrayObject *specin, *ipartout;  // The python objects to be extracted from the args
  float * spec;
  int * ipart;
  int ihmax;
  int nk, nth, dims[2], i, j;

  if (!PyArg_ParseTuple(args, "O!i", &PyArray_Type, &specin, &ihmax))
    return NULL;
  if (NULL == specin)
    return NULL;

  nk = dims[0] = PyArray_DIMS(specin)[0];
  nth = dims[1] = PyArray_DIMS(specin)[1];

  /* Make a new int vector of same dimension */
  ipartout = (PyArrayObject *) PyArray_ZEROS(PyArray_NDIM(specin),
					     PyArray_DIMS(specin),
					     NPY_INT,
					     1);
  
  spec = (float *) PyArray_DATA(specin);
  ipart = (int *) PyArray_DATA(ipartout);

  // Do the calculation
  partition(spec, ipart, nk, nth, ihmax);
  
  
  /* Free memory, close file and return */
  // Don't think that is necessary
  //PyArray_free(spec);
  
  return PyArray_Return(ipartout);
}
