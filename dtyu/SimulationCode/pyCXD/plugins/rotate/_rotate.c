#include <Python.h>
#include <numpy/arrayobject.h>
//#define NPY_NO_DEPRECATED_API NPY_1_7_API_VERSION
#include "rotate.h"


static char module_docstring[] =
    "This module rotates your array.";
static char rotate_docstring[] = 
    "rotate(img, imgr, phi, cenx, ceny):\n\
        This function rotates your array img by phi around [cenx, ceny]\n\
        and saves it into imgr.";

static PyObject *_rotate_rotate(PyObject *self, PyObject *args);

static PyMethodDef module_methods[] = {
    {"rotate", _rotate_rotate, METH_VARARGS, rotate_docstring},
    {NULL, NULL, 0, NULL}
};


PyMODINIT_FUNC 
#if PY_MAJOR_VERSION >= 3
PyInit__rotate(void)
#else
init_rotate(void)
#endif
{
#if PY_MAJOR_VERSION >= 3
    static struct PyModuleDef moduledef = {
        PyModuleDef_HEAD_INIT,
        "_rotate",              /* m_name */
        module_docstring,     /* m_doc */
        -1,                     /* m_size */
        module_methods,       /* m_methods */
        NULL,                   /* m_reload */
        NULL,                   /* m_traverse */
        NULL,                   /* m_clear */
        NULL,                   /* m_free */
    };
#endif

#if PY_MAJOR_VERSION >= 3
    PyObject *m = PyModule_Create(&moduledef);
#else
    PyObject *m = Py_InitModule3("_rotate", module_methods, module_docstring);
#endif

    if(m == NULL)
        return (PyMODINIT_FUNC)-1;

    /* Load numpy functionality. */
    import_array();
    return m;
}

static PyObject *_rotate_rotate(PyObject *self, PyObject *args){
    /* The test wrapper for a python module*/
    /* Takes in a (nparray, nparray, theta, cenx, ceny)
    * Takes in a (img1, imgr,pixellist, theta, cenx, ceny)
    *   (See ParseTuple for the arguments)
    */
    
    int dimx,dimy;
    npy_intp *dims;
    double theta, cenx, ceny;
    PyObject *pyobj1, *pyobj2;
    int val;

    /* Parse the input tuple */
    if(!PyArg_ParseTuple(args, "OOddd", &pyobj1, &pyobj2, &theta, &cenx, &ceny))
        return NULL;
    
    /* Interpret the numpy array*/
    PyObject *np_arr1 = PyArray_FROM_OTF(pyobj1, NPY_DOUBLE, NPY_IN_ARRAY);
    PyObject *np_arr2 = PyArray_FROM_OTF(pyobj2, NPY_DOUBLE, NPY_IN_ARRAY);
    if(!PyArray_ISFLOAT(np_arr1) || !PyArray_ISFLOAT(np_arr2)){
        printf("Error, input array is not float64, exiting...\n");
        Py_XDECREF(np_arr1);
        Py_XDECREF(np_arr2);
        return NULL;
    }

    dims = PyArray_DIMS(np_arr1);
    //fastest index is rightermost, and we let x denote fastest varying index
    dimx = dims[1];
    dimy = dims[0];
    //printf("Got dims %d,%d, nopixels %d\n",dimx,dimy,nopixels);

    /* Exception if it didn't work*/
    if(np_arr1 == NULL){
        Py_XDECREF(np_arr1);
        return NULL;
    }
    if(np_arr2 == NULL){
        Py_XDECREF(np_arr2);
        return NULL;
    }

    /* How long is array*/
    //int N2 = (int)PyArray_DIM(arr1,0);

    /* Grab the pointers*/
    double * img = (double *)PyArray_DATA(np_arr1);
    double * imgr = (double *)PyArray_DATA(np_arr2);

    //printf("Zeroing array\n");
    zero_elems(imgr,dimx*dimy);
    //printf("Rotating array\n");
    val = rotate(img,imgr,theta,cenx,ceny,dimx,dimy);
    //printf("Done\n");


    /* Clean up ALL python objects, avoid memory leaks!!*/
    Py_DECREF(np_arr1);
    Py_DECREF(np_arr2);
    //Py_DECREF(dims);
    //Py_DECREF(img);
    //Py_DECREF(imgr);


//    if(value == NULL){
//        PyErr_SetString(PyExc_RuntimeError,
//            "Some error");
//        return NULL;
//    }

    /* Make the output*/
    PyObject *ret  = Py_BuildValue("d",val);
    return ret;
}
