//#include "stdafx.h"
#ifdef _DEBUG
#undef _DEBUG
#include <Python.h>
#define _DEBUG
#else
#include <Python.h>
#endif

// Produce deprecation warnings (needs to come before arrayobject.h inclusion).
#define NPY_NO_DEPRECATED_API NPY_1_7_API_VERSION

#include "numpy/arrayobject.h"
#include <stdio.h>
#include <stdlib.h>
#include <string>
#include <vector>
#include <functional>
#include <algorithm>
#include <sstream>
#include "autoarray.h"

#define NUMPY_PI 3.141592653589793

#if PY_MAJOR_VERSION >= 3
#define PY3K
#endif

#ifdef PY3K
#define PyString_AsString PyBytes_AsString
#endif

static PyObject *HoughTransform_hough_trans_lines(PyObject *self, PyObject *args)
{
    PyArrayObject *ImageArray;
    int HoughSpace;
    int retValue = 0;

    if (!PyArg_ParseTuple(args, "iO!", &HoughSpace, &PyArray_Type, &ImageArray))
        return NULL;

    unsigned char *image_data = reinterpret_cast<unsigned char *>(PyArray_DATA(ImageArray));
    int imH = static_cast<int>(PyArray_DIM(ImageArray, 0));
    int imW = static_cast<int>(PyArray_DIM(ImageArray, 1));

    int stride_h = static_cast<int>(PyArray_STRIDE(ImageArray, 0));
    int stride_w = static_cast<int>(PyArray_STRIDE(ImageArray, 1));

    double hough_interval = NUMPY_PI / double(HoughSpace);
    int max_length = (int)(sqrt(2.0) * std::max(imH, imW));

    npy_intp Dims[1] = { 2 * max_length * HoughSpace };
    PyArrayObject *PyArray = (PyArrayObject *)PyArray_SimpleNew(1, Dims, NPY_UBYTE);
    unsigned char *pNormImage5Pt = reinterpret_cast<unsigned char *>(PyArray_DATA(PyArray));
    
    AutoArray<float> trans(2 * max_length * HoughSpace);
    memset(trans.begin(), 0, 2 * max_length * HoughSpace);
    int loc = 0;
    for (int h = 0; h < imH; ++h) {
        for (int w = 0; w < imW; ++w) {
            unsigned char P = image_data[h * stride_h + w * stride_w];

            if (P < 255)
                continue;

            for (int cell = 0; cell < HoughSpace; ++cell) {
                loc = (int)(w * cos(cell * hough_interval) + h * sin(cell * hough_interval) + max_length);

                if (loc < 0 || (loc >= 2 * max_length))
                    continue;

                trans[loc * HoughSpace + cell] += 1.0;
            }
        
        }
    }

    // 接下来进行变换
    float max_value = 0;
    for (int loc = 0; loc < 2 * max_length * HoughSpace; ++loc) {
        if (max_value < trans[loc])
            max_value = trans[loc];
    }

    for (int loc = 0; loc < 2 * max_length * HoughSpace; ++loc) {
        pNormImage5Pt[loc] = unsigned char(int(255.0 * trans[loc] / max_value));
    }

    if (retValue != 0) {
        // printf("Normlize face error!");
        Py_DECREF(Py_None);
        return Py_None;
    }
    else {
        return PyArray_Return(PyArray);
    }
}

static PyObject *HoughTransform_hough_trans_circles(PyObject *self, PyObject *args)
{
    PyArrayObject *ImageArray;
    int retValue = 0;
    int minRadius, maxRadius;

    if (!PyArg_ParseTuple(args, "iiO!", &minRadius, &maxRadius, &PyArray_Type, &ImageArray))
        return NULL;

    unsigned char *image_data = reinterpret_cast<unsigned char *>(PyArray_DATA(ImageArray));
    int imH = static_cast<int>(PyArray_DIM(ImageArray, 0));
    int imW = static_cast<int>(PyArray_DIM(ImageArray, 1));

    int stride_h = static_cast<int>(PyArray_STRIDE(ImageArray, 0));
    int stride_w = static_cast<int>(PyArray_STRIDE(ImageArray, 1));

    int depth = maxRadius - minRadius + 1;
    npy_intp Dims[1] = { depth * imH * imW };
    PyArrayObject *PyArray = (PyArrayObject *)PyArray_SimpleNew(1, Dims, NPY_UBYTE);
    unsigned char *pNormImage5Pt = reinterpret_cast<unsigned char *>(PyArray_DATA(PyArray));

    AutoArray<float> trans(depth * imH * imW);
    memset(trans.begin(), 0, depth * imH * imW);
    int x0, y0;
    double t;
    for (int h = int(0.4 * imH); h < int(0.75 * imH); ++h) {
        for (int w = int(0.5 * imW - 0.8 * maxRadius); w < int(0.5 * imW + 0.8 * maxRadius); ++w) {
            unsigned char P = image_data[h * stride_h + w * stride_w];

            if (P < 255)
                continue;

            for (int r = minRadius; r <= maxRadius; ++r) {
                for (int theta = 0; theta < 360; theta++) {
                    t = (theta * NUMPY_PI) / 180; // 角度值0 ~ 2*PI  
                    x0 = int(w - r * cos(t) + 0.5);
                    y0 = int(h - r * sin(t) + 0.5);
                    if (x0 < imW && x0 > 0 && y0 < imH && y0 > 0) {
                        trans[(r - minRadius) * imH * imW + y0 * imW + x0] += 1;
                    }
                }
            }
            
        }
    }

    // 接下来进行变换
    float max_value = 0;
    for (int loc = 0; loc < depth * imH * imW; ++loc) {
        if (max_value < trans[loc])
            max_value = trans[loc];
    }

    for (int loc = 0; loc < depth * imH * imW; ++loc) {
        pNormImage5Pt[loc] = unsigned char(int(255.0 * trans[loc] / max_value));
    }

    if (retValue != 0) {
        // printf("Normlize face error!");
        Py_DECREF(Py_None);
        return Py_None;
    }
    else {
        return PyArray_Return(PyArray);
    }
}

static PyMethodDef
HoughTransformMethods[] = {
    { "hough_trans_lines", HoughTransform_hough_trans_lines, METH_VARARGS },
    { "hough_trans_circles", HoughTransform_hough_trans_circles, METH_VARARGS },
    { NULL, NULL },
};

#ifdef PY3K
// module definition structure for python3
static struct PyModuleDef hough_trans_def = {
    PyModuleDef_HEAD_INIT,
    "HoughTransform",				// name of module
    NULL,							// module documentation, may be NULL
    -1, 	// size of per-interpreter state of the module,	or -1 if the module keeps state in global variables.
    HoughTransformMethods,
    NULL,
    NULL,
    NULL,
    NULL
};

/*
*	Initialize the SelectTriplet functions,
*  Module name must be SelectTriplet in compile and linked
*/
// module initializer for python3
PyMODINIT_FUNC PyInit_HoughTransform()
{
    PyObject *obj = PyModule_Create(&hough_trans_def);
    import_array();  // Must be present for NumPy.  Called first after above line.
    return obj;
}
#else	//	PY3K
// module initializer for python2
extern "C" void initHoughTransform(void)
{
    Py_InitModule3("HoughTransform", HoughTransformMethods, "hough transformation");
    import_array();  // Must be present for NumPy.  Called first after above line.
}
#endif	//	PY3K