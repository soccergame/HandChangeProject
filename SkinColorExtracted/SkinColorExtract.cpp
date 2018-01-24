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

#if PY_MAJOR_VERSION >= 3
#define PY3K
#endif

#ifdef PY3K
#define PyString_AsString PyBytes_AsString
#endif

static PyObject *SkinColorExtract_extract_skin_color_YCbCr(PyObject *self, PyObject *args)
{
    PyArrayObject *ImageArray;

    int retValue = 0;

    if (!PyArg_ParseTuple(args, "O!", &PyArray_Type, &ImageArray))
        return NULL;

    unsigned char *image_data = reinterpret_cast<unsigned char *>(PyArray_DATA(ImageArray));
    int imH = static_cast<int>(PyArray_DIM(ImageArray, 0));
    int imW = static_cast<int>(PyArray_DIM(ImageArray, 1));
    int imC = static_cast<int>(PyArray_DIM(ImageArray, 2));

    int stride_h = static_cast<int>(PyArray_STRIDE(ImageArray, 0));
    int stride_w = static_cast<int>(PyArray_STRIDE(ImageArray, 1));
    int stride_c = static_cast<int>(PyArray_STRIDE(ImageArray, 2));

    /*printf("%d, %d, %d\n", stride_h, stride_w, stride_c);

    printf("%d\n", strNormMethod);*/

    float Wcb = 46.97f;
    float Wcr = 38.76f;
    float WHCb = 14.0f;
    float WHCr = 10.0f;
    float WLCb = 23.0f;
    float WLCr = 20.0f;
    float Ymin = 16.0f;
    float Ymax = 235.0f;
    float Kl = 125.0f;
    float Kh = 188.0f;
    float WCb = 0.0f;
    float WCr = 0.0f;
    float CbCenter = 0.0f;
    float CrCenter = 0.0f;

    npy_intp Dims[1] = { imH * imW };
    PyArrayObject *PyArray = (PyArrayObject *)PyArray_SimpleNew(1, Dims, NPY_UBYTE);
    unsigned char *pNormImage5Pt = reinterpret_cast<unsigned char *>(PyArray_DATA(PyArray));
    memset(pNormImage5Pt, 0, imH * imW);
    
    for (int h = 0; h < imH; ++h) {
        for (int w = 0; w < imW; ++w) {
            float Y = float(image_data[h * stride_h + w * stride_w]);
            float Cr = float(image_data[stride_c + h * stride_h + w * stride_w]);
            float Cb = float(image_data[2 * stride_c + h * stride_h + w * stride_w]);

            if (Y < Kl) {
                WCr = WLCr + (Y - Ymin) * (Wcr - WLCr) / (Kl - Ymin);
                WCb = WLCb + (Y - Ymin) * (Wcb - WLCb) / (Kl - Ymin);

                CrCenter = 154.0f - (Kl - Y) * (154.0f - 144.0f) / (Kl - Ymin);
                CbCenter = 108.0f + (Kl - Y) * (118.0f - 108.0f) / (Kl - Ymin);
            }
            else if (Y > Kh) {
                WCr = WHCr + (Y - Ymax) * (Wcr - WHCr) / (Ymax - Kh);
                WCb = WHCb + (Y - Ymax) * (Wcb - WHCb) / (Ymax - Kh);

                CrCenter = 154.0f + (Y - Kh) * (154.0f - 132.0f) / (Ymax - Kh);
                CbCenter = 108.0f + (Y - Kh) * (118.0f - 108.0f) / (Ymax - Kh);
            }
            

            if (Y < Kl || Y > Kh) {
                Cr = (Cr - CrCenter) * Wcr / WCr + 154.0f;
                Cb = (Cb - CbCenter) * Wcb / WCb + 108.0f;
            }
            
            // if (Cb > 77 && Cb < 127 && Cr > 133 && Cr < 173)
            if (Cb > 77 && Cb < 135 && Cr > 130 && Cr < 180/* && Y > 80*/) {
                pNormImage5Pt[h * imW + w] = 255;
            }
        }
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

static PyObject *SkinColorExtract_extract_skin_color_RGB(PyObject *self, PyObject *args)
{
    PyArrayObject *ImageArray;

    int retValue = 0;

    if (!PyArg_ParseTuple(args, "O!", &PyArray_Type, &ImageArray))
        return NULL;

    unsigned char *image_data = reinterpret_cast<unsigned char *>(PyArray_DATA(ImageArray));
    int imH = static_cast<int>(PyArray_DIM(ImageArray, 0));
    int imW = static_cast<int>(PyArray_DIM(ImageArray, 1));
    int imC = static_cast<int>(PyArray_DIM(ImageArray, 2));

    int stride_h = static_cast<int>(PyArray_STRIDE(ImageArray, 0));
    int stride_w = static_cast<int>(PyArray_STRIDE(ImageArray, 1));
    int stride_c = static_cast<int>(PyArray_STRIDE(ImageArray, 2));

    /*printf("%d, %d, %d\n", stride_h, stride_w, stride_c);

    printf("%d\n", strNormMethod);*/

    // BGR GMM Ä£ÐÍ
    int nCenterNum = 16;
    double Means[3][16] = {
        { 73.53, 249.71, 161.68, 186.07, 189.26, 247.00, 150.10, 206.85,
        212.78, 234.87, 151.19, 120.52, 192.20, 214.29, 99.57, 238.88 },
        { 29.94, 233.94, 116.25, 136.62, 98.37, 152.20, 72.66, 171.09,
        152.82, 175.43, 97.74, 77.55, 119.62, 136.08, 54.33, 203.08 },
        { 17.76, 217.49, 96.95, 114.40, 51.18, 90.84, 37.76, 156.34,
        120.04, 138.94, 74.59, 59.82, 82.32, 87.24, 38.06, 176.91 }
    };
    double Variances[3][16] = {
        { 765.40, 39.94, 291.03, 274.95, 633.18, 65.23, 408.63, 530.08,
        160.57, 163.80, 425.40, 330.45, 152.76, 204.90, 448.13, 178.38 },
        { 121.44, 154.44, 60.48, 64.60, 222.40, 691.53, 200.77, 155.08,
        84.52, 121.57, 73.56, 70.34, 92.14, 140.17, 90.18, 156.27 },
        { 112.80, 396.05, 162.85, 198.27, 250.69, 609.92, 257.57, 572.79,
        243.90, 279.22, 175.11, 151.82, 259.15, 270.19, 151.29, 404.99 }
    };
    double Weight[16] = { 0.0294, 0.0331, 0.0654, 0.0756, 0.0554, 0.0314, 0.0454, 0.0469,
        0.0956, 0.0763, 0.1100, 0.0676, 0.0755, 0.0500, 0.0667, 0.0749 };

    double dVariancesInv[3][16];
    for (int n = 0; n < 3; n++)
    {
        for (int k = 0; k < 16; k++)
        {
            dVariancesInv[n][k] = 1.0f / Variances[n][k];
        }
    }

    const double dConst = ((2 * 3.1415) * (2 * 3.1415) * (2 * 3.1415));
    double pConst[16];
    for (int k = 0; k < 16; k++)
    {
        pConst[k] = Weight[k] * 1.0f / sqrt(dConst * Variances[0][k] * Variances[1][k] * Variances[2][k]);
    }


    npy_intp Dims[1] = { imH * imW };
    PyArrayObject *PyArray = (PyArrayObject *)PyArray_SimpleNew(1, Dims, NPY_UBYTE);
    unsigned char *pNormImage5Pt = reinterpret_cast<unsigned char *>(PyArray_DATA(PyArray));
    memset(pNormImage5Pt, 0, imH * imW);

    for (int h = 0; h < imH; ++h) {
        for (int w = 0; w < imW; ++w) {
            float R = float(image_data[h * stride_h + w * stride_w]);
            float G = float(image_data[stride_c + h * stride_h + w * stride_w]);
            float B = float(image_data[2 * stride_c + h * stride_h + w * stride_w]);

            double dProb = 0.0f;
            for (int k = 0; k < nCenterNum; k++)
            {
                double lamda = ((R - Means[0][k])*(R - Means[0][k])) * dVariancesInv[0][k] +
                    ((G - Means[1][k]) * (G - Means[1][k])) * dVariancesInv[1][k] +
                    ((B - Means[2][k]) * (B - Means[2][k])) * dVariancesInv[2][k];

                dProb = dProb + (exp(-0.5 * lamda)) * pConst[k];
            }
            dProb = dProb * 1000000;

            if (dProb > 0.5) {
                pNormImage5Pt[h * imW + w] = 1;
            }
   
        }
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
SkinColorExtractMethods[] = {
    { "extract_skin_color_YCbCr", SkinColorExtract_extract_skin_color_YCbCr, METH_VARARGS },
    { "extract_skin_color_RGB", SkinColorExtract_extract_skin_color_RGB, METH_VARARGS },
    { NULL, NULL },
};

#ifdef PY3K
// module definition structure for python3
static struct PyModuleDef extract_skin_color_def = {
    PyModuleDef_HEAD_INIT,
    "SkinColorExtract",				// name of module
    NULL,							// module documentation, may be NULL
    -1, 	// size of per-interpreter state of the module,	or -1 if the module keeps state in global variables.
    SkinColorExtractMethods,
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
PyMODINIT_FUNC PyInit_SkinColorExtract()
{
    PyObject *obj = PyModule_Create(&extract_skin_color_def);
    import_array();  // Must be present for NumPy.  Called first after above line.
    return obj;
}
#else	//	PY3K
// module initializer for python2
extern "C" void initSkinColorExtract(void)
{
    Py_InitModule3("SkinColorExtract", SkinColorExtractMethods, "extract skin color area");
    import_array();  // Must be present for NumPy.  Called first after above line.
}
#endif	//	PY3K