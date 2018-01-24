# This is an example of a distutils 'setup' script for the example_nt
# sample.  This provides a simpler way of building your extension
# and means you can avoid keeping MSVC solution files etc in source-control.
# It also means it should magically build with all compilers supported by
# python.

# USAGE: you probably want 'setup.py install' - but execute 'setup.py --help'
# for all the details.

# NOTE: This is *not* a sample for distutils - it is just the smallest
# script that can build this.  See distutils docs for more info.

from __future__ import absolute_import
from distutils.core import setup, Extension
import numpy

HoughTransform_mod = Extension('HoughTransform',
                        sources = ['HoughTransform.cpp'],
                        include_dirs = [numpy.get_include(), '/usr/include'],
                        library_dirs = ['D:/work/THIDFaceCollection/common/SDKLib/x64/lib/', '/usr/lib/'],
                        language="c++",
                        extra_compile_args=['-std=c++11', '-fopenmp'], 
                        extra_link_args=['-lgomp'])


setup(name = "HoughTransform",
    version = "1.0",
    description = "A sample extension module",
    ext_modules = [HoughTransform_mod],
)
