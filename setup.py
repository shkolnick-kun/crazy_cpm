#!/usr/bin/env python3
# -*- coding: utf-8 -*-
#==============================================================================
"""
    This file is based on
    https://github.com/FedericoStra/cython-package-example/blob/master/setup.py
"""
#==============================================================================
"""
    MIT License

    Copyright (c) 2019 Federico Stra

    Permission is hereby granted, free of charge, to any person obtaining a copy
    of this software and associated documentation files (the "Software"), to deal
    in the Software without restriction, including without limitation the rights
    to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
    copies of the Software, and to permit persons to whom the Software is
    furnished to do so, subject to the following conditions:

        The above copyright notice and this permission notice shall be included in all
        copies or substantial portions of the Software.

    THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
    IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
    FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
    AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
    LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
    OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
    SOFTWARE.
"""
#==============================================================================
"""
    CrazyCPM
    Copyright (C) 2025 anonimous

    This program is free software: you can redistribute it and/or modify
    it under the terms of the GNU General Public License as published by
    the Free Software Foundation, either version 3 of the License, or
    (at your option) any later version.

    This program is distributed in the hope that it will be useful,
    but WITHOUT ANY WARRANTY; without even the implied warranty of
    MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
    GNU General Public License for more details.

    You should have received a copy of the GNU General Public License
    along with this program.  If not, see <http://www.gnu.org/licenses/>.

    Please contact with me by E-mail: shkolnick.kun@gmail.com
"""

#==============================================================================
import numpy
import os
from os.path import abspath, dirname, isfile, join, relpath, splitext
from setuptools import Extension, setup

try:
    from Cython.Build import cythonize
except ImportError:
    cythonize = None

# https://cython.readthedocs.io/en/latest/src/userguide/source_files_and_compilation.html#distributing-cython-modules
def no_cythonize(extensions, **_ignore):
    for extension in extensions:
        sources = []
        for sfile in extension.sources:
            path, ext = splitext(sfile)
            if ext in (".pyx", ".py"):
                if extension.language == "c++":
                    ext = ".cpp"
                else:
                    ext = ".c"
                sfile = path + ext
            sources.append(sfile)
        extension.sources[:] = sources
    return extensions

EXT_BASE  = "ccpm"
EXT_NAME  = "_" + EXT_BASE
SETUP_DIR = abspath(dirname(__file__))

src_dir   = join(SETUP_DIR, "src")
ext_dir   = join(src_dir, "crazy_cpm")
pyx_file  = join(ext_dir, EXT_NAME + ".pyx")
ext_file  = join(ext_dir, EXT_NAME + ".c")

#Nowadays setup needs relative paths
src_dir   = relpath(src_dir,  SETUP_DIR)
ext_dir   = relpath(ext_dir,  SETUP_DIR)
pyx_file  = relpath(pyx_file, SETUP_DIR)
ext_file  = relpath(ext_file, SETUP_DIR)

extensions = [
    Extension(EXT_NAME, [pyx_file],
        include_dirs=[numpy.get_include(), src_dir, ext_dir],
        define_macros=[("NPY_NO_DEPRECATED_API", "NPY_1_7_API_VERSION")])
    ]

if not isfile(ext_file):
    CYTHONIZE = cythonize is not None
else:
    CYTHONIZE = bool(int(os.getenv("CYTHONIZE", 0))) and cythonize is not None

if CYTHONIZE:
    compiler_directives = {"language_level": 3, "embedsignature": True}
    extensions = cythonize(extensions, compiler_directives=compiler_directives)
else:
    extensions = no_cythonize(extensions)

setup(ext_modules=extensions)
