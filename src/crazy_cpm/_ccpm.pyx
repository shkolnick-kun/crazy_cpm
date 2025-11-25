# -*- coding: utf-8 -*-
"""
Cython module for high-performance CPM/PERT computations.

This module provides C-accelerated functions for network analysis,
including Activity-on-Arrow (AoA) network generation and dependency
mapping for Critical Path Method calculations.

.. note::
    This module requires Cython and a C compiler for building.
    The C backend provides significant performance improvements
    for large network models.

Functions:
    compute_aoa: Generate Activity-on-Arrow network from activity dependencies
    make_full_map: Create complete dependency matrix for network analysis

Constants:
    OK: Operation completed successfully
    EINVAL: Invalid input parameters
    ENOMEM: Memory allocation failure
    ELOOP: Circular dependency detected in network

"""
###############################################################################
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
###############################################################################

#cython: language_level=3
#distutils: language=c

from libc cimport stdint
from libcpp cimport bool

ctypedef stdint.uint8_t  _uint8_t
ctypedef stdint.uint16_t _uint16_t
ctypedef stdint.uint32_t _uint32_t

cdef extern from "ccpm.c":
    ctypedef enum ccpmResultEn:
        CCPM_OK=0
        CCPM_EINVAL
        CCPM_ENOMEM
        CCPM_ELOOP
        CCPM_ELIM
        CCPM_EUNK

    cdef ccpmResultEn ccpm_make_aoa(_uint16_t * act_ids,
                                    _uint16_t * lnk_src,
                                    _uint16_t * lnk_dst,
                                    size_t      n_lnk,
                                    _uint16_t * act_src,
                                    _uint16_t * act_dst
                                    )

    cdef ccpmResultEn ccpm_make_full_map(_uint16_t * act_ids,
                                         _uint16_t * lnk_src,
                                         _uint16_t * lnk_dst,
                                         size_t      n_lnk,
                                         size_t      n_max,
                                         _uint32_t * full_act_dep,
                                         _uint8_t  * full_dep_map
                                         )

import  numpy as np
cimport numpy as cnp

# Define constants
OK     = CCPM_OK
EINVAL = CCPM_EINVAL
ENOMEM = CCPM_ENOMEM
ELOOP  = CCPM_ELOOP
ELIM   = CCPM_ELIM
EUNK   = CCPM_EUNK

###############################################################################
def make_aoa(act_ids, lnk_src, lnk_dst):
    """
    Cython wrapper for ccpm_make_aoa - converts Python lists to C arrays and back

    Args:
        act_ids: List of activity IDs
        lnk_src: List of link sources
        lnk_dst: List of link destinations

    Returns:
        tuple: (status_code, status_message, act_ids, act_src, act_dst)
        where:
          status_code: integer status code (0 = success)
          status_message: human-readable status description
          act_ids: resulting activity IDs
          act_src: resulting activity source events
          act_dst: resulting activity destination events
    """
    cdef size_t n_act = len(act_ids)
    cdef size_t n_lnk = len(lnk_src)
    cdef size_t n_max = n_act + (n_lnk if n_lnk > n_act else n_act)

    # Create buffer arrays
    act_ids_arr = np.zeros(n_max + 1, dtype=np.uint16)
    lnk_src_arr = np.zeros(n_lnk, dtype=np.uint16)
    lnk_dst_arr = np.zeros(n_lnk, dtype=np.uint16)
    act_src_arr = np.zeros(n_max + 1, dtype=np.uint16)
    act_dst_arr = np.zeros(n_max + 1, dtype=np.uint16)

    # Prepare input data
    act_ids_arr[0] = n_act
    for i in range(n_act):
        act_ids_arr[i + 1] = act_ids[i]

    for i in range(n_lnk):
        lnk_src_arr[i] = lnk_src[i]
        lnk_dst_arr[i] = lnk_dst[i]

    # Memory views
    cdef _uint16_t[:] act_ids_view = act_ids_arr
    cdef _uint16_t[:] lnk_src_view = lnk_src_arr
    cdef _uint16_t[:] lnk_dst_view = lnk_dst_arr
    cdef _uint16_t[:] act_src_view = act_src_arr
    cdef _uint16_t[:] act_dst_view = act_dst_arr

    # Make AoA network
    cdef ccpmResultEn result = ccpm_make_aoa(&act_ids_view[0],
                                             &lnk_src_view[0],
                                             &lnk_dst_view[0],
                                             n_lnk,
                                             &act_src_view[0],
                                             &act_dst_view[0]
                                             )

    # Get output data
    py_act_ids = []
    for i in range(act_ids_arr[0]):
        py_act_ids.append(act_ids_arr[i + 1])

    py_act_src = []
    py_act_dst = []

    for i in range(act_src_arr[0]):
        py_act_src.append(act_src_arr[i + 1])
        py_act_dst.append(act_dst_arr[i + 1])

    return result, py_act_ids, py_act_src, py_act_dst

###############################################################################
def make_full_map(act_ids, lnk_src, lnk_dst):
    """
    Build full dependency map for activities

    Args:
        act_ids: List of activity IDs
        lnk_src: List of link sources
        lnk_dst: List of link destinations

    Returns:
        tuple: (status_code, status_message, full_dep_map)
        where:
          status_code: integer status code (0 = success)
          status_message: human-readable status description
          full_dep_map: 2D numpy array with dtype=bool representing the full dependency matrix
    """
    cdef size_t n_act = len(act_ids)
    cdef size_t n_lnk = len(lnk_src)
    cdef size_t n_max = n_act

    # Create buffer arrays
    act_ids_arr      = np.zeros(n_act + 1, dtype=np.uint16)
    lnk_src_arr      = np.zeros(n_lnk, dtype=np.uint16)
    lnk_dst_arr      = np.zeros(n_lnk, dtype=np.uint16)
    full_act_dep_arr = np.zeros(n_max * n_max, dtype=np.uint32)
    full_dep_map_arr = np.zeros(n_max * n_max, dtype=np.uint8)

    # Prepare input data
    act_ids_arr[0] = n_act
    for i in range(n_act):
        act_ids_arr[i + 1] = act_ids[i]

    for i in range(n_lnk):
        lnk_src_arr[i] = lnk_src[i]
        lnk_dst_arr[i] = lnk_dst[i]

    # Memory views
    cdef _uint16_t[:] act_ids_view      = act_ids_arr
    cdef _uint16_t[:] lnk_src_view      = lnk_src_arr
    cdef _uint16_t[:] lnk_dst_view      = lnk_dst_arr
    cdef _uint32_t[:] full_act_dep_view = full_act_dep_arr
    cdef _uint8_t[:]  full_dep_map_view = full_dep_map_arr

    # Compute dependency map
    cdef ccpmResultEn result = ccpm_make_full_map(&act_ids_view[0],
                                                  &lnk_src_view[0],
                                                  &lnk_dst_view[0],
                                                  n_lnk,
                                                  n_max,
                                                  &full_act_dep_view[0],
                                                  &full_dep_map_view[0]
                                                  )

    # Конвертируем результат в numpy bool array
    full_dep_map_np = np.zeros((n_act, n_act), dtype=np.bool_)
    for i in range(n_act):
        for j in range(n_act):
            full_dep_map_np[i, j] = full_dep_map_arr[i * n_max + j]

    return result, full_dep_map_np
