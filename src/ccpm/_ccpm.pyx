# -*- coding: utf-8 -*-
"""
    CrazyCPM
    Copyright (C) 2021 anonimous

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
#cython: language_level=3
#distutils: language=c

from libc cimport stdint
from libcpp cimport bool

ctypedef stdint.uint16_t _uint16_t
ctypedef bool _bool

cdef extern from "ccpm.c":
    ctypedef enum ccpmResultEn:
        CCPM_OK=0
        CCPM_EINVAL
        CCPM_ENOMEM
        CCPM_ELOOP

    cdef ccpmResultEn ccpm_check_act_ids(_uint16_t * act_id, _uint16_t n_act)
    
    cdef ccpmResultEn ccpm_check_links(_uint16_t * lnk_src, _uint16_t * lnk_dst, \
                                       _uint16_t   n_lnk)
    
    cdef ccpmResultEn ccpm_links_prepare(_uint16_t *  act_id, _uint16_t     n_act, \
                                         _uint16_t * lnk_src, _uint16_t * lnk_dst, \
                                         _uint16_t     n_lnk)
    
    cdef ccpmResultEn ccpm_populate_dep_info(_uint16_t *  act_id, _uint16_t *     dep, \
                                             _uint16_t *   n_dep, _bool     * dep_map, \
                                             _uint16_t * lnk_src, _uint16_t * lnk_dst, \
                                             _uint16_t     n_lnk)
        
    cdef ccpmResultEn ccpm_sort(_uint16_t * tmp, _uint16_t * key, \
                                _uint16_t * val, _uint16_t     n)

    cdef ccpmResultEn ccpm_build_dep(_uint16_t     n_act, _uint16_t    map_len, \
                                     _uint16_t *     tmp, _uint16_t *   act_id, \
                                     _uint16_t * act_pos, _uint16_t *    opt_n, \
                                     _uint16_t * opt_dep, _bool     *  opt_map, \
                                     _uint16_t *  full_n, _uint16_t * full_dep, \
                                     _bool     * full_map)

    cdef ccpmResultEn ccpm_make_aoa(_uint16_t * act_id,  \
                                    _uint16_t * act_src, \
                                    _uint16_t * act_dst, \
                                    _uint16_t n_act,     \
                                    _uint16_t * n_dum,   \
                                    _uint16_t * lnk_src, \
                                    _uint16_t * lnk_dst, \
                                    _uint16_t *n_lnk)

    cdef ccpmResultEn ccpm_make_full_map(_uint16_t *  act_id, _uint16_t n_act, \
                                         _uint16_t * lnk_src, _uint16_t * lnk_dst, _uint16_t n_lnk,
                        _bool * full_dep_map)

import  numpy as np#WTF??
cimport numpy as np

#Define constants
OK     = CCPM_OK
EINVAL = CCPM_EINVAL
ENOMEM = CCPM_ENOMEM
ELOOP  = CCPM_ELOOP

###############################################################################
def compute_aoa(np.ndarray act_id, np.ndarray lnk_src, np.ndarray lnk_dst):
    #assert act_id.dtype  == np.uint16
    #assert lnk_src.dtype == np.uint16
    #assert lnk_dst.dtype == np.uint16
    assert len(lnk_src)  == len(lnk_dst)

    cdef _uint16_t n_act = len(act_id)
    cdef _uint16_t n_lnk = len(lnk_src)
    cdef _uint16_t n_dum = 0

    _act_id  = act_id.astype(np.uint16)
    _lnk_src = lnk_src.astype(np.uint16)
    _lnk_dst = lnk_dst.astype(np.uint16)

    cdef _uint16_t [::1] v_act_id  = _act_id
    cdef _uint16_t [::1] v_lnk_src = _lnk_src
    cdef _uint16_t [::1] v_lnk_dst = _lnk_dst

    act_src = np.zeros((n_act + n_lnk,), dtype=np.uint16)
    act_dst = np.zeros((n_act + n_lnk,), dtype=np.uint16)

    cdef _uint16_t [::1] v_act_src = act_src
    cdef _uint16_t [::1] v_act_dst = act_dst

    status = ccpm_make_aoa(&v_act_id[0], &v_act_src[0], &v_act_dst[0], n_act, \
                           &n_dum, &v_lnk_src[0], &v_lnk_dst[0], &n_lnk)

    return status, \
        act_src[:n_act + n_dum].copy(), act_dst[:n_act + n_dum].copy(), \
            _lnk_src[:n_lnk].copy(), _lnk_dst[:n_lnk].copy()

###############################################################################
def make_full_map(np.ndarray act_id, np.ndarray lnk_src, np.ndarray lnk_dst):
    assert len(lnk_src) == len(lnk_dst)

    cdef _uint16_t n_act = len(act_id)
    cdef _uint16_t n_lnk = len(lnk_src)

    _act_id  = act_id.astype(np.uint16)
    _lnk_src = lnk_src.astype(np.uint16) 
    _lnk_dst = lnk_dst.astype(np.uint16)

    cdef _uint16_t [::1] v_act_id  = _act_id
    cdef _uint16_t [::1] v_lnk_src = _lnk_src
    cdef _uint16_t [::1] v_lnk_dst = _lnk_dst
    
    full_dep_map = np.zeros((n_act, n_act), dtype=np.bool_)
    cdef _bool [:, ::1] full_dep_map_view = full_dep_map

    cdef ccpmResultEn status
    status = ccpm_make_full_map(&v_act_id[0], n_act,
                               &v_lnk_src[0], &v_lnk_dst[0], n_lnk,
                               &full_dep_map_view[0, 0])

    return status, full_dep_map