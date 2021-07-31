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

cdef extern from "ccpm.c":
    ctypedef enum ccpmResultEn:
        CCMP_OK=0
        CCMP_EINVAL
        CCMP_ENOMEM
        CCMP_ELOOP

    cdef ccpmResultEn ccpm_make_aoa(stdint.uint16_t * act_id,  \
                                    stdint.uint16_t * act_src, \
                                    stdint.uint16_t * act_dst, \
                                    stdint.uint16_t n_act,     \
                                    stdint.uint16_t * n_dum,   \
                                    stdint.uint16_t * lnk_src, \
                                    stdint.uint16_t * lnk_dst, \
                                    stdint.uint16_t *n_lnk)

cimport numpy as np
import  numpy as np#WTF??

def ccpm_compute_aoa(np.ndarray act_id, np.ndarray lnk_src, np.ndarray lnk_dst):
    assert act_id.dtype == np.int
    assert lnk_src.dtype == np.int
    assert lnk_dst.dtype == np.int
    assert len(lnk_src) == len(lnk_dst)

    cdef stdint.uint16_t n_act = len(act_id)
    cdef stdint.uint16_t n_lnk = len(lnk_src)
    cdef stdint.uint16_t n_dum = 0

    _act_id  = act_id.astype(np.uint16)
    _lnk_src = lnk_src.astype(np.uint16)
    _lnk_dst = lnk_dst.astype(np.uint16)

    cdef stdint.uint16_t [::1] v_act_id  = _act_id
    cdef stdint.uint16_t [::1] v_lnk_src = _lnk_src
    cdef stdint.uint16_t [::1] v_lnk_dst = _lnk_dst

    act_src = np.zeros((n_act + n_lnk,), dtype=np.uint16)
    act_dst = np.zeros((n_act + n_lnk,), dtype=np.uint16)

    cdef stdint.uint16_t [::1] v_act_src = act_src
    cdef stdint.uint16_t [::1] v_act_dst = act_dst

    status = ccpm_make_aoa(&v_act_id[0], &v_act_src[0], &v_act_dst[0], n_act, \
                           &n_dum, &v_lnk_src[0], &v_lnk_dst[0], &n_lnk)

    return status, \
        act_src[:n_act].copy(), \
            act_dst[:n_act].copy(), \
                act_src[n_act : n_act + n_dum].copy(), \
                    act_dst[n_act : n_act + n_dum].copy(), \
                        lnk_src[:n_lnk].copy(), lnk_dst[:n_lnk].copy()