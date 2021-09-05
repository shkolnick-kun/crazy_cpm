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

ctypedef stdint.uint16_t _uint16_t

cdef extern from "ccpm.c":
    ctypedef enum ccpmResultEn:
        CCMP_OK=0
        CCMP_EINVAL
        CCMP_ENOMEM
        CCMP_ELOOP

    cdef ccpmResultEn ccpm_make_aoa(_uint16_t * act_id,  \
                                    _uint16_t * act_src, \
                                    _uint16_t * act_dst, \
                                    _uint16_t n_act,     \
                                    _uint16_t * n_dum,   \
                                    _uint16_t * lnk_src, \
                                    _uint16_t * lnk_dst, \
                                    _uint16_t *n_lnk)

    cdef double ccpm_viz_loss(double * p,             \
                              _uint16_t * node_layer, \
                              _uint16_t n_node,       \
                              _uint16_t * edge_src,   \
                              _uint16_t * edge_dst,   \
                              double * edge_w,        \
                              _uint16_t n_edge)

import  numpy as np#WTF??
cimport numpy as np

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
cdef class vizGraphLoss:
    cdef _uint16_t _node_n
    cdef _uint16_t _edge_n

    cdef np.ndarray _node_layer
    cdef np.ndarray _edge_src
    cdef np.ndarray _edge_dst
    cdef np.ndarray _edge_w

    cdef _uint16_t [::1] v_node_layer
    cdef _uint16_t [::1] v_edge_src
    cdef _uint16_t [::1] v_edge_dst
    cdef double    [::1] v_edge_w

    def __init__(self, np.ndarray node_id, np.ndarray node_layer,\
                 np.ndarray edge_src, np.ndarray edge_dst, np.ndarray edge_w):

        assert len(node_id) == len(node_layer)
        assert len(edge_w)  == len(edge_src)
        assert len(edge_w)  == len(edge_dst)

        self._node_layer  = node_layer.astype(np.uint16)
        self.v_node_layer = self._node_layer

        self._edge_src  = edge_src.astype(np.uint16)
        self.v_edge_src = self._edge_src

        self._edge_dst  = edge_dst.astype(np.uint16)
        self.v_edge_dst = self._edge_dst

        self._edge_w  = edge_w.astype(np.float64)
        self.v_edge_w = self._edge_w

        #Тут не нужна особая эффективность: выполняем 1 раз
        self._node_n = len(node_id)
        self._edge_n = len(edge_w)

        _node_id = list(node_id)

        for i in range(self._edge_n):
            self._edge_src[i] = _node_id.index(self._edge_src[i])
            self._edge_dst[i] = _node_id.index(self._edge_dst[i])

    def run(self, np.ndarray p):
        assert len(p) == self._node_n

        cdef np.ndarray _p = p.astype(np.float64)
        cdef double    [::1] v_p = _p

        cdef float loss

        #TODO: with nogil:
        loss = ccpm_viz_loss(&v_p[0],
                             &self.v_node_layer[0],
                             self._node_n,
                             &self.v_edge_src[0],
                             &self.v_edge_dst[0],
                             &self.v_edge_w[0],
                             self._edge_n)

        return loss


