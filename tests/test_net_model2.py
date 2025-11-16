#!/usr/bin/env python3
# -*- coding: utf-8 -*-
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
from betapert import mpert
import graphviz
import numpy as np
import pandas as pd
import os
from crazy_cpm import NetworkModel

def reduce_dependencies(_act_ids, _lnk_src, _lnk_dst):
    assert isinstance(_act_ids, list)
    assert isinstance(_lnk_src, list)
    assert isinstance(_lnk_dst, list)

    assert len(_lnk_src) == len(_lnk_dst)

    # Copy data
    lnk_src = _lnk_src.copy()
    lnk_dst = _lnk_dst.copy()
    act_ids = list(set(_act_ids + lnk_src + lnk_dst))

    # Number of actions
    n_act = len(act_ids)

    # Full size
    nf = max(n_act, len(lnk_src)) + len(lnk_src)

    # Dependenci lists
    full_act_dep = []
    for i in range(nf):
        full_act_dep.append([])

    full_dep_map = np.zeros((nf, nf), dtype=bool)

    for i in range(len(lnk_src)):
        # Raplace action ids by positions
        lnk_src[i] = act_ids.index(lnk_src[i])
        lnk_dst[i] = act_ids.index(lnk_dst[i])

        # Construct dependency matrix and lists
        full_dep_map[lnk_dst[i], lnk_src[i]] = True
        full_act_dep[lnk_dst[i]].append(lnk_src[i])

    # Construct full dependency map and lists of full dependencies
    for i in range(n_act):
        if not len(full_act_dep[i]):
            continue
        j = 0
        while j < len(full_act_dep[i]):
            d = full_act_dep[i][j]
            for k in full_act_dep[d]:
                if full_dep_map[i, k]:
                    continue
                full_dep_map[i, k] = True
                if i == k:
                    return False, act_ids, None, None, None, None, None
                full_act_dep[i].append(k)
            j += 1

    #Action positions
    act_pos = sorted(list(range(n_act)), key=lambda i: len(full_act_dep[i]))

    # Compute minimal dependency map
    min_dep_map = full_dep_map.copy()
    for p in range(n_act-1, -1, -1):
        i = act_pos[p]
        for l in range(len(full_act_dep[i])):
            j = full_act_dep[i][l]
            for m in range(len(full_act_dep[i])):
                k = full_act_dep[i][m]
                if k == j:
                    continue
                if full_dep_map[k, j]:
                    min_dep_map[i, j] = False

    # Compute minimal dependency lists
    min_act_dep = []
    for m in range(nf):
        min_act_dep.append([])

    for m in range(n_act):
        i = act_pos[m]
        for j in range(nf):
            if min_dep_map[i, j]:
                min_act_dep[i].append(j)

    # We need succesors to go later and action with shorter dep lists to go earlier
    act_pos = sorted(act_pos, key=lambda i: len(min_dep_map[i]))
    act_pos = sorted(act_pos, key=lambda i: len(full_act_dep[i]))

    # Reduce common action dependencies by adding dummies
    n_mix = n_act
    #
    def _handle_deps(min_deps, target):
        #
        # Append to target predeceptors
        full_dep_map[target, n_mix] = True
        full_act_dep[target].append(n_mix)
        #
        # Replace target min dependeoncies with dummy action
        for d in min_deps:
            min_dep_map[target, d] = False
            min_dep_map[target, n_mix] = True
            # Recompute target min deps
        min_act_dep[target] = []
        for d in range(nf):
            if min_dep_map[target,  d]:
                min_act_dep[target].append(d)


    def _add_a_dummy(full_deps, min_deps):
        act_ids.append(65535)
        act_pos.append(n_mix)
        # Set dummmy information if needed
        min_act_dep[n_mix] = min_deps.copy()
        for d in min_deps:
            min_dep_map[n_mix,  d] = True

        full_act_dep[n_mix] = full_deps.copy()
        for d in full_deps:
            full_dep_map[n_mix, d] = True

    # Firsh reduce nested list of dependencies
    p = 0
    while p < n_act:
        i = act_pos[p]
        if not len(full_act_dep[i]):
            p += 1
            continue
        #
        # Searching for nested list
        min_com_deps  = []
        q = p + 1
        while q < n_act:
            j = act_pos[q]
            #
            if 0 == len(full_act_dep[j]):
                q += 1
                continue
            #
            min_com_deps = []
            for d in min_act_dep[i]:
                if min_dep_map[j,  d]:
                    min_com_deps.append(d)
            #
            if 0 == len(min_com_deps):
                q += 1
                continue
            #
            if len(min_act_dep[i]) == len(min_com_deps):
                #Skip equal lists
                if len(min_act_dep[j]) == len(min_com_deps):
                    min_com_deps = []
                    q += 1
                    continue
                # Found nested list
                _handle_deps(min_com_deps, j)
                q += 1
                break
            #
            min_com_deps = []
            q += 1
        #
        # Now handle deps for next actions with the same deps nested
        while q < n_act:
            j = act_pos[q]
            #
            com_deps = []
            for d in min_com_deps:
                if min_dep_map[j,  d]:
                    com_deps.append(d)
            #
            if 0 == len(com_deps):
                q += 1
                continue
            #
            if len(min_com_deps) == len(com_deps):
                #Skip equal lists
                if len(min_com_deps) == len(min_act_dep[j]):
                    q += 1
                    continue
                # Found nested list
                _handle_deps(min_com_deps, j)
            #
            q += 1
        # Now we may add s new dummy
        if 0 < len(min_com_deps):
            _add_a_dummy(full_act_dep[i], min_com_deps)
            n_mix += 1

        p += 1

    # Next reduce overlaping lists of dependencies
    #
    n_rnl = n_mix
    i = 0
    while i < n_rnl:
        #
        if not len(full_act_dep[i]):
            i += 1
            continue
        # Search for overlapping lists
        min_com_deps  = []
        j = 0
        while j < n_rnl:
            #
            if not len(full_act_dep[j]):
                j += 1
                continue
            #
            min_com_deps = []
            for d in min_act_dep[i]:
                if min_dep_map[j,  d]:
                    min_com_deps.append(d)
            #
            if 0 == len(min_com_deps) or len(min_act_dep[i]) == len(min_com_deps):
                # Skip equal or nonoverlapping lists
                min_com_deps = []
                j += 1
                continue
            #
            # Reduce first two actions dependencies overlap
            full_com_deps = []
            for d in full_act_dep[i]:
                if full_dep_map[j,  d]:
                    full_com_deps.append(d)

            _handle_deps(min_com_deps, i)
            _add_a_dummy(full_com_deps, min_com_deps)
            n_mix += 1

            _handle_deps(min_com_deps, j)
            _add_a_dummy(full_com_deps, min_com_deps)
            n_mix += 1
            j += 1
            break
        #
        while j < n_rnl:
            #
            if not len(full_act_dep[j]):
                j += 1
                continue
            #
            com_deps = []
            for d in min_com_deps:
                if min_dep_map[j,  d]:
                    com_deps.append(d)
            #
            if 0 == len(com_deps) or len(min_act_dep[i]) == len(com_deps):
                # Skip equal or nonoverlapping lists
                j += 1
                continue
            #
            # Reduce remaining action dependencies overlap
            full_com_deps = []
            for d in full_act_dep[i]:
                if full_dep_map[j,  d]:
                    full_com_deps.append(d)

            _handle_deps(min_com_deps, j)
            _add_a_dummy(full_com_deps, min_com_deps)
            n_mix += 1
            j += 1
        i += 1

    # Now build aoa network
    started = [False for i in act_ids]
    num_dep = [len(i) for i in min_act_dep]
    act_src = [0 for i in range(nf)]
    act_dst = [0 for i in range(nf)]

    evt = 1
    def _find_started():
        group = []
        for i in range(len(started)):
            if (0 == num_dep[i]) and (not started[i]):
                started[i] = True
                act_src[i] = evt
                group.append(i)
        return group

    chk = _find_started()
    evt += 1
    dum = len(act_ids)
    i = 0
    while i < len(chk):
        for j in range(len(min_act_dep)):
            if chk[i] in min_act_dep[j]:
                num_dep[j] -= 1
        #
        start = _find_started()
        if len(start):
            for a in min_act_dep[start[0]]:
                act_dst[a] = evt
            evt += 1
        chk += start
        i += 1

    for i in range(dum):
        if 0 == act_dst[i]:
            act_dst[i] = evt

    return True, act_src[:dum].copy(), act_dst[:dum].copy(), act_ids, act_pos

#==============================================================================
if __name__ == '__main__':
    act = list(range(14))

    #src = []
    #dst = []

    #src = [1, 2, 3,  5, 6, 7,   4, 4, 4 ]
    #dst = [5, 6, 7,  8, 9, 10,  8, 9, 10]

    #src = [1, 2, 3,  5, 6, 7,   4, 4, 4, 10,]
    #dst = [5, 6, 7,  8, 9, 10,  8, 9, 10, 3,]

    #src = [1, 1, 1,  2, 3, 4,  5, 5, 5,  6, 7, 8,  9,  9,  9, ]
    #dst = [2, 3, 4,  5, 5, 5,  6, 7, 8,  9, 9, 9,  10, 11, 12 ]

    #src = [1,2,3, 2,3, 3,4, 1,6,7, 5,6,7,  3, 6, 7,  6, 8, 9,  7, 8, 9,10]
    #dst = [5,5,5, 6,6, 7,7, 8,8,8, 9,9,9, 10,10,10, 11,11,11, 12,12,12,12]

    #
    src = [1, 1, 1, 2, 2, 3,]
    dst = [4, 5, 6, 5, 6, 6,]

    #src = [1,2,3, 2,3, 3, 4, 5, 6,  4, 5, 6,  0, 7,  0, 8,10, 0, 9, 10]
    #dst = [4,4,4, 5,5, 6, 7, 8, 9, 10,10,10, 11,11, 12,12,12, 13,13,13]

    uni = set(src + dst)
    act = list(range(min(uni), max(uni) + 1))

    status, act_src, act_dst, act_ids, act_pos = reduce_dependencies(act, src, dst)
