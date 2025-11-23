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
import numpy as np

CCPM_FAKE = 65535

def make_aoa(__act_ids, _lnk_src, _lnk_dst):

    #==========================================================================
    # begin: ignore
    assert isinstance(__act_ids, list)
    assert isinstance(_lnk_src, list)
    assert isinstance(_lnk_dst, list)

    assert len(_lnk_src) == len(_lnk_dst)

    # Copy data
    lnk_src = _lnk_src.copy()
    lnk_dst = _lnk_dst.copy()
    _act_ids = list(set(__act_ids + lnk_src + lnk_dst))

    # Number of actions
    n_act = len(_act_ids)

    # Maximum size
    n_max = max(n_act, len(lnk_src)) + len(lnk_src)

    # Dependenci lists
    full_act_dep = []
    for i in range(n_max):
        full_act_dep.append([])

    full_dep_map = np.zeros((n_max, n_max), dtype=bool)

    for i in range(len(lnk_src)):
        # Raplace action ids by positions
        lnk_src[i] = _act_ids.index(lnk_src[i])
        lnk_dst[i] = _act_ids.index(lnk_dst[i])

        # Construct dependency matrix and lists
        full_dep_map[lnk_dst[i], lnk_src[i]] = True
        full_act_dep[lnk_dst[i]].append(lnk_src[i])
    # end: ignore
    #==========================================================================

    #==========================================================================
    # begin: ccpm_build_full_deps
    #
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
                    return False, _act_ids, None, None
                full_act_dep[i].append(k)
            j += 1
    # end: ccpm_build_full_deps
    #==========================================================================

    #==========================================================================
    # begin: ccpm_optimize_deps
    #
    # Action positions
    _act_pos = sorted(list(range(n_act)), key=lambda i: len(full_act_dep[i]))

    # Compute minimal dependency map
    min_dep_map = full_dep_map.copy()
    for p in range(n_act-1, -1, -1):
        i = _act_pos[p]
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
    for m in range(n_max):
        min_act_dep.append([])

    for m in range(n_act):
        i = _act_pos[m]
        for j in range(n_max):
            if min_dep_map[i, j]:
                min_act_dep[i].append(j)
    # and: ccpm_optimize_deps
    #==========================================================================

    # We need succesors to go later and action with shorter dep lists to go earlier
    _act_pos = sorted(_act_pos, key=lambda i: len(min_act_dep[i]))
    _act_pos = sorted(_act_pos, key=lambda i: len(full_act_dep[i]))

    #==========================================================================
    # Auxilary functions
    #
    # Reduce common action dependencies by adding dummies
    n_cur = 0

    #--------------------------------------------------------------------------
    # ccpm_hadle_deps
    def _handle_deps(min_deps, target):
        #
        # Append to target predeceptors
        full_dep_map[target, n_cur] = True
        full_act_dep[target].append(n_cur)
        #
        # Replace target min dependeoncies with dummy action
        for d in min_deps:
            min_dep_map[target, d] = False
            min_dep_map[target, n_cur] = True
            # Recompute target min deps
        min_act_dep[target] = []
        for d in range(n_max):
            if min_dep_map[target,  d]:
                min_act_dep[target].append(d)

    #--------------------------------------------------------------------------
    #ccpm_add_a_dummy
    def _add_a_dummy(min_deps, deps, dep_map):
        _act_ids.append(CCPM_FAKE)
        _act_pos.append(n_cur)
        # Set dummmy information if needed
        min_act_dep[n_cur] = min_deps.copy()
        for d in min_deps:
            min_dep_map[n_cur,  d] = True

        full_act_dep[n_cur] = deps.copy()
        full_dep_map[n_cur] = dep_map.copy()

    #--------------------------------------------------------------------------
    #ccpm_full_deps
    def _full_deps(min_deps):
        deps    = min_deps.copy()
        dep_map = np.zeros((n_max,), dtype=bool)
        #
        i = 0
        while i < len(deps):
            j = deps[i]
            for k in full_act_dep[j]:
                if dep_map[k]:
                    continue
                #
                dep_map[k] = True
                #
                if k == j:
                    raise RuntimeError("")
                #
                deps.append(k)
            i += 1
        #
        return deps, dep_map

    #==========================================================================
    # begin: ccpm_process_nested_deps
    #
    # Firsh reduce nested list of dependencies
    p = 0
    while p < n_act:
        i = _act_pos[p]
        if not len(min_act_dep[i]):
            p += 1
            continue
        #
        # Searching for nested list
        found_nested = False
        min_com_deps  = []
        q = p + 1
        while q < n_act:
            j = _act_pos[q]
            #
            if 0 == len(min_act_dep[j]):
                q += 1
                continue
            #
            min_com_deps = []
            for d in min_act_dep[i]:
                if min_dep_map[j,  d]:
                    min_com_deps.append(d)

            lcd = len(min_com_deps)
            if len(min_act_dep[i]) == lcd or len(min_act_dep[j]) == lcd:
                if len(min_act_dep[i]) != len(min_act_dep[j]):
                    # Nested lists, will reduce nesting lists
                    found_nested = True
                    break
            #
            # end while
            q += 1

        if not found_nested:
            # No nested lists found, continue
            p += 1
            continue

        # Found nested lists will reduce them
        lmcd = len(min_com_deps)
        deps, dep_map = _full_deps(min_com_deps)
        q = p + 1
        while q < n_act:
            j = _act_pos[q]
            #
            lmad = len(min_act_dep[j])
            if 0 == lmad or lmad == lmcd:
                # Skip empty, equal, non nested lists
                q += 1
                continue
            #
            com_deps = []
            for d in min_com_deps:
                if min_dep_map[j,  d]:
                    com_deps.append(d)
            #
            if len(com_deps) != lmcd:
                # Skip non nested lists
                q += 1
                continue

            # Reduce nested lists
            n_cur = len(_act_ids)
            _handle_deps(min_com_deps, j)
            _add_a_dummy(min_com_deps, deps, dep_map)
            # End while
            q += 1
        # End while
        p += 1
    # end: ccpm_process_nested_deps
    #==========================================================================

    #==========================================================================
    # begin: ccpm_process_overlapping_deps
    #
    # Next reduce overlaping lists of dependencies
    n_last = len(_act_ids)
    p = 0
    while p < n_last:
        #
        i = _act_pos[p]
        if not len(min_act_dep[i]):
            p += 1
            continue
        # Search for overlapping lists
        found_overlap = False
        min_com_deps  = []
        q = 0
        while q < n_last:
            #
            j = _act_pos[q]
            if not len(min_act_dep[j]):
                q += 1
                continue
            #
            min_com_deps = []
            for d in min_act_dep[i]:
                if min_dep_map[j,  d]:
                    min_com_deps.append(d)
            #
            lmcd = len(min_com_deps)
            if 0 < lmcd and len(min_act_dep[j]) != lmcd and len(min_act_dep[i]) != lmcd:
                # Found overlapping lists
                found_overlap = True
                break
            #
            # End while
            q += 1
        #
        if not found_overlap:
            n_last = len(_act_ids)
            p += 1
            continue
        #
        lmcd = len(min_com_deps)
        deps, dep_map = _full_deps(min_com_deps)
        q = 0
        while q < n_last:
            #
            j = _act_pos[q]
            if not len(min_act_dep[j]):
                q += 1
                continue
            #
            com_deps = []
            for d in min_com_deps:
                if min_dep_map[j,  d]:
                    com_deps.append(d)
            #
            if lmcd == len(com_deps) and len(min_act_dep[j]) != lmcd:
                # Reduce action dependencies overlap
                n_cur = len(_act_ids)
                _handle_deps(min_com_deps, j)
                _add_a_dummy(min_com_deps, deps, dep_map)
            # End while
            q += 1
        # End while
        n_last = len(_act_ids)
        p += 1

    # and: ccpm_process_overlapping_deps
    #==========================================================================

    #==========================================================================
    # begin: ccpm_build_network
    # Now build aoa network
    started = [False for i in _act_ids]
    num_dep = [len(i) for i in min_act_dep]
    _act_src = [0 for i in range(n_max)]
    _act_dst = [0 for i in range(n_max)]
    events  = []

    evt = 1
    def _find_started():
        group = []
        for i in range(len(started)):
            if (0 == num_dep[i]) and (not started[i]):
                started[i] = True
                _act_src[i] = evt
                group.append(i)
        return group

    dum = len(_act_ids)
    chk = _find_started()
    #
    events.append(evt)
    evt += 1

    i = 0
    while i < len(chk):
        for j in range(len(min_act_dep)):
            if chk[i] in min_act_dep[j]:
                num_dep[j] -= 1
        #
        start = _find_started()
        if len(start):
            for a in min_act_dep[start[0]]:
                if _act_dst[a]:
                    # Now we have to add an extra dummy
                    _act_pos.append(dum)
                    _act_ids.append(CCPM_FAKE)
                    _act_src[dum] = _act_dst[a]
                    _act_dst[dum] = evt
                    started.append(True)
                    dum += 1
                else:
                    _act_dst[a] = evt
            #
            events.append(evt)
            evt += 1
        #
        chk += start
        i += 1

    for i in range(dum):
        if 0 == _act_dst[i]:
            _act_dst[i] = evt;

    #Append last event
    events.append(evt)
    # end: ccpm_build_network
    #==========================================================================

    #==========================================================================
    # begin: ccpm_optimize_network_stage_1
    #
    # OK, we have a basic network now
    # Let us try to optimize it

    evt_deps  = [[] for e in events]
    evt_dins  = [[] for e in events]
    evt_real  = [False for e in events]
    dum = len(_act_ids)

    for i in range(n_max):
        for j in range(n_max):
            full_dep_map[i,j] = False
    #
    for k in range(dum):
        i = _act_dst[k] - 1
        j = _act_src[k] - 1

        #
        if CCPM_FAKE != _act_ids[k]:
            evt_real[i] = True
            continue
        #
        evt_dins[i].append(k)
        evt_deps[i].append(j)
        full_dep_map[i,j] = True


    # If some events have only dummy inputs and have equal dependencies
    # (earlier events) then we can "glue" them together
    for i in range(len(events)):
        if evt_real[i]:
            continue
        if not len(evt_deps[i]):
            continue
        #
        for j in range(i + 1, len(events)):
            if evt_real[j]:
                continue
            if len(evt_deps[i]) < 2:
                continue
            #
            if len(evt_deps[i]) != len(evt_deps[j]):
                continue
            #
            s = 0
            for d in evt_deps[i]:
                if full_dep_map[j, d]:
                    s += 1

            # Will redirect _act_src later
            # (events[i] != (i + 1)) is feature of redundant event
            if len(evt_deps[i]) == s:
                events[j] = events[i]

                for d in evt_dins[j]:
                    # Don't need these dummies anymore
                    _act_src[d] = CCPM_FAKE
                    _act_dst[d] = CCPM_FAKE


    # Work with redundant events and dummies (part 2)
    # "Glue" events each of which has only one input which is dummy
    # to their predeceptors
    for i in range(len(events)):
        if evt_real[i]:
            continue
        if 1 == len(evt_deps[i]):
            d = evt_dins[i][0]
            events[i] = _act_src[d]
            # Don't need this dummy anymore
            _act_src[d] = CCPM_FAKE
            _act_dst[d] = CCPM_FAKE

    #Do "Glue": Redirect IOs of redundant events
    def _do_glue():
        for i in range(dum):
            # Skip redundant dummies
            if CCPM_FAKE == _act_src[i] or CCPM_FAKE == _act_dst[i]:
                continue
            _act_src[i] = events[_act_src[i] - 1]
            _act_dst[i] = events[_act_dst[i] - 1]

    _do_glue()
    # and: ccpm_optimize_network_stage_1
    #==========================================================================

    #==========================================================================
    # begin: ccpm_optimize_network_stage_2
    #
    # Work with redundant events and dummies (part 3)
    # "Glue" events each of which has only one output which is dummy
    # to their successors
    evt_douts = [[] for e in events]
    evt_nout  = [0 for e in events]
    dum = len(_act_ids)

    for k in range(dum):
        if CCPM_FAKE == _act_src[k] or CCPM_FAKE == _act_dst[k]:
            continue
        #
        j = _act_src[k] - 1
        #
        evt_nout[j] += 1
        #
        if CCPM_FAKE != _act_ids[k]:
            continue
        #
        evt_douts[j].append(k)

    for i in range(len(events)):
        if evt_nout[i] > 1:
            continue
        if len(evt_douts[i]) < 1:
            continue
        d = evt_douts[i][0]
        events[i] = _act_dst[d]
        # Don't need this dummy anymore
        _act_src[d] = CCPM_FAKE
        _act_dst[d] = CCPM_FAKE

    _do_glue()
    # end: ccpm_optimize_network_stage_2
    #==========================================================================

    #==========================================================================
    # begin: ccpm_add_needed_dummies
    #
    # Add needed dummies
    _act_pos = sorted(_act_pos, key=lambda i: _act_dst[i])
    _act_pos = sorted(_act_pos, key=lambda i: _act_src[i])

    d = len(_act_ids)
    for i in range(d):
        if CCPM_FAKE == _act_src[i] or CCPM_FAKE == _act_dst[i]:
            continue
        if not started[i]:
            continue

        for j in range(i + 1, d):
            if CCPM_FAKE == _act_src[j] or CCPM_FAKE == _act_dst[j]:
                continue
            if _act_dst[i] == _act_dst[j] and _act_src[i] == _act_src[j]:
                started[j] = False

                evt += 1
                _act_dst[j] = evt

                dum = len(_act_ids)
                _act_pos.append(dum)
                _act_ids.append(CCPM_FAKE)
                _act_src[dum] = evt
                _act_dst[dum] = _act_dst[i]
                events.append(evt)
    # end: ccpm_add_needed_dummies
    #==========================================================================

    #==========================================================================
    # begin: ccpm_finalize_network
    #
    # Renumerate events
    evt = 1
    for i in range(len(events)):
        if events[i] != (i + 1):
            events[i] = CCPM_FAKE
            continue
        events[i] = evt
        evt += 1

    for i in range(len(_act_ids)):
        # Skip redundant dummies and events
        if CCPM_FAKE == _act_src[i] or CCPM_FAKE == _act_dst[i]:
            continue
        if CCPM_FAKE == events[_act_src[i] - 1] or \
            CCPM_FAKE == events[_act_dst[i] - 1]:
            continue
        #
        _act_src[i] = events[_act_src[i] - 1]
        _act_dst[i] = events[_act_dst[i] - 1]

    # Constuct result
    #act_pos = sorted(_act_pos, key=lambda i: _act_ids[i])
    act_ids = []
    act_src = []
    act_dst = []
    for p in _act_pos:
        if CCPM_FAKE == _act_ids[p]:
            continue
        act_ids.append(_act_ids[p])
        act_src.append(_act_src[p])
        act_dst.append(_act_dst[p])

    for p in _act_pos:
        if CCPM_FAKE != _act_ids[p] or \
            CCPM_FAKE == _act_src[p] or CCPM_FAKE == _act_dst[p]:
            continue
        act_src.append(_act_src[p])
        act_dst.append(_act_dst[p])

    # end: ccpm_finalize_network
    #==========================================================================

    return True, act_ids, act_src, act_dst

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

    src = [1,2,3, 2,3, 3,4, 1,6,7, 5,6,7,  3, 6, 7,  6, 8, 9,  7, 8, 9,10]
    dst = [5,5,5, 6,6, 7,7, 8,8,8, 9,9,9, 10,10,10, 11,11,11, 12,12,12,12]

    #
    #src = [1, 1, 1, 2, 2, 3,]
    #dst = [4, 5, 6, 5, 6, 6,]

    #src = [1,2,3, 2,3, 3, 4, 5, 6,  4, 5, 6,  0, 7,  0, 8,10, 0, 9, 10]
    #dst = [4,4,4, 5,5, 6, 7, 8, 9, 10,10,10, 11,11, 12,12,12, 13,13,13]

    uni = set(src + dst)
    act = list(range(min(uni), max(uni) + 1))

    status, _act_ids, _act_src, _act_dst = make_aoa(act, src, dst)
