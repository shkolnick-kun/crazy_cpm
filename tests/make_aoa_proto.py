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

def make_aoa(_act_ids, _lnk_src, _lnk_dst):
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
                    return False, act_ids, None, None
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

    #==========================================================================
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
        act_ids.append(CCPM_FAKE)
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

    #==========================================================================
    # Now build aoa network
    started = [False for i in act_ids]
    num_dep = [len(i) for i in min_act_dep]
    act_src = [0 for i in range(nf)]
    act_dst = [0 for i in range(nf)]
    events  = []

    evt = 1
    def _find_started():
        group = []
        for i in range(len(started)):
            if (0 == num_dep[i]) and (not started[i]):
                started[i] = True
                act_src[i] = evt
                group.append(i)
        return group

    dum = len(act_ids)
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
                if act_dst[a]:
                    # Now we have to add an extra dummy
                    act_pos.append(dum)
                    act_ids.append(CCPM_FAKE)
                    act_src[dum] = act_dst[a]
                    act_dst[dum] = evt
                    started.append(True)
                    dum += 1
                else:
                    act_dst[a] = evt
            #
            events.append(evt)
            evt += 1
        #
        chk += start
        i += 1

    for i in range(dum):
        if 0 == act_dst[i]:
            act_dst[i] = evt;

    #Append last event
    events.append(evt)

    #==========================================================================
    # OK, we have a basic network now
    # Let us try to optimize it

    evt_deps  = [[] for e in events]
    evt_dins  = [[] for e in events]
    evt_real  = [False for e in events]
    evt_pos   = [i for i in range(len(events))]

    for i in range(nf):
        for j in range(nf):
            full_dep_map[i,j] = False
    #
    for k in range(dum):
        i = act_dst[k] - 1
        j = act_src[k] - 1

        #
        if CCPM_FAKE != act_ids[k]:
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

            # Will redirect act_src later
            # (events[i] != (i + 1)) is feature of redundant event
            if len(evt_deps[i]) == s:
                events[j] = events[i]

                for d in evt_dins[j]:
                    # Don't need these dummies anymore
                    act_src[d] = CCPM_FAKE
                    act_dst[d] = CCPM_FAKE


    # Work with redundant events and dummies (part 2)
    # "Glue" events each of which has only one input which is dummy
    # to their predeceptors
    for i in range(len(events)):
        if evt_real[i]:
            continue
        if 1 == len(evt_deps[i]):
            d = evt_dins[i][0]
            events[i] = act_src[d]
            # Don't need this dummy anymore
            act_src[d] = CCPM_FAKE
            act_dst[d] = CCPM_FAKE

    #Do "Glue": Redirect IOs of redundant events
    def _do_glue():
        for i in range(dum):
            # Skip redundant dummies
            if CCPM_FAKE == act_src[i] or CCPM_FAKE == act_dst[i]:
                continue
            act_src[i] = events[act_src[i] - 1]
            act_dst[i] = events[act_dst[i] - 1]

    _do_glue()

    # Work with redundant events and dummies (part 3)
    # "Glue" events each of which has only one output which is dummy
    # to their successors
    evt_douts = [[] for e in events]
    evt_nout  = [0 for e in events]
    for k in range(dum):
        if CCPM_FAKE == act_src[k] or CCPM_FAKE == act_dst[k]:
            continue
        #
        j = act_src[k] - 1
        #
        evt_nout[j] += 1
        #
        if CCPM_FAKE != act_ids[k]:
            continue
        #
        evt_douts[j].append(k)

    for i in range(len(events)):
        if evt_nout[i] > 1:
            continue
        if len(evt_douts[i]) < 1:
            continue
        d = evt_douts[i][0]
        events[i] = act_dst[d]
        # Don't need this dummy anymore
        act_src[d] = CCPM_FAKE
        act_dst[d] = CCPM_FAKE


    _do_glue()

    #==========================================================================
    # Renumerate events
    evt = 1
    for i in range(len(events)):
        if events[i] != (i + 1):
            events[i] = CCPM_FAKE
            continue
        events[i] = evt
        evt += 1

    for i in range(dum):
        # Skip redundant dummies
        if CCPM_FAKE == act_src[i] or CCPM_FAKE == act_dst[i]:
            continue
        act_src[i] = events[act_src[i] - 1]
        act_dst[i] = events[act_dst[i] - 1]

    #==========================================================================
    # Remove redundant dummies
    j = 0
    rm = 0
    rm_src = []
    rm_dst = []
    rm_ids = []
    rm_pos = []

    for p in range(dum):
        i = act_pos[p]
        if CCPM_FAKE == act_src[i] or CCPM_FAKE == act_dst[i]:
            rm += 1
            continue
        rm_src.append(act_src[i])
        rm_dst.append(act_dst[i])
        rm_ids.append(act_ids[i])
        rm_pos.append(j)
        j += 1
    dum -= rm

    act_src = rm_src.copy()
    act_dst = rm_dst.copy()
    act_ids = rm_ids.copy()
    act_pos = rm_pos.copy()

    #==========================================================================
    # Add needed dummies
    act_pos = sorted(act_pos[:dum], key=lambda i: act_dst[i])
    act_pos = sorted(act_pos[:dum], key=lambda i: act_src[i])
    d = dum
    for i in range(d):
        if not started[i]:
            continue

        for j in range(i + 1, d):
            if act_dst[i] == act_dst[j] and act_src[i] == act_src[j]:
                started[j] = False

                evt += 1
                act_dst[j] = evt

                act_pos.append(dum)
                act_ids.append(CCPM_FAKE)
                act_src[dum] = evt
                act_dst[dum] = act_dst[i]
                dum += 1

    # Constuct result
    res_pos = sorted(act_pos, key=lambda i: act_ids[i])
    res_src = []
    res_dst = []
    for p in res_pos:
        res_src.append(act_src[p])
        res_dst.append(act_dst[p])

    res_ids = [i for i in act_ids if i != CCPM_FAKE]

    return True, res_ids, res_src, res_dst

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

    status, act_ids, act_src, act_dst = make_aoa(act, src, dst)
