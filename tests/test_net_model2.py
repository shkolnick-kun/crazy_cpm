#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
CrazyCPM - Critical Path Method and PERT analysis library
=========================================================

This module provides comprehensive CPM (Critical Path Method) and PERT
(Program Evaluation and Review Technique) analysis capabilities for project
management.

Features:
    - Multiple activity duration input formats (direct, PERT three-point, PERT two-point)
    - Automatic network diagram generation using Graphviz
    - Statistical analysis with variance propagation
    - Multiple link format support
    - Export to dictionaries and pandas DataFrames

Classes:
    - NetworkModel: Main class for network analysis
    - _Activity: Represents activities in the network (internal)
    - _Event: Represents events/milestones in the network (internal)

Usage Example:
    >>> wbs = {
    ...     1: {'letter': 'A', 'duration': 5.0, 'variance': 1.0},
    ...     2: {'letter': 'B', 'optimistic': 3.0, 'most_likely': 4.0, 'pessimistic': 8.0}
    ... }
    >>> links = [[1], [2]]
    >>> model = NetworkModel(wbs, links=links)
    >>> activities_df, events_df = model.to_dataframe()

.. note::
    This module uses a C extension (_ccpm) for performance-critical computations.
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
    n = max(n_act, len(lnk_src)) + len(lnk_src)

    # Dependenci lists
    full_act_dep = []
    for i in range(n):
        full_act_dep.append([])

    full_dep_map = np.zeros((n, n), dtype=bool)

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
                full_dep_map[i, k] = True
                if i == k:
                    return False #############################################
                full_act_dep[i].append(k)
            j += 1

    #Action positions
    act_pos = sorted(list(range(n_act)), key=lambda i: len(full_act_dep[i]))

    # Compute minimal dependency map
    min_dep_map = full_dep_map.copy()
    for m in range(n_act-1, -1, -1):
        i = act_pos[m]
        for p in range(len(full_act_dep[i])):
            j = full_act_dep[i][p]
            for q in range(len(full_act_dep[i])):
                k = full_act_dep[i][q]
                if k == j:
                    continue
                if full_dep_map[k, j]:
                    min_dep_map[i, j] = False

    # Compute minimal dependency lists
    min_act_dep = []
    for m in range(n):
        min_act_dep.append([])

    for m in range(n_act):
        i = act_pos[m]
        for j in range(n):
            if min_dep_map[i, j]:
                min_act_dep[i].append(j)

    # Reduce work dependencies

    return True

#==============================================================================
if __name__ == '__main__':
    # Example usage with all link formats and new duration input methods
    wbs = {
        # Standard format (backward compatibility)
         1: {'letter': 'A', 'duration': 1., 'name': 'A1'},
         2: {'letter': 'B', 'duration': 1., 'name': 'A2'},
         3: {'letter': 'C', 'duration': 2., 'name': 'A3'},

         4: {'letter': 'D', 'duration': 3., 'name': 'A4'},

         5: {'letter': 'E', 'duration': 1., 'name': 'A5'},
         6: {'letter': 'F', 'duration': 1., 'name': 'A6'},
         7: {'letter': 'G', 'duration': 2., 'name': 'A7'},

         8: {'letter': 'H', 'duration': 3., 'name': 'A8'},
         9: {'letter': 'J', 'duration': 1., 'name': 'A9'},
        10: {'letter': 'K', 'duration': 4., 'name': 'A10'},
    }

    act = list(wbs.keys())
    src = [1, 2, 3,  5, 6, 7,   4, 4, 4 ]
    dst = [5, 6, 7,  8, 9, 10,  8, 9, 10]

    reduce_dependencies(act, src, dst)
