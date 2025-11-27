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



#==============================================================================
if __name__ == '__main__':
    # Example usage with all link formats and new duration input methods
    wbs = {
        # Standard format (backward compatibility)
         1: {'letter': 'A', 'expected': 1., 'name': 'A1'},
         2: {'letter': 'B', 'expected': 1., 'name': 'A2', 'vacation':[3]},
         3: {'letter': 'C', 'expected': 2., 'name': 'A3'},
         4: {'letter': 'D', 'expected': 3., 'name': 'A4'},
         5: {'letter': 'E', 'expected': 1., 'name': 'A5'},
         6: {'letter': 'F', 'expected': 1., 'name': 'A6', 'vacation':[6,7]},
         7: {'letter': 'G', 'expected': 2., 'name': 'A7'},
         8: {'letter': 'H', 'expected': 3., 'name': 'A8'},
         9: {'letter': 'J', 'expected': 1., 'name': 'A9'},
        10: {'letter': 'K', 'expected': 4., 'name': 'A10'},
        11: {'letter': 'L', 'expected': 5., 'name': 'A11'},
        12: {'letter': 'M', 'expected': 5., 'name': 'A12'},
        13: {'letter': 'O', 'expected': 5., 'name': 'A13'},
        14: {'letter': 'P', 'expected': 5., 'name': 'A14'},
    }

    def resource_aware_duration(effort, activity, base_time):
        if None == base_time:
            return effort

        vacation = activity.data.get('vacation', [])
        if not vacation:
            return effort

        vacation.sort()

        if effort > 0.:
            start = base_time
            end   = start + effort
            for d in vacation:
                if d <= end and d >= start:
                    end += 1
            return end - start
        else:
            end   = base_time
            start = end + effort
            for d in vacation:
                if d <= end and d >= start:
                    start -= 1
            return start - end

    src = np.array([1, 1, 1,  2, 3, 4,  5, 5, 5,  6, 7, 8,  9,  9,  9, ])
    dst = np.array([2, 3, 4,  5, 5, 5,  6, 7, 8,  9, 9, 9,  10, 11, 12 ])
    net = NetworkModel(wbs, src, dst, duration=resource_aware_duration)
    a,e = net.to_dataframe()
