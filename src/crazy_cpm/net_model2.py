#!/usr/bin/env python3
# -*- coding: utf-8 -*-

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
import graphviz
import numpy as np
import pandas as pd
import scipy
import os

import _ccpm

# Machine epsilon
EPS = np.finfo(float).eps

# Constants for array indexing in time computations
RES = 0  # Result of time computation
VAR = 1  # Result variance estimation (used for PERT)
ERR = 2  # Computation error upper limit

# Supported file formats for network visualization
_VIZ_FORMATS = ('png', 'svg', 'pdf')

#==============================================================================
def fit_beta(M, D, a, b, err):
    if a > b:
        raise ValueError(f"Invalid bounds: optimistic ({a}) must be <= pessimistic ({b})")
    if not (a <= M <= b):
        raise ValueError(f"Mean ({M}) must be between optimistic ({a}) and pessimistic ({b})")
    if err < 0.0:
        raise ValueError(f"Invalid error ({err}) must be >= 0.0")

    ul = max(abs(a), abs(b))
    if b - a <= 2 * EPS * ul:  # b - a must be big enough
        # This is the chain of deterministic processes, use M without g computation
        return None, None

    if err < 2 * EPS * ul:
        _thr = 2 * EPS * ul
    else:
        _thr = err

    if np.sqrt(D) < _thr:
        # This is the chain of deterministic processes and D is computation error
        return None, None

    mu = (M - a) / (b - a)
    var = D / (b - a) ** 2

    ratio = mu * (1 - mu) / var
    if ratio <= 1:
        ratio = 1 + 2 * EPS

    alpha = mu * (ratio - 1)
    beta = (1 - mu) * (ratio - 1)

    return alpha, beta

#==============================================================================
def calc_ppf(p, M, D, a, b, err=0.0):
    if np.sqrt(D) <= EPS * M:
        return M

    alpha, beta = fit_beta(M, D, a, b, err)
    if not alpha:
        return M

    estimate = scipy.stats.beta.ppf(p, alpha, beta)
    if np.isnan(estimate):
        return M

    return a + (b - a) * estimate

#==============================================================================
def calc_cdf(val, M, D, a, b, err=0.0):
    if np.sqrt(D) <= EPS * M:
        return 1.0 if val > M else 0.0

    alpha, beta = fit_beta(M, D, a, b, err)
    if not alpha:
        return 1.0 if val > M else 0.0

    prob = scipy.stats.beta.cdf((val - a) / (b - a), alpha, beta)
    if np.isnan(prob):
        return 0

    return prob

#==============================================================================
def _p_quantile_estimate(p, tm, optimistic, pessimistic):
    return calc_ppf(p, tm[RES], tm[VAR], optimistic, pessimistic, err=tm[ERR])

#==============================================================================
def _prob_estimate(val, tm, optimistic, pessimistic):
    return calc_cdf(val, tm[RES], tm[VAR], optimistic, pessimistic, err=tm[ERR])

#==============================================================================
def _choice(old, new, delta):
    e = new[ERR] + old[ERR]
    if delta >= e:
        return new  # Certain result
    elif delta >= -e:
        # Uncertain result, use mixing
        ret = np.zeros((3,), dtype=float)
        ret[RES] = 0.5 * (new[RES] + old[RES])
        ret[VAR] = max(old[VAR], new[VAR])
        ret[ERR] = 0.5 * e
        return ret
    else:
        return old  # Certain result

#==============================================================================
def _default_duration(effort, activity, base_time, target):
    # Return effort directly (default: duration = effort)
    # This preserves the sign of effort for forward/backward pass calculations
    return effort

#==============================================================================
def _default_style(element):
    if isinstance(element, Event):
        res = element.reserve
        prob_reserve = element.early_prob(element.late[RES])
        tooltip  = f"Event {element.id}\n"
        tooltip += f"early = {element.early[RES]:.2f}\n"
        tooltip += f"late = {element.late[RES]:.2f}\n"
        tooltip += f"reserve = {element.reserve[RES]:.2f}\n"
    elif isinstance(element, Activity):
        res = element.reserve
        prob_reserve = element.early_end_prob(element.late_end[RES])
        tooltip = element.letter
        if 'name' in element.data:
            tooltip += ' ' + element.data['name']

    else:
        raise TypeError(
            f"element must be Event or Activity, got {type(element)}"
        )

    prob_thr = element.model.p

    if abs(res[RES]) <= res[ERR]:
        # Critical path element
        return {
            'color': '#ff0000',
            'penwidth': '4',
            'fontsize': '16',
            'weight': '3',
            'tooltip': tooltip
        }
    elif prob_reserve < prob_thr:
        # Sub-critical element (may consume reserve)
        return {
            'color': '#ffa000',
            'penwidth': '3',
            'fontsize': '16',
            'weight': '2',
            'tooltip': tooltip
        }
    else:
        # Non-critical element
        return {
            'color': '#000000',
            'penwidth': '2',
            'fontsize': '14',
            'weight': '1',
            'tooltip': tooltip
        }

#==============================================================================
class Activity:
    def __init__(self, id, wbs_id, letter, model, src, dst, expected=0.0,
                 exp_var=0.0, optimistic=0.0, pessimistic=0.0, data=None, is_dummy=False):
        # Validate types and values (explicit checks instead of asserts)
        if not isinstance(id, int):
            raise TypeError(f"id must be int, got {type(id)}")
        if not isinstance(wbs_id, int):
            raise TypeError(f"wbs_id must be int, got {type(wbs_id)}")
        if not isinstance(letter, str):
            raise TypeError(f"letter must be str, got {type(letter)}")
        if not isinstance(model, NetworkModel):
            raise TypeError(f"model must be NetworkModel, got {type(model)}")
        if not isinstance(src, Event):
            raise TypeError(f"src must be Event, got {type(src)}")
        if not isinstance(dst, Event):
            raise TypeError(f"dst must be Event, got {type(dst)}")
        if not isinstance(expected, (int, float)):
            raise TypeError(f"expected must be a real number, got {type(expected)}")
        expected = float(expected)
        if expected < 0.0:
            raise ValueError(f"expected effort must be non-negative, got {expected}")
        if not isinstance(exp_var, (int, float)):
            raise TypeError(f"exp_var must be a real number, got {type(exp_var)}")
        exp_var = float(exp_var)
        if exp_var < 0.0:
            raise ValueError(f"variance must be non-negative, got {exp_var}")
        if data is not None and not isinstance(data, dict):
            raise TypeError(f"data must be dict or None, got {type(data)}")
        if not isinstance(optimistic, (int, float)):
            raise TypeError(f"optimistic must be a real number, got {type(optimistic)}")
        optimistic = float(optimistic)
        if not isinstance(pessimistic, (int, float)):
            raise TypeError(f"pessimistic must be a real number, got {type(pessimistic)}")
        pessimistic = float(pessimistic)

        self.id = id
        self.wbs_id = wbs_id
        self.letter = letter
        self.model = model

        self._src = src
        self._src._out_activities.append(self)

        self._dst = dst
        self._dst._in_activities.append(self)

        self.expected = np.zeros((3,), dtype=float)
        self.expected[RES] = expected
        self.expected[VAR] = exp_var
        self.expected[ERR] = EPS * expected

        self.duration = np.zeros_like(self.expected)

        self.data = data if data is not None else {}

        self._is_dummy = is_dummy

        # CPM/PERT parameters
        self.early_start = np.zeros_like(self.expected)
        self.late_start = np.zeros_like(self.expected)
        self.early_end = np.zeros_like(self.expected)
        self.late_end = np.zeros_like(self.expected)
        self.reserve = np.zeros_like(self.expected)

        self.optimistic = optimistic
        self.opt_start = 0.
        self.opt_end = 0.

        self.pessimistic = pessimistic
        self.pes_start = 0.
        self.pes_end = 0.

    @property
    def src(self):
        return self._src

    @src.setter
    def src(self, new_src):
        if isinstance(self._src, Event):
            self._src._out_activities.remove(self)

        self._src = new_src
        new_src._out_activities.append(self)

    @property
    def dst(self):
        return self._dst

    @dst.setter
    def dst(self, new_dst):
        if isinstance(self._dst, Event):
            self._dst._in_activities.remove(self)

        self._dst = new_dst
        new_dst._in_activities.append(self)

    @property
    def is_dummy(self):
        return self._is_dummy


    def compute_final_duration(self):
        # Early duration calculation
        # Note:
        # early_end[[RES,VAR]] == early_start[[RES,VAR]] + early[[RES,VAR]]
        early = self.early_end - self.early_start
        early[ERR] = self.early_end[ERR] + self.early_start[ERR]

        # Late duration calculation
        # Note:
        # late_start[RES] == late_end[RES] - late[RES]
        # late_start[VAR] == late_end[VAR] + late[VAR]
        late = np.zeros_like(early)
        late[RES] = self.late_end[RES] - self.late_start[RES]
        late[VAR] = self.late_start[VAR] - self.late_end[VAR]
        late[ERR] = self.late_end[ERR] + self.late_start[ERR]

        # Choose most certain estimate
        self.duration = _choice(early, late, late[RES] - early[RES])

    @property
    def early_start_pqe(self):
        return _p_quantile_estimate(self.model.p, self.early_start, self.opt_start, self.pes_start)

    def early_start_prob(self, val):
        return _prob_estimate(val, self.early_start, self.opt_start, self.pes_start)

    @property
    def early_end_pqe(self):
        return _p_quantile_estimate(self.model.p, self.early_end, self.opt_end, self.pes_end)

    def early_end_prob(self, val):
        return _prob_estimate(val, self.early_end, self.opt_end, self.pes_end)

    def __repr__(self):
        return str(self.to_dict())

    def to_dict(self):
        # Start with user data
        ret = self.data.copy()

        ret['id'      ] = self.id
        ret['wbs_id'  ] = self.wbs_id
        ret['letter'  ] = self.letter
        ret['src_id'  ] = self.src.id
        ret['dst_id'  ] = self.dst.id
        ret['expected'] = self.expected[RES] # Resource effort estimate
        ret['duration'] = self.duration[RES] # Actual duration
        ret['is_dummy'] = self._is_dummy

        # CPM timing parameters
        ret['early_start'] = self.early_start[RES]
        ret['late_start' ] = self.late_start[RES]
        ret['early_end'  ] = self.early_end[RES]
        ret['late_end'   ] = self.late_end[RES]
        ret['reserve'    ] = self.reserve[RES]

        if self.model.is_pert:
            # PERT analysis fields
            ret['exp_var'   ] = self.expected[VAR]   # Variance of effort estimate
            ret['variance'  ] = self.duration[VAR]  # Variance of duration
            ret['optimistic'] = self.optimistic   # Optimistic effort
            ret['opt_start' ] = self.opt_start     # Optimistic start time
            ret['opt_end'   ] = self.opt_end         # Optimistic end time

            ret['pessimistic'] = self.pessimistic  # Pessimistic effort
            ret['pes_start'  ] = self.pes_start    # Pessimistic start time
            ret['pes_end'    ] = self.pes_end      # Pessimistic end time

            ret['early_start_var'] = self.early_start[VAR]
            ret['early_end_var'  ] = self.early_end[VAR]
            ret['early_start_pqe'] = self.early_start_pqe
            ret['early_end_pqe'  ] = self.early_end_pqe

            ret['late_end_prob'  ] = self.early_end_prob(self.late_end[RES])

        if self.model.debug:
            # Debug information
            ret['early_start_err'] = self.early_start[ERR]
            ret['late_start_err' ] = self.late_start[ERR]
            ret['early_end_err'  ] = self.early_end[ERR]
            ret['late_end_err'   ] = self.late_end[ERR]

        return ret

#==============================================================================
class Event:
    def __init__(self, id, model):
        if not isinstance(id, int):
            raise TypeError(f"id must be int, got {type(id)}")
        if not isinstance(model, NetworkModel):
            raise TypeError(f"model must be NetworkModel, got {type(model)}")

        self.id = id
        self.model = model
        self._in_activities = []
        self._out_activities = []

        # CPM time parameters (calculated later)
        self.early_pre = np.zeros((3,), dtype=float)
        self.late_pre = np.zeros((3,), dtype=float)
        self.early = np.zeros((3,), dtype=float)
        self.late = np.zeros((3,), dtype=float)
        self.reserve = np.zeros((3,), dtype=float)
        self.stage = 0

        self.optimistic = 0.0
        self.pessimistic = 0.0

        self.data = {} #User data

    @property
    def in_activities(self):
        return tuple(self._in_activities)

    @property
    def out_activities(self):
        return tuple(self._out_activities)

    @property
    def early_pqe(self):
        return _p_quantile_estimate(self.model.p, self.early, self.optimistic, self.pessimistic)

    def early_prob(self, val):
        return _prob_estimate(val, self.early, self.optimistic, self.pessimistic)

    def __repr__(self):
        return str(self.to_dict())

    def to_dict(self):
        # Start with user data
        ret = self.data.copy()

        # Basic CPM parameters
        ret['id'     ] = self.id
        ret['stage'  ] = self.stage
        ret['early'  ] = self.early[RES]
        ret['late'   ] = self.late[RES]
        ret['reserve'] = self.reserve[RES]

        if self.model.is_pert:
            # PERT analysis fields
            ret['optimistic' ] = self.optimistic
            ret['pessimistic'] = self.pessimistic
            ret['early_var'  ] = self.early[VAR]
            ret['early_pqe'  ] = self.early_pqe
            ret['late_prob'  ] = self.early_prob(self.late[RES])

        if self.model.debug:
            # Debug information
            ret['early_err'] = self.early[ERR]
            ret['late_err' ] = self.late[ERR]

            ret['early_pre'     ] = self.early_pre[RES]
            ret['early_pre_var' ] = self.early_pre[VAR]
            ret['early_pre_err' ] = self.early_pre[ERR]

            ret['late_pre'     ] = self.late_pre[RES]
            ret['late_pre_var' ] = self.late_pre[VAR]
            ret['late_pre_err' ] = self.late_pre[ERR]

        return ret

#==============================================================================
def _calculate_action_time_params(work_data, default_risk=0.3):
    if not isinstance(work_data, dict):
        raise TypeError(f"work_data must be dict, got {type(work_data)}")
    if not isinstance(default_risk, (int, float)) or default_risk < 0.0 or default_risk > 1.0:
        raise ValueError(f"default_risk must be a real number between 0 and 1, got {default_risk}")
    default_risk = float(default_risk)

    # 1. Check for three-point PERT estimation (highest priority)
    if all(key in work_data for key in ['optimistic', 'most_likely', 'pessimistic']):
        # Normalize inputs to float
        a = float(work_data['optimistic'])
        m = float(work_data['most_likely'])
        b = float(work_data['pessimistic'])

        # Validate inputs
        if not (a <= m <= b):
            raise ValueError(f"Invalid PERT estimates: must satisfy optimistic <= most_likely <= pessimistic. Got: {a}, {m}, {b}")

        mean = (a + 4 * m + b) / 6
        variance = ((b - a) / 6) ** 2
        return mean, variance, a, m, b

    # 2. Check for two-point PERT estimation (medium priority)
    elif all(key in work_data for key in ['optimistic', 'pessimistic']):
        # Normalize inputs to float
        a = float(work_data['optimistic'])
        b = float(work_data['pessimistic'])

        # Validate inputs
        if not (a <= b):
            raise ValueError(f"Invalid PERT estimates: must satisfy optimistic <= pessimistic. Got: {a}, {b}")

        m = (2 * a + b) / 3
        mean = (3 * a + 2 * b) / 5
        variance = ((b - a) / 5) ** 2

        return mean, variance, a, m, b

    # 3. Direct parameters (lowest priority - backward compatibility)
    elif 'expected' in work_data:
        # Normalize inputs to float
        mean     = float(work_data['expected'])
        variance = float(work_data.get('exp_var', 0.0))

        # Validate inputs
        if mean < 0:
            raise ValueError(f"Expected effort must be non-negative. Got: {mean}")
        if variance < 0:
            raise ValueError(f"Variance must be non-negative. Got: {variance}")

        if variance > 0:
            d = 3 * np.sqrt(variance)
            if d > default_risk * mean:
                a = (1 - default_risk) * mean
            else:
                a = mean - d

            b = a + 2 * d
            m = (mean * 6 - a - b) / 4
        else:
            a = mean
            b = mean
            m = mean

        return mean, variance, a, m, b

    # 4. Error - insufficient data
    else:
        raise ValueError(f"Insufficient data for determining action effort parameters. Available keys: {list(work_data.keys())}")

#==============================================================================
class NetworkModel:
    def __init__(self, wbs_dict, lnk_src=None, lnk_dst=None, links=None,
                 duration=_default_duration, p=0.95, default_risk=0.3,
                 next_act_id=1, debug=False):
        # Validate wbs_dict
        if not isinstance(wbs_dict, dict):
            raise TypeError(f"wbs_dict must be dict, got {type(wbs_dict)}")
        if not wbs_dict:
            raise ValueError("wbs_dict cannot be empty")
        # Validate each key/value
        for act_id, act_data in wbs_dict.items():
            if not isinstance(act_id, int):
                raise TypeError(f"Activity ID must be int, got {type(act_id)} for key {act_id}")
            if not isinstance(act_data, dict):
                raise TypeError(f"Activity data must be dict, got {type(act_data)} for activity {act_id}")
            if 'letter' not in act_data:
                raise ValueError(f"Activity {act_id} missing required 'letter' field")
            if not isinstance(act_data['letter'], str):
                raise TypeError(f"Activity {act_id} 'letter' must be str, got {type(act_data['letter'])}")

        if not callable(duration):
            raise TypeError(f"duration must be callable, got {type(duration)}")

        if not isinstance(p, (int, float)) or not (0.0 < p < 1.0):
            raise ValueError(f"p must be a real number between 0 and 1, got {p}")
        p = float(p)
        if not isinstance(default_risk, (int, float)) or default_risk < 0.0 or default_risk > 1.0:
            raise ValueError(f"default_risk must be a real number between 0 and 1, got {default_risk}")
        default_risk = float(default_risk)
        if not isinstance(next_act_id, int) or next_act_id < 1:
            raise ValueError(f"next_act_id must be positive int, got {next_act_id}")
        if not isinstance(debug, bool):
            raise TypeError(f"debug must be bool, got {type(debug)}")

        self.debug = debug
        self._is_pert = False
        self._p = p
        self._duration = duration  # Resource-aware duration callback

        # Parse links into standard format
        lnk_src, lnk_dst = self._parse_links(lnk_src, lnk_dst, links)

        # Create network model
        self._create_model(wbs_dict, lnk_src, lnk_dst, default_risk, next_act_id)

        # After _create_model, events must be non-empty
        if len(self._events) == 0:
            raise RuntimeError("Network construction resulted in no events. Check input data.")

        # Compute stages of project
        self._compute_target('stage')

        # Renumerate events according to the rules of network modeling
        self._events.sort(key=lambda e: e.stage)
        for i, e in enumerate(self._events, 1):
            e.id = i

        # Compute Event and Activity time parameters
        self._compute_time_params()

    @property
    def is_pert(self):
        return self._is_pert

    @property
    def p(self):
        return self._p

    @property
    def activities(self):
        return self._activities

    @property
    def events(self):
        return self._events

    def _parse_links(self, lnk_src, lnk_dst, links):
        if (lnk_src is not None or lnk_dst is not None) and links is not None:
            raise ValueError(
                "Provide either (lnk_src, lnk_dst) or links, not both"
            )

        # Case 1: Old format (lnk_src and lnk_dst provided)
        if lnk_src is not None and lnk_dst is not None:
            # They must be iterables
            try:
                src_list = list(lnk_src)
                dst_list = list(lnk_dst)
            except TypeError as e:
                raise TypeError(f"lnk_src and lnk_dst must be iterable, got {type(lnk_src)} and {type(lnk_dst)}") from e
            if len(src_list) != len(dst_list):
                raise ValueError(f"lnk_src and lnk_dst must have same length, got {len(src_list)} and {len(dst_list)}")
            return src_list, dst_list

        # Case 2: New formats via links parameter
        if links is None:
            raise ValueError("Either (lnk_src, lnk_dst) or links must be provided")

        # Format 1: Two rows [[src...], [dst...]]
        if (isinstance(links, (list, tuple)) and len(links) == 2 and
                isinstance(links[0], (list, tuple, np.ndarray)) and
                isinstance(links[1], (list, tuple, np.ndarray))):
            try:
                src_list = list(links[0])
                dst_list = list(links[1])
            except TypeError as e:
                raise TypeError("Could not convert links rows to lists") from e
            if len(src_list) != len(dst_list):
                raise ValueError(f"Two-row links: source and destination rows must have same length, got {len(src_list)} and {len(dst_list)}")
            return src_list, dst_list

        # Format 2: Two columns [[src, dst], [src, dst], ...]
        elif isinstance(links, (list, tuple, np.ndarray)) and len(links) > 0:
            if not all(isinstance(item, (list, tuple, np.ndarray)) and len(item) == 2
                       for item in links):
                raise ValueError(
                    f"Unrecognized links format: expected two rows "
                    f"[[src, ...], [dst, ...]] or a list of [src, dst] pairs, "
                    f"got {type(links).__name__} of length {len(links)}"
                )
            src_list = [item[0] for item in links]
            dst_list = [item[1] for item in links]
            return src_list, dst_list

        # Format 3: Dictionary {'src': [...], 'dst': [...]}
        elif isinstance(links, dict):
            if 'src' in links and 'dst' in links:
                try:
                    src_list = list(links['src'])
                    dst_list = list(links['dst'])
                except TypeError as e:
                    raise TypeError("Dictionary links 'src' and 'dst' must be iterable") from e
                if len(src_list) != len(dst_list):
                    raise ValueError(f"Dictionary links: 'src' and 'dst' must have same length, got {len(src_list)} and {len(dst_list)}")
                return src_list, dst_list
            else:
                raise ValueError("Dictionary links must contain 'src' and 'dst' keys")

        else:
            raise ValueError(f"Unsupported links format: {type(links)}")

    def _create_model(self, wbs_dict, lnk_src, lnk_dst, default_risk, next_act_id):
        # Validate link lengths
        if len(lnk_src) != len(lnk_dst):
            raise ValueError(f"lnk_src and lnk_dst must have same length, got {len(lnk_src)} and {len(lnk_dst)}")

        act_ids = list(wbs_dict.keys())

        # Generate network graph using C extension
        act_ids, net_src, net_dst = _ccpm.make_aoa(act_ids, lnk_src, lnk_dst)

        self._events = []
        self.next_act = next_act_id
        self._activities = []

        # Create events
        if len(net_dst) == 0:
            raise RuntimeError("No destination events generated. Check input links.")
        max_event = max(net_dst)
        for i in range(max_event):
            self._add_event(int(i + 1))

        # Create activities (real and dummy)
        na = len(act_ids)  # Number of actions
        nd = 0  # Number of dummy actions
        dsrc = set()  # Dummy event sources
        for i in range(len(net_src)):
            if i < na:
                # Real activity - get data from WBS
                act_id = act_ids[i]
                wbs_data = wbs_dict.get(act_id)
                if wbs_data is None:
                    raise ValueError(f"Activity ID {act_id} not found in wbs_dict")

                # Calculate expected effort and variance using unified function
                try:
                    expected, exp_var, optimistic, _, pessimistic = \
                        _calculate_action_time_params(wbs_data, default_risk)
                except ValueError as e:
                    raise ValueError(f"Error processing activity {act_id} ({wbs_data.get('letter', 'unknown')}): {e}") from e

                letter = wbs_data.get('letter', '')

                # Even one wbs item with nonzero variance is enough to compute PERT
                if exp_var > 0.0:
                    self._is_pert = True

                # Create data dict without fields stored as separate attributes
                data_without_duplicates = self._remove_duplicate_fields(wbs_data)

                self._add_activity(int(act_id), int(net_src[i]), int(net_dst[i]),
                                   expected, exp_var, optimistic, pessimistic,
                                   letter, data_without_duplicates)
            else:
                # Add a dummy activity (no effort, no letter, no data)
                nd += 1  # One more dummy work
                self._add_activity(0, int(net_src[i]), int(net_dst[i]),
                                   0., 0., 0., 0., '#' + str(nd), {}, is_dummy=True)
                dsrc.add(int(net_src[i]))

        # Network postprocessing
        # Make sure that activities with largest efforts are on straight paths between events
        grpoi = {}
        # Find groups of triangles on dummies
        for e in self._events:
            # Watch only dummy sources
            if e.id not in dsrc:
                continue

            bck = e._in_activities
            fwd = e._out_activities
            # Watch only events with one incoming and one outgoing action
            if 1 < len(bck) or 1 < len(fwd):
                continue

            key = (bck[0].src.id, fwd[0].dst.id)
            if key not in grpoi.keys():
                for a in bck[0].src._out_activities:
                    if a.dst.id == fwd[0].dst.id:
                        grpoi[key] = (a, [bck[0]])
                        break
            else:
                grpoi[key][1].append(bck[0])

        # Place maximum duration activities on long side of triangle groups
        for k in grpoi.keys():
            aoi = grpoi[k][1]
            # Maximum duration candidate
            maxa = grpoi[k][0]
            for a in aoi:
                # target is None because this is post-processing without scenario
                if self._duration(a.expected[RES], a, None, None) > self._duration(maxa.expected[RES], maxa, None, None):
                    a.dst, maxa.dst = maxa.dst, a.dst
                    maxa = a

    def _remove_duplicate_fields(self, wbs_data):
        # Create a copy to avoid modifying the original data
        data_copy = wbs_data.copy()

        # Remove fields that are stored as separate attributes
        fields_to_remove = ['expected', 'exp_var', 'letter',
                            'optimistic', 'most_likely', 'pessimistic']
        for field in fields_to_remove:
            if field in data_copy:
                del data_copy[field]

        return data_copy

    def _compute_time_params(self):
        if self.is_pert:
            # For PERT models we must compute optimistic and pessimistic scenarios
            self._compute_target('optimistic')
            self._compute_target('pessimistic')

        self._compute_target('early_pre')

        # Set late times starting from project completion
        late_pre = np.zeros((3,), dtype=float)
        for e in self._events:
            if e.early_pre[RES] > late_pre[RES]:
                late_pre = e.early_pre.copy()

        for e in self._events:
            e.late_pre = late_pre.copy()

        self._compute_target('late_pre')

        for a in self._activities:
            a.compute_final_duration()

        self._compute_target('early')

        # Set late times starting from project completion
        late = np.zeros((3,), dtype=float)
        for e in self._events:
            if e.early[RES] > late[RES]:
                late = e.early.copy()

        for e in self._events:
            e.late = late.copy()

        self._compute_target('late')

        # Compute reserves
        for e in self._events:
            e.reserve[VAR] = e.late[VAR] + e.early[VAR]
            e.reserve[ERR] = e.late[ERR] + e.early[ERR]
            # Round off insignificant values
            r = e.late[RES] - e.early[RES]
            e.reserve[RES] = r if abs(r) > e.reserve[ERR] else 0.0

        for a in self._activities:
            # Compute start and end reserve values separately
            # These values may differ due to resource availability time dependence
            start_res = np.zeros((3,), dtype=float)
            start_res[RES] = a.late_start[RES] - a.early_start[RES]
            start_res[VAR] = a.late_start[VAR] + a.early_start[VAR]
            start_res[ERR] = a.late_start[ERR] + a.early_start[ERR]

            end_res = np.zeros((3,), dtype=float)
            end_res[RES] = a.late_end[RES] - a.early_end[RES]
            end_res[VAR] = a.late_end[VAR] + a.early_end[VAR]
            end_res[ERR] = a.late_end[ERR] + a.early_end[ERR]

            # Choose minimum reserve value
            a.reserve = _choice(start_res, end_res, start_res[RES] - end_res[RES])

            # Round off insignificant values
            r = a.reserve[RES]
            a.reserve[RES] = r if abs(r) > a.reserve[ERR] else 0.0

            # Check for programming errors
            if r < -a.reserve[ERR]:
                raise RuntimeError(f"Activity {a.id} has negative time reserve ({r})")



    def _compute_target(self, target=None):
        def _choice_early(old, new):
            return _choice(old, new, new[RES] - old[RES])

        def _choice_late(old, new):
            return _choice(old, new, old[RES] - new[RES])

        def _delta_late(a):
            ret = a.expected.copy()
            ret[RES] = -a.expected[RES]
            return ret

        def _duration_late(effort, activity, base_time, target):
            ret = activity.duration.copy()
            ret[RES] = -activity.duration[RES]
            return ret

        def _duration_vec(effort, activity, base_time, target):
            # Optimize for default duration function
            if _default_duration == self._duration:
                return effort

            dur = np.zeros((3,), dtype=float)

            # Compute duration value and error bound
            # effort[RES] is float: positive for forward pass, negative for backward pass
            dur[RES] = self._duration(effort[RES], activity, base_time[RES], target)
            dur[ERR] = EPS * abs(dur[RES])  # Error bound based on absolute duration

            if 0. == effort[VAR] or not self.is_pert:
                # Deterministic or fake activity
                # VAR is zero already
                return dur

            # Compute shape parameter for modified PERT distribution
            alpha, beta = fit_beta(activity.expected[RES], activity.expected[VAR],
                                   activity.optimistic, activity.pessimistic,
                                   activity.expected[ERR])

            if not alpha:
                # Deterministic activity
                return dur

            # Model is PERT and activity is not deterministic,
            # will compute duration variance using modified PERT formula

            # Compute optimistic and pessimistic duration estimates
            if effort[RES] >= 0.:
                # Forward pass: use positive effort values
                a = self._duration(activity.optimistic , activity, base_time[RES], target)
                b = self._duration(activity.pessimistic, activity, base_time[RES], target)
            else:
                # Backward pass: use negative effort values
                a = self._duration(-activity.optimistic , activity, base_time[RES], target)
                b = self._duration(-activity.pessimistic, activity, base_time[RES], target)

            # Use beta-distribution formula for variance calculation:
            var_beta = alpha * beta / (alpha + beta + 1) / ((alpha + beta) ** 2)
            dur[VAR] = var_beta * ((b - a) ** 2)

            return dur

        def _check_and_fix_evt(e, target):
            if 'early' == target:
                if e.early[RES] < 0.0:
                    e.early[RES] = 0.0

            elif 'late' == target:
                if e.early[RES] - e.late[RES] >  e.late[ERR] + e.early[ERR]:
                    raise RuntimeError(f"Event {e.id} has abnormal late time ({e.late[RES]} < {e.early[RES]}) ")

                if e.late[RES] < e.early[RES]:
                    e.late[RES] = e.early[RES]

            return

        def _check_and_fix_act(a, target):
            # Time params lower limit

            if 'early' == target:
                if a.early_start[RES] < 0.0:
                    a.early_start[RES] = 0.0

                if a.early_start[RES] - a.early_end[RES] >  a.early_start[ERR] + a.early_end[ERR]:
                    raise RuntimeError(f"Activity {a.id} has abnormal early end time ({a.early_end[RES]} < {a.early_start[RES]}) ")

                if a.early_end[RES] < a.early_start[RES]:
                    a.early_end[RES] = a.early_start[RES]

            elif 'late' == target:
                if a.early_start[RES] - a.late_start[RES] >  a.early_start[ERR] + a.late_start[ERR]:
                    raise RuntimeError(f"Activity {a.id} has abnormal late start time ({a.late_start[RES]} < {a.early_start[RES]}) ")

                if a.late_start[RES] < a.early_start[RES]:
                    a.late_start[RES] = a.early_start[RES]

                if a.late_start[RES] - a.late_end[RES] >  a.late_start[ERR] + a.late_end[ERR]:
                    raise RuntimeError(f"Activity {a.id} has abnormal late end time ({a.late_end[RES]} < {a.late_start[RES]}) ")

                if a.late_end[RES] < a.late_start[RES]:
                    a.late_end[RES] = a.late_start[RES]

            return

        if 'stage' == target:
            act_base = None
            act_new = None
            act_next = 'dst'
            fwd = '_out_activities'
            rev = '_in_activities'
            choice = max
            delta = lambda a: 1
            process_delta = lambda d, a, b, t: d

        elif 'early_pre' == target:
            act_base = 'early_start'
            act_new = 'early_end'
            act_next = 'dst'
            fwd = '_out_activities'
            rev = '_in_activities'
            choice = _choice_early
            delta = lambda a: a.expected
            process_delta = lambda d, a, b, t: _duration_vec(d, a, b, t)

        elif 'late_pre' == target:
            act_base = 'late_end'
            act_new = 'late_start'
            act_next = 'src'
            fwd = '_in_activities'
            rev = '_out_activities'
            choice = _choice_late
            delta = _delta_late
            process_delta = lambda d, a, b, t: _duration_vec(d, a, b, t)

        elif 'early' == target:
            act_base = 'early_start'
            act_new = 'early_end'
            act_next = 'dst'
            fwd = '_out_activities'
            rev = '_in_activities'
            choice = _choice_early
            delta = lambda a: a.expected
            process_delta = lambda d, a, b, t: a.duration

        elif 'late' == target:
            act_base = 'late_end'
            act_new = 'late_start'
            act_next = 'src'
            fwd = '_in_activities'
            rev = '_out_activities'
            choice = _choice_late
            delta = _delta_late
            process_delta = _duration_late

        elif 'optimistic' == target:
            act_base = 'opt_start'
            act_new = 'opt_end'
            act_next = 'dst'
            fwd = '_out_activities'
            rev = '_in_activities'
            choice = max
            delta = lambda a: a.optimistic
            process_delta = lambda d, a, b, t: self._duration(d, a, b, t)

        elif 'pessimistic' == target:
            act_base = 'pes_start'
            act_new = 'pes_end'
            act_next = 'dst'
            fwd = '_out_activities'
            rev = '_in_activities'
            choice = max
            delta = lambda a: a.pessimistic
            process_delta = lambda d, a, b, t: self._duration(d, a, b, t)
        else:
            raise ValueError(f"Unknown 'target' value: {target}")

        # Initialize activity parameters and processing function
        if target != 'stage':
            for a in self._activities:
                setattr(a, act_base, -1)
                setattr(a, act_new, -1)

        # Count dependencies for topological sorting
        n_dep = {e:len(getattr(e, rev)) for e in self._events}

        # Find starting events (no dependencies)
        evt = [e for e, n in n_dep.items() if 0 == n]
        # Check for programming errors
        if 1 != len(evt):
            raise RuntimeError(f"The project must have exactly one starting event, but found {len(evt)}")

        # Process events in topological order
        i = 0
        while True:

            e = evt[i]

            base_val = getattr(e, target)

            for a in getattr(e, fwd):
                if act_base:
                    setattr(a, act_base, base_val)

                # Calculate new value using appropriate delta processing
                new_val = base_val + process_delta(delta(a), a, base_val, target)

                if act_new:
                    setattr(a, act_new, new_val)

                if act_base or act_new:
                    _check_and_fix_act(a, target)

                next_evt = getattr(a, act_next)

                setattr(next_evt, target, choice(getattr(next_evt, target), new_val))

                n_dep[next_evt] -= 1

                if 0 >= n_dep[next_evt]:
                    evt.append(next_evt)

            _check_and_fix_evt(e, target)

            i += 1
            if i >= len(evt):
                break

    def _add_event(self, i):
        self._events.append(Event(i, self))

    def _add_activity(self, wbs_id, src_id, dst_id, expected, exp_var,
                      optimistic, pessimistic, letter, data, is_dummy=False):
        if not isinstance(wbs_id, int):
            raise TypeError(f"wbs_id must be int, got {type(wbs_id)}")
        if not isinstance(src_id, int):
            raise TypeError(f"src_id must be int, got {type(src_id)}")
        if not isinstance(dst_id, int):
            raise TypeError(f"dst_id must be int, got {type(dst_id)}")
        if not isinstance(expected, (int, float)):
            raise TypeError(f"expected must be a real number, got {type(expected)}")
        expected = float(expected)
        if expected < 0.0:
            raise ValueError(f"expected must be non-negative, got {expected}")
        if not isinstance(exp_var, (int, float)):
            raise TypeError(f"exp_var must be a real number, got {type(exp_var)}")
        exp_var = float(exp_var)
        if exp_var < 0.0:
            raise ValueError(f"exp_var must be non-negative, got {exp_var}")
        if not isinstance(optimistic, (int, float)):
            raise TypeError(f"optimistic must be a real number, got {type(optimistic)}")
        optimistic = float(optimistic)
        if not isinstance(pessimistic, (int, float)):
            raise TypeError(f"pessimistic must be a real number, got {type(pessimistic)}")
        pessimistic = float(pessimistic)
        if not isinstance(letter, str):
            raise TypeError(f"letter must be str, got {type(letter)}")
        if not isinstance(data, dict):
            raise TypeError(f"data must be dict, got {type(data)}")

        act = Activity(self.next_act, wbs_id, letter, self,
                        self._events[src_id - 1], self._events[dst_id - 1],
                        expected, exp_var, optimistic, pessimistic, data, is_dummy=is_dummy)
        self._activities.append(act)
        self.next_act += 1

    def __repr__(self):
        _repr = 'Events:{\n'
        for e in self._events:
            _repr += '        ' + str(e) + '\n'
        _repr += '}\n'

        _repr += 'Activities:\n'
        for a in self._activities:
            _repr += '        ' + str(a) + '\n'
        _repr += '}\n'

        return _repr

    def to_dict(self):
        activities_data = [activity.to_dict() for activity in self._activities]
        events_data = [event.to_dict() for event in self._events]

        return {
            'activities': activities_data,
            'events': events_data
        }

    def to_dataframe(self):
        # Convert to dictionaries first
        model_dict = self.to_dict()

        # Create events DataFrame (straightforward)
        events_df = pd.DataFrame(model_dict['events'], dtype=object)

        activities_df = pd.DataFrame(model_dict['activities'], dtype=object)
        try:
            with pd.option_context('future.no_silent_downcasting', True):
                activities_df = activities_df.replace({np.nan: ''})
        except pd.errors.OptionError:
            activities_df = activities_df.replace({np.nan: ''})

        return activities_df, events_df

    def viz(self, output_path=None, group_by_stage=True, get_style=_default_style):
        if not callable(get_style):
            raise TypeError(f"Parameter get_style must be callable, got {type(get_style)}")

        dot = graphviz.Digraph(node_attr={'shape': 'record', 'style': 'rounded'})
        dot.graph_attr['rankdir'] = 'LR'
        dot.graph_attr['splines'] = 'polyline'

        def _label_event(e):
            style = get_style(e)
            # Simple record label without HTML tags
            return '{{%d |{%.1f|%.1f}| %.2f}}' % (
                e.id, e.early[RES], e.late[RES], e.reserve[RES]
            ), style

        # 1. Add event nodes (optionally grouped by stage)
        if group_by_stage:
            stages = {}
            for e in self._events:
                stages.setdefault(e.stage, []).append(e)

            for stage in sorted(stages.keys()):
                with dot.subgraph() as s:
                    s.attr(rank='same')
                    s.attr(id=f'stage_{stage}')
                    for e in stages[stage]:
                        label, style = _label_event(e)
                        s.node(str(e.id), label,
                               tooltip=style['tooltip'],
                               color=style['color'],
                               penwidth=style['penwidth'],
                               fontsize=style['fontsize'])
        else:
            for e in self._events:
                label, style = _label_event(e)
                dot.node(str(e.id), label,
                         color=style['color'],
                         tooltip=style['tooltip'],
                         penwidth=style['penwidth'],
                         fontsize=style['fontsize'])

        # 2. Add edges with embedded labels
        for a in self._activities:
            # Get style for this activity based on its criticality
            activity_style = get_style(a)

            # Build label text for the label node
            if a.wbs_id:   # real activity
                lbl = f"{a.letter}\\nt={a.duration[RES]:.1f}\\nr={a.reserve[RES]:.2f}"
            else:          # dummy activity
                lbl = f"{a.letter}\\nr={a.reserve[RES]:.2f}"

            # Unique identifier for the label node
            label_node_id = f"label_{a.id}"

            # Determine edge style (solid for real, dashed for dummy)
            edge_style = 'dashed' if a.is_dummy else 'solid'

            # Create visible label node (light gray, rounded box)
            # Label uses fontsize=12 for all activities as specified
            dot.node(label_node_id,
                     label=lbl,
                     tooltip=activity_style['tooltip'],
                     shape='box',
                     style='filled,rounded',
                     color=activity_style['color'],
                     penwidth=activity_style['penwidth'],
                     fillcolor='#f0f0f0',
                     fontsize='12',
                     margin='0.05,0.05')

            # Main edge: source -> label node (no arrowhead)
            dot.edge(str(a.src.id), label_node_id,
                     tooltip=activity_style['tooltip'],
                     style=edge_style,
                     color=activity_style['color'],
                     penwidth=activity_style['penwidth'],
                     weight=activity_style['weight'],
                     arrowhead='none')

            # Main edge: label node -> destination (with arrowhead)
            dot.edge(label_node_id, str(a.dst.id),
                     tooltip=activity_style['tooltip'],
                     style=edge_style,
                     color=activity_style['color'],
                     penwidth=activity_style['penwidth'],
                     weight=activity_style['weight'],
                     arrowhead='normal')

        # 3. Save to file if output path provided
        if output_path is not None:
            base, ext = os.path.splitext(output_path)
            fmt = ext.lstrip('.').lower() if ext else 'png'
            if fmt not in _VIZ_FORMATS:
                raise ValueError(
                    f"Unsupported visualization format: {fmt!r}. "
                    f"Supported formats: {_VIZ_FORMATS}"
                )
            dot.render(base, format=fmt, cleanup=True)

        return dot

#==============================================================================
if __name__ == '__main__':
    import math

    """
    Demonstration module for CrazyCPM library.

    This section provides comprehensive examples of library usage including:
    - Multiple link format demonstrations
    - Resource-aware scheduling examples
    - PERT analysis with statistical estimates
    - Visualization and export capabilities
    """

    # Example usage with all link formats and new time input methods
    wbs = {
        # Standard format (backward compatibility)
        1: {'letter': 'A', 'expected': 3.84, 'exp_var': 0.00, 'name': 'Heating and frames study'},

        # Three-point PERT format
        2: {'letter': 'B', 'optimistic': 1.5, 'most_likely': 2.0, 'pessimistic': 3.0,
            'name': 'Scouring and installation of building site establishment'},

        # Two-point PERT format
        3: {'letter': 'C', 'optimistic': 3.0, 'pessimistic': 5.0, 'name': 'Earthwork and concrete well'},

        # Standard format with zero exp_var
        4: {'letter': 'D', 'expected': 4., 'name': 'Earthwork and concrete longitudinal beams'},

        # Three-point PERT
        5: {'letter': 'E', 'optimistic': 5.0, 'most_likely': 6.0, 'pessimistic': 8.0, 'name': 'Frame construction'},

        # Standard format
        6: {'letter': 'F', 'expected': 6., 'exp_var': 0.01, 'name': 'Frame transport'},
        7: {'letter': 'G', 'expected': 6., 'exp_var': 0.01, 'name': 'Assemblage'},

        # Two-point PERT
        8: {'letter': 'H', 'optimistic': 1.5, 'pessimistic': 3.0, 'name': 'Earthwork and pose drains'},

        # Three-point PERT
        9: {'letter': 'I', 'optimistic': 4.0, 'most_likely': 5.0, 'pessimistic': 7.0,
            'name': 'Heating provisioning and assembly'},

        # Standard format
        10: {'letter': 'J', 'expected': 5., 'exp_var': 0.01, 'name': 'Electric installation'},
        11: {'letter': 'K', 'expected': 2., 'exp_var': 0.01, 'name': 'Painting'},
        12: {'letter': 'L', 'expected': 1., 'exp_var': 0.01, 'name': 'Pavement'}
    }

    print("=== Demonstration of all link formats and new time input methods ===")

    # Old format
    print("\n1. Old format with new time inputs:")
    src_old = np.array([1,2,3, 2,3, 3,4, 1,6,7, 5,6,7,  3, 6, 7,  6, 8, 9,  7, 8, 9,10])
    dst_old = np.array([5,5,5, 6,6, 7,7 ,8,8,8, 9,9,9, 10,10,10, 11,11,11, 12,12,12,12])
    n_old = NetworkModel(wbs, src_old, dst_old)
    print("Successfully created model with mixed time input formats")

    # Test the new time calculation function
    print("\n=== Testing time calculation function ===")
    test_cases = [
        {'optimistic': 1, 'most_likely': 2, 'pessimistic': 3},
        {'optimistic': 3, 'pessimistic': 5},
        {'expected': 4.0, 'exp_var': 0.5},
        {'expected': 2.0}  # exp_var defaults to 0
    ]

    for i, test_case in enumerate(test_cases):
        try:
            mean, var, a, m, b = _calculate_action_time_params(test_case)
            print(f"Test case {i + 1}: {test_case} -> mean={mean:.3f}, variance={var:.3f}")
        except ValueError as e:
            print(f"Test case {i + 1} error: {e}")

    # Demonstration of the new data formats in activities
    print("\n=== Demonstration of new time formats in activities ===")
    for i, activity in enumerate(n_old._activities[:8]):  # Show first 8 activities
        if activity.wbs_id != 0:  # Skip dummy activities
            print(f"Activity {i + 1}: wbs_id={activity.wbs_id}, letter='{activity.letter}'")
            print(f"  Expected effort: {activity.expected[RES]:.3f}, Variance: {activity.expected[VAR]:.3f}")
            print(f"  Actual duration: {activity.duration[RES]:.3f}")
            if activity.data:
                print(f"  Data fields: {list(activity.data.keys())}")

    # Advanced example: Resource-aware scheduling
    print("\n=== Advanced Example: Resource-Aware Scheduling ===")

    def resource_aware_duration(effort, activity, base_time, target):
        team = activity.data.get('team_size', 1)
        prod = activity.data.get('productivity', 1.0)

        if base_time is None or effort == 0 or team * prod <= 0:
            return effort

        daily_cap = team * prod
        remaining = abs(effort)

        def day_avail(t):
            return 0.5 if int(t) % 7 >= 5 else 1.0

        day = int(math.floor(base_time))

        if effort > 0.0:
            step = 1
            time_left = 1.0 - (base_time - day)
        else:
            step = -1
            time_left = base_time - day

        dur = 0.0
        while True:
            cap = daily_cap * day_avail(day)
            avail = cap * time_left
            if remaining <= avail:
                return (dur + remaining / cap) * float(step)
            dur += time_left
            time_left = 1.0
            remaining -= avail

            day += step

    # WBS with resource information
    wbs_resource = {
        1: {'letter': 'A', 'optimistic': 40., 'pessimistic': 60., 'team_size': 2, 'productivity': 0.8, 'name': 'Design'},
        2: {'letter': 'B', 'expected': 60., 'team_size': 3, 'productivity': 0.9, 'name': 'Development'},
        3: {'letter': 'C', 'expected': 20., 'team_size': 1, 'productivity': 1.0, 'name': 'Testing'}
    }

    links_resource = [[1, 2], [2, 3]]

    print("Creating resource-aware model...")
    model_resource = NetworkModel(wbs_resource, links=links_resource, duration=resource_aware_duration)

    # Show resource-aware durations
    for activity in model_resource._activities:
        if activity.wbs_id != 0:
            print(f"Activity {activity.letter}: "
                  f"Effort={activity.expected[RES]:.1f}h, "
                  f"Duration={activity.duration[RES]:.1f} days, "
                  f"Team={activity.data.get('team_size', 1)} people, "
                  f"Productivity={activity.data.get('productivity', 1.0):.1f}")

    print("Full dependency map for old format:")
    act_id = np.array(list(wbs.keys()))
    status, full_dep_map = _ccpm.make_full_map(act_id, src_old, dst_old)
    print(status)
    print(act_id)
    print(full_dep_map)

    src_old[0] = 12
    act_id = np.array(list(wbs.keys()))
    status, full_dep_map = _ccpm.make_full_map(act_id, src_old, dst_old)
    print(status)
    print(act_id)
    print(full_dep_map)

    # New format 1 (two rows)
    print("\n2. New format 1 (two rows):")
    links_format1 = [
        [1, 2, 3, 2, 3, 3, 4, 1, 6, 7, 5, 6, 7, 3, 6, 7, 6, 8, 9, 7, 8, 9, 10],
        [5, 5, 5, 6, 6, 7, 7, 8, 8, 8, 9, 9, 9, 10, 10, 10, 11, 11, 11, 12, 12, 12, 12]
    ]
    n_new1 = NetworkModel(wbs, links=links_format1)
    print("Successfully created model with format 1")

    # New format 2 (two columns)
    print("\n3. New format 2 (two columns):")
    links_format2 = [
        [1, 5], [2, 5], [3, 5], [2, 6], [3, 6], [3, 7], [4, 7],
        [1, 8], [6, 8], [7, 8], [5, 9], [6, 9], [7, 9], [3, 10],
        [6, 10], [7, 10], [6, 11], [8, 11], [9, 11], [7, 12],
        [8, 12], [9, 12], [10, 12]
    ]
    n_new2 = NetworkModel(wbs, links=links_format2)
    print("Successfully created model with format 2")

    # New format 3 (dictionary)
    print("\n4. New format 3 (dictionary):")
    links_format3 = {
        'src': [1, 2, 3, 2, 3, 3, 4, 1, 6, 7, 5, 6, 7, 3, 6, 7, 6, 8, 9, 7, 8, 9, 10],
        'dst': [5, 5, 5, 6, 6, 7, 7, 8, 8, 8, 9, 9, 9, 10, 10, 10, 11, 11, 11, 12, 12, 12, 12]
    }
    n_new3 = NetworkModel(wbs, links=links_format3)
    print("Successfully created model with format 3")

    # Check model equivalence
    print("\n=== Checking model equivalence ===")
    print(f"Old == New1: {len(n_old._activities) == len(n_new1._activities)}")
    print(f"Old == New2: {len(n_old._activities) == len(n_new2._activities)}")
    print(f"Old == New3: {len(n_old._activities) == len(n_new3._activities)}")

    print("\n=== Demonstration of dictionary export ===")
    model_dict = n_old.to_dict()

    print(f"Number of activities: {len(model_dict['activities'])}")
    print(f"Number of events: {len(model_dict['events'])}")

    # Show data structure
    print("\nActivity data structure:")
    if len(model_dict['activities']) > 0:
        first_activity = model_dict['activities'][0]
        print(f"Keys: {list(first_activity.keys())}")

    print("\nEvent data structure:")
    if len(model_dict['events']) > 0:
        first_event = model_dict['events'][0]
        print(f"Keys: {list(first_event.keys())}")

    # Demonstration of DataFrame export (if pandas is available)
    print("\n=== Demonstration of DataFrame export ===")
    try:
        n_old.debug = True
        activities_df, events_df = n_old.to_dataframe()
        n_old.debug = False

        print("\nActivities DataFrame:")
        print(f"Size: {activities_df.shape}")
        print(f"Columns: {list(activities_df.columns)}")
        print("\nFirst 5 rows:")
        print(activities_df.head())

        print("\nEvents DataFrame:")
        print(f"Size: {events_df.shape}")
        print(f"Columns: {list(events_df.columns)}")
        print("\nFirst 5 rows:")
        print(events_df.head())

        # Show that data from 'data' field is now in separate columns
        print("\nChecking data expansion:")
        if 'name' in activities_df.columns:
            print("Data from 'data' successfully expanded into separate columns:")
            print(activities_df[['letter', 'name', 'expected', 'duration', 'reserve']].head())

    except Exception as e:
        print(f"Error exporting to DataFrame: {e}")

    # Create visualization in specific directory
    print("\n=== Creating visualization ===")

    # Get the directory of the current module
    module_dir = os.path.dirname(os.path.abspath(__file__))

    # Construct the target directory path: <module directory>/../../tests/data
    target_dir = os.path.normpath(os.path.join(module_dir, '../../tests/data'))

    # Create the directory if it doesn't exist
    os.makedirs(target_dir, exist_ok=True)

    # Base file path (without extension)
    file_path = os.path.join(target_dir, 'cpm_network')

    # Create and save the visualization in all supported formats
    for ext in _VIZ_FORMATS:
        viz_path = f'{file_path}.{ext}'
        n_old.viz(output_path=viz_path)
        print(f"Visualization saved as '{viz_path}'")
    print(f"Target directory: {target_dir}")
