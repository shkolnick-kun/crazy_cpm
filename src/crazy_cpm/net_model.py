#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
CrazyCPM - Critical Path Method and PERT analysis library
=========================================================

This module provides comprehensive CPM (Critical Path Method) and PERT
(Program Evaluation and Review Technique) analysis capabilities for project
management with resource-aware time calculations.

Features
--------
- Multiple activity resource input formats (direct, PERT three-point, PERT two-point)
- Resource-aware duration calculations via callback function
- Automatic network diagram generation using Graphviz
- Statistical analysis with variance propagation
- Multiple link format support
- Export to dictionaries and pandas DataFrames

Classes
-------
- :class:`NetworkModel`: Main class for network analysis
- :class:`_Activity`: Represents activities in the network (internal)
- :class:`_Event`: Represents events/milestones in the network (internal)

Key Concepts
------------
Resource Effort vs Duration:
    - ``expected``, ``optimistic``, ``pessimistic``: Estimates of resource effort (e.g., person-hours, machine-hours)
    - ``duration``: Actual time considering resource availability and allocation

Resource Dependency:
    - Duration calculations can account for resource constraints, productivity, and availability
    - Custom duration callback allows modeling of complex resource scenarios

Usage Example
-------------
Basic usage with direct resource effort estimates:

>>> wbs = {
...     1: {'letter': 'A', 'expected': 5.0, 'variance': 1.0},
...     2: {'letter': 'B', 'optimistic': 3.0, 'most_likely': 4.0, 'pessimistic': 8.0}
... }
>>> links = [[1, 2]]
>>> model = NetworkModel(wbs, links=links)
>>> activities_df, events_df = model.to_dataframe()

Advanced usage with resource-aware duration:

>>> def resource_duration(effort, activity, base_time):
...
...     # Calculate actual duration based on resource allocation
...     resource_count = 2  # Two people assigned
...     productivity = 0.8  # 80% productivity
...
...     return effort / (resource_count * productivity)
...
>>> model = NetworkModel(wbs, links=links, duration=resource_duration)

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
import _ccpm


# Constants for array indexing in time computations
EPS = np.finfo(float).eps
RES = 0  # Result of time computation
VAR = 1  # Result variance estimation (used for PERT)
ERR = 2  # Computation error upper limit

#==============================================================================
def fit_mpert(M, D, a, b):
    """
    Fit modified PERT distribution parameters to match mean and variance.

    This function calculates the most likely value and shape parameter
    for a modified PERT distribution given target mean and variance.

    Parameters
    ----------
    M : float
        Target mean value
    D : float
        Target variance
    a : float
        Optimistic (minimum) value
    b : float
        Pessimistic (maximum) value

    Returns
    -------
    tuple
        (m, g) where:
        - m: most likely value
        - g: shape parameter (None for deterministic case)

    Raises
    ------
    ValueError
        If invalid bounds are provided or mean is outside [a,b] range

    Notes
    -----
    The modified PERT distribution uses the formula:
    M = (a + g * m + b) / (2 + g)
    D = (M - a) * (b - M) / (3 + g)
    """
    if a > b:
        raise ValueError(f"Invalid bounds: optimistic ({a}) must be <= pessimistic ({b})")

    if not (a <= M <= b):
        raise ValueError(f"Mean ({M}) must be between optimistic ({a}) and pessimistic ({b})")

    _tol = np.sqrt(EPS)

    if b - a < 2 * _tol * (a + b):
        # This is the chain of deterministic processes, use M without g computation
        return M, None

    _thr = _tol * (b - a)

    if np.sqrt(D) < _thr:
        # This is the chain of deterministic processes and D is computation error
        return M, None

    # Make sure that M is in range of (a + _thr, b - _thr)
    if (M - a) < _thr:
        M = a + _thr

    if (b - M) < _thr:
        M = b - _thr

    # Compute g lower limit, make sure that ml is far from M enough
    if M < (a + b) / 2:
        ml = a + _thr / 2
    else:
        ml = b - _thr / 2
    # Now compute min safe g value
    gmin = (a + b - 2 * M) / (M - ml)

    # Compute g, make sure that computed m will be in range of (a,b)
    g = (M - a) * (b - M) / D - 3
    g = g if g > gmin else gmin

    # Compute m
    m = ((2 + g) * M - a - b) / g

    # Make sure that m is in range of (a,b) even in case of dramatic roundoff errors
    m = m if m >= a else (a + _thr / 2)
    m = m if m <= b else (b - _thr / 2)

    return m, g

#==============================================================================
def _p_quantile_estimate(p, tm, optimistic, pessimistic):
    """
    Calculate quantile estimate for time with given probability.

    Parameters
    ----------
    p : float
        Probability value (0 < p < 1)
    tm : numpy.ndarray
        Time array with [RES, VAR, ERR] components
    optimistic : float
        Optimistic time estimate
    pessimistic : float
        Pessimistic time estimate

    Returns
    -------
    float
        Quantile estimate for the given probability
    """
    s = np.sqrt(tm[VAR])
    if s <= EPS * tm[RES]:
        return tm[RES]

    mode, lambd = fit_mpert(tm[RES], tm[VAR], optimistic, pessimistic)
    if not lambd:
        return tm[RES]

    estimate = mpert.ppf(p, optimistic, mode, pessimistic, lambd)
    if np.isnan(estimate):
        return tm[RES]

    return estimate

#==============================================================================
def _prob_estimate(val, tm, optimistic, pessimistic):
    """
    Estimate probability that time is less than given value.

    Parameters
    ----------
    val : float
        Value to compare against
    tm : numpy.ndarray
        Time array with [RES, VAR, ERR] components
    optimistic : float
        Optimistic time estimate
    pessimistic : float
        Pessimistic time estimate

    Returns
    -------
    float
        Probability P(tm < val)
    """
    s = np.sqrt(tm[VAR])
    if s <= EPS * tm[RES]:
        return 1.0 if val > tm[RES] else 0.0

    mode, lambd = fit_mpert(tm[RES], tm[VAR], optimistic, pessimistic)
    if not lambd:
        return 1.0 if val > tm[RES] else 0.0

    prob = mpert.cdf(val, optimistic, mode, pessimistic, lambd)
    if np.isnan(prob):
        return 0
    return prob

#==============================================================================
def _choice(old, new, delta):
    """
    Choose between two time estimates based on certainty criteria.

    This function implements a decision mechanism for selecting between
    competing time estimates in the presence of uncertainty.

    Parameters
    ----------
    old : numpy.ndarray
        Existing time estimate [value, variance, error_bound]
    new : numpy.ndarray
        New time estimate [value, variance, error_bound]
    delta : float
        Difference threshold for decision making

    Returns
    -------
    numpy.ndarray
        Selected time estimate
    """
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
def _default_duration(effort, activity, base_time):
    """
    Default duration callback function.

    This function provides the default behavior where duration equals effort.
    Override this with custom logic to model resource constraints.

    Parameters
    ----------
    effort : float
        Resource effort estimate:
        - Positive number or zero for forward pass (early times calculation)
        - Negative number or zero for backward pass (late times calculation)
    activity : _Activity
        Activity object for context-aware calculations
    base_time : float or None
        Base time from which the activity starts:
        - float for time computations during network traversal
        - None for network post-processing (optimistic/pessimistic scenarios)

    Returns
    -------
    float
        Activity duration with the same sign as effort:
        - Zero if effort is zero
        - Positive number if effort is positive
        - Negative number if effort is negative

    Notes
    -----
    The function must preserve the sign of the effort parameter to ensure
    correct forward and backward pass calculations in the network.

    When base_time is None, the function should return a duration estimate
    without considering time-based resource availability constraints.

    Examples
    --------
    Simple productivity model:

    >>> def custom_duration(effort, activity, base_time):
    ...     team_size = activity.data.get('team_size', 1)
    ...     productivity = activity.data.get('productivity', 1.0)
    ...
    ...     # Calculate absolute duration
    ...     abs_duration = abs(effort) / (team_size * productivity)
    ...
    ...     # Return with original sign
    ...     return abs_duration if effort >= 0 else -abs_duration
    """
    # Return effort directly (default: duration = effort)
    # This preserves the sign of effort for forward/backward pass calculations
    return effort

#==============================================================================
class _Activity:
    """
    Represents an activity (task) in the network model.

    This class stores all information about a network activity including
    timing parameters, statistical properties, and custom data.

    Parameters
    ----------
    id : int
        Unique activity identifier
    wbs_id : int
        Work Breakdown Structure identifier (0 for dummy activities)
    letter : str
        Activity letter/code for visualization
    model : NetworkModel
        Parent network model instance
    src : _Event
        Source event of the activity
    dst : _Event
        Destination event of the activity
    expected : float
        Activity expected resource effort (mathematical expectation)
    variance : float
        Activity effort variance
    optimistic : float
        Optimistic effort estimate
    pessimistic : float
        Pessimistic effort estimate
    data : dict, optional
        Additional activity data from WBS

    Attributes
    ----------
    id : int
        Unique activity identifier
    wbs_id : int
        WBS identifier
    letter : str
        Activity letter/code
    model : NetworkModel
        Parent network model
    src : _Event
        Source event
    dst : _Event
        Destination event
    expected : numpy.ndarray
        Array containing [effort_value, variance, error_bound]
    data : dict
        Additional activity data
    early_start : numpy.ndarray
        Early start time [value, variance, error_bound]
    late_start : numpy.ndarray
        Late start time [value, variance, error_bound]
    early_end : numpy.ndarray
        Early end time [value, variance, error_bound]
    late_end : numpy.ndarray
        Late end time [value, variance, error_bound]
    reserve : numpy.ndarray
        Time reserve [value, variance, error_bound]
    optimistic : float
        Optimistic effort estimate
    pessimistic : float
        Pessimistic effort estimate
    opt_start : float
        Optimistic start time
    opt_end : float
        Optimistic end time
    pes_start : float
        Pessimistic start time
    pes_end : float
        Pessimistic end time

    Notes
    -----
    - Resource effort arrays follow the format: [RES (value), VAR (variance), ERR (error bound)]
    - Duration is calculated dynamically considering resource allocation and availability
    - The ``duration`` property computes actual time from early and late time analyses
    """

    def __init__(self, id, wbs_id, letter, model, src, dst, expected=0.0,
                 exp_var=0.0, optimistic=0.0, pessimistic=0.0, data=None):
        assert isinstance(id, int)
        assert isinstance(wbs_id, int)
        assert isinstance(letter, str)
        assert isinstance(model, NetworkModel)
        assert isinstance(src, _Event)
        assert isinstance(dst, _Event)
        assert isinstance(expected, float)
        assert expected >= 0.0
        assert exp_var >= 0.0
        assert data is None or isinstance(data, dict)

        self.id = id
        self.wbs_id = wbs_id
        self.letter = letter
        self.model = model
        self.src = src
        self.dst = dst

        self.expected = np.zeros((3,), dtype=float)
        self.expected[RES] = expected
        self.expected[VAR] = exp_var
        self.expected[ERR] = EPS * expected

        self.data = data if data is not None else {}

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
    def duration(self):
        """
        Calculate activity duration from early and late time analyses.

        This property computes the actual duration by comparing
        early and late time calculations and selecting the most certain
        estimate.

        Returns
        -------
        numpy.ndarray
            Duration array [value, variance, error_bound]

        Notes
        -----
        The duration is calculated as:
        - Early approach: early_end - early_start
        - Late approach: late_end - late_start
        - Final selection uses certainty-based decision making
        """
        # Early duration calculation
        # Note:
        # ealy_end[[RES,VAR]] == ealy_atsrt[[RES,VAR]] + early[[RES,VAR]]
        early      = self.early_end - self.early_start
        early[ERR] = self.early_end[ERR] + self.early_start[ERR]

        # Late duration calculation
        # Note:
        # late_start[RES] == late_end[RES] - late[RES]
        # late_start[VAR] == late_end[VAR] + late[VAR]
        late = np.zeros_like(early)
        late[RES] = self.late_end[RES]   - self.late_start[RES]
        late[VAR] = self.late_start[VAR] - self.late_end[VAR]
        late[ERR] = self.late_end[ERR]   + self.late_start[ERR]

        # Choose most certain estimate
        return _choice(early, late, late[RES] - early[RES])

    @property
    def early_start_pqe(self):
        """
        Get early start probabilistic quantile estimate.

        Returns
        -------
        float
            Early start time quantile for model's probability level
        """
        return _p_quantile_estimate(self.model.p, self.early_start, self.opt_start, self.pes_start)

    def early_start_prob(self, val):
        """
        Get probability that early start is less than given value.

        Parameters
        ----------
        val : float
            Value to compare against

        Returns
        -------
        float
            P(early_start < val)
        """
        return _prob_estimate(val, self.early_start, self.opt_start, self.pes_start)

    @property
    def early_end_pqe(self):
        """
        Get early end probabilistic quantile estimate.

        Returns
        -------
        float
            Early end time quantile for model's probability level
        """
        return _p_quantile_estimate(self.model.p, self.early_end, self.opt_end, self.pes_end)

    def early_end_prob(self, val):
        """
        Get probability that early end is less than given value.

        Parameters
        ----------
        val : float
            Value to compare against

        Returns
        -------
        float
            P(early_end < val)
        """
        return _prob_estimate(val, self.early_end, self.opt_end, self.pes_end)

    def __repr__(self):
        """String representation of the activity."""
        return str(self.to_dict())

    def to_dict(self):
        """
        Convert activity to dictionary representation.

        Returns
        -------
        dict
            Dictionary containing all activity data with structure:

            - ``id``: Activity ID
            - ``wbs_id``: WBS ID
            - ``letter``: Activity letter
            - ``src_id``: Source event ID
            - ``dst_id``: Destination event ID
            - ``expected``: Activity expected resource effort
            - ``duration``: Actual duration
            - ``early_start``, ``late_start``, ``early_end``, ``late_end``: Timing parameters
            - ``reserve``: Time reserve
            - ``data``: Additional activity data
            - Additional PERT fields if applicable

        Notes
        -----
        PERT-specific fields (early_start_var, early_end_var, early_start_pqe, early_end_pqe)
        are only included when PERT analysis is enabled.
        """
        duration = self.duration
        ret = {
            'id': self.id,
            'wbs_id': self.wbs_id,
            'letter': self.letter,
            'src_id': self.src.id,
            'dst_id': self.dst.id,
            'expected': self.expected[RES],  # Resource effort estimate
            'duration': duration[RES],       # Actual duration

            # CPM timing parameters
            'early_start': self.early_start[RES],
            'late_start': self.late_start[RES],
            'early_end': self.early_end[RES],
            'late_end': self.late_end[RES],
            'reserve': self.reserve[RES],

            # Additional data copy
            'data': self.data.copy()  # Return a copy to avoid modifying original
        }

        if self.model.is_pert:
            # PERT analysis fields
            ret['exp_var'] = self.expected[VAR]   # Variance of effort estimate
            ret['variance'] = duration[VAR]       # Variance of duration
            ret['optimistic'] = self.optimistic   # Optimistic effort
            ret['opt_start'] = self.opt_start     # Optimistic start time
            ret['opt_end'] = self.opt_end         # Optimistic end time

            ret['pessimistic'] = self.pessimistic # Pessimistic effort
            ret['pes_start'] = self.pes_start     # Pessimistic start time
            ret['pes_end'] = self.pes_end         # Pessimistic end time

            ret['early_start_var'] = self.early_start[VAR]
            ret['early_end_var'] = self.early_end[VAR]
            ret['early_start_pqe'] = self.early_start_pqe
            ret['early_end_pqe'] = self.early_end_pqe

            ret['late_end_prob'] = self.early_end_prob(self.late_end[RES])

        if self.model.debug:
            # Debug information
            ret['early_start_err'] = self.early_start[ERR]
            ret['late_start_err'] = self.late_start[ERR]
            ret['early_end_err'] = self.early_end[ERR]
            ret['late_end_err'] = self.late_end[ERR]

        return ret

#==============================================================================
class _Event:
    """
    Represents an event (milestone) in the network model.

    Events mark the beginning or end of activities and are used to
    calculate critical paths and timing parameters.

    Parameters
    ----------
    id : int
        Unique event identifier
    model : NetworkModel
        Parent network model instance

    Attributes
    ----------
    id : int
        Unique event identifier
    model : NetworkModel
        Parent network model
    early : numpy.ndarray
        Early time [value, variance, error_bound]
    late : numpy.ndarray
        Late time [value, variance, error_bound]
    reserve : numpy.ndarray
        Time reserve [value, variance, error_bound]
    stage : int
        Event stage in topological order
    optimistic : float
        Optimistic time estimate
    pessimistic : float
        Pessimistic time estimate
    """

    def __init__(self, id, model):
        assert isinstance(id, int)
        assert isinstance(model, NetworkModel)

        self.id = id
        self.model = model

        # CPM time parameters (calculated later)
        self.early = np.zeros((3,), dtype=float)
        self.late = np.zeros((3,), dtype=float)
        self.reserve = np.zeros((3,), dtype=float)
        self.stage = 0

        self.optimistic = 0.0
        self.pessimistic = 0.0

    @property
    def in_activities(self):
        """Get all activities entering this event."""
        return [a for a in self.model.activities if a.dst == self]

    @property
    def out_activities(self):
        """Get all activities leaving this event."""
        return [a for a in self.model.activities if a.src == self]

    @property
    def early_pqe(self):
        """
        Get early time probabilistic quantile estimate.

        Returns
        -------
        float
            Early time quantile for model's probability level
        """
        return _p_quantile_estimate(self.model.p, self.early, self.optimistic, self.pessimistic)

    def early_prob(self, val):
        """
        Get probability that early time is less than given value.

        Parameters
        ----------
        val : float
            Value to compare against

        Returns
        -------
        float
            P(early < val)
        """
        return _prob_estimate(val, self.early, self.optimistic, self.pessimistic)

    def __repr__(self):
        """String representation of the event."""
        return str(self.to_dict())

    def to_dict(self):
        """
        Convert event to dictionary representation.

        Returns
        -------
        dict
            Dictionary containing event data with structure:

            - ``id``: Event ID
            - ``stage``: Topological stage
            - ``early``: Early time
            - ``late``: Late time
            - ``reserve``: Time reserve
            - Additional PERT fields if applicable

        Notes
        -----
        PERT-specific fields (early_var, early_pqe) are only included
        when PERT analysis is enabled.
        """
        # Basic CPM parameters
        ret = {
            'id': self.id,
            'stage': self.stage,
            'early': self.early[RES],
            'late': self.late[RES],
            'reserve': self.reserve[RES],
        }

        if self.model.is_pert:
            # PERT analysis fields
            ret['optimistic'] = self.optimistic
            ret['pessimistic'] = self.pessimistic
            ret['early_var'] = self.early[VAR]
            ret['early_pqe'] = self.early_pqe
            ret['late_prob'] = self.early_prob(self.late[RES])

        if self.model.debug:
            # Debug information
            ret['early_err'] = self.early[ERR]
            ret['late_err'] = self.late[ERR]

        return ret

#==============================================================================
def _calculate_action_time_params(work_data, default_risk=0.3):
    """
    Calculate expected resource effort and its variance from various input formats.

    Supports three formats in order of priority:

    1. **Three-point PERT**: Uses optimistic, most_likely, and pessimistic effort estimates
       with formula: mean = (optimistic + 4*most_likely + pessimistic)/6,
       variance = ((pessimistic - optimistic)/6)²

    2. **Two-point PERT**: Uses optimistic and pessimistic effort estimates
       with formula: mean = (optimistic + 4*pessimistic)/5,
       variance = ((pessimistic - optimistic)/5)²

    3. **Direct parameters**: Uses directly provided expected and variance values

    Parameters
    ----------
    work_data : dict
        Dictionary containing work data with one of these combinations:

        - For three-point PERT: ``optimistic``, ``most_likely``, ``pessimistic``
        - For two-point PERT: ``optimistic``, ``pessimistic``
        - For direct parameters: ``expected``, ``exp_var`` (optional)
    default_risk : float, default=0.3
        Default risk factor for effort estimation when variance is provided

    Returns
    -------
    tuple
        (mean_effort, variance, optimistic, most_likely, pessimistic)

    Raises
    ------
    ValueError
        If insufficient data is provided or estimates are invalid

    Examples
    --------
    Three-point PERT effort estimation:

    >>> data = {'optimistic': 5, 'most_likely': 7, 'pessimistic': 12}
    >>> mean, var, a, m, b = _calculate_action_time_params(data)
    >>> print(f"Mean effort: {mean:.2f}, Variance: {var:.2f}")
    Mean effort: 7.50, Variance: 1.36

    Two-point PERT effort estimation:

    >>> data = {'optimistic': 3, 'pessimistic': 8}
    >>> mean, var, a, m, b = _calculate_action_time_params(data)
    >>> print(f"Mean effort: {mean:.2f}, Variance: {var:.2f}")
    Mean effort: 7.00, Variance: 1.00

    Direct parameters:

    >>> data = {'expected': 6.5, 'exp_var': 0.5}
    >>> mean, var, a, m, b = _calculate_action_time_params(data)
    >>> print(f"Mean effort: {mean:.2f}, Variance: {var:.2f}")
    Mean effort: 6.50, Variance: 0.50
    """
    # 1. Check for three-point PERT estimation (highest priority)
    if all(key in work_data for key in ['optimistic', 'most_likely', 'pessimistic']):
        a = work_data['optimistic']
        m = work_data['most_likely']
        b = work_data['pessimistic']

        # Validate inputs
        if not (a <= m <= b):
            raise ValueError(f"Invalid PERT estimates: must satisfy optimistic <= most_likely <= pessimistic. Got: {a}, {m}, {b}")

        mean = (a + 4 * m + b) / 6
        variance = ((b - a) / 6) ** 2
        return mean, variance, a, m, b

    # 2. Check for two-point PERT estimation (medium priority)
    elif all(key in work_data for key in ['optimistic', 'pessimistic']):
        a = work_data['optimistic']
        b = work_data['pessimistic']

        # Validate inputs
        if not (a <= b):
            raise ValueError(f"Invalid PERT estimates: must satisfy optimistic <= pessimistic. Got: {a}, {b}")

        m = (2 * a + b) / 3
        mean = (3 * a + 2 * b) / 5
        variance = ((b - a) / 5) ** 2

        return mean, variance, a, m, b

    # 3. Direct parameters (lowest priority - backward compatibility)
    elif 'expected' in work_data:
        mean = work_data['expected']
        variance = work_data.get('exp_var', 0.0)

        # Validate inputs
        if mean < 0:
            raise ValueError(f"Expected effort must be non-negative. Got: {mean}")
        if variance < 0:
            raise ValueError(f"Variance must be non-negative. Got: {variance}")

        if variance > 0:
            d = 6 * np.sqrt(variance) / 2
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
    """
    Main class for CPM/PERT network analysis with resource-aware scheduling.

    This class constructs and analyzes network models using Critical Path
    Method (CPM) and Program Evaluation and Review Technique (PERT) with
    support for resource-dependent duration calculations.

    Parameters
    ----------
    wbs_dict : dict
        Work Breakdown Structure dictionary with activity data.
        Each key is an activity ID and value is a dictionary containing:

        - ``letter``: Activity letter/code (required)
        - One of these resource effort specifications:
            - Direct: ``expected`` and optional ``exp_var``
            - Three-point PERT: ``optimistic``, ``most_likely``, ``pessimistic``
            - Two-point PERT: ``optimistic``, ``pessimistic``
        - ``name``: Activity description (optional)
        - Any other custom fields for resource modeling

    lnk_src : array-like, optional
        Source activity IDs for dependencies (old format)
    lnk_dst : array-like, optional
        Destination activity IDs for dependencies (old format)
    links : various, optional
        Dependency links in various formats:

        - Two rows: ``[[src1, src2, ...], [dst1, dst2, ...]]``
        - Two columns: ``[[src1, dst1], [src2, dst2], ...]``
        - Dictionary: ``{'src': [src1, src2, ...], 'dst': [dst1, dst2, ...]}``

    duration : callable, default=_default_duration
        Callback function for resource-aware duration calculation.
        Signature: duration(effort, activity, base_time) -> float
        - effort: float value representing resource effort:
          * Positive number or zero for forward pass (early times calculation)
          * Negative number or zero for backward pass (late times calculation)
        - activity: _Activity object for context
        - base_time: float for time-based availability checks during network traversal,
          or None for network post-processing (optimistic/pessimistic scenarios)
        Returns: float value representing actual duration with the same sign as effort:
          * Zero if effort is zero
          * Positive number if effort is positive
          * Negative number if effort is negative
    p : float, default=0.95
        Probability level for PERT quantile estimates
    default_risk : float, default=0.3
        Default risk factor for effort estimation
    debug : bool, default=False
        Enable debug mode to include computation error bounds

    Raises
    ------
    ValueError
        If insufficient link data is provided or links format is invalid
    AssertionError
        If network construction fails

    Attributes
    ----------
    activities : list
        List of _Activity objects in the network
    events : list
        List of _Event objects in the network
    is_pert : bool
        True if PERT analysis is enabled (variance > 0 for any activity)
    debug : bool
        Debug mode flag
    p : float
        Probability level for PERT

    Examples
    --------
    Basic usage with direct effort parameters:

    >>> wbs = {
    ...     1: {'letter': 'A', 'expected': 5.0, 'exp_var': 1.0},
    ...     2: {'letter': 'B', 'expected': 3.0}
    ... }
    >>> links = [[1, 2]]
    >>> model = NetworkModel(wbs, links=links)

    Three-point PERT effort estimates:

    >>> wbs_pert = {
    ...     1: {'letter': 'A', 'optimistic': 3, 'most_likely': 5, 'pessimistic': 8},
    ...     2: {'letter': 'B', 'optimistic': 2, 'pessimistic': 6}
    ... }
    >>> model_pert = NetworkModel(wbs_pert, links=links)

    Resource-aware scheduling with custom duration callback:

    >>> def team_duration(effort, activity, base_time):
    ...     team_size = activity.data.get('team_size', 1)
    ...     productivity = activity.data.get('productivity', 1.0)
    ...
    ...     # Calculate absolute duration
    ...     abs_duration = abs(effort) / (team_size * productivity)
    ...
    ...     # Return with original sign
    ...     return abs_duration if effort >= 0 else -abs_duration
    ...
    >>> wbs_resource = {
    ...     1: {'letter': 'A', 'expected': 40.0, 'team_size': 2, 'productivity': 0.8},
    ...     2: {'letter': 'B', 'expected': 30.0, 'team_size': 1, 'productivity': 1.0}
    ... }
    >>> model_resource = NetworkModel(wbs_resource, links=links, duration=team_duration)

    Notes
    -----
    The duration callback function must preserve the sign of the effort parameter
    to ensure correct forward and backward pass calculations in the network:
    - For forward pass (early times): effort >= 0, duration >= 0
    - For backward pass (late times): effort <= 0, duration <= 0
    - For zero effort: duration = 0

    When base_time is None in the duration callback, it indicates that
    the function should return a duration estimate without considering
    time-based resource availability constraints.

    For PERT analysis, variance is automatically propagated through the
    network using modified PERT distribution formulas.
    """

    def __init__(self, wbs_dict, lnk_src=None, lnk_dst=None, links=None,
                 duration=_default_duration, p=0.95, default_risk=0.3, debug=False):
        assert isinstance(wbs_dict, dict)

        self.debug = debug
        self.is_pert = False
        self.p = p
        self._duration = duration  # Resource-aware duration callback

        # Parse links into standard format
        lnk_src, lnk_dst = self._parse_links(lnk_src, lnk_dst, links)

        # Create network model
        self._create_model(wbs_dict, lnk_src, lnk_dst, default_risk)
        assert 0 < len(self.events)

        # Compute stages of project
        self._compute_target('stage')

        # Renumerate events according to the rules of network modeling
        self.events.sort(key=lambda e: e.stage)
        for i, e in enumerate(self.events, 1):
            e.id = i

        # Compute Event and Activity time parameters
        self._compute_time_params()

    def _parse_links(self, lnk_src, lnk_dst, links):
        """
        Parse links from various formats into standard lnk_src, lnk_dst arrays.

        Parameters
        ----------
        lnk_src : array-like, optional
            Source activity IDs (old format)
        lnk_dst : array-like, optional
            Destination activity IDs (old format)
        links : various, optional
            Links in various new formats

        Returns
        -------
        tuple
            (lnk_src, lnk_dst) as lists

        Raises
        ------
        ValueError
            If links format is invalid or insufficient data provided
        """
        # Case 1: Old format (lnk_src and lnk_dst provided)
        if lnk_src is not None and lnk_dst is not None:
            return list(lnk_src), list(lnk_dst)

        # Case 2: New formats via links parameter
        if links is None:
            raise ValueError("Either (lnk_src, lnk_dst) or links must be provided")

        # Format 1: Two rows [[src...], [dst...]]
        if (isinstance(links, (list, tuple)) and len(links) == 2 and
                isinstance(links[0], (list, tuple, np.ndarray)) and
                isinstance(links[1], (list, tuple, np.ndarray))):
            return list(links[0]), list(links[1])

        # Format 2: Two columns [[src, dst], [src, dst], ...]
        elif (isinstance(links, (list, tuple, np.ndarray)) and
              len(links) > 0 and
              isinstance(links[0], (list, tuple, np.ndarray)) and
              len(links[0]) == 2):
            src_list = [item[0] for item in links]
            dst_list = [item[1] for item in links]
            return src_list, dst_list

        # Format 3: Dictionary {'src': [...], 'dst': [...]}
        elif isinstance(links, dict):
            if 'src' in links and 'dst' in links:
                return list(links['src']), list(links['dst'])
            else:
                raise ValueError("Dictionary links must contain 'src' and 'dst' keys")

        else:
            raise ValueError(f"Unsupported links format: {type(links)}")

    def _create_model(self, wbs_dict, lnk_src, lnk_dst, default_risk):
        """
        Create network model from WBS data and links.

        Internal method that constructs the network graph, creates events
        and activities, and sets up the model for analysis.

        Parameters
        ----------
        wbs_dict : dict
            Work Breakdown Structure data
        lnk_src : list
            Source activity IDs
        lnk_dst : list
            Destination activity IDs
        default_risk : float
            Default risk factor for effort estimation

        Notes
        -----
        This method uses the C++ extension _ccpm for efficient AOA
        (Activity-on-Arrow) network generation and automatically creates
        dummy activities where needed.
        """
        assert len(lnk_src) == len(lnk_dst)

        act_ids = list(wbs_dict.keys())

        # Generate network graph using C extension
        status, act_ids, net_src, net_dst = _ccpm.make_aoa(act_ids, lnk_src, lnk_dst)
        assert _ccpm.OK == status

        self.events = []
        self.next_act = 1
        self.activities = []

        # Create events
        for i in range(np.max(net_dst)):
            self._add_event(int(i + 1))

        # Create activities (real and dummy)
        na = len(act_ids)  # Number of actions
        nd = 0  # Number of dummy actions
        dsrc = [] # Dummy event sources
        for i in range(len(net_src)):
            if i < na:
                # Real activity - get data from WBS
                act_id = act_ids[i]
                wbs_data = wbs_dict[act_id]  # Get complete WBS data

                # Calculate expected effort and variance using unified function
                try:
                    expected, exp_var, optimistic, _, pessimistic = \
                        _calculate_action_time_params(wbs_data, default_risk)
                except ValueError as e:
                    raise ValueError(f"Error processing activity {act_id} ({wbs_data.get('letter', 'unknown')}): {e}")

                letter = wbs_data.get('letter', '')

                # Even one wbs item with nonzero variance is enough to compute PERT
                if exp_var > 0.0:
                    self.is_pert = True

                # Create data dict without fields stored as separate attributes
                data_without_duplicates = self._remove_duplicate_fields(wbs_data, expected, exp_var, letter)

                self._add_activity(int(act_id), int(net_src[i]), int(net_dst[i]),
                                   expected, exp_var, optimistic, pessimistic,
                                   letter, data_without_duplicates)
            else:
                # Add a dummy activity (no effort, no letter, no data)
                nd += 1  # One more dummy work
                self._add_activity(0, int(net_src[i]), int(net_dst[i]),
                                   0., 0., 0., 0., '#' + str(nd), {})
                dsrc.append(int(net_src[i]))

        # Network postprocessing
        # Make sure that activities with largest efforts are on straight paths between events
        grpoi = {}
        # Find groups of triangles on dummies
        for e in self.events:
            # Watch only dummy sources
            if e.id not in dsrc:
                continue

            bck = e.in_activities
            fwd = e.out_activities
            # Watch only events with one incoming and one outgoing action
            if 1 < len(bck) or 1 < len(fwd):
                continue

            key = (bck[0].src.id, fwd[0].dst.id)
            if key not in grpoi.keys():
                for a in bck[0].src.out_activities:
                    if a.dst.id == fwd[0].dst.id:
                        grpoi[key] = (a, [bck[0]])
                        break
            else:
                grpoi[key][1].append(bck[0])

        # Place maximum duration activities on long side of triangle groups
        for k in grpoi.keys():
            aoi = grpoi[k][1]
            if len(aoi) < 1:
                raise RuntimeError("Action of interest group must contain at least one action!")
            # Maximum duration candidate
            maxa = grpoi[k][0]
            for a in aoi:
                if self._duration(a.expected[RES], a, None) > self._duration(maxa.expected[RES], maxa, None):
                    a.dst, maxa.dst = maxa.dst, a.dst
                    maxa = a

    def _remove_duplicate_fields(self, wbs_data, expected, exp_var, letter):
        """
        Remove fields from WBS data that are stored as separate activity attributes.

        Parameters
        ----------
        wbs_data : dict
            Complete WBS data for an activity
        expected : float
            Activity expected effort (already extracted)
        exp_var : float
            Activity effort variance (already extracted)
        letter : str
            Activity letter (already extracted)

        Returns
        -------
        dict
            WBS data without fields stored as separate attributes
        """
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
        """
        Compute all time parameters for events and activities.

        This method performs both forward pass (early times) and
        backward pass (late times) through the network, then calculates
        time reserves for both events and activities.

        Notes
        -----
        The computation is performed in two phases:
        1. Forward pass: Compute early times starting from project beginning
        2. Backward pass: Compute late times starting from project completion

        For PERT models, additional optimistic and pessimistic scenarios
        are computed to provide statistical analysis.
        """
        self._compute_target('early')

        # Set late times starting from project completion
        late = np.zeros((3,), dtype=float)
        for e in self.events:
            if e.early[RES] > late[RES]:
                late = e.early.copy()

        for e in self.events:
            e.late = late

        self._compute_target('late')

        # Compute reserves
        for e in self.events:
            e.reserve[VAR] = e.late[VAR] + e.early[VAR]
            e.reserve[ERR] = e.late[ERR] + e.early[ERR]
            # Round off insignificant values
            r = e.late[RES] - e.early[RES]
            e.reserve[RES] = r if abs(r) > e.reserve[ERR] else 0.0
            # Check for programming errors
            if r < -e.reserve[ERR]:
                raise RuntimeError("Events can not have negative time reserves!!!")

        for a in self.activities:
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
                raise RuntimeError("Actions can not have negative time reserves!!!")

        if self.is_pert:
            # For PERT models we must compute optimistic and pessimistic scenarios
            self._compute_target('optimistic')
            self._compute_target('pessimistic')


    def _compute_target(self, target=None):
        """
        Compute CPM parameters for events and activities.

        This method performs topological sorting and computes either
        early times (forward pass) or late times (backward pass).

        Parameters
        ----------
        target : str
            What to compute: 'stage', 'early', 'late', 'optimistic', or 'pessimistic'

        Raises
        ------
        ValueError
            If target parameter is invalid

        Notes
        -----
        The computation uses different strategies for stage calculation
        vs time parameter calculation. For PERT analysis, variance is
        propagated using modified PERT distribution formulas.
        """
        def _choice_early(old, new):
            return _choice(old, new, new[RES] - old[RES])

        def _choice_late(old, new):
            return _choice(old, new, old[RES] - new[RES])

        def _delta_late(a):
            ret = a.expected.copy()
            ret[RES] = -a.expected[RES]
            return ret

        def _duration_vec(effort, activity, base_time):
            """
            Compute duration vector with variance propagation for PERT analysis.

            This function handles the conversion from resource effort to actual
            duration while properly propagating variance through the network.

            Parameters
            ----------
            effort : numpy.ndarray
                Resource effort array [value, variance, error_bound]
            activity : _Activity
                Activity object for context
            base_time : numpy.ndarray or None
                Base time array [value, variance, error_bound] for time computations,
                or None for network post-processing

            Returns
            -------
            numpy.ndarray
                Duration array [value, variance, error_bound]
            """
            # Optimize for default duration function
            if _default_duration == self._duration:
                return effort

            dur = np.zeros((3,), dtype=float)

            # Compute duration value and error bound
            # effort[RES] is float: positive for forward pass, negative for backward pass
            dur[RES] = self._duration(effort[RES], activity, base_time[RES])
            dur[ERR] = EPS * abs(dur[RES])  # Error bound based on absolute duration

            if 0. == effort[VAR] or not self.is_pert:
                # Deterministic or fake activity
                # VAR is zero already
                return dur

            # Compute shape parameter for modified PERT distribution
            _,g = fit_mpert(activity.expected[RES], activity.expected[VAR],
                            activity.optimistic, activity.pessimistic)

            if not g:
                # Deterministic activity
                return dur

            # Model is PERT and activity is not deterministic,
            # will compute duration variance using modified PERT formula

            # Compute optimistic and pessimistic duration estimates
            # For variance calculation, use base_time=None to get estimates
            # without time-based constraints
            if effort[RES] >= 0.:
                # Forward pass: use positive effort values
                opt_dur  = self._duration(activity.optimistic,  activity, base_time[RES])
                pess_dur = self._duration(activity.pessimistic, activity, base_time[RES])
            else:
                # Backward pass: use negative effort values
                opt_dur  = self._duration( -activity.optimistic,  activity, base_time[RES])
                pess_dur = self._duration( -activity.pessimistic, activity, base_time[RES])

            # Use modified PERT formula for variance calculation:
            # D = (M - a) * (b - M) / (3 + g)
            # Note:
            # We don't care positive or negative values here as
            # sign(dur[RES]) == sign(opt_dur) == sign(pess_dur)
            # and so numerator is allways positive
            dur[VAR] = (dur[RES] - opt_dur) * (pess_dur - dur[RES]) / (3 + g)

            return dur

        if 'stage' == target:
            act_base = None
            act_new = None
            act_next = 'dst'
            fwd = 'out_activities'
            rev = 'in_activities'
            choice = max
            delta = lambda a: 1
            process_delta = lambda x, a, b: x

        elif 'early' == target:
            act_base = 'early_start'
            act_new = 'early_end'
            act_next = 'dst'
            fwd = 'out_activities'
            rev = 'in_activities'
            choice = _choice_early
            delta = lambda a: a.expected
            process_delta = _duration_vec

        elif 'late' == target:
            act_base = 'late_end'
            act_new = 'late_start'
            act_next = 'src'
            fwd = 'in_activities'
            rev = 'out_activities'
            choice = _choice_late
            delta = _delta_late
            process_delta = _duration_vec

        elif 'optimistic' == target:
            act_base = 'opt_start'
            act_new = 'opt_end'
            act_next = 'dst'
            fwd = 'out_activities'
            rev = 'in_activities'
            choice = max
            delta = lambda a: a.optimistic
            process_delta = self._duration

        elif 'pessimistic' == target:
            act_base = 'pes_start'
            act_new = 'pes_end'
            act_next = 'dst'
            fwd = 'out_activities'
            rev = 'in_activities'
            choice = max
            delta = lambda a: a.pessimistic
            process_delta = self._duration
        else:
            raise ValueError("Unknown 'target' value!!!")

        # Initialize activity parameters and processing function
        if target != 'stage':
            for a in self.activities:
                setattr(a, act_base, -1)
                setattr(a, act_new, -1)

        # Count dependencies for topological sorting
        n_dep = [len(getattr(e, rev)) for e in self.events]

        # Find starting events (no dependencies)
        evt = [i for i, n in enumerate(n_dep) if 0 == n]
        # Check for programming errors
        if 1 != len(evt):
            raise RuntimeError("The project can not have more than one starting event!!!")

        # Process events in topological order
        i = 0
        while True:
            e = self.events[evt[i]]
            base_val = getattr(e, target)

            for a in getattr(e, fwd):
                if act_base:
                    setattr(a, act_base, base_val)

                # Calculate new value using appropriate delta processing
                # For stage: direct value, for time: resource-aware duration callback
                new_val = base_val + process_delta(delta(a), a, base_val)

                if act_new:
                    setattr(a, act_new, new_val)

                next_evt = getattr(a, act_next)
                next_i = self.events.index(next_evt)

                setattr(next_evt, target, choice(getattr(next_evt, target), new_val))

                n_dep[next_i] -= 1

                if 0 >= n_dep[next_i]:
                    evt.append(next_i)

            i += 1
            if i >= len(evt):
                break

    def _add_event(self, i):
        """Add a new event to the network."""
        self.events.append(_Event(i, self))

    def _add_activity(self, wbs_id, src_id, dst_id, expected, exp_var,
                      optimistic, pessimistic, letter, data):
        """
        Add a new activity to the network.

        Parameters
        ----------
        wbs_id : int
            Work Breakdown Structure identifier (0 for dummy activities)
        src_id : int
            Source event ID
        dst_id : int
            Destination event ID
        expected : float
            Activity expected resource effort
        exp_var : float
            Activity effort variance
        optimistic : float
            Optimistic effort estimate
        pessimistic : float
            Pessimistic effort estimate
        letter : str
            Activity letter/code for visualization
        data : dict
            WBS data excluding fields stored as separate attributes
        """
        assert isinstance(wbs_id, int)
        assert isinstance(src_id, int)
        assert isinstance(dst_id, int)
        assert isinstance(expected, float)
        assert isinstance(exp_var, float)
        assert isinstance(optimistic, float)
        assert isinstance(pessimistic, float)
        assert isinstance(letter, str)
        assert isinstance(data, dict)

        act = _Activity(self.next_act, wbs_id, letter, self,
                        self.events[src_id - 1], self.events[dst_id - 1],
                        expected, exp_var, optimistic, pessimistic, data)
        self.activities.append(act)
        self.next_act += 1

    def __repr__(self):
        """String representation of the network model."""
        _repr = 'Events:{\n'
        for e in self.events:
            _repr += '        ' + str(e) + '\n'
        _repr += '}\n'

        _repr += 'Activities:\n'
        for a in self.activities:
            _repr += '        ' + str(a) + '\n'
        _repr += '}\n'

        return _repr

    def to_dict(self):
        """
        Convert network model to dictionary representation.

        Returns
        -------
        dict
            Dictionary with structure:

            .. code-block:: python

                {
                    'activities': [
                        {activity1_data},
                        {activity2_data},
                        ...
                    ],
                    'events': [
                        {event1_data},
                        {event2_data},
                        ...
                    ]
                }
        """
        activities_data = [activity.to_dict() for activity in self.activities]
        events_data = [event.to_dict() for event in self.events]

        return {
            'activities': activities_data,
            'events': events_data
        }

    def to_dataframe(self):
        """
        Convert network model to pandas DataFrames.

        Returns
        -------
        tuple
            (activities_df, events_df) - pandas DataFrames for activities and events

        Notes
        -----
        The activities DataFrame expands all custom data fields from the
        'data' attribute into separate columns for easy analysis.
        """
        # Convert to dictionaries first
        model_dict = self.to_dict()

        # Create events DataFrame (straightforward)
        events_df = pd.DataFrame(model_dict['events'])

        # Create activities DataFrame with data fields expanded
        activities_list = model_dict['activities']

        # Expand data fields into separate columns
        expanded_activities = []
        for activity in activities_list:
            # Start with basic activity data
            activity_data = {k: v for k, v in activity.items() if k != 'data'}

            # Add data fields as separate columns
            if 'data' in activity and activity['data']:
                activity_data.update(activity['data'])

            expanded_activities.append(activity_data)

        activities_df = pd.DataFrame(expanded_activities).fillna(value='')

        return activities_df, events_df

    def viz(self, output_path=None):
        """
        Create Graphviz visualization of the CPM network.

        Parameters
        ----------
        output_path : str, optional
            Custom output path for saving the visualization file.
            If None, uses default location.

        Returns
        -------
        graphviz.Digraph
            Graphviz object for rendering or saving

        Notes
        -----
        The visualization uses the following color coding:
        - Red: Critical path (zero reserve)
        - Orange: Near-critical activities
        - Black: Non-critical activities
        - Dashed lines: Dummy activities

        The visualization shows:
        - Event nodes with early/late times and reserves
        - Activity edges with duration and reserve information
        - Critical paths highlighted in red
        """
        dot = graphviz.Digraph(node_attr={'shape': 'record', 'style': 'rounded'})
        dot.graph_attr['rankdir'] = 'LR'
        #dot.graph_attr['dpi'] = '300'

        def _cl(res, p):
            """Choose color based on reserve (red for critical path)"""
            if abs(res[RES]) <= res[ERR]:  # Absolute precision is nonsense
                return '#ff0000'  # Critical path
            elif p < self.p:
                return '#ffa000'  # May consume the time reserve before completion
            else:
                return '#000000'

        # Add events/nodes
        for e in self.events:
            # Format time values to 1 decimal place
            dot.node(str(e.id),
                     '{{%d |{%.1f|%.1f}| %.2f}}' % (e.id,
                                                     e.early[RES],
                                                     e.late[RES],
                                                     e.reserve[RES]),
                     color=_cl(e.reserve, e.early_prob(e.late[RES])))

        # Add activities/edges
        for a in self.activities:
            lbl = a.letter
            if a.wbs_id:  # Real activity
                # Use letter instead of wbs_id in visualization
                # Format duration and reserve to 1 decimal place
                lbl += '\n t=' + format(a.duration[RES], '.1f') + '\n r=' + format(a.reserve[RES], '.2f')
            else:  # Dummy activity
                lbl += '\n r=' + format(a.reserve[RES], '.2f')

            dot.edge(str(a.src.id), str(a.dst.id),
                     label=lbl,
                     color=_cl(a.reserve, a.early_end_prob(a.late_end[RES])),
                     style='dashed' if a.expected[RES] == 0.0 else 'solid'
                     )

        # If output path is specified, render to that location
        if output_path is not None:
            dot.render(output_path, format='png', cleanup=True)

        return dot

#==============================================================================
if __name__ == '__main__':
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
    src_old = np.array([1,2,3, 2,3,3, 4,1,6, 7, 5,6,7,  3, 6, 7,  6, 8, 9,  7, 8, 9,10])
    dst_old = np.array([5,5,5, 6,6,7, 7,8,8, 8, 9,9,9, 10,10,10, 11,11,11, 12,12,12,12])
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
    for i, activity in enumerate(n_old.activities[:8]):  # Show first 8 activities
        if activity.wbs_id != 0:  # Skip dummy activities
            print(f"Activity {i + 1}: wbs_id={activity.wbs_id}, letter='{activity.letter}'")
            print(f"  Expected effort: {activity.expected[RES]:.3f}, Variance: {activity.expected[VAR]:.3f}")
            print(f"  Actual duration: {activity.duration[RES]:.3f}")
            if activity.data:
                print(f"  Data fields: {list(activity.data.keys())}")

    # Advanced example: Resource-aware scheduling
    print("\n=== Advanced Example: Resource-Aware Scheduling ===")

    def resource_aware_duration(effort, activity, base_time):
        """
        Custom duration callback function for resource-aware scheduling.

        This function demonstrates how to model resource constraints including:
        - Team size allocation
        - Productivity factors
        - Resource availability based on time

        Parameters
        ----------
        effort : float
            Resource effort estimate:
            - Positive number or zero for forward pass
            - Negative number or zero for backward pass
        activity : _Activity
            Activity object containing resource data
        base_time : float or None
            Base time for availability calculations during network traversal,
            or None for network post-processing

        Returns
        -------
        float
            Actual duration with the same sign as effort
        """
        # Get resource allocation from activity data
        team_size = activity.data.get('team_size', 1)
        productivity = activity.data.get('productivity', 1.0)

        # Calculate absolute duration from absolute effort
        abs_effort = abs(effort)

        if base_time is not None:
            # Time-dependent resource availability during network traversal
            # Simulate resource availability (e.g., weekends, holidays)
            availability = 1.0
            # Example: reduce availability on weekends (simplified)
            day_of_week = int(base_time) % 7
            if day_of_week >= 5:  # Weekend
                availability = 0.5

            # Calculate absolute duration with time-based availability
            if team_size * productivity * availability > 0:
                abs_duration = abs_effort / (team_size * productivity * availability)
            else:
                abs_duration = abs_effort  # Fallback to direct mapping
        else:
            # No time-based constraints for post-processing
            if team_size * productivity > 0:
                abs_duration = abs_effort / (team_size * productivity)
            else:
                abs_duration = abs_effort  # Fallback to direct mapping

        # Return duration with original sign to preserve forward/backward pass logic
        return abs_duration if effort >= 0 else -abs_duration

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
    for activity in model_resource.activities:
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
    print(f"Old == New1: {len(n_old.activities) == len(n_new1.activities)}")
    print(f"Old == New2: {len(n_old.activities) == len(n_new2.activities)}")
    print(f"Old == New3: {len(n_old.activities) == len(n_new3.activities)}")

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

    # Construct the full file path
    file_path = os.path.join(target_dir, 'cpm_network')

    # Create and save the visualization
    dot = n_old.viz(output_path=file_path)
    print(f"Visualization saved as '{file_path}.png'")
    print(f"Target directory: {target_dir}")
