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
- Automatic network diagram generation using Graphviz (PNG, SVG, PDF)
- Statistical analysis with variance propagation
- Multiple link format support
- User-defined data fields on activities and events
- Export to dictionaries and pandas DataFrames

Classes
-------
- :class:`NetworkModel`: Main class for network analysis
- :class:`Activity`: Represents activities in the network (internal)
- :class:`Event`: Represents events/milestones in the network (internal)

Key Concepts
------------
Resource Effort vs Duration:
    - ``expected``, ``optimistic``, ``pessimistic``: Estimates of resource effort
      (e.g., person-hours, machine-hours). These are inputs to the model.
    - ``duration``: Actual calendar time computed from effort via the
      ``duration`` callback, taking into account resource availability,
      allocation, and productivity.

Resource Dependency:
    - Duration calculations can account for resource constraints, productivity, and availability
    - Custom duration callback allows modeling of complex resource scenarios

Time Arrays:
    Throughout this module, time and effort values are represented as
    numpy arrays of shape ``(3,)`` with the following layout:

    - ``[RES]`` — point estimate (mean value)
    - ``[VAR]`` — variance (used only when PERT analysis is enabled)
    - ``[ERR]`` — upper bound on computation error

    The constants ``RES``, ``VAR``, ``ERR`` are defined at module level.

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

>>> def resource_duration(effort, activity, base_time, target):
...     # Calculate actual duration based on resource allocation
...     resource_count = 2  # Two people assigned
...     productivity = 0.8  # 80% productivity
...     return effort / (resource_count * productivity)
...
>>> model = NetworkModel(wbs, links=links, duration=resource_duration)

.. note::
    This module uses a C extension (_ccpm) for performance-critical computations.

.. note::
    The model uses a **two-phase time analysis** to support resource-aware
    duration callbacks whose result depends on the calendar context
    (start or end time of the activity). Phase 1 computes preliminary
    ``early_pre`` / ``late_pre`` passes; a single per-activity duration
    is then frozen via :meth:`Activity.compute_final_duration`; phase 2
    computes the final ``early`` / ``late`` passes using those frozen
    durations. This guarantees that forward and backward passes use
    identical durations and therefore that all time reserves are
    non-negative. See :meth:`NetworkModel._compute_time_params` for the
    algorithm and :meth:`NetworkModel._compute_target` for the per-target
    semantics.

.. note::
    During the **final** ``'early'`` and ``'late'`` passes (phase 2) the
    model checks the following CPM invariants:

    - ``early_start >= 0``
    - ``early_end >= early_start``
    - ``late_start >= early_start``
    - ``late_end >= late_start``
    - ``late >= early`` for events

    Violations within the accumulated error bound are silently clamped;
    violations beyond it raise :exc:`RuntimeError`, which signals a
    structural inconsistency (bad topology or a misbehaving ``duration``
    callback) rather than a purely numerical issue.

    The preliminary ``'early_pre'`` / ``'late_pre'`` passes are **not**
    subject to these checks: they may temporarily hold values that
    violate ``late_pre >= early_pre``, which is exactly what the freeze
    step between phases corrects.
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
    """
    Fit modified beta distribution parameters to match mean and variance.

    This function calculates the shape parameters (alpha, beta) for a
    beta distribution supported on ``[a, b]`` such that its mean equals
    ``M`` and its variance equals ``D``.

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
    err : float
        Computation error threshold (non-negative). If ``sqrt(D) < err``,
        the distribution is treated as deterministic.

    Returns
    -------
    tuple
        ``(alpha, beta)`` — shape parameters of the beta distribution,
        or ``(None, None)`` if the distribution is considered deterministic
        (i.e., ``b - a`` is too small, or the standard deviation is below
        the computation error threshold).

    Raises
    ------
    ValueError
        If ``a > b``, if ``M`` is outside ``[a, b]``, or if ``err < 0``.

    Notes
    -----
    The function uses the standard method of moments for the beta
    distribution. If the resulting ratio ``mu * (1 - mu) / var`` is
    less than or equal to 1 (which would correspond to a degenerate
    distribution), the ratio is clamped to ``1 + 2 * EPS`` to ensure
    positive shape parameters.
    """
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
    """
    Calculate the quantile (percent point function) for a modified beta distribution.

    Parameters
    ----------
    p : float
        Probability level (0 < p < 1)
    M : float
        Mean value
    D : float
        Variance
    a : float
        Optimistic (minimum) value
    b : float
        Pessimistic (maximum) value
    err : float, optional
        Computation error threshold (default 0.0)

    Returns
    -------
    float
        Quantile estimate for the given probability.

        If the distribution is treated as deterministic (see
        :func:`fit_beta`), returns ``M``.

    Raises
    ------
    ValueError
        If :func:`fit_beta` raises an error due to invalid bounds or parameters.
    """
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
    """
    Calculate the cumulative distribution function for a modified beta distribution.

    Parameters
    ----------
    val : float
        Value at which to evaluate the CDF
    M : float
        Mean value
    D : float
        Variance
    a : float
        Optimistic (minimum) value
    b : float
        Pessimistic (maximum) value
    err : float, optional
        Computation error threshold (default 0.0)

    Returns
    -------
    float
        Probability ``P(X <= val)``.

        For a deterministic distribution (see :func:`fit_beta`),
        returns ``1.0`` if ``val > M`` and ``0.0`` otherwise.

    Raises
    ------
    ValueError
        If :func:`fit_beta` raises an error due to invalid bounds or parameters.
    """
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
    """
    Calculate quantile estimate for time with given probability.

    Parameters
    ----------
    p : float
        Probability value (0 < p < 1)
    tm : numpy.ndarray
        Time array with ``[RES, VAR, ERR]`` components
    optimistic : float
        Optimistic time estimate
    pessimistic : float
        Pessimistic time estimate

    Returns
    -------
    float
        Quantile estimate for the given probability

    Raises
    ------
    ValueError
        If :func:`calc_ppf` raises an error due to invalid parameters.
    """
    return calc_ppf(p, tm[RES], tm[VAR], optimistic, pessimistic, err=tm[ERR])

#==============================================================================
def _prob_estimate(val, tm, optimistic, pessimistic):
    """
    Estimate probability that time is less than given value.

    Parameters
    ----------
    val : float
        Value to compare against
    tm : numpy.ndarray
        Time array with ``[RES, VAR, ERR]`` components
    optimistic : float
        Optimistic time estimate
    pessimistic : float
        Pessimistic time estimate

    Returns
    -------
    float
        Probability ``P(tm < val)``

    Raises
    ------
    ValueError
        If :func:`calc_cdf` raises an error due to invalid parameters.
    """
    return calc_cdf(val, tm[RES], tm[VAR], optimistic, pessimistic, err=tm[ERR])

#==============================================================================
def _choice(old, new, delta):
    """
    Choose between two time estimates based on certainty criteria.

    This function implements a heuristic decision mechanism for selecting
    between competing time estimates in the presence of uncertainty.

    Parameters
    ----------
    old : numpy.ndarray
        Existing time estimate ``[value, variance, error_bound]``
    new : numpy.ndarray
        New time estimate ``[value, variance, error_bound]``
    delta : float
        Difference between the new and old estimates
        (``new[RES] - old[RES]``). Used to decide which estimate is
        more certain.

    Returns
    -------
    numpy.ndarray
        Selected time estimate:

        - If ``delta >= e`` where ``e = new[ERR] + old[ERR]`` — returns ``new``
          (the difference is statistically significant).
        - If ``delta <= -e`` — returns ``old``.
        - Otherwise — returns a mixed estimate with mean ``0.5 * (new + old)``,
          maximum variance, and half the combined error bound.

    Notes
    -----
    This is an *engineering heuristic*, not a rigorous statistical
    combination of distributions. The mixed estimate deliberately
    overestimates variance (by taking the maximum) and halves the
    error bound to reflect the loss of precision when two nearly
    equal estimates are combined.
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
def _default_duration(effort, activity, base_time, target):
    """
    Default duration callback function.

    This function provides the default behavior where duration equals effort.
    It is used when the user does not supply a custom ``duration`` callback
    to :class:`NetworkModel`.

    Parameters
    ----------
    effort : float
        Resource effort estimate:

        - Positive number or zero for forward passes
        - Negative number or zero for backward passes
    activity : Activity
        Activity object for context-aware calculations
    base_time : float or None
        Calendar context for the duration calculation:

        - ``float`` with ``target='early_pre'``: the *start* time of
          the activity (phase-1 forward pass).
        - ``float`` with ``target='late_pre'``: the *end* time of the
          activity (phase-1 backward pass).
        - ``float`` (scalar) with ``target='optimistic'`` or
          ``'pessimistic'``: the scenario start time.
        - ``None`` with ``target=None``: no calendar context (network
          post-processing, e.g. triangle optimisation in
          :meth:`NetworkModel._create_model`).

        .. note::
            The callback is **not** invoked during the final (phase-2)
            ``'early'`` / ``'late'`` passes — those use the duration
            value frozen on the activity between phases (see
            :meth:`Activity.compute_final_duration`). This is what
            guarantees that forward and backward passes use identical
            durations and, consequently, that reserves are non-negative.
    target : str or None
        Target scenario identifier. One of ``'early_pre'``, ``'late_pre'``,
        ``'optimistic'``, ``'pessimistic'``, or ``None``.

    Returns
    -------
    float
        Activity duration with the same sign as ``effort``:

        - Zero if ``effort`` is zero
        - Positive number if ``effort`` is positive
        - Negative number if ``effort`` is negative

    Notes
    -----
    The function must preserve the sign of the ``effort`` parameter to ensure
    correct forward and backward pass calculations in the network.

    When ``base_time`` is ``None``, the function should return a duration
    estimate without considering time-based resource availability constraints.

    Examples
    --------
    Simple productivity model:

    >>> def custom_duration(effort, activity, base_time, target):
    ...     team_size = activity.data.get('team_size', 1)
    ...     productivity = activity.data.get('productivity', 1.0)
    ...     # Calculate absolute duration
    ...     abs_duration = abs(effort) / (team_size * productivity)
    ...     # Return with original sign
    ...     return abs_duration if effort >= 0 else -abs_duration
    """
    # Return effort directly (default: duration = effort)
    # This preserves the sign of effort for forward/backward pass calculations
    return effort

#==============================================================================
def _default_style(element):
    """
    Determine styling (color, penwidth, fontsize, weight) based on criticality.

    Parameters
    ----------
    element : Event or Activity
        Graph element whose style is being computed. The styling decision
        is based on the element's time reserve and the probability that
        the reserve is positive:

        - For an ``Event`` the probability is ``element.early_prob(element.late[RES])``
        - For an ``Activity`` the probability is ``element.early_end_prob(element.late_end[RES])``

        The threshold probability is taken from ``element.model.p``.

    Returns
    -------
    dict
        Dictionary with styling attributes for Graphviz.
        Contains the following keys:

        - ``color``: string with CSS color (e.g., ``'#ff0000'``,
          ``'#ffa000'``, ``'#000000'``)
        - ``penwidth``: string with line width (e.g., ``'4'``, ``'3'``, ``'2'``)
        - ``fontsize``: string with font size (e.g., ``'16'``, ``'14'``)
        - ``weight``: string with edge weight for Graphviz
          (e.g., ``'3'``, ``'2'``, ``'1'``)
        - ``tooltip``: string with hover tooltip text. For an ``Event``
          it lists the event id, early/late times, and reserve; for an
          ``Activity`` it contains the activity letter and, if present,
          the ``name`` field from ``element.data``. Tooltips are rendered
          only in SVG output.

    Raises
    ------
    TypeError
        If ``element`` is neither an ``Event`` nor an ``Activity``.

    Notes
    -----
    Elements that are formally critical (``abs(res[RES]) <= res[ERR]``) are
    drawn red. Elements with ``prob_reserve < prob_thr`` are drawn as
    sub-critical (orange). Remaining elements are drawn as non-critical
    (black).
    """
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
    """
    Represents an activity (task) in the network model.

    This class stores all information about a network activity including
    timing parameters, statistical properties, and custom data.

    Parameters
    ----------
    id : int
        Unique activity identifier (internal, assigned by :class:`NetworkModel`)
    wbs_id : int
        Work Breakdown Structure identifier (0 for dummy activities)
    letter : str
        Activity letter/code for visualization
    model : NetworkModel
        Parent network model instance
    src : Event
        Source event of the activity
    dst : Event
        Destination event of the activity
    expected : int or float
        Activity expected resource effort (mathematical expectation).
        Integer values are accepted and converted to ``float``.
    exp_var : int or float
        Activity effort variance. Must be non-negative.
    optimistic : int or float
        Optimistic effort estimate
    pessimistic : int or float
        Pessimistic effort estimate
    data : dict, optional
        Additional activity data from WBS (custom user fields)
    is_dummy : bool, default=False
        Whether this is a dummy activity (used in AoA network construction)

    Attributes
    ----------
    id : int
        Unique activity identifier
    wbs_id : int
        WBS identifier (0 for dummy activities)
    letter : str
        Activity letter/code
    model : NetworkModel
        Parent network model
    src : Event
        Source event (property, setter updates the event's activity lists)
    dst : Event
        Destination event (property, setter updates the event's activity lists)
    expected : numpy.ndarray
        Array containing ``[effort_value, variance, error_bound]``
    data : dict
        Additional activity data
    early_start : numpy.ndarray
        Early start time ``[value, variance, error_bound]`` (phase-2 result)
    late_start : numpy.ndarray
        Late start time ``[value, variance, error_bound]`` (phase-2 result)
    early_end : numpy.ndarray
        Early end time ``[value, variance, error_bound]`` (phase-2 result)
    late_end : numpy.ndarray
        Late end time ``[value, variance, error_bound]`` (phase-2 result)
    reserve : numpy.ndarray
        Time reserve ``[value, variance, error_bound]`` (phase-2 result)
    duration : numpy.ndarray
        Frozen final duration ``[value, variance, error_bound]``, used
        verbatim by both phase-2 passes.
    optimistic : float
        Optimistic effort estimate
    pessimistic : float
        Pessimistic effort estimate
    opt_start : float
        Optimistic start time (scenario)
    opt_end : float
        Optimistic end time (scenario)
    pes_start : float
        Pessimistic start time (scenario)
    pes_end : float
        Pessimistic end time (scenario)
    is_dummy : bool
        Whether this is a dummy activity

    Notes
    -----
    - Resource effort arrays follow the format
      ``[RES (value), VAR (variance), ERR (error bound)]``.
    - The final time attributes (``early_start``, ``late_start``,
      ``early_end``, ``late_end``, ``reserve``) and ``duration`` are the
      result of the phase-2 passes of
      :meth:`NetworkModel._compute_time_params`. Because both phase-2
      passes use the same frozen ``duration``, the invariant
      ``late_end - late_start == early_end - early_start`` holds for
      every activity, so time reserves are non-negative.
    - The preliminary phase-1 estimates are not exposed on the activity;
      they live on the events as ``early_pre`` / ``late_pre`` and are
      used only inside :meth:`compute_final_duration`.
    - Setters for ``src`` and ``dst`` maintain the corresponding event's
      internal ``_in_activities`` / ``_out_activities`` lists. Do not
      mutate those lists directly.
    - During the phase-2 network traversal, the CPM invariants
      ``early_end >= early_start``, ``late_start >= early_start`` and
      ``late_end >= late_start`` are enforced by
      :meth:`NetworkModel._compute_target`. After a successful model
      construction these relations hold for the activity's time
      attributes as well (up to the accumulated error bound).

    Raises
    ------
    TypeError
        If any argument has an incorrect type.
    ValueError
        If ``expected`` or ``exp_var`` is negative, or if ``data`` is not
        a dict when provided.
    """

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
        """Source event of the activity.

        Setting this property updates the corresponding event's
        internal activity lists (removes from old, appends to new).
        """
        return self._src

    @src.setter
    def src(self, new_src):
        if isinstance(self._src, Event):
            self._src._out_activities.remove(self)

        self._src = new_src
        new_src._out_activities.append(self)

    @property
    def dst(self):
        """Destination event of the activity.

        Setting this property updates the corresponding event's
        internal activity lists (removes from old, appends to new).
        """
        return self._dst

    @dst.setter
    def dst(self, new_dst):
        if isinstance(self._dst, Event):
            self._dst._in_activities.remove(self)

        self._dst = new_dst
        new_dst._in_activities.append(self)

    @property
    def is_dummy(self):
        """Whether this activity is a dummy (used in AoA construction)."""
        return self._is_dummy


    def compute_final_duration(self):
        """
        Freeze the final activity duration for the phase-2 passes.

        This method is called by :meth:`NetworkModel._compute_time_params`
        **between** the preliminary (``'early_pre'`` / ``'late_pre'``)
        and final (``'early'`` / ``'late'``) passes. It compares the
        duration implied by the preliminary early schedule
        (``early_end - early_start``) with the duration implied by the
        preliminary late schedule (``late_end - late_start``) and
        selects the more certain estimate via :func:`_choice`.

        The selected value is stored in :attr:`duration` and is used
        verbatim by both phase-2 passes, so that forward and backward
        calculations use exactly the same duration and time reserves
        are guaranteed non-negative.

        Returns
        -------
        None
            The result is written to :attr:`duration`.

        Notes
        -----
        This is not a public "recalculate" method — calling it outside
        the two-phase sequence will overwrite :attr:`duration` with a
        value derived from stale ``early_pre`` / ``late_pre`` /
        ``early`` / ``late`` fields.

        The returned array's ``VAR`` component reflects the propagated
        variance of the duration (relevant for PERT models). Its ``RES``
        component is the point estimate.
        """

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
        """
        Get early start probabilistic quantile estimate.

        Returns
        -------
        float
            Early start time quantile for the model's probability level ``p``.
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
            ``P(early_start < val)``
        """
        return _prob_estimate(val, self.early_start, self.opt_start, self.pes_start)

    @property
    def early_end_pqe(self):
        """
        Get early end probabilistic quantile estimate.

        Returns
        -------
        float
            Early end time quantile for the model's probability level ``p``.
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
            ``P(early_end < val)``
        """
        return _prob_estimate(val, self.early_end, self.opt_end, self.pes_end)

    def __repr__(self):
        """String representation of the activity (a dict-like view)."""
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
            - ``duration``: Actual (frozen) duration
            - ``early_start``, ``late_start``, ``early_end``, ``late_end``:
              Timing parameters (phase-2 result)
            - ``reserve``: Time reserve (phase-2 result)
            - ``is_dummy``: Whether this is a dummy activity
            - ``data``: Additional activity data
            - Additional PERT fields if applicable

        Notes
        -----
        PERT-specific fields (``exp_var``, ``variance``, ``optimistic``,
        ``opt_start``, ``opt_end``, ``pessimistic``, ``pes_start``,
        ``pes_end``, ``early_start_var``, ``early_end_var``,
        ``early_start_pqe``, ``early_end_pqe``, ``late_end_prob``) are only
        included when PERT analysis is enabled (``model.is_pert``).

        Debug-specific fields (``early_start_err``, ``late_start_err``,
        ``early_end_err``, ``late_end_err``) are only included when
        ``model.debug`` is ``True``. Preliminary phase-1 estimates are
        not exposed on the activity — see ``Event.to_dict`` for
        ``early_pre`` / ``late_pre``.

        The returned dictionary is a copy of ``self.data`` extended with
        the computed CPM/PERT fields. If a user field name collides with
        a computed field name, the computed value takes precedence.
        """

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
        Unique event identifier (renumerated by :class:`NetworkModel`
        after construction, according to topological stage order)
    model : NetworkModel
        Parent network model
    data : dict
        Additional user data associated with the event. This dictionary
        holds arbitrary user-defined fields; they are merged into the
        output of :meth:`to_dict` (and therefore appear as separate
        columns in the events DataFrame produced by
        :meth:`NetworkModel.to_dataframe`).
    early_pre : numpy.ndarray
        Preliminary early time ``[value, variance, error_bound]`` from
        phase 1 (see :meth:`NetworkModel._compute_time_params`).
    late_pre : numpy.ndarray
        Preliminary late time ``[value, variance, error_bound]`` from
        phase 1. This is an internal service value: it may be negative
        or larger than ``early_pre`` and is used only to freeze
        per-activity durations. Do not compare it against ``early_pre``
        — the final, API-facing fields are ``early`` and ``late``.
    early : numpy.ndarray
        Early time ``[value, variance, error_bound]`` (phase-2 result)
    late : numpy.ndarray
        Late time ``[value, variance, error_bound]`` (phase-2 result)
    reserve : numpy.ndarray
        Time reserve ``[value, variance, error_bound]`` (phase-2 result)
    stage : int
        Event stage in topological order (0 = start event)
    optimistic : float
        Optimistic time estimate (scenario)
    pessimistic : float
        Pessimistic time estimate (scenario)

    Raises
    ------
    TypeError
        If ``id`` is not an integer or ``model`` is not a NetworkModel
        instance.

    Notes
    -----
    This class uses identity-based ``__eq__``/``__hash__`` (default from
    ``object``). Do NOT define ``__eq__`` without also defining a
    compatible ``__hash__`` — instances are used as dict keys in
    ``NetworkModel._compute_target``. Do NOT add ``@dataclass`` without
    ``eq=False``.

    The :attr:`in_activities` and :attr:`out_activities` properties return
    **immutable tuples** (snapshots) of the current activity lists. The
    underlying lists are maintained by ``Activity.src`` / ``Activity.dst``
    setters. To mutate the topology, reassign ``activity.src`` or
    ``activity.dst`` — never mutate the internal lists directly.

    The model uses a two-phase time analysis (see
    :meth:`NetworkModel._compute_time_params`). ``early_pre`` and
    ``late_pre`` hold the phase-1 (preliminary) estimates and may not
    satisfy ``late_pre >= early_pre``; they exist so that
    :meth:`Activity.compute_final_duration` can freeze a single duration
    per activity before phase 2. The public API exposes the phase-2
    values ``early`` and ``late``, which do satisfy ``late >= early``
    (enforced by :meth:`NetworkModel._compute_target`).

    User-defined keys in ``data`` must not collide with the reserved
    field names inserted by :meth:`to_dict` (``id``, ``stage``,
    ``early``, ``late``, ``reserve``, ``optimistic``, ``pessimistic``,
    ``early_var``, ``early_pqe``, ``late_prob``, ``early_err``,
    ``late_err``, ``early_pre``, ``early_pre_var``, ``early_pre_err``,
    ``late_pre``, ``late_pre_var``, ``late_pre_err``), otherwise the
    computed values take precedence in the output dictionary.

    During the phase-2 network traversal, the CPM invariant
    ``late >= early`` is enforced by :meth:`NetworkModel._compute_target`.
    After successful model construction, event late times therefore
    always satisfy ``late[RES] >= early[RES] - (late[ERR] + early[ERR])``
    and, in the common case, ``late[RES] >= early[RES]``.
    """

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
        """Activities entering this event.

        Returns an **immutable tuple** (snapshot) of the current incoming
        activities. The underlying list is maintained by the
        ``Activity.dst`` setter. To mutate topology, reassign
        ``activity.dst = new_event`` — do not attempt to modify the
        returned tuple (it is immutable) or the internal list directly.
        """
        return tuple(self._in_activities)

    @property
    def out_activities(self):
        """Activities leaving this event.

        Returns an **immutable tuple** (snapshot) of the current outgoing
        activities. The underlying list is maintained by the
        ``Activity.src`` setter. To mutate topology, reassign
        ``activity.src = new_event`` — do not attempt to modify the
        returned tuple (it is immutable) or the internal list directly.
        """
        return tuple(self._out_activities)

    @property
    def early_pqe(self):
        """
        Get early time probabilistic quantile estimate.

        Returns
        -------
        float
            Early time quantile for the model's probability level ``p``.
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
            ``P(early < val)``
        """
        return _prob_estimate(val, self.early, self.optimistic, self.pessimistic)

    def __repr__(self):
        """String representation of the event (a dict-like view)."""
        return str(self.to_dict())

    def to_dict(self):
        """
        Convert event to dictionary representation.

        Returns
        -------
        dict
            Dictionary containing event data with structure:

            - All user-defined fields from ``self.data``
            - ``id``: Event ID
            - ``stage``: Topological stage
            - ``early``: Early time (phase-2 result)
            - ``late``: Late time (phase-2 result)
            - ``reserve``: Time reserve (phase-2 result)
            - Additional PERT fields if applicable

        Notes
        -----
        PERT-specific fields (``optimistic``, ``pessimistic``, ``early_var``,
        ``early_pqe``, ``late_prob``) are only included when PERT analysis
        is enabled (``model.is_pert``).

        Debug-specific fields:

        - ``early_err``, ``late_err`` — phase-2 error bounds.
        - ``early_pre``, ``early_pre_var``, ``early_pre_err`` — phase-1
          preliminary early time and its statistics.
        - ``late_pre``, ``late_pre_var``, ``late_pre_err`` — phase-1
          preliminary late time and its statistics. Note that
          ``late_pre`` is an internal service value and may be negative
          or larger than ``early_pre``; the public API value is ``late``.

        The returned dictionary is a copy of ``self.data`` extended with
        the computed CPM/PERT fields. Reserved field names (``id``,
        ``stage``, ``early``, ``late``, ``reserve``, ``optimistic``,
        ``pessimistic``, ``early_var``, ``early_pqe``, ``late_prob``,
        ``early_err``, ``late_err``, ``early_pre``, ``early_pre_var``,
        ``early_pre_err``, ``late_pre``, ``late_pre_var``,
        ``late_pre_err``) must not be used as user data keys, otherwise
        the computed values take precedence in the output dictionary.
        """
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
    """
    Calculate expected resource effort and its variance from various input formats.

    Supports three formats in order of priority:

    1. **Three-point PERT**: Uses ``optimistic``, ``most_likely``, and
       ``pessimistic`` effort estimates with formulas:

       - ``mean = (optimistic + 4 * most_likely + pessimistic) / 6``
       - ``variance = ((pessimistic - optimistic) / 6) ** 2``

    2. **Two-point PERT**: Uses ``optimistic`` and ``pessimistic`` effort
       estimates with formulas:

       - ``most_likely = (2 * optimistic + pessimistic) / 3``
       - ``mean = (3 * optimistic + 2 * pessimistic) / 5``
       - ``variance = ((pessimistic - optimistic) / 5) ** 2``

    3. **Direct parameters**: Uses directly provided ``expected`` and
       optional ``exp_var`` values. If ``exp_var`` is positive, the
       implicit ``optimistic`` / ``pessimistic`` bounds are derived
       using ``default_risk`` as follows:

       - ``d = 3 * sqrt(exp_var)``
       - ``optimistic = mean - min(d, default_risk * mean)``
       - ``pessimistic = optimistic + 2 * d``
       - ``most_likely = (6 * mean - optimistic - pessimistic) / 4``

       If ``exp_var == 0``, the distribution is treated as deterministic:
       ``optimistic = most_likely = pessimistic = expected``.

    Parameters
    ----------
    work_data : dict
        Dictionary containing work data with one of these combinations:

        - For three-point PERT: ``optimistic``, ``most_likely``, ``pessimistic``
        - For two-point PERT: ``optimistic``, ``pessimistic``
        - For direct parameters: ``expected``, ``exp_var`` (optional)
    default_risk : float, default=0.3
        Default risk factor (in ``[0, 1]``) used to derive bounds when
        only ``expected`` and ``exp_var`` are provided.

    Returns
    -------
    tuple
        ``(mean_effort, variance, optimistic, most_likely, pessimistic)``
        — all floats representing **resource effort**, not duration.

    Raises
    ------
    TypeError
        If ``work_data`` is not a dict, or if any numeric field cannot
        be converted to ``float``.
    ValueError
        If ``default_risk`` is out of range, or if insufficient data is
        provided, or if estimates are invalid (e.g., the ordering
        ``optimistic <= most_likely <= pessimistic`` is not satisfied,
        or negative mean/variance are given).

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
    Mean effort: 5.80, Variance: 1.00

    Direct parameters:

    >>> data = {'expected': 6.5, 'exp_var': 0.5}
    >>> mean, var, a, m, b = _calculate_action_time_params(data)
    >>> print(f"Mean effort: {mean:.2f}, Variance: {var:.2f}")
    Mean effort: 6.50, Variance: 0.50
    """
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
    """
    Main class for CPM/PERT network analysis with resource-aware scheduling.

    This class constructs and analyzes network models using Critical Path
    Method (CPM) and Program Evaluation and Review Technique (PERT) with
    support for resource-dependent duration calculations.

    Parameters
    ----------
    wbs_dict : dict
        Work Breakdown Structure dictionary with activity data.
        Each key is an activity ID (``int``) and value is a dictionary
        containing:

        - ``letter``: Activity letter/code (required, ``str``)
        - One of these resource effort specifications:

          - Direct: ``expected`` and optional ``exp_var``
          - Three-point PERT: ``optimistic``, ``most_likely``, ``pessimistic``
          - Two-point PERT: ``optimistic``, ``pessimistic``
        - ``name``: Activity description (optional)
        - Any other custom fields for resource modeling

    lnk_src : array-like, optional
        Source activity IDs for dependencies (legacy format).
        Must be provided together with ``lnk_dst``.
    lnk_dst : array-like, optional
        Destination activity IDs for dependencies (legacy format).
        Must be provided together with ``lnk_src``.
    links : various, optional
        Dependency links in various formats:

        - Two rows: ``[[src1, src2, ...], [dst1, dst2, ...]]``
        - Two columns: ``[[src1, dst1], [src2, dst2], ...]``
        - Dictionary: ``{'src': [src1, src2, ...], 'dst': [dst1, dst2, ...]}``

        Cannot be used together with ``lnk_src``/``lnk_dst``.
    duration : callable, default=_default_duration
        Callback function for resource-aware duration calculation.
        Signature: ``duration(effort, activity, base_time, target) -> float``

        - ``effort``: float value representing resource effort:

          * Positive number or zero for forward passes
          * Negative number or zero for backward passes
        - ``activity``: :class:`Activity` object for context
        - ``base_time``: context-dependent:

          * ``float`` with ``target='early_pre'`` — activity *start*
            time;
          * ``float`` with ``target='late_pre'`` — activity *end* time;
          * ``float`` (scalar) with ``target='optimistic'`` or
            ``'pessimistic'`` — scenario start time;
          * ``None`` with ``target=None`` — no calendar context
            (post-processing).
        - ``target``: str or None — one of ``'early_pre'``, ``'late_pre'``,
          ``'optimistic'``, ``'pessimistic'``, or ``None``.

        Returns: float value representing actual duration with the same
        sign as ``effort``:

        - Zero if ``effort`` is zero
        - Positive number if ``effort`` is positive
        - Negative number if ``effort`` is negative

        .. note::
            The callback is invoked only during phase-1 passes
            (``'early_pre'`` / ``'late_pre'``), scenario passes
            (``'optimistic'`` / ``'pessimistic'``), and post-processing
            (``None``). It is **not** invoked during the final phase-2
            ``'early'`` / ``'late'`` passes — those use the frozen
            per-activity duration computed by
            :meth:`Activity.compute_final_duration`. This design
            guarantees that forward and backward passes always use
            identical durations for the same activity.

    p : float, default=0.95
        Probability level for PERT quantile estimates (``0 < p < 1``)
    default_risk : float, default=0.3
        Default risk factor for effort estimation (``0 <= default_risk <= 1``)
    next_act_id : int, default=1
        Starting ID for automatically generated internal activity numbering.
        Must be positive.
    debug : bool, default=False
        Enable debug mode to include computation error bounds and
        preliminary phase-1 estimates in output

    Raises
    ------
    TypeError
        If input types are incorrect.
    ValueError
        If insufficient link data is provided, links format is invalid,
        or if network construction fails (e.g., circular dependencies).
    RuntimeError
        If internal network consistency checks fail. In addition to the
        construction-time checks (exactly one source event, no cycles,
        valid AoA), the **final** ``'early'`` / ``'late'`` passes
        (phase 2) raise :exc:`RuntimeError` when the following CPM
        invariants are violated beyond the accumulated error bound:

        - ``early_start >= 0``
        - ``early_end >= early_start`` (activity)
        - ``late_start >= early_start`` (activity)
        - ``late_end >= late_start`` (activity)
        - ``late >= early`` (event)

        Such violations typically indicate a structural inconsistency
        (bad topology or a misbehaving ``duration`` callback) rather
        than a purely numerical issue. The preliminary
        ``'early_pre'`` / ``'late_pre'`` passes are not subject to these
        checks.

    Attributes
    ----------
    activities : list of Activity
        List of activity objects in the network (real + dummy).
    events : list of Event
        List of event objects in the network, sorted by topological stage.
    is_pert : bool
        ``True`` if PERT analysis is enabled (any activity has non-zero
        variance).
    debug : bool
        Debug mode flag.
    p : float
        Probability level for PERT quantile estimates.

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

    >>> def team_duration(effort, activity, base_time, target):
    ...     team_size = activity.data.get('team_size', 1)
    ...     productivity = activity.data.get('productivity', 1.0)
    ...     # Calculate absolute duration
    ...     abs_duration = abs(effort) / (team_size * productivity)
    ...     # Return with original sign
    ...     return abs_duration if effort >= 0 else -abs_duration
    ...
    >>> wbs_resource = {
    ...     1: {'letter': 'A', 'expected': 40.0, 'team_size': 2, 'productivity': 0.8},
    ...     2: {'letter': 'B', 'expected': 30.0, 'team_size': 1, 'productivity': 1.0}
    ... }
    >>> model_resource = NetworkModel(wbs_resource, links=links, duration=team_duration)

    Visualization in various formats:

    >>> model.viz('network.png')  # PNG (default if extension omitted)
    >>> model.viz('network.svg')  # SVG
    >>> model.viz('network.pdf')  # PDF

    Notes
    -----
    The duration callback function must preserve the sign of the effort
    parameter to ensure correct forward and backward pass calculations:

    - For forward passes: ``effort >= 0``, ``duration >= 0``
    - For backward passes: ``effort <= 0``, ``duration <= 0``
    - For zero effort: ``duration == 0``

    When ``base_time`` is ``None`` in the duration callback, it indicates
    that the function should return a duration estimate without considering
    time-based resource availability constraints.

    For PERT analysis, variance is automatically propagated through the
    network using modified PERT distribution formulas.

    The model uses a two-phase time analysis (see
    :meth:`_compute_time_params` for the algorithm):

    **Phase 1** — preliminary ``early_pre`` / ``late_pre`` passes. The
    ``duration`` callback is invoked at each activity with the calendar
    context natural for that direction (start time for ``early_pre``,
    end time for ``late_pre``). This produces two potentially different
    duration estimates per activity.

    **Freeze** — a single final duration is selected per activity via
    :meth:`Activity.compute_final_duration` (certainty-based choice) and
    stored in ``activity.duration``.

    **Phase 2** — final ``early`` / ``late`` passes. These use the
    frozen ``activity.duration`` and do **not** call the callback again.
    As a result, for every activity
    ``late_end - late_start == early_end - early_start``, so time
    reserves are guaranteed non-negative even for resource-aware
    callbacks whose result depends on calendar context.

    Numerical consistency of time parameters (non-negativity,
    ``early_end >= early_start``, ``late_start >= early_start``,
    ``late_end >= late_start``, ``late >= early``) is enforced *during*
    the phase-2 traversal inside :meth:`_compute_target` by the local
    helpers ``_check_and_fix_act`` and ``_check_and_fix_evt``. The
    phase-1 passes are intentionally not sanitized — ``late_pre`` may
    temporarily be less than ``early_pre``, which is exactly what the
    freeze step reconciles.

    Because the ``_check_and_fix_act`` / ``_check_and_fix_evt`` helpers
    only react to the ``'early'`` and ``'late'`` targets, the ``'stage'``,
    ``'early_pre'``, ``'late_pre'``, ``'optimistic'`` and ``'pessimistic'``
    passes are not subject to these checks.

    Reserved WBS field names (``expected``, ``exp_var``, ``letter``,
    ``optimistic``, ``most_likely``, ``pessimistic``) are consumed during
    parsing and removed from ``activity.data``. Do not use these names
    for custom fields.

    User-defined event data is stored in ``Event.data``. Its keys must not
    collide with the reserved field names inserted by ``Event.to_dict``
    (``id``, ``stage``, ``early``, ``late``, ``reserve``, ``optimistic``,
    ``pessimistic``, ``early_var``, ``early_pqe``, ``late_prob``,
    ``early_err``, ``late_err``, ``early_pre``, ``early_pre_var``,
    ``early_pre_err``, ``late_pre``, ``late_pre_var``, ``late_pre_err``),
    otherwise the computed values take precedence in the output
    dictionary.
    """

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
        """Whether PERT analysis is enabled (any activity has non-zero variance)."""
        return self._is_pert

    @property
    def p(self):
        """Probability level for PERT quantile estimates."""
        return self._p

    @property
    def activities(self):
        """List of activity objects in the network."""
        return self._activities

    @property
    def events(self):
        """List of event objects, sorted by topological stage."""
        return self._events

    def _parse_links(self, lnk_src, lnk_dst, links):
        """
        Parse links from various formats into standard ``(lnk_src, lnk_dst)`` lists.

        Parameters
        ----------
        lnk_src : array-like, optional
            Source activity IDs (legacy format)
        lnk_dst : array-like, optional
            Destination activity IDs (legacy format)
        links : various, optional
            Links in one of the supported new formats

        Returns
        -------
        tuple
            ``(lnk_src, lnk_dst)`` as lists of activity IDs

        Raises
        ------
        TypeError
            If any input is not an iterable or cannot be converted to a list.
        ValueError
            If link data is insufficient, lengths mismatch, format is
            unsupported, or both legacy and new formats are provided.
        """

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
        """
        Create network model from WBS data and links.

        Internal method that constructs the network graph, creates events
        and activities, and sets up the model for analysis. It delegates
        the AoA network generation to the C extension ``_ccpm.make_aoa``,
        then performs a post-processing step to place maximum-effort
        activities on the long side of triangle groups.

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
        next_act_id : int
            Starting ID for automatically generated activities

        Raises
        ------
        ValueError
            If any activity data is invalid or links contain invalid IDs.
        RuntimeError
            If C library returns an error or network consistency fails.
        """
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
        """
        Remove fields from WBS data that are stored as separate activity attributes.

        Parameters
        ----------
        wbs_data : dict
            Complete WBS data for an activity

        Returns
        -------
        dict
            Shallow copy of ``wbs_data`` with the following keys removed
            (they are consumed during parsing and stored as separate
            attributes on the :class:`Activity` instance):
            ``expected``, ``exp_var``, ``letter``, ``optimistic``,
            ``most_likely``, ``pessimistic``.
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

        This method performs the two-phase time analysis: preliminary
        forward and backward passes, a freeze step that fixes one
        duration per activity, and final forward and backward passes.
        Time reserves are computed after phase 2 from the final times.

        Raises
        ------
        RuntimeError
            If any activity has a negative time reserve after phase 2,
            indicating a programming error or corrupted network. Also
            propagates :exc:`RuntimeError` from :meth:`_compute_target`,
            which is raised during the phase-2 ``'early'`` / ``'late'``
            passes when a CPM invariant is violated beyond the
            accumulated error bound.

        Notes
        -----
        The computation proceeds in four passes with a freeze step in
        between:

        1. **``'early_pre'`` (forward).** The ``duration`` callback is
           invoked with ``base_time`` = the activity's start time
           (source event's ``early_pre``). Produces preliminary early
           times.
        2. **``'late_pre'`` (backward).** The ``duration`` callback is
           invoked with ``base_time`` = the activity's end time
           (destination event's ``late_pre``). Produces preliminary
           late times.

        **Freeze.** For every activity,
        :meth:`Activity.compute_final_duration` selects a single final
        duration from the two preliminary estimates (via
        :func:`_choice`, which prefers the more certain of the two).
        The result is stored in ``activity.duration``.

        3. **``'early'`` (forward).** Uses the frozen
           ``activity.duration``; the callback is **not** re-invoked.
        4. **``'late'`` (backward).** Also uses the frozen
           ``activity.duration``.

        This design guarantees that forward and backward passes use
        exactly the same duration for each activity, and therefore that
        ``late_end - late_start == early_end - early_start`` for every
        activity — i.e. time reserves are non-negative, even for
        resource-aware callbacks whose result depends on calendar
        context (start vs. end time, weekends, etc.).

        For PERT models (``is_pert == True``) the ``'optimistic'`` and
        ``'pessimistic'`` scenario passes are computed **before** phase 1
        and use the callback directly (they are not affected by the
        freeze step, since they represent separate scenarios rather than
        the nominal schedule).

        Numerical consistency (non-negativity, ``early_end >=
        early_start``, ``late_start >= early_start``, ``late_end >=
        late_start``, ``late >= early``) is enforced *during* the
        phase-2 traversal inside :meth:`_compute_target` by
        ``_check_and_fix_act`` / ``_check_and_fix_evt``. The phase-1
        passes are not sanitized: ``late_pre`` may temporarily fall
        below ``early_pre`` (for activities whose early schedule
        crosses a calendar boundary that the backward callback
        legitimately treats as a longer working period). The freeze
        step reconciles the two estimates so that phase 2 starts from
        a consistent per-activity duration.

        Once all time parameters are computed, time reserves smaller
        than their error bound (in absolute value) are rounded to zero.
        """
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
        """
        Compute CPM parameters for events and activities.

        This method performs topological sorting and computes either
        preliminary early/late times, final early/late times, stage
        numbers, or optimistic/pessimistic scenario times.

        Parameters
        ----------
        target : str
            What to compute. One of:

            - ``'stage'`` — topological stage of each event
            - ``'early_pre'`` — preliminary early times (forward pass,
              phase 1; the ``duration`` callback is invoked with
              ``base_time`` = activity start time)
            - ``'late_pre'`` — preliminary late times (backward pass,
              phase 1; the ``duration`` callback is invoked with
              ``base_time`` = activity end time)
            - ``'early'`` — final early times (forward pass, phase 2;
              uses the frozen ``activity.duration``, no callback)
            - ``'late'`` — final late times (backward pass, phase 2;
              uses the frozen ``activity.duration``, no callback)
            - ``'optimistic'`` — optimistic scenario times (callback
              invoked with scalar ``base_time``)
            - ``'pessimistic'`` — pessimistic scenario times (callback
              invoked with scalar ``base_time``)

        Raises
        ------
        ValueError
            If ``target`` parameter is invalid.
        RuntimeError
            If the network does not have exactly one starting event
            (i.e., the project does not have exactly one source event),
            or if it contains cycles not detected earlier.

            In addition, during the **final** ``'early'`` and ``'late'``
            passes (phase 2) the following CPM invariants are checked
            for every event and activity; a violation that exceeds the
            accumulated error bound raises :exc:`RuntimeError`:

            - ``early_start >= 0`` (event and activity)
            - ``early_end >= early_start`` (activity)
            - ``late_start >= early_start`` (activity)
            - ``late_end >= late_start`` (activity)
            - ``late >= early`` (event)

            Violations within the accumulated error bound are silently
            clamped. The corresponding messages identify the offending
            event/activity and the specific invariant.

        Notes
        -----
        The computation uses different strategies per target:

        - ``'stage'`` — each activity contributes a delta of ``1`` to
          the stage of its destination event.
        - ``'early_pre'`` / ``'late_pre'`` — delta is a time vector
          ``[RES, VAR, ERR]`` processed by ``_duration_vec``, which
          calls the ``duration`` callback with the appropriate
          ``base_time``.
        - ``'early'`` / ``'late'`` — delta is the frozen
          ``activity.duration`` (negated for ``'late'``). The callback
          is **not** invoked: forward and backward passes use exactly
          the same duration, which is what guarantees non-negative
          reserves.
        - ``'optimistic'`` / ``'pessimistic'`` — delta is a scalar and
          the callback receives ``base_time`` as a scalar.

        The algorithm maintains a count of unresolved dependencies for
        each event (``n_dep``). Events with zero dependencies are pushed
        to the processing queue. When an activity is processed, the
        destination event's dependency counter is decremented; when it
        reaches zero, the event is added to the queue.

        Numerical consistency is enforced *in place* during the
        ``'early'`` and ``'late'`` traversals (phase 2 only) by
        ``_check_and_fix_act`` / ``_check_and_fix_evt``. The phase-1
        passes (``'early_pre'`` / ``'late_pre'``) are intentionally not
        sanitized: they may temporarily hold values that violate
        ``late_pre >= early_pre``, which the freeze step reconciles
        before phase 2.

        - ``_check_and_fix_act(a, target)`` — called after each activity
          update. Ensures ``early_start >= 0`` and enforces
          ``early_end >= early_start``, ``late_start >= early_start``,
          ``late_end >= late_start``. Violations beyond the accumulated
          error bound raise :exc:`RuntimeError`; smaller violations are
          silently clamped.
        - ``_check_and_fix_evt(e, target)`` — called after each event is
          processed. Ensures ``early >= 0`` and, on the late pass,
          enforces ``late >= early``. If ``late < early`` by more than
          the accumulated error bound, :exc:`RuntimeError` is raised;
          otherwise ``late`` is raised to ``early``.

        Correcting values during the traversal keeps downstream
        calculations consistent and prevents small numerical
        inconsistencies from propagating through the rest of the
        network. As a consequence, callers do not need to run a
        separate post-processing pass to sanitize early/late values.

        Because these helpers only react to ``'early'`` and ``'late'``,
        the ``'stage'``, ``'early_pre'``, ``'late_pre'``, ``'optimistic'``
        and ``'pessimistic'`` passes are unaffected by them.

        For PERT analysis, variance is propagated during the phase-1
        passes (and scenario passes) by ``_duration_vec`` using the
        modified PERT distribution formula. The phase-2 passes reuse the
        frozen duration vector, including its ``VAR`` component.
        """
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
            """
            Compute duration vector with variance propagation for PERT analysis.

            This function handles the conversion from resource effort to actual
            duration while properly propagating variance through the network.

            Parameters
            ----------
            effort : numpy.ndarray
                Resource effort array ``[value, variance, error_bound]``
            activity : Activity
                Activity object for context
            base_time : numpy.ndarray or None
                Base time array ``[value, variance, error_bound]`` for time
                computations, or ``None`` for network post-processing
            target : str or None
                Target scenario identifier

            Returns
            -------
            numpy.ndarray
                Duration array ``[value, variance, error_bound]``

            Notes
            -----
            - If the model uses the default duration callback, the effort
              array is returned as-is (no conversion needed).
            - Otherwise, the callback is invoked on ``effort[RES]`` and the
              result is stored in ``dur[RES]``. The error bound ``dur[ERR]``
              is set to ``EPS * abs(dur[RES])``.
            - For PERT models with non-zero effort variance, the duration
              variance is computed using the shape parameters of the fitted
              beta distribution (via :func:`fit_beta`).
            """
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
            """
            Sanitize event time parameters after processing.

            Ensures non-negativity of early times and consistency between
            ``early`` and ``late`` values.

            Parameters
            ----------
            e : Event
                Event to sanitize
            target : str
                Which parameter set was just computed (``'early'`` or
                ``'late'``). Any other value is ignored.

            Raises
            ------
            RuntimeError
                On the late pass, if ``late`` falls below ``early`` by
                more than the accumulated error bound
                (``early[RES] - late[RES] > late[ERR] + early[ERR]``).
                This indicates a structural inconsistency in the network
                (e.g., a misbehaving ``duration`` callback).

            Notes
            -----
            On the ``'early'`` pass, negative ``early[RES]`` is clamped
            to zero. On the ``'late'`` pass, a violation of
            ``late >= early`` that is within the error bound is
            resolved by raising ``late`` to ``early``.
            """
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
            """
            Sanitize activity time parameters after update.

            Ensures non-negativity of early times, and enforces the
            ordering constraints ``early_end >= early_start``,
            ``late_start >= early_start`` and ``late_end >= late_start``.

            Parameters
            ----------
            a : Activity
                Activity to sanitize
            target : str
                Which parameter set was just computed (``'early'`` or
                ``'late'``). Any other value is ignored.

            Raises
            ------
            RuntimeError
                If any of the following invariants is violated beyond the
                accumulated error bound:

                - ``early_end >= early_start`` (early pass)
                - ``late_start >= early_start`` (late pass)
                - ``late_end >= late_start`` (late pass)

                Such a violation indicates a structural inconsistency in
                the network (e.g., a ``duration`` callback that maps
                non-negative effort to a negative duration, or a topology
                that contradicts the signed backward-pass deltas).

            Notes
            -----
            Violations within the accumulated error bound are silently
            clamped:

            - early pass: ``early_end = max(early_end, early_start)``
            - late pass: ``late_start = max(late_start, early_start)``,
              then ``late_end = max(late_end, late_start)``

            Negative ``early_start`` is clamped to zero regardless of the
            target.
            """
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
        """
        Add a new event to the network.

        Parameters
        ----------
        i : int
            Unique event identifier (positive integer).
        """
        self._events.append(Event(i, self))

    def _add_activity(self, wbs_id, src_id, dst_id, expected, exp_var,
                      optimistic, pessimistic, letter, data, is_dummy=False):
        """
        Add a new activity to the network.

        Parameters
        ----------
        wbs_id : int
            Work Breakdown Structure identifier (0 for dummy activities)
        src_id : int
            Source event ID (1-based index into ``self._events``)
        dst_id : int
            Destination event ID (1-based index into ``self._events``)
        expected : int or float
            Activity expected resource effort
        exp_var : int or float
            Activity effort variance
        optimistic : int or float
            Optimistic effort estimate
        pessimistic : int or float
            Pessimistic effort estimate
        letter : str
            Activity letter/code for visualization
        data : dict
            WBS data excluding fields stored as separate attributes
        is_dummy : bool, default=False
            Whether this is a dummy activity

        Raises
        ------
        TypeError
            If arguments have wrong types.
        ValueError
            If expected values are invalid (e.g., negative).
        """
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
        """String representation of the network model (events + activities)."""
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

        Notes
        -----
        Each activity dictionary is a copy of the activity's ``data``
        extended with the computed CPM/PERT fields. Each event dictionary
        is likewise a copy of the event's ``data`` extended with the
        computed CPM/PERT fields (see :meth:`Event.to_dict`). The public
        CPM fields (``early``, ``late``, ``reserve``) correspond to the
        phase-2 results; phase-1 preliminary values appear on events only
        under ``debug=True``.
        """
        activities_data = [activity.to_dict() for activity in self._activities]
        events_data = [event.to_dict() for event in self._events]

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
            ``(activities_df, events_df)`` — pandas DataFrames for
            activities and events.

        Notes
        -----
        Both DataFrames expand custom user data fields from the
        corresponding ``data`` attributes (``Activity.data`` and
        ``Event.data``) into separate columns for easy analysis.

        Activities' missing values are replaced with empty strings
        for object-typed columns (where possible, subject to the
        installed pandas version).
        """
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
        """
        Create Graphviz visualization of the CPM/PERT network.

        Parameters
        ----------
        output_path : str, optional
            Output path for saving the visualization file. The rendering
            format is determined by the file extension:

            - ``.png`` — Portable Network Graphics (default if no extension
              is given)
            - ``.svg`` — Scalable Vector Graphics
            - ``.pdf`` — Portable Document Format

            If ``None``, the graph is not rendered to a file (only the
            ``Digraph`` object is returned).
        group_by_stage : bool, default=False
            If ``True``, events are grouped into "layers" by their
            topological order (stage). If ``False``, events are placed
            freely by the Graphviz default layout.
        get_style : callable, default=_default_style
            Callback that computes styling attributes for a graph element.
            Signature: ``get_style(element) -> dict``, where ``element``
            is either an ``Event`` or an ``Activity``. The returned
            dictionary must contain keys ``color``, ``penwidth`` and
            ``fontsize``; the key ``weight`` is used for activity edges;
            the key ``tooltip`` is used for hover tooltips (rendered
            only in SVG output).

        Returns
        -------
        graphviz.Digraph
            Graphviz object for rendering or saving.

        Raises
        ------
        TypeError
            If ``get_style`` is not callable.
        ValueError
            If ``output_path`` has an unsupported extension.

        Notes
        -----
        The visualization shows:

        - Event nodes with early/late times and reserves (phase-2 values)
        - Activity edges with duration and reserve information

        The visualization uses the following coding:

        - Critical paths: red (``#ff0000``), penwidth=4, fontsize=16, weight=3
        - Sub-critical paths: orange (``#ffa000``), penwidth=3, fontsize=16,
          weight=2
        - Non-critical paths: black (``#000000``), penwidth=2, fontsize=14,
          weight=1
        - Dashed arrows: dummy activities

        Activity labels are embedded directly on edges through an invisible
        intermediate node; the label itself is drawn as a light-gray
        rounded box attached to the bend point of the edge. The label
        font size is fixed at 12 for all activity labels.

        When ``group_by_stage=True``, the topological order of events is
        preserved by clustering events with the same stage together,
        improving readability for large networks.

        Tooltips provided by ``get_style`` are rendered only in SVG output.
        """
        if not callable(get_style):
            raise TypeError(f"Parameter get_style must be callable, got {type(get_style)}")

        dot = graphviz.Digraph(node_attr={'shape': 'record', 'style': 'rounded'})
        dot.graph_attr['rankdir'] = 'LR'
        dot.graph_attr['splines'] = 'polyline'

        def _label_event(e):
            """Format event node label (early/late times, reserve)."""
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
        activity : Activity
            Activity object containing resource data
        base_time : float or None
            Base time for availability calculations during network
            traversal, or ``None`` for network post-processing. For
            optimistic/pessimistic scenarios, ``base_time`` is a scalar.
        target : str or None
            Target scenario identifier (can be used for scenario-specific
            logic)

        Returns
        -------
        float
            Actual duration with the same sign as ``effort``.
        """
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
