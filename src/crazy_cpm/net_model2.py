#!/usr/bin/env python3
# -*- coding: utf-8 -*-

#==============================================================================
"""
    CrazyCPM
    Copyright (C) 2026 anonimous

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
"""
CrazyCPM — Critical Path Method and PERT analysis library
=========================================================

This module provides comprehensive CPM (Critical Path Method) and PERT
(Program Evaluation and Review Technique) analysis capabilities for project
management with resource-aware time calculations.

Features
--------
- Multiple activity resource input formats (direct, PERT three-point, PERT two-point)
- Resource-aware duration calculations via a callback function
- Two-phase time computation for time-varying resource availability
- Automatic network diagram generation using Graphviz (PNG, SVG, PDF)
- Statistical analysis with variance and scenario propagation (EXP / VAR /
  LWR / UPR / ERR components)
- Multiple link format support
- User-defined data fields on activities and events
- Export to dictionaries and pandas DataFrames

Classes
-------
- :class:`ActivityOnArrowModel`: Main class for network analysis
- :class:`ArrowActivity`: Represents activities in the network
- :class:`Event`: Represents events/milestones in the network

Key Concepts
------------
Resource Effort vs Duration:
    - ``expected``, ``optimistic``, ``pessimistic``: Estimates of resource
      effort (e.g., person-hours, machine-hours). These are inputs to the
      model and are stored in the :attr:`ArrowActivity.expected` array.
    - ``duration``: Actual calendar time computed from effort via the
      ``duration`` callback, taking into account resource availability,
      allocation, and productivity.

Component Arrays:
    Throughout this module, effort, time, and reserve values are
    represented as numpy arrays of shape ``(5,)`` with the following
    layout:

    - ``[EXP]`` — expected (mean) value
    - ``[VAR]`` — variance (used for PERT analysis)
    - ``[LWR]`` — lower limit (optimistic scenario)
    - ``[UPR]`` — upper limit (pessimistic scenario)
    - ``[ERR]`` — upper bound on computation error

    The constants ``EXP``, ``VAR``, ``LWR``, ``UPR``, ``ERR`` are defined
    at module level. Whenever the ``LWR`` / ``UPR`` components are used,
    they represent the scenario bounds of the distribution (optimistic
    and pessimistic estimates respectively). In particular, the
    ``duration`` callback is invoked for the ``EXP``, ``UPR``, and
    ``LWR`` components of the effort estimate; all three calls receive
    the **same** ``base_time`` (the ``EXP`` component), which
    guarantees the ordering ``dur[LWR] <= dur[EXP] <= dur[UPR]``.

Two-Phase Time Computation:
    When the ``duration`` callback depends on ``base_time`` (i.e. on
    *when* the activity is scheduled), the same activity could receive
    different durations in the forward and backward passes. The model
    avoids this inconsistency by computing time parameters in two
    phases:

    1. **Preliminary (``_pre``) phase:** early/late times are computed
       from the raw effort estimates. The user's ``duration`` callback
       is invoked here (via :func:`ActivityOnArrowModel._compute_target`'s
       inner ``_duration_vec``), once for each of ``EXP``, ``UPR``, and
       ``LWR``.
    2. **Final phase:** :meth:`ArrowActivity.compute_final_duration`
       freezes a single duration value (``duration``) from the
       preliminary times, and the network is re-traversed using the
       frozen value. The backward pass applies :func:`_negative` to the
       frozen duration (negating ``EXP`` and swapping / negating
       ``LWR`` / ``UPR``). The user's callback is **not** invoked again.

    This linearizes the feedback "schedule -> duration -> schedule" in a
    single step and keeps the CPM invariants (``late >= early``)
    satisfied even for time-dependent resource models. It is a *fast
    estimator*: the "hidden critical path" effect at resource calendar
    boundaries is captured only in the first approximation (see
    :meth:`ArrowActivity.compute_final_duration`).

Usage Example
-------------
Basic usage with direct resource effort estimates::

    wbs = {
        1: {'letter': 'A', 'expected': 5.0, 'exp_var': 1.0},
        2: {'letter': 'B', 'optimistic': 3.0, 'most_likely': 4.0, 'pessimistic': 8.0}
    }
    links = [[1, 2]]
    model = ActivityOnArrowModel(wbs, links=links)
    activities_df, events_df = model.to_dataframe()

Advanced usage with resource-aware duration::

    def resource_duration(effort, activity, base_time, target):
        resource_count = 2
        productivity = 0.8
        return effort / (resource_count * productivity)

    model = ActivityOnArrowModel(wbs, links=links, duration=resource_duration)

.. note::
    This module uses a C extension (``_ccpm``) for performance-critical
    computations (AoA network construction and full dependency maps).

.. note::
    Numerical consistency of early/late time parameters is enforced
    *during* the forward and backward passes rather than as a separate
    post-processing step. This keeps downstream calculations consistent
    (see :meth:`ActivityOnArrowModel._compute_target` for details).

.. note::
    During the ``'early_pre'``, ``'late_pre'``, ``'early'`` and ``'late'``
    passes the model checks the following CPM invariants:

    - On early passes:

      - ``early_start[LWR] >= 0`` (clamped, not raised);
      - ``early_end[c] >= early_start[c]`` for ``c in {EXP, UPR, LWR}``
        (raised on violation beyond the error bound);
      - ``early_start[EXP]`` and ``early_end[EXP]`` must lie within
        ``[LWR, UPR]`` (clamped within the error bound, raised
        otherwise).

    - On late passes:

      - ``late_start[EXP] >= early_start[EXP]``;
      - ``late_end[EXP] >= late_start[EXP]``;
      - ``late_start[EXP]`` and ``late_end[EXP]`` must lie within
        ``[LWR, UPR]`` (clamped within the error bound, raised
        otherwise).

    - On both late passes (for events): ``late[EXP] >= early[EXP]``.
    - On the final ``'early'`` / ``'late'`` passes (for events):
      ``early[EXP]`` / ``late[EXP]`` must lie within ``[LWR, UPR]``
      (clamped within the error bound, raised otherwise).

    Violations within the accumulated error bound are silently clamped;
    violations beyond it raise :exc:`RuntimeError`, which signals a
    structural inconsistency (bad topology or a misbehaving ``duration``
    callback) rather than a purely numerical issue.

.. note::
    The two-phase scheme is a *first approximation* for resource
    calendars with time-varying availability. It captures the feedback
    "schedule -> duration -> schedule" with a single linearization step
    and keeps the overall cost linear in the size of the network. Full
    handling of discontinuities (activity start or end coinciding with
    the boundary of a resource availability window, i.e. the "hidden
    critical path" effect) would require either constraint programming
    or Monte-Carlo simulation over resource calendars; those are
    intentionally out of scope — this module is a fast estimator, not a
    scheduler.
"""

#==============================================================================
import graphviz
import numpy as np
import pandas as pd
import scipy
import os

import _ccpm

# Supported file formats for network visualization
_VIZ_FORMATS = ('png', 'svg', 'pdf')

# Machine epsilon
EPS = np.finfo(float).eps

# Component indices for effort / time / reserve arrays
EXP = 0  # Expected (mean) value
VAR = 1  # Variance
LWR = 2  # Lower limit (optimistic scenario)
UPR = 3  # Upper limit (pessimistic scenario)
ERR = 4  # Error bound

#==============================================================================
def _unpack_components(components):
    """
    Validate and unpack a component array.

    Parameters
    ----------
    components : array-like
        Array of shape ``(5,)`` with the layout
        ``[EXP, VAR, LWR, UPR, ERR]``.

    Returns
    -------
    tuple
        ``(EXP, VAR, LWR, UPR, ERR)`` -- five float scalars.

    Raises
    ------
    ValueError
        If ``components`` is not a 1-D array with at least five
        components.
    """
    components = np.asarray(components, dtype=float)

    if components.ndim != 1 or components.size < 5:
        raise ValueError(
            "Expected a 1D array with at least 5 components: "
            "EXP, VAR, LWR, UPR, ERR"
        )

    return components[EXP], components[VAR], components[LWR], components[UPR], components[ERR]

#==============================================================================
def fit_beta(M, D, a, b, err):
    """
    Public scalar-interface beta-distribution fitting.

    Fits the shape parameters ``(alpha, beta)`` of a beta distribution on
    ``[a, b]`` so that its mean equals ``M`` and its variance equals
    ``D``.

    Parameters
    ----------
    M : float
        Target mean (expected value)
    D : float
        Target variance
    a : float
        Lower limit (optimistic / minimum value)
    b : float
        Upper limit (pessimistic / maximum value)
    err : float
        Computation error bound (non-negative)

    Returns
    -------
    tuple
        ``(alpha, beta)`` -- shape parameters, or ``(None, None)`` if the
        distribution is treated as deterministic.

    Raises
    ------
    ValueError
        If ``a > b``, if ``M`` is outside ``[a, b]``, or if ``err < 0``.

    Notes
    -----
    The function uses the method of moments for the beta distribution.
    If the resulting ratio ``mu * (1 - mu) / var`` is less than or equal
    to 1 (which would correspond to a degenerate distribution), the
    ratio is clamped to ``1 + 2 * EPS`` to ensure positive shape
    parameters. If the resulting shape parameters are still non-positive
    (numerically degenerate), ``(None, None)`` is returned, treating the
    activity as deterministic.

    .. warning::
        The method-of-moments fit assumes that the underlying duration
        distribution is smooth and unimodal. When the ``duration``
        callback is discontinuous (e.g., an activity can jump across a
        resource availability window boundary), the resulting variance
        is systematically underestimated. This is acceptable for the
        fast-estimator use case but should be kept in mind when
        interpreting PERT quantiles near calendar boundaries.
    """
    if a > b:
        raise ValueError(
            f"Invalid bounds: optimistic ({a}) must be <= pessimistic ({b})"
        )

    if not (a <= M <= b):
        raise ValueError(
            f"Mean ({M}) must be between optimistic ({a}) "
            f"and pessimistic ({b})"
        )

    if err < 0.0:
        raise ValueError(f"Invalid error ({err}) must be >= 0.0")

    span = b - a

    ul = max(abs(a), abs(b), abs(span))

    if span <= 2.0 * EPS * ul:
        # Deterministic chain: use M without computing g
        return None, None

    # Effective error threshold: user-supplied bound or machine degeneracy
    thr = max(err, 2.0 * EPS * ul)

    if np.sqrt(D) < thr:
        # Deterministic chain, D is within computation error
        return None, None

    mu = (M - a) / span
    var = D / span ** 2
    ratio = max(mu * (1.0 - mu) / var, 1.0 + 2.0 * EPS)

    alpha = mu * (ratio - 1.0)
    beta = (1.0 - mu) * (ratio - 1.0)

    if alpha <= 0.0 or beta <= 0.0:
        return None, None

    return alpha, beta

#==============================================================================
def _fit_beta(components):
    """
    Array-interface wrapper around :func:`fit_beta`.

    Accepts the component vector ``[EXP, VAR, LWR, UPR, ERR]``, unpacks
    it, and forwards the scalar arguments to :func:`fit_beta`.

    Parameters
    ----------
    components : array-like
        Array of shape ``(5,)`` with the layout
        ``[EXP, VAR, LWR, UPR, ERR]``.

    Returns
    -------
    tuple
        ``(alpha, beta)`` -- shape parameters, or ``(None, None)`` if the
        distribution is considered deterministic. See :func:`fit_beta`
        for the full description of the fitting logic and its
        ``Notes``/``.. warning::``.
    """
    return fit_beta(*_unpack_components(components))

#==============================================================================
def calc_ppf(p, M, D, a, b, err=0.0):
    """
    Public scalar-interface quantile (PPF) calculation.

    Parameters
    ----------
    p : float
        Probability level (``0 < p < 1``)
    M : float
        Mean value
    D : float
        Variance
    a : float
        Lower limit (optimistic)
    b : float
        Upper limit (pessimistic)
    err : float, optional
        Computation error bound (default ``0.0``)

    Returns
    -------
    float
        Quantile estimate for the given probability.

        If the distribution is treated as deterministic (see
        :func:`fit_beta`), returns ``M``.

    Raises
    ------
    ValueError
        If :func:`fit_beta` raises an error due to invalid bounds or
        parameters.
    """
    alpha, beta = fit_beta(M, D, a, b, err)
    if alpha is None:
        return M

    q = scipy.stats.beta.ppf(p, alpha, beta)
    if np.isnan(q):
        return M

    return a + (b - a) * q

#==============================================================================
def _calc_ppf(p, components):
    """
    Array-interface wrapper around :func:`calc_ppf`.

    Parameters
    ----------
    p : float
        Probability level (``0 < p < 1``)
    components : array-like
        Component array ``[EXP, VAR, LWR, UPR, ERR]``

    Returns
    -------
    float
        Quantile estimate for the given probability. Returns ``EXP`` if
        the distribution is treated as deterministic (see
        :func:`fit_beta`).
    """
    return calc_ppf(p, *_unpack_components(components))

#==============================================================================
def calc_cdf(val, M, D, a, b, err=0.0):
    """
    Public scalar-interface CDF calculation.

    Parameters
    ----------
    val : float
        Value at which to evaluate the CDF
    M : float
        Mean value
    D : float
        Variance
    a : float
        Lower limit (optimistic)
    b : float
        Upper limit (pessimistic)
    err : float, optional
        Computation error bound (default ``0.0``)

    Returns
    -------
    float
        Probability ``P(X <= val)``.

        For a deterministic distribution (see :func:`fit_beta`), returns
        ``1.0`` if ``val > M`` and ``0.0`` otherwise.

    Raises
    ------
    ValueError
        If :func:`fit_beta` raises an error due to invalid bounds or
        parameters.
    """
    alpha, beta = fit_beta(M, D, a, b, err)
    if alpha is None:
        return 1.0 if val > M else 0.0

    prob = scipy.stats.beta.cdf((val - a) / (b - a), alpha, beta)
    if np.isnan(prob):
        return 0.0

    return prob

#==============================================================================
def _calc_cdf(val, components):
    """
    Array-interface wrapper around :func:`calc_cdf`.

    Parameters
    ----------
    val : float
        Value at which to evaluate the CDF
    components : array-like
        Component array ``[EXP, VAR, LWR, UPR, ERR]``

    Returns
    -------
    float
        Probability ``P(X <= val)``. Returns ``1.0`` if ``val > EXP``
        and ``0.0`` otherwise for deterministic distributions.
    """
    return calc_cdf(val, *_unpack_components(components))

#==============================================================================
def _choice(old, new, delta):
    """
    Choose between two time estimates based on certainty criteria.

    This function implements a heuristic decision mechanism for selecting
    between competing time estimates in the presence of uncertainty. The
    returned value is a 5-component array ``[EXP, VAR, LWR, UPR, ERR]``.

    Parameters
    ----------
    old : numpy.ndarray
        Existing time estimate
    new : numpy.ndarray
        New time estimate
    delta : float
        Difference between the new and old estimates. Used to decide
        which estimate is more certain.

    Returns
    -------
    numpy.ndarray
        Selected time estimate:

        - If ``delta >= e`` where ``e = new[ERR] + old[ERR]`` -- returns
          a copy of ``new`` (the difference is statistically significant).
        - If ``delta < -e`` -- returns a copy of ``old``.
        - Otherwise -- returns a mixed estimate with

          - ``EXP = 0.5 * (new[EXP] + old[EXP])``
          - ``VAR = max(old[VAR], new[VAR])``
          - ``LWR = min(old[LWR], new[LWR])``
          - ``UPR = max(old[UPR], new[UPR])``
          - ``ERR = 0.5 * (new[ERR] + old[ERR])``

    Notes
    -----
    This is an *engineering heuristic*, not a rigorous statistical
    combination of distributions. The mixed estimate deliberately
    overestimates variance (by taking the maximum), takes the widest
    scenario range (``min`` of lower bounds and ``max`` of upper bounds),
    and halves the error bound to reflect the loss of precision when two
    nearly equal estimates are combined.
    """
    err = new[ERR] + old[ERR]
    if delta >= err:
        return new.copy()  # Certain result
    elif delta >= -err:
        # Uncertain result, use mixing
        ret = np.zeros((5,), dtype=float)
        ret[EXP] = 0.5 * (new[EXP] + old[EXP])
        ret[VAR] = max(old[VAR], new[VAR])
        ret[LWR] = min(old[LWR], new[LWR])
        ret[UPR] = max(old[UPR], new[UPR])
        ret[ERR] = 0.5 * err
        return ret
    else:
        return old.copy()  # Certain result

#==============================================================================
def _negative(vec):
    """
    Return the "negative" of a component vector.

    This is the signed-negation operator used by the backward (late) pass.
    It reflects the distribution around zero: the expected value is
    negated, the scenario bounds are swapped and negated (so that the new
    ``LWR <= UPR`` still holds), and the variance / error bound are left
    unchanged.

    Parameters
    ----------
    vec : numpy.ndarray
        Component array ``[EXP, VAR, LWR, UPR, ERR]``

    Returns
    -------
    numpy.ndarray
        New component array ``[-EXP, VAR, -UPR, -LWR, ERR]``
    """
    ret = np.zeros((5,), dtype=float)
    ret[EXP] = -vec[EXP]
    ret[LWR] = -vec[UPR]
    ret[UPR] = -vec[LWR]

    ret[VAR] = vec[VAR]
    ret[ERR] = vec[ERR]
    return ret

#==============================================================================
def _default_duration(effort, activity, base_time, target):
    """
    Default duration callback function.

    Provides the default behavior where duration equals effort. Used when
    the user does not supply a custom ``duration`` callback to
    :class:`ActivityOnArrowModel`.

    Parameters
    ----------
    effort : float
        Resource effort estimate:

        - Positive number or zero for forward pass (early times calculation)
        - Negative number or zero for backward pass (late times calculation)
    activity : ArrowActivity
        Activity object for context-aware calculations
    base_time : float
        Base time from which the activity starts -- the ``EXP`` component
        of the event's base time (the callback is invoked only during
        the preliminary ``'early_pre'`` / ``'late_pre'`` passes).
    target : str
        Target scenario identifier. Currently one of ``'early_pre'`` or
        ``'late_pre'`` (the only passes in which the callback is
        invoked).

    Returns
    -------
    float
        Activity duration with the same sign as ``effort``.

    Notes
    -----
    The function must preserve the sign of the ``effort`` parameter to
    ensure correct forward and backward pass calculations in the network.
    ``base_time`` is always a ``float``: the callback is invoked only
    during the preliminary (``'early_pre'`` / ``'late_pre'``) passes, and
    the model never passes ``None``.

    Examples
    --------
    Simple productivity model::

        def custom_duration(effort, activity, base_time, target):
            team_size = activity.data.get('team_size', 1)
            productivity = activity.data.get('productivity', 1.0)
            abs_duration = abs(effort) / (team_size * productivity)
            return abs_duration if effort >= 0 else -abs_duration
    """
    # Return effort directly (default: duration = effort)
    # This preserves the sign of effort for forward/backward pass calculations
    return effort

#==============================================================================
def _default_style(element):
    """
    Determine Graphviz styling (color, penwidth, fontsize, weight) based
    on criticality.

    Parameters
    ----------
    element : Event or ArrowActivity
        Graph element whose style is being computed. The styling decision
        is based on the element's time reserve and the probability that
        the reserve is positive:

        - For an :class:`Event` the probability is
          ``element.early_prob(element.late[EXP])``
        - For an :class:`ArrowActivity` the probability is
          ``element.early_end_prob(element.late_end[EXP])``

        The threshold probability is taken from ``element.model.p``.

    Returns
    -------
    dict
        Dictionary with styling attributes for Graphviz. Contains keys
        ``color``, ``penwidth``, ``fontsize``, ``weight``, ``tooltip``.

        Elements that are formally critical (``abs(res[EXP]) <=
        res[ERR]``) are drawn red. Elements with ``prob_reserve <
        prob_thr`` are drawn as sub-critical (orange). Remaining elements
        are drawn as non-critical (black).

        Tooltips are rendered only in SVG output.

    Raises
    ------
    TypeError
        If ``element`` is neither an :class:`Event` nor an
        :class:`ArrowActivity`.
    """
    if isinstance(element, Event):
        res = element.reserve
        prob_reserve = element.early_prob(element.late[EXP])
        tooltip  = f"Event {element.id}\n"
        tooltip += f"early = {element.early[EXP]:.2f}\n"
        tooltip += f"late = {element.late[EXP]:.2f}\n"
        tooltip += f"reserve = {element.reserve[EXP]:.2f}\n"
    elif isinstance(element, ArrowActivity):
        res = element.reserve
        prob_reserve = element.early_end_prob(element.late_end[EXP])
        tooltip = element.letter
        if 'name' in element.data:
            tooltip += ' ' + element.data['name']

    else:
        raise TypeError(
            f"element must be Event or Activity, got {type(element)}"
        )

    prob_thr = element.model.p

    if abs(res[EXP]) <= res[ERR]:
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
class ArrowActivity(object):
    """
    Represents an activity (task) in the network model.

    This class stores all information about a network activity including
    timing parameters, statistical properties, and custom data.

    Parameters
    ----------
    id : int
        Unique activity identifier (internal, assigned by
        :class:`ActivityOnArrowModel`)
    wbs_id : int
        Work Breakdown Structure identifier (``0`` for dummy activities)
    letter : str
        Activity letter / code for visualization
    model : ActivityOnArrowModel
        Parent network model instance
    src : Event
        Source event of the activity
    dst : Event
        Destination event of the activity
    expected : numpy.ndarray or None, optional
        Resource effort estimate as a 5-component float array
        ``[EXP, VAR, LWR, UPR, ERR]``. If ``None``, a zero array is used
        (used for dummy activities).
    data : dict, optional
        Additional activity data from WBS (custom user fields)
    is_dummy : bool, default=False
        Whether this is a dummy activity (used in AoA network construction)

    Attributes
    ----------
    id : int
        Unique activity identifier
    wbs_id : int
        WBS identifier (``0`` for dummy activities)
    letter : str
        Activity letter / code
    model : ActivityOnArrowModel
        Parent network model
    src : Event
        Source event (property, setter updates the event's activity lists)
    dst : Event
        Destination event (property, setter updates the event's activity
        lists)
    expected : numpy.ndarray
        Resource effort estimate array ``[EXP, VAR, LWR, UPR, ERR]``
    data : dict
        Additional activity data
    early_start : numpy.ndarray
        Early start time ``[EXP, VAR, LWR, UPR, ERR]``. In the final
        phase the ``LWR`` / ``UPR`` components hold the optimistic and
        pessimistic scenario start times respectively.
    late_start : numpy.ndarray
        Late start time ``[EXP, VAR, LWR, UPR, ERR]``
    early_end : numpy.ndarray
        Early end time ``[EXP, VAR, LWR, UPR, ERR]``
    late_end : numpy.ndarray
        Late end time ``[EXP, VAR, LWR, UPR, ERR]``
    reserve : numpy.ndarray
        Time reserve ``[EXP, VAR, LWR, UPR, ERR]``. The ``LWR`` / ``UPR``
        components hold the scenario reserve bounds.
    duration : numpy.ndarray
        Frozen activity duration ``[EXP, VAR, LWR, UPR, ERR]``, computed
        by :meth:`compute_final_duration` from the preliminary (``_pre``)
        phase results. The final forward pass (``'early'``) uses it as-is;
        the final backward pass (``'late'``) uses
        :func:`_negative` of the same array. This is also the value
        reported by :meth:`to_dict` as the actual activity duration.
    is_dummy : bool
        Whether this is a dummy activity

    Notes
    -----
    - Effort, time and reserve arrays follow the format
      ``[EXP, VAR, LWR, UPR, ERR]``.
    - The :attr:`duration` attribute holds the frozen duration computed
      by :meth:`compute_final_duration`. It is used as the delta in both
      final passes of :meth:`ActivityOnArrowModel._compute_target`
      (``'early'`` uses it as-is, ``'late'`` applies :func:`_negative`).
      It is also the value reported as the actual activity duration.
    - Setters for :attr:`src` and :attr:`dst` maintain the corresponding
      event's internal ``_in_activities`` / ``_out_activities`` lists.
      Do not mutate those lists directly.
    - During the network traversal, the CPM invariants are enforced by
      :meth:`ActivityOnArrowModel._compute_target` (see its docstring for
      the exact invariants and their handling). After a successful model
      construction these relations hold for the activity's time
      attributes as well, up to the accumulated error bound.

    Raises
    ------
    TypeError
        If any argument has an incorrect type.
    """

    def __init__(self, id, wbs_id, letter, model, src, dst, expected=None, data=None, is_dummy=False):
        # Validate types and values (explicit checks instead of asserts)
        if not isinstance(id, int):
            raise TypeError(f"id must be int, got {type(id)}")

        if not isinstance(wbs_id, int):
            raise TypeError(f"wbs_id must be int, got {type(wbs_id)}")

        if not isinstance(letter, str):
            raise TypeError(f"letter must be str, got {type(letter)}")

        if not isinstance(model, ActivityOnArrowModel):
            raise TypeError(f"model must be ActivityOnArrowModel, got {type(model)}")

        if not isinstance(src, Event):
            raise TypeError(f"src must be Event, got {type(src)}")

        if not isinstance(dst, Event):
            raise TypeError(f"dst must be Event, got {type(dst)}")

        if expected is None:
            expected_arr = np.zeros((5,), dtype=float)

        elif (isinstance(expected, np.ndarray)
              and expected.shape == (5,)
              and np.issubdtype(expected.dtype, np.floating)):
            expected_arr = expected.astype(float, copy=True)

        else:
            raise TypeError(
                "expected must be a float numpy.ndarray of shape (5,), "
                f"got {type(expected).__name__} with shape "
                f"{getattr(expected, 'shape', None)} and dtype "
                f"{getattr(expected, 'dtype', None)}"
            )

        self.id = id
        self.wbs_id = wbs_id
        self.letter = letter
        self.data = data if data is not None else {}
        self._is_dummy = is_dummy

        self.model = model

        # Events
        self._src = src
        self._src._out_activities.append(self)

        self._dst = dst
        self._dst._in_activities.append(self)

        # Effort
        self.expected = expected_arr.copy()

        # CPM/PERT parameters
        self.duration    = np.zeros_like(self.expected)
        self.early_start = np.zeros_like(self.expected)
        self.late_start  = np.zeros_like(self.expected)
        self.early_end   = np.zeros_like(self.expected)
        self.late_end    = np.zeros_like(self.expected)
        self.reserve     = np.zeros_like(self.expected)

    #==========================================================================
    @property
    def src(self):
        """Source event of the activity.

        Setting this property updates the corresponding event's internal
        activity lists (removes from old, appends to new).
        """
        return self._src

    @src.setter
    def src(self, new_src):
        if isinstance(self._src, Event):
            self._src._out_activities.remove(self)

        self._src = new_src
        new_src._out_activities.append(self)

    #==========================================================================
    @property
    def dst(self):
        """Destination event of the activity.

        Setting this property updates the corresponding event's internal
        activity lists (removes from old, appends to new).
        """
        return self._dst

    @dst.setter
    def dst(self, new_dst):
        if isinstance(self._dst, Event):
            self._dst._in_activities.remove(self)

        self._dst = new_dst
        new_dst._in_activities.append(self)

    #==========================================================================
    @property
    def is_dummy(self):
        """Whether this activity is a dummy (used in AoA construction)."""
        return self._is_dummy

    #==========================================================================
    @property
    def early_start_pqe(self):
        """
        Get the early-start probabilistic quantile estimate.

        Returns
        -------
        float
            Early-start quantile for the model's probability level
            ``p``, computed from the full ``[EXP, VAR, LWR, UPR, ERR]``
            component array of :attr:`early_start`.
        """
        return _calc_ppf(self.model.p, self.early_start)

    #==========================================================================
    def early_start_prob(self, val):
        """
        Get the probability that the early start is less than ``val``.

        Parameters
        ----------
        val : float
            Value to compare against

        Returns
        -------
        float
            ``P(early_start < val)``
        """
        return _calc_cdf(val, self.early_start)

    #==========================================================================
    @property
    def early_end_pqe(self):
        """
        Get the early-end probabilistic quantile estimate.

        Returns
        -------
        float
            Early-end quantile for the model's probability level ``p``.
        """
        return _calc_ppf(self.model.p, self.early_end)

    #==========================================================================
    def early_end_prob(self, val):
        """
        Get the probability that the early end is less than ``val``.

        Parameters
        ----------
        val : float
            Value to compare against

        Returns
        -------
        float
            ``P(early_end < val)``
        """
        return _calc_cdf(val, self.early_end)

    #==========================================================================
    def __repr__(self):
        """String representation of the activity (a dict-like view)."""
        return str(self.to_dict())

    #==========================================================================
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
            - ``expected``: Activity expected resource effort (``EXP``)
            - ``duration``: Frozen activity duration (``EXP``)
            - ``early_start``, ``late_start``, ``early_end``, ``late_end``:
              timing parameters (``EXP``)
            - ``reserve``: time reserve (``EXP``)
            - ``is_dummy``: whether this is a dummy activity
            - All user-defined fields from ``self.data``
            - Additional PERT fields if applicable

        Notes
        -----
        PERT-specific fields are only included when PERT analysis is
        enabled (``model.is_pert``):

        - ``exp_var`` -- variance of the effort estimate
        - ``variance`` -- variance of the frozen duration
        - ``optimistic`` -- optimistic effort (``expected[LWR]``)
        - ``opt_start`` -- optimistic scenario start (``early_start[LWR]``)
        - ``opt_end`` -- optimistic scenario end (``early_end[LWR]``)
        - ``pessimistic`` -- pessimistic effort (``expected[UPR]``)
        - ``pes_start`` -- pessimistic scenario start (``early_start[UPR]``)
        - ``pes_end`` -- pessimistic scenario end (``early_end[UPR]``)
        - ``early_start_var``, ``early_end_var`` -- variances of start/end
        - ``early_start_pqe``, ``early_end_pqe`` -- probabilistic
          quantile estimates
        - ``late_end_prob`` -- probability that ``early_end`` is less than
          ``late_end[EXP]``

        Debug-specific fields (only when ``model.debug`` is true):
        ``early_start_err``, ``late_start_err``, ``early_end_err``,
        ``late_end_err`` -- error bounds of the corresponding timing
        parameters.

        The returned dictionary is a copy of ``self.data`` extended with
        the computed CPM/PERT fields. If a user field name collides with
        a computed field name, the computed value takes precedence.

        .. note::
            The ``duration`` field is the frozen value stored in
            :attr:`duration` (see :meth:`compute_final_duration`).
            During the final backward pass :func:`_negative` is applied
            *locally* inside
            :func:`ActivityOnArrowModel._compute_target`; the attribute
            itself always holds the forward-pass value.
        """
        # Start with user data
        ret = self.data.copy()

        ret['id'      ] = self.id
        ret['wbs_id'  ] = self.wbs_id
        ret['letter'  ] = self.letter
        ret['src_id'  ] = self.src.id
        ret['dst_id'  ] = self.dst.id
        ret['expected'] = self.expected[EXP] # Resource effort estimate
        ret['duration'] = self.duration[EXP] # Actual duration
        ret['is_dummy'] = self._is_dummy

        # CPM timing parameters
        ret['early_start'] = self.early_start[EXP]
        ret['late_start' ] = self.late_start[EXP]
        ret['early_end'  ] = self.early_end[EXP]
        ret['late_end'   ] = self.late_end[EXP]
        ret['reserve'    ] = self.reserve[EXP]

        if self.model.is_pert:
            # PERT analysis fields
            ret['exp_var'   ] = self.expected[VAR]    # Variance of effort estimate
            ret['variance'  ] = self.duration[VAR]    # Variance of duration

            ret['optimistic'] = self.expected[LWR]    # Optimistic effort
            ret['opt_start' ] = self.early_start[LWR] # Optimistic start time
            ret['opt_end'   ] = self.early_end[LWR]   # Optimistic end time

            ret['pessimistic'] = self.expected[UPR]    # Pessimistic effort
            ret['pes_start'  ] = self.early_start[UPR] # Pessimistic start time
            ret['pes_end'    ] = self.early_end[UPR]   # Pessimistic end time

            ret['early_start_var'] = self.early_start[VAR]
            ret['early_end_var'  ] = self.early_end[VAR]
            ret['early_start_pqe'] = self.early_start_pqe
            ret['early_end_pqe'  ] = self.early_end_pqe

            ret['late_end_prob'  ] = self.early_end_prob(self.late_end[EXP])

        if self.model.debug:
            # Debug information
            ret['early_start_err'] = self.early_start[ERR]
            ret['late_start_err' ] = self.late_start[ERR]
            ret['early_end_err'  ] = self.early_end[ERR]
            ret['late_end_err'   ] = self.late_end[ERR]

        return ret

    #==========================================================================
    def compute_final_duration(self):
        """
        Freeze forward- and backward-pass durations from the preliminary
        (``_pre``) phase results.

        This method is the **bridge between the two phases**: it is
        called at the end of
        :meth:`ActivityOnArrowModel._compute_time_params_pre`, and the
        second phase (:meth:`ActivityOnArrowModel._compute_time_params`)
        then uses the frozen value instead of re-invoking the user's
        ``duration`` callback. It reads the preliminary times
        (``early_start``/``early_end``, ``late_start``/``late_end`` as
        populated by the ``_pre`` passes), derives a single fixed
        duration, and stores it in :attr:`duration`.

        Freezing is required whenever the ``duration`` callback depends
        on ``base_time`` -- i.e. when resource availability varies over
        time (work calendars, shift patterns, day/night cycles, etc.).
        Without freezing, the same activity would receive different
        durations in the forward and backward passes, which would break
        the CPM invariants (``late_start >= early_start``,
        ``late_end >= late_start``). Freezing restores *internal
        consistency* of the snapshot at a cost linear in the network
        size.

        The preliminary forward duration is computed component-wise as
        ``early = early_end - early_start`` (with error bounds summed),
        and the backward duration as

        - ``late[EXP] = late_end[EXP] - late_start[EXP]``
        - ``late[LWR] = late_end[UPR] - late_start[UPR]``
        - ``late[UPR] = late_end[LWR] - late_start[LWR]``
        - ``late[VAR] = late_start[VAR] - late_end[VAR]``
        - ``late[ERR] = late_end[ERR] + late_start[ERR]``

        The most certain of the two estimates is then selected via
        :func:`_choice` and stored in :attr:`duration`.

        Because the **same** frozen ``duration`` is used by both final
        passes (the forward pass uses it as-is, the backward pass applies
        :func:`_negative`), the start and end reserves are equal *by
        construction*::

            start_res = late_start - early_start
            end_res   = late_end   - early_end
            start_res - end_res == 0

        Notes
        -----
        The method does **not** clamp the frozen duration to a
        scenario range; the previous generation of this class performed
        an additional scenario-range clamp and ``b``-growth step, but the
        current implementation keeps :attr:`duration` exactly as chosen
        by :func:`_choice`. Any residual calendar-induced inconsistency
        therefore manifests as a *shift* of the frozen duration away
        from the raw forward-pass value, not as a start/end reserve
        mismatch.

        The ``VAR`` component is inherited from the chosen estimate and
        is not modified by this method.
        """
        # Early duration calculation
        # Note:
        # early_end[[EXP, VAR, UPR, LWR]] == early_start[[EXP, VAR, UPR, LWR]] + early[[EXP, VAR, UPR, LWR]]
        early = self.early_end - self.early_start
        early[ERR] = self.early_end[ERR] + self.early_start[ERR]

        # Late duration calculation
        # Note:
        # late_start[EXP] == late_end[EXP] - late[EXP]
        # late_start[UPR] == late_end[UPR] - late[LWR]
        # late_start[LWR] == late_end[LWR] - late[UPR]
        # late_start[VAR] == late_end[VAR] + late[VAR]
        late = np.zeros_like(early)
        late[EXP] = self.late_end[EXP]   - self.late_start[EXP]
        late[LWR] = self.late_end[UPR]   - self.late_start[UPR]
        late[UPR] = self.late_end[LWR]   - self.late_start[LWR]
        late[VAR] = self.late_start[VAR] - self.late_end[VAR]
        late[ERR] = self.late_end[ERR]   + self.late_start[ERR]

        # Choose most certain estimate
        self.duration = _choice(early, late, late[EXP] - early[EXP])


#==============================================================================
class Event(object):
    """
    Represents an event (milestone) in the network model.

    Events mark the beginning or end of activities and are used to
    calculate critical paths and timing parameters.

    Parameters
    ----------
    id : int
        Unique event identifier
    model : ActivityOnArrowModel
        Parent network model instance

    Attributes
    ----------
    id : int
        Unique event identifier (renumerated by
        :class:`ActivityOnArrowModel` after construction, according to
        topological stage order)
    model : ActivityOnArrowModel
        Parent network model
    data : dict
        Additional user data associated with the event. This dictionary
        holds arbitrary user-defined fields; they are merged into the
        output of :meth:`to_dict` (and therefore appear as separate
        columns in the events DataFrame produced by
        :meth:`ActivityOnArrowModel.to_dataframe`).
    early_pre : numpy.ndarray
        Preliminary early time ``[EXP, VAR, LWR, UPR, ERR]`` (phase-1
        result; used by :meth:`ArrowActivity.compute_final_duration`
        to freeze durations)
    late_pre : numpy.ndarray
        Preliminary late time ``[EXP, VAR, LWR, UPR, ERR]`` (phase-1
        result). This is an internal service value; the public API value
        is :attr:`late`.
    early : numpy.ndarray
        Final early time ``[EXP, VAR, LWR, UPR, ERR]`` (phase-2 result).
        The ``LWR`` / ``UPR`` components hold the optimistic and
        pessimistic scenario times respectively.
    late : numpy.ndarray
        Final late time ``[EXP, VAR, LWR, UPR, ERR]`` (phase-2 result)
    reserve : numpy.ndarray
        Time reserve ``[EXP, VAR, LWR, UPR, ERR]``
    stage : int
        Event stage in topological order (``0`` = start event)

    Raises
    ------
    TypeError
        If ``id`` is not an integer or ``model`` is not an
        :class:`ActivityOnArrowModel` instance.

    Notes
    -----
    This class uses identity-based ``__eq__``/``__hash__`` (default from
    ``object``). Do NOT define ``__eq__`` without also defining a
    compatible ``__hash__`` -- instances are used as dict keys in
    ``ActivityOnArrowModel._compute_target``. Do NOT add ``@dataclass``
    without ``eq=False``.

    The :attr:`in_activities` and :attr:`out_activities` properties
    return **immutable tuples** (snapshots) of the current activity
    lists. The underlying lists are maintained by ``ArrowActivity.src`` /
    ``ArrowActivity.dst`` setters. To mutate the topology, reassign
    ``activity.src`` or ``activity.dst`` -- never mutate the internal
    lists directly.

    User-defined keys in ``data`` must not collide with the reserved
    field names inserted by :meth:`to_dict` (``id``, ``stage``,
    ``early``, ``late``, ``reserve``, ``optimistic``, ``pessimistic``,
    ``early_var``, ``early_pqe``, ``late_prob``, ``early_err``,
    ``late_err``, ``early_pre``, ``early_pre_var``, ``early_pre_err``,
    ``late_pre``, ``late_pre_var``, ``late_pre_err``), otherwise the
    computed values take precedence in the output dictionary.

    During the network traversal, the CPM invariant ``late >= early``
    is enforced by :meth:`ActivityOnArrowModel._compute_target` for both
    the preliminary (``late_pre >= early_pre``) and the final
    (``late >= early``) passes (``EXP`` component only). After
    successful model construction, event late times therefore always
    satisfy ``late[EXP] >= early[EXP] - (late[ERR] + early[ERR])`` and,
    in the common case, ``late[EXP] >= early[EXP]``.
    """

    def __init__(self, id, model):
        if not isinstance(id, int):
            raise TypeError(f"id must be int, got {type(id)}")
        if not isinstance(model, ActivityOnArrowModel):
            raise TypeError(f"model must be ActivityOnArrowModel, got {type(model)}")

        self.id = id
        self.model = model
        self._in_activities = []
        self._out_activities = []

        # CPM time parameters (calculated later)
        #
        # _pre values are the phase-1 (preliminary) results. They are
        # consumed by ArrowActivity.compute_final_duration() to freeze
        # the per-activity durations, and are kept around for debug
        # inspection only. The public API is exposed via the final
        # 'early' / 'late' arrays.
        self.early_pre = np.zeros((5,), dtype=float)
        self.late_pre = np.zeros((5,), dtype=float)
        self.early = np.zeros((5,), dtype=float)
        self.late = np.zeros((5,), dtype=float)
        self.reserve = np.zeros((5,), dtype=float)
        self.stage = 0

        self.data = {} #User data

    #==========================================================================
    @property
    def in_activities(self):
        """Activities entering this event.

        Returns an **immutable tuple** (snapshot) of the current incoming
        activities. The underlying list is maintained by the
        ``ArrowActivity.dst`` setter. To mutate topology, reassign
        ``activity.dst = new_event`` -- do not attempt to modify the
        returned tuple (it is immutable) or the internal list directly.
        """
        return tuple(self._in_activities)

    #==========================================================================
    @property
    def out_activities(self):
        """Activities leaving this event.

        Returns an **immutable tuple** (snapshot) of the current outgoing
        activities. The underlying list is maintained by the
        ``ArrowActivity.src`` setter. To mutate topology, reassign
        ``activity.src = new_event`` -- do not attempt to modify the
        returned tuple (it is immutable) or the internal list directly.
        """
        return tuple(self._out_activities)

    #==========================================================================
    @property
    def early_pqe(self):
        """
        Get the early-time probabilistic quantile estimate.

        Returns
        -------
        float
            Early-time quantile for the model's probability level ``p``.
        """
        return _calc_ppf(self.model.p, self.early)

    #==========================================================================
    def early_prob(self, val):
        """
        Get the probability that the early time is less than ``val``.

        Parameters
        ----------
        val : float
            Value to compare against

        Returns
        -------
        float
            ``P(early < val)``
        """
        return _calc_cdf(val, self.early)

    #==========================================================================
    def __repr__(self):
        """String representation of the event (a dict-like view)."""
        return str(self.to_dict())

    #==========================================================================
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
            - ``early``: Early time (phase-2, ``EXP``)
            - ``late``: Late time (phase-2, ``EXP``)
            - ``reserve``: Time reserve (``EXP``)
            - Additional PERT fields if applicable

        Notes
        -----
        PERT-specific fields are only included when PERT analysis is
        enabled (``model.is_pert``):

        - ``optimistic`` -- optimistic scenario early time
          (``early[LWR]``)
        - ``pessimistic`` -- pessimistic scenario early time
          (``early[UPR]``)
        - ``early_var`` -- variance of the early time
        - ``early_pqe`` -- probabilistic quantile of the early time
        - ``late_prob`` -- probability that ``early`` is less than
          ``late[EXP]``

        Debug-specific fields (only when ``model.debug`` is true):

        - ``early_err``, ``late_err`` -- phase-2 error bounds.
        - ``early_pre``, ``early_pre_var``, ``early_pre_err`` -- phase-1
          preliminary early time and its statistics.
        - ``late_pre``, ``late_pre_var``, ``late_pre_err`` -- phase-1
          preliminary late time and its statistics. Note that
          ``late_pre`` is an internal service value; the public API
          value is ``late``.

        The returned dictionary is a copy of ``self.data`` extended with
        the computed CPM/PERT fields. Reserved field names (listed in
        the class-level Notes) must not be used as user data keys,
        otherwise the computed values take precedence in the output
        dictionary.
        """
        # Start with user data
        ret = self.data.copy()

        # Basic CPM parameters
        ret['id'     ] = self.id
        ret['stage'  ] = self.stage
        ret['early'  ] = self.early[EXP]
        ret['late'   ] = self.late[EXP]
        ret['reserve'] = self.reserve[EXP]

        if self.model.is_pert:
            # PERT analysis fields
            ret['optimistic' ] = self.early[LWR]
            ret['pessimistic'] = self.early[UPR]
            ret['early_var'  ] = self.early[VAR]
            ret['early_pqe'  ] = self.early_pqe
            ret['late_prob'  ] = self.early_prob(self.late[EXP])

        if self.model.debug:
            # Debug information
            ret['early_err'] = self.early[ERR]
            ret['late_err' ] = self.late[ERR]

            ret['early_pre'     ] = self.early_pre[EXP]
            ret['early_pre_var' ] = self.early_pre[VAR]
            ret['early_pre_err' ] = self.early_pre[ERR]

            ret['late_pre'     ] = self.late_pre[EXP]
            ret['late_pre_var' ] = self.late_pre[VAR]
            ret['late_pre_err' ] = self.late_pre[ERR]

        return ret

#==============================================================================
def _calculate_action_time_params(work_data, default_risk=0.3):
    """
    Calculate the resource effort component array from various input
    formats.

    Supports three formats, in order of priority:

    1. **Three-point PERT**: uses ``optimistic``, ``most_likely``, and
       ``pessimistic`` effort estimates with

       - ``mean = (optimistic + 4 * most_likely + pessimistic) / 6``
       - ``variance = ((pessimistic - optimistic) / 6) ** 2``

    2. **Two-point PERT**: uses ``optimistic`` and ``pessimistic`` effort
       estimates with

       - ``mean = (3 * optimistic + 2 * pessimistic) / 5``
       - ``variance = ((pessimistic - optimistic) / 5) ** 2``

    3. **Direct parameters**: uses directly provided ``expected`` and
       optional ``exp_var``. If ``exp_var`` is positive, the implicit
       ``optimistic`` / ``pessimistic`` bounds are derived using
       ``default_risk``:

       - ``d = 3 * sqrt(exp_var)``
       - ``optimistic = mean - min(d, default_risk * mean)``
       - ``pessimistic = optimistic + 2 * d``

       If ``exp_var == 0``, the distribution is treated as
       deterministic: ``optimistic = pessimistic = mean``.

    After the raw component array is built, a deterministic check is
    performed: if ``sqrt(VAR) <= ERR``, the activity is forced to
    deterministic (``LWR = UPR = EXP``, ``VAR = 0``). Otherwise, an
    MPERT-style mode correction is applied to make the beta fit
    well-posed:

    - ``gamma = (mean - a) * (b - mean) / variance - 3`` (clamped from
      below by ``6 * EPS``);
    - if ``mean`` is too close to ``a``, ``mean`` is pushed inside by
      ``2 * ERR``;
    - if ``mean`` is too close to ``b``, ``b`` is grown so that the mean
      stays strictly inside;
    - ``variance`` is recomputed as ``(mean - a) * (b - mean) /
      (gamma + 3)``.

    Parameters
    ----------
    work_data : dict
        Dictionary containing work data with one of these combinations:

        - For three-point PERT: ``optimistic``, ``most_likely``,
          ``pessimistic``
        - For two-point PERT: ``optimistic``, ``pessimistic``
        - For direct parameters: ``expected``, ``exp_var`` (optional)
    default_risk : float, default=0.3
        Default risk factor (in ``[0, 1]``) used to derive bounds when
        only ``expected`` and ``exp_var`` are provided.

    Returns
    -------
    numpy.ndarray
        Component array of shape ``(5,)`` with layout
        ``[EXP, VAR, LWR, UPR, ERR]``, representing **resource effort**
        (not duration). ``ERR`` is set to
        ``EPS * max(|LWR|, |UPR|, |UPR - LWR|)``.

    Raises
    ------
    TypeError
        If ``work_data`` is not a dict, or if any numeric field cannot
        be converted to ``float``.
    ValueError
        If ``default_risk`` is out of range, if insufficient data is
        provided, or if estimates are invalid (e.g. the ordering
        ``optimistic <= most_likely <= pessimistic`` is not satisfied,
        or negative mean/variance are given).

    Examples
    --------
    Three-point PERT effort estimation::

        data = {'optimistic': 5, 'most_likely': 7, 'pessimistic': 12}
        effort = _calculate_action_time_params(data)
        print(effort[EXP], effort[VAR])

    Two-point PERT effort estimation::

        data = {'optimistic': 3, 'pessimistic': 8}
        effort = _calculate_action_time_params(data)

    Direct parameters::

        data = {'expected': 6.5, 'exp_var': 0.5}
        effort = _calculate_action_time_params(data)
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

    # 2. Check for two-point PERT estimation (medium priority)
    elif all(key in work_data for key in ['optimistic', 'pessimistic']):
        # Normalize inputs to float
        a = float(work_data['optimistic'])
        b = float(work_data['pessimistic'])

        # Validate inputs
        if not (a <= b):
            raise ValueError(f"Invalid PERT estimates: must satisfy optimistic <= pessimistic. Got: {a}, {b}")

        mean = (3 * a + 2 * b) / 5
        variance = ((b - a) / 5) ** 2

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
        else:
            a = mean
            b = mean

    # 4. Error - insufficient data
    else:
        raise ValueError(f"Insufficient data for determining action effort parameters. Available keys: {list(work_data.keys())}")

    ret = np.zeros((5,), dtype=float)
    ret[EXP] = mean
    ret[VAR] = variance
    ret[LWR] = a
    ret[UPR] = b
    ret[ERR] = EPS * max(abs(a), abs(b), abs(b - a))

    gamma_nom = (mean - a) * (b - mean)

    if np.sqrt(ret[VAR]) <= ret[ERR] or ret[VAR] <= np.sqrt(EPS) * gamma_nom:
        # Deterministic activity
        ret[LWR] = ret[EXP]
        ret[UPR] = ret[EXP]
        ret[VAR] = 0.0
        return ret

    # Make sure that mode is in (a,b)
    # Use MPERT math for corrections
    gamma = gamma_nom / variance - 3.0

    # Gamma must be > 0, so:
    # the first part of gamma must be at least 3.0 * (1 + 2.0 * EPS)
    # thus gamma threshold is 6.0 * EPS
    if gamma < 6.0 * EPS:
        gamma = 6.0 * EPS

    # Now values corrections
    thr = a + (b - a) / (gamma + 2.0)
    if mean <= thr:
        mean = thr + 2.0 * ret[ERR]

    if mean >= b - (b - a) / (gamma + 2.0):
        b = ((gamma + 2.0) * mean - a) / (gamma + 1.0) + 2.0 * ret[ERR]

    variance = (mean - a) * (b - mean) / (gamma + 3.0)

    ret[EXP] = mean
    ret[VAR] = variance
    ret[UPR] = b

    return ret

#==============================================================================
def _sanitize_ordering(vec, label, where, target):
    """
    Enforce the component ordering ``LWR <= EXP <= UPR`` on a component
    vector in place.

    The function handles three cases:

    1. The interval is inverted beyond the error bound
       (``LWR - UPR > ERR``): raises :exc:`RuntimeError`.
    2. The interval is inverted within the error bound: collapses
       ``LWR`` and ``UPR`` to ``EXP`` (after verifying that ``EXP`` also
       lies within ``ERR`` of the inverted interval).
    3. ``EXP`` lies outside ``[LWR, UPR]`` but within the error bound:
       shifts the whole interval so that ``EXP`` falls inside, preserving
       its width. A shift larger than ``ERR`` raises
       :exc:`RuntimeError`.

    Parameters
    ----------
    vec : numpy.ndarray
        Component array ``[EXP, VAR, LWR, UPR, ERR]``, modified in place.
    label : str
        Human-readable element label used in error messages (e.g.
        ``"Activity A"`` or ``"Event 3"``).
    where : str
        Name of the quantity being sanitized (e.g. ``'early start'``,
        ``'late end'``), used in error messages.
    target : str
        Name of the current pass (e.g. ``'early_pre'``, ``'late'``),
        used in error messages.

    Raises
    ------
    RuntimeError
        If the violation exceeds the accumulated error bound.
    """
    # Step 1: interval inverted beyond the error bound
    if vec[LWR] - vec[UPR] > vec[ERR]:
        raise RuntimeError(
            f"{label} has abnormal {target} {where} interval "
            f"({vec[LWR]} > {vec[UPR]})"
        )

    # Step 2: interval inverted within the error bound — collapse to EXP
    if vec[LWR] > vec[UPR]:
        # EXP must also lie within ERR of the (inverted) interval,
        # otherwise the noise assumption does not hold
        if vec[UPR] - vec[EXP] > vec[ERR]:
            raise RuntimeError(
                f"{label} has abnormal {target} {where} "
                f"({vec[EXP]} < {vec[UPR]})"
            )
        if vec[EXP] - vec[LWR] > vec[ERR]:
            raise RuntimeError(
                f"{label} has abnormal {target} {where} "
                f"({vec[EXP]} > {vec[LWR]})"
            )
        vec[LWR] = vec[EXP]
        vec[UPR] = vec[EXP]
        return

    # Step 3: clamp EXP into [LWR, UPR] by shifting the whole interval
    if vec[EXP] < vec[LWR]:
        delta = vec[LWR] - vec[EXP]
        if delta > vec[ERR]:
            raise RuntimeError(
                f"{label} has abnormal {target} {where} "
                f"({vec[EXP]} < {vec[LWR]})"
            )
        vec[EXP] += delta
        vec[UPR] += delta
    elif vec[EXP] > vec[UPR]:
        delta = vec[EXP] - vec[UPR]
        if delta > vec[ERR]:
            raise RuntimeError(
                f"{label} has abnormal {target} {where} "
                f"({vec[EXP]} > {vec[UPR]})"
            )
        vec[EXP] -= delta
        vec[LWR] -= delta

    return

#==============================================================================
def _sanitize_end_ge_start(start_vec, end_vec, label, target):
    """
    Enforce the invariant ``end >= start`` component-wise, in place.

    For each of the components ``EXP``, ``UPR``, ``LWR`` the function
    checks that ``start <= end``. Violations larger than the sum of the
    two error bounds raise :exc:`RuntimeError`. A violation of the
    ``EXP`` component within the error bound is corrected by shifting
    the entire ``end`` vector (``EXP``, ``UPR``, ``LWR``) by the
    required delta.

    Parameters
    ----------
    start_vec : numpy.ndarray
        Start-time component array ``[EXP, VAR, LWR, UPR, ERR]``.
    end_vec : numpy.ndarray
        End-time component array ``[EXP, VAR, LWR, UPR, ERR]``, modified
        in place.
    label : str
        Human-readable element label used in error messages.
    target : str
        Name of the current pass, used in error messages.

    Raises
    ------
    RuntimeError
        If ``end < start`` for any checked component beyond the sum of
        the corresponding error bounds.
    """
    for c in (EXP, UPR, LWR):
        diff = start_vec[c] - end_vec[c]
        if diff > start_vec[ERR] + end_vec[ERR]:
            raise RuntimeError(
                f"{label} has abnormal {target} end time "
                f"({end_vec[c]} < {start_vec[c]}) for component {c}"
            )

    if start_vec[EXP] > end_vec[EXP]:
        delta = start_vec[EXP] - end_vec[EXP]
        end_vec[EXP] += delta
        end_vec[UPR] += delta
        end_vec[LWR] += delta

    return

#==============================================================================
def _sanitize_act(a, target):
    """
    Enforce the CPM invariants on an activity's time parameters in place.

    Depending on ``target``, the function sanitizes either the early
    (``'early_pre'`` / ``'early'``) or the late (``'late_pre'`` /
    ``'late'``) parameters of ``a``:

    For early passes:

    - decouples ``early_start`` / ``early_end`` from the event arrays
      (``copy()``) before modifying them;
    - clamps ``early_start[LWR]`` to ``>= 0`` by shifting the whole
      vector;
    - normalizes component ordering, enforces ``early_end >=
      early_start``, then re-normalizes ordering.

    For late passes:

    - decouples ``late_start`` / ``late_end`` from the event arrays;
    - checks ``late_start[EXP] >= early_start[EXP]`` and
      ``late_end[EXP] >= late_start[EXP]`` (shifting the scenario bounds
      together with ``EXP`` if within the error bound, raising
      otherwise);
    - normalizes component ordering, enforces ``late_end >=
      late_start``, then re-normalizes ordering.

    Parameters
    ----------
    a : ArrowActivity
        Activity to sanitize.
    target : str
        One of ``'early_pre'``, ``'early'``, ``'late_pre'``, ``'late'``.

    Raises
    ------
    RuntimeError
        If any invariant is violated beyond the accumulated error bound.
    """
    label = f"Activity {a.letter}"

    if target in ['early_pre', 'early']:
        # Decouple activities from events
        a.early_start = a.early_start.copy()
        a.early_end   = a.early_end.copy()

        # Time params lower limit
        if a.early_start[LWR] < 0.0:
            a.early_start[EXP] -= a.early_start[LWR]
            a.early_start[UPR] -= a.early_start[LWR]
            a.early_start[LWR] = 0.0

        # Normalize component ordering, then enforce end >= start,
        # then re-normalize ordering (the end shift may perturb it)
        _sanitize_ordering(a.early_start, label, 'early start', target)
        _sanitize_ordering(a.early_end,   label, 'early end',   target)
        _sanitize_end_ge_start(a.early_start, a.early_end, label, target)
        _sanitize_ordering(a.early_start, label, 'early start', target)
        _sanitize_ordering(a.early_end,   label, 'early end',   target)

    elif target in ['late_pre', 'late']:
        # Decouple activities from events
        a.late_start = a.late_start.copy()
        a.late_end   = a.late_end.copy()

        # Invariant is preserved only for expected value
        if a.early_start[EXP] - a.late_start[EXP] > a.early_start[ERR] + a.late_start[ERR]:
            raise RuntimeError(
                f"{label} has abnormal {target} expected start time "
                f"({a.late_start[EXP]} < {a.early_start[EXP]})"
            )

        if a.late_start[EXP] < a.early_start[EXP]:
            delta = a.early_start[EXP] - a.late_start[EXP]
            a.late_start[EXP] += delta
            a.late_start[UPR] += delta
            a.late_start[LWR] += delta

        # Invariant is preserved only for expected value
        if a.late_start[EXP] - a.late_end[EXP] > a.late_start[ERR] + a.late_end[ERR]:
            raise RuntimeError(
                f"{label} has abnormal {target} expected end time "
                f"({a.late_end[EXP]} < {a.late_start[EXP]})"
            )

        if a.late_end[EXP] < a.late_start[EXP]:
            delta = a.late_start[EXP] - a.late_end[EXP]
            a.late_end[EXP] += delta
            a.late_end[UPR] += delta
            a.late_end[LWR] += delta

        # Normalize component ordering, then enforce end >= start,
        # then re-normalize ordering (the end shift may perturb it)
        _sanitize_ordering(a.late_start, label, 'late start', target)
        _sanitize_ordering(a.late_end,   label, 'late end',   target)
        _sanitize_end_ge_start(a.late_start, a.late_end, label, target)
        _sanitize_ordering(a.late_start, label, 'late start', target)
        _sanitize_ordering(a.late_end,   label, 'late end',   target)

    return

#==============================================================================
def _sanitize_evt(e, target):
    """
    Enforce the CPM invariants on an event's time parameters in place.

    Depending on ``target``:

    - ``'early_pre'``: clamps ``early_pre[LWR]`` to ``>= 0`` by shifting
      ``EXP`` and ``UPR``.
    - ``'late_pre'``: checks ``late_pre[EXP] >= early_pre[EXP]``
      (shifting the scenario bounds if within the error bound, raising
      otherwise).
    - ``'early'``: clamps ``early[LWR]`` to ``>= 0`` and then normalizes
      component ordering.
    - ``'late'``: checks ``late[EXP] >= early[EXP]`` (shifting the
      scenario bounds if within the error bound, raising otherwise) and
      then normalizes component ordering.

    Parameters
    ----------
    e : Event
        Event to sanitize.
    target : str
        One of ``'early_pre'``, ``'late_pre'``, ``'early'``, ``'late'``.

    Raises
    ------
    RuntimeError
        If ``late < early`` beyond the accumulated error bound.
    """
    label = f"Event {e.id}"

    if 'early_pre' == target:
        if e.early_pre[LWR] < 0.0:
            e.early_pre[EXP] -= e.early_pre[LWR]
            e.early_pre[UPR] -= e.early_pre[LWR]
            e.early_pre[LWR] = 0.0

    elif 'late_pre' == target:
        # Invariant is preserved only for expected value
        if e.early_pre[EXP] - e.late_pre[EXP] > e.late_pre[ERR] + e.early_pre[ERR]:
            raise RuntimeError(
                f"{label} has abnormal late expected preliminary time "
                f"({e.late_pre[EXP]} < {e.early_pre[EXP]})"
            )

        if e.late_pre[EXP] < e.early_pre[EXP]:
            delta = e.early_pre[EXP] - e.late_pre[EXP]
            e.late_pre[EXP] += delta
            e.late_pre[UPR] += delta
            e.late_pre[LWR] += delta

    elif 'early' == target:
        if e.early[LWR] < 0.0:
            e.early[EXP] -= e.early[LWR]
            e.early[UPR] -= e.early[LWR]
            e.early[LWR] = 0.0

        _sanitize_ordering(e.early, label, 'early', target)

    elif 'late' == target:
        # Invariant is preserved only for expected value
        if e.early[EXP] - e.late[EXP] > e.late[ERR] + e.early[ERR]:
            raise RuntimeError(
                f"{label} has abnormal expected late time "
                f"({e.late[EXP]} < {e.early[EXP]})"
            )

        if e.late[EXP] < e.early[EXP]:
            delta = e.early[EXP] - e.late[EXP]
            e.late[EXP] += delta
            e.late[UPR] += delta
            e.late[LWR] += delta

        _sanitize_ordering(e.late, label, 'late', target)

    return
#==============================================================================
def _choice_early(old, new):
    """
    Forward-pass wrapper around :func:`_choice`.

    Selects between ``old`` and ``new`` early-time estimates using
    ``delta = new[EXP] - old[EXP]`` (the new estimate is "better" when
    it is larger).

    Parameters
    ----------
    old : numpy.ndarray
        Existing early-time estimate ``[EXP, VAR, LWR, UPR, ERR]``.
    new : numpy.ndarray
        Candidate early-time estimate ``[EXP, VAR, LWR, UPR, ERR]``.

    Returns
    -------
    numpy.ndarray
        Selected (or mixed) estimate. See :func:`_choice`.
    """
    return _choice(old, new, new[EXP] - old[EXP])

#==============================================================================
def _choice_late(old, new):
    """
    Backward-pass wrapper around :func:`_choice`.

    Selects between ``old`` and ``new`` late-time estimates using
    ``delta = old[EXP] - new[EXP]`` (the new estimate is "better" when
    it is smaller).

    Parameters
    ----------
    old : numpy.ndarray
        Existing late-time estimate ``[EXP, VAR, LWR, UPR, ERR]``.
    new : numpy.ndarray
        Candidate late-time estimate ``[EXP, VAR, LWR, UPR, ERR]``.

    Returns
    -------
    numpy.ndarray
        Selected (or mixed) estimate. See :func:`_choice`.
    """
    return _choice(old, new, old[EXP] - new[EXP])

#==============================================================================
def _delta_early(a):
    """
    Forward-pass delta for an activity.

    Parameters
    ----------
    a : ArrowActivity
        Activity whose effort is used as the delta.

    Returns
    -------
    numpy.ndarray
        ``a.expected`` -- the raw resource effort component array.
    """
    return a.expected

#==============================================================================
def _delta_late(a):
    """
    Backward-pass delta for an activity.

    Parameters
    ----------
    a : ArrowActivity
        Activity whose effort is used as the delta.

    Returns
    -------
    numpy.ndarray
        :func:`_negative` of ``a.expected``, so that the signed
        backward-pass arithmetic yields a non-negative duration.
    """
    return _negative(a.expected)

#==============================================================================
class ActivityOnArrowModel:
    """
    Main class for CPM/PERT network analysis with resource-aware
    scheduling.

    This class constructs and analyzes network models using Critical Path
    Method (CPM) and Program Evaluation and Review Technique (PERT) with
    support for resource-dependent duration calculations.

    Parameters
    ----------
    wbs_dict : dict
        Work Breakdown Structure dictionary with activity data.
        Each key is an activity ID (``int``) and value is a dictionary
        containing:

        - ``letter``: Activity letter / code (required, ``str``)
        - One of these resource effort specifications:

          - Direct: ``expected`` and optional ``exp_var``
          - Three-point PERT: ``optimistic``, ``most_likely``,
            ``pessimistic``
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

        Cannot be used together with ``lnk_src`` / ``lnk_dst``.
    duration : callable, default=_default_duration
        Callback function for resource-aware duration calculation.
        Signature: ``duration(effort, activity, base_time, target) -> float``

        - ``effort``: float value representing resource effort:

          * Positive number or zero for forward pass (early times
            calculation)
          * Negative number or zero for backward pass (late times
            calculation)
        - ``activity``: :class:`ArrowActivity` object for context
        - ``base_time``: ``float`` -- the ``EXP`` component of the
          event's base time. Currently the callback is invoked only
          during the preliminary passes (``'early_pre'`` /
          ``'late_pre'``), and all three calls (for ``effort[EXP]``,
          ``effort[UPR]``, ``effort[LWR]``) receive the **same**
          ``base_time``.
        - ``target``: ``str`` -- currently one of ``'early_pre'`` or
          ``'late_pre'`` (the only passes in which the callback is
          invoked). During the final passes the model uses the frozen
          ``ArrowActivity.duration`` and does not call the callback.

        Returns: float value representing actual duration with the same
        sign as ``effort``:

        - Zero if ``effort`` is zero
        - Positive number if ``effort`` is positive
        - Negative number if ``effort`` is negative

        .. note::
            The callback is invoked during the **preliminary** (``_pre``)
            passes only. It is **not** invoked during the **final**
            (``'early'`` / ``'late'``) passes and **not** during network
            post-processing: the final passes use the frozen
            :attr:`ArrowActivity.duration` value produced by
            :meth:`ArrowActivity.compute_final_duration` (the backward
            pass applies :func:`_negative`). See the class-level Notes
            for the rationale.

        .. note::
            Within :func:`ActivityOnArrowModel._compute_target`'s inner
            ``_duration_vec``, the callback is called three times per
            activity: for the ``EXP``, ``UPR``, and ``LWR`` components of
            the effort estimate. All three calls receive the **same**
            ``base_time`` (the ``EXP`` component of the event's base
            time), which guarantees ``dur[LWR] <= dur[EXP] <= dur[UPR]``.

    p : float, default=0.95
        Probability level for PERT quantile estimates (``0 < p < 1``)
    default_risk : float, default=0.3
        Default risk factor for effort estimation
        (``0 <= default_risk <= 1``)
    next_act_id : int, default=1
        Starting ID for automatically generated internal activity
        numbering. Must be positive.
    debug : bool, default=False
        Enable debug mode to include computation error bounds in output

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
        valid AoA), the time-parameter passes raise :exc:`RuntimeError`
        when the CPM invariants are violated beyond the accumulated
        error bound:

        - On early passes:

          - ``early_start[LWR] >= 0`` (clamped, not raised);
          - ``early_end[c] >= early_start[c]`` for
            ``c in {EXP, UPR, LWR}`` (raised on violation beyond error
            bound);
          - ``early_start[EXP]`` and ``early_end[EXP]`` must lie within
            ``[LWR, UPR]`` (clamped within the error bound, raised
            otherwise).

        - On late passes:

          - ``late_start[EXP] >= early_start[EXP]``;
          - ``late_end[EXP] >= late_start[EXP]``;
          - ``late_start[EXP]`` and ``late_end[EXP]`` must lie within
            ``[LWR, UPR]`` (clamped within the error bound, raised
            otherwise).

        - On both late passes (for events): ``late[EXP] >= early[EXP]``.
        - On the final ``'early'`` / ``'late'`` passes (for events):
          ``early[EXP]`` / ``late[EXP]`` must lie within ``[LWR, UPR]``
          (clamped within the error bound, raised otherwise).

        These checks apply to both the preliminary (``_pre``) and the
        final passes. Such violations typically indicate a structural
        inconsistency (bad topology or a misbehaving ``duration``
        callback) rather than a purely numerical issue.

    Attributes
    ----------
    activities : list of ArrowActivity
        List of activity objects in the network (real + dummy).
    events : list of Event
        List of event objects in the network, sorted by topological
        stage.
    is_pert : bool
        ``True`` if PERT analysis is enabled (any activity has non-zero
        variance).
    debug : bool
        Debug mode flag.
    p : float
        Probability level for PERT quantile estimates.

    Examples
    --------
    Basic usage with direct effort parameters::

        wbs = {
            1: {'letter': 'A', 'expected': 5.0, 'exp_var': 1.0},
            2: {'letter': 'B', 'expected': 3.0}
        }
        links = [[1, 2]]
        model = ActivityOnArrowModel(wbs, links=links)

    Three-point PERT effort estimates::

        wbs_pert = {
            1: {'letter': 'A', 'optimistic': 3, 'most_likely': 5, 'pessimistic': 8},
            2: {'letter': 'B', 'optimistic': 2, 'pessimistic': 6}
        }
        model_pert = ActivityOnArrowModel(wbs_pert, links=links)

    Resource-aware scheduling with custom duration callback::

        def team_duration(effort, activity, base_time, target):
            team_size = activity.data.get('team_size', 1)
            productivity = activity.data.get('productivity', 1.0)
            abs_duration = abs(effort) / (team_size * productivity)
            return abs_duration if effort >= 0 else -abs_duration

        wbs_resource = {
            1: {'letter': 'A', 'expected': 40.0, 'team_size': 2, 'productivity': 0.8},
            2: {'letter': 'B', 'expected': 30.0, 'team_size': 1, 'productivity': 1.0}
        }
        model_resource = ActivityOnArrowModel(wbs_resource, links=links,
                                              duration=team_duration)

    Visualization in various formats::

        model.viz('network.png')  # PNG (default if extension omitted)
        model.viz('network.svg')  # SVG
        model.viz('network.pdf')  # PDF

    Notes
    -----
    **Two-phase time computation.** Time parameters are computed in two
    phases separated by the network post-processing (see
    :meth:`_compute_time_params_pre` and :meth:`_compute_time_params`):

    1. The preliminary (``_pre``) passes compute early/late times using
       the raw resource effort estimates. The user's ``duration``
       callback is invoked here (via the inner ``_duration_vec`` of
       :meth:`_compute_target`), once for each of the ``EXP``, ``UPR``,
       and ``LWR`` components of the effort estimate, all with the same
       ``base_time``.
    2. :meth:`ArrowActivity.compute_final_duration` freezes a single
       duration value (``ArrowActivity.duration``) derived from those
       preliminary values.
    3. :meth:`_network_postprocess` moves maximum-duration activities
       onto the straight edges of the triangle groups. It compares the
       already frozen durations, so it can only run at this point.
    4. The final passes re-traverse the post-processed network using the
       frozen duration (the backward pass applies :func:`_negative`).
       The user's ``duration`` callback is **not** invoked again in the
       final passes.

    Freezing is required whenever the ``duration`` callback depends on
    ``base_time`` (resource availability that varies over time):
    without it, the same activity could receive different durations in
    the forward and backward passes, breaking the CPM invariants. The
    two-phase scheme therefore linearizes the feedback
    "schedule -> duration -> schedule" with a single step and keeps the
    overall cost linear in the network size.

    **Scope of the approximation.** This is a *fast estimator*, not a
    scheduler. The "hidden critical path" effect -- where an activity's
    start or end coincides with a resource calendar boundary and the
    reserve in front of it is misleading -- is captured only in the
    first approximation, through the single-duration freeze. A rigorous
    treatment would require constraint programming or Monte-Carlo
    simulation over resource calendars, which are intentionally out of
    scope. In particular:

    - The beta approximation in :func:`_fit_beta` underestimates the
      variance of durations that are discontinuous in ``base_time``
      (see the warning in :func:`_fit_beta`).
    - Because the same frozen ``duration`` is used by both final
      passes, the start and end reserves of every activity are equal
      by construction. Any residual calendar-induced inconsistency
      therefore manifests as a *shift* of the frozen duration away
      from the raw forward-pass value, not as a start/end reserve
      mismatch. The value reported by :meth:`ArrowActivity.to_dict` is
      the frozen one.

    **Construction order.** :meth:`__init__` performs the following
    steps in exactly this order:

    1. :meth:`_create_model` -- build the AoA network (real and dummy
       activities) and collect the source events of the dummy
       activities.
    2. :meth:`_compute_time_params_pre` -- preliminary passes, then
       freezing of the durations.
    3. :meth:`_network_postprocess` -- move maximum-duration activities
       onto the straight edges of the triangle groups.
    4. :meth:`_compute_time_params` -- final passes and time reserves.
    5. :meth:`_compute_target` with ``'stage'`` -- topological stage of
       every event, computed on the post-processed topology.
    6. Sort the events by ``stage`` and renumber ``Event.id`` starting
       from ``1``.

    Note that the triangle-group optimization runs after the durations
    have been frozen (it compares them) and before both the stage pass
    and the final passes, so the stages and the event times both
    describe the post-processed topology, while the final event numbering
    is assigned only after the topology has been modified.

    **Duration callback contract.** The duration callback function must
    preserve the sign of the effort parameter to ensure correct forward
    and backward pass calculations:

    - For forward pass (early times): ``effort >= 0``,
      ``duration >= 0``
    - For backward pass (late times): ``effort <= 0``,
      ``duration <= 0``
    - For zero effort: ``duration == 0``

    ``base_time`` is always a ``float`` -- the ``EXP`` component of the
    source event's base time. The callback is invoked only during the
    preliminary (``'early_pre'`` / ``'late_pre'``) passes, so it is never
    called with ``None``.

    **Numerical consistency.** Numerical consistency of early/late time
    parameters is enforced *during* the forward and backward passes (see
    :meth:`_compute_target`), rather than through a separate post-hoc
    sanitization pass. Both the preliminary (``early_pre`` /
    ``late_pre``) and the final (``early`` / ``late``) passes are
    covered. This ensures that downstream computations always operate
    on consistent values:

    - Values whose violation is within the accumulated error bound are
      silently clamped to satisfy the invariants.
    - Values whose violation exceeds the accumulated error bound cause
      a :exc:`RuntimeError`. This indicates a structural problem -- for
      example, a ``duration`` callback that maps non-negative effort
      to a negative duration, or a topology that is inconsistent with
      the signed backward-pass deltas.

    Because the ``_sanitize_act`` / ``_sanitize_evt`` helpers only
    react to the ``'early_pre'``, ``'late_pre'``, ``'early'`` and
    ``'late'`` targets, the ``'stage'`` pass is not subject to these
    checks.

    **Reserved field names.** Reserved WBS field names (``expected``,
    ``exp_var``, ``letter``, ``optimistic``, ``most_likely``,
    ``pessimistic``) are consumed during parsing and removed from
    ``activity.data``. Do not use these names for custom fields.

    User-defined event data is stored in ``Event.data``. Its keys must
    not collide with the reserved field names inserted by
    :meth:`Event.to_dict` (listed in the ``Event`` class notes),
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
        dummy_srcs = self._create_model(wbs_dict, lnk_src, lnk_dst, default_risk, next_act_id)

        # After _create_model, events must be non-empty
        if len(self._events) == 0:
            raise RuntimeError("Network construction resulted in no events. Check input data.")

        # Compute Event and Activity time parameters (phase 1)
        self._compute_time_params_pre()

        # Postprocess created network
        self._network_postprocess(dummy_srcs)

        # Compute Event and Activity time parameters (phase 2)
        self._compute_time_params()

        # Compute stages of project on the post-processed topology
        self._compute_target('stage')

        # Renumerate events according to the rules of network modeling
        self._events.sort(key=lambda e: e.stage)
        for i, e in enumerate(self._events, 1):
            e.id = i

    #==========================================================================
    @property
    def is_pert(self):
        """Whether PERT analysis is enabled (any activity has non-zero
        variance)."""
        return self._is_pert

    #==========================================================================
    @property
    def p(self):
        """Probability level for PERT quantile estimates."""
        return self._p

    #==========================================================================
    @property
    def activities(self):
        """List of activity objects in the network."""
        return self._activities

    #==========================================================================
    @property
    def events(self):
        """List of event objects, sorted by topological stage."""
        return self._events

    #==========================================================================
    def _parse_links(self, lnk_src, lnk_dst, links):
        """
        Parse links from various formats into standard
        ``(lnk_src, lnk_dst)`` lists.

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
            If any input is not an iterable or cannot be converted to a
            list.
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

    #==========================================================================
    def _create_model(self, wbs_dict, lnk_src, lnk_dst, default_risk, next_act_id):
        """
        Create the network model from WBS data and links.

        Internal method that constructs the network graph, creates events
        and activities, and sets up the model for analysis. It delegates
        the AoA network generation to the C extension ``_ccpm.make_aoa``.
        The triangle-group optimization that places maximum-duration
        activities on the straight edges is performed separately, by
        :meth:`_network_postprocess`, between the two time-computation
        phases.

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

        Returns
        -------
        set of int
            IDs of the events that are sources of dummy activities. The
            set is consumed by :meth:`_network_postprocess` and is
            expressed in the event IDs assigned during construction, i.e.
            before the final renumbering.

        Raises
        ------
        ValueError
            If any activity data is invalid or links contain invalid IDs.
        RuntimeError
            If the C extension returns an error or network consistency
            fails.
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
                    expected = _calculate_action_time_params(wbs_data, default_risk)
                except ValueError as e:
                    raise ValueError(f"Error processing activity {act_id} ({wbs_data.get('letter', 'unknown')}): {e}") from e

                letter = wbs_data.get('letter', '')

                # Even one wbs item with nonzero variance is enough to compute PERT
                if expected[VAR] > 0.0:
                    self._is_pert = True

                # Create data dict without fields stored as separate attributes
                data_without_duplicates = self._remove_duplicate_fields(wbs_data)

                self._add_activity(int(act_id), letter, int(net_src[i]), int(net_dst[i]),
                                   expected, data_without_duplicates)
            else:
                # Add a dummy activity (no effort, no letter, no data)
                nd += 1  # One more dummy work
                self._add_activity(0, '#' + str(nd), int(net_src[i]), int(net_dst[i]),
                                   None, {}, is_dummy=True)
                dsrc.add(int(net_src[i]))

        return dsrc

    def _network_postprocess(self, dsrc):
        """
        Place maximum-duration activities on the straight edges of
        triangle groups.

        A *triangle group* is a pair of events ``(W, Z)`` connected both
        directly, by a single activity ``a`` (``W -> Z``), and
        indirectly, through an intermediate event ``X``
        (``W -> X -> Z``). Every event that is a source of a dummy
        activity and has at most one incoming and at most one outgoing
        activity is treated as a candidate intermediate event; the group
        key is ``(bck[0].src.id, fwd[0].dst.id)``, i.e. ``(W, Z)``. All
        candidates sharing a key are collected in one group, but only if
        a direct ``W -> Z`` activity exists.

        For each group the activity with the largest frozen
        :attr:`ArrowActivity.duration` is moved onto the direct edge by
        swapping destinations: the ``W -> X`` candidate becomes
        ``W -> Z``, and the previously direct ``W -> Z`` activity becomes
        ``W -> X``.

        Parameters
        ----------
        dsrc : set of int
            IDs of the events that are sources of dummy activities, as
            returned by :meth:`_create_model`. The IDs are the ones
            assigned during construction, i.e. before the final
            renumbering.

        Notes
        -----
        - This method is called from :meth:`__init__` *between* the two
          time-computation phases: after
          :meth:`_compute_time_params_pre` (which freezes
          :attr:`ArrowActivity.duration`) and before
          :meth:`_compute_time_params` (which recomputes the event times
          for the post-processed topology).
        - Durations are compared using the **frozen**
          ``duration[EXP]`` values produced by
          :meth:`ArrowActivity.compute_final_duration`; the ``duration``
          callback is not invoked here.
        - The swap does not invalidate the frozen durations: only the
          destination events are exchanged, while both swapped activities
          keep their source event and therefore their ``base_time`` (the
          ``EXP`` component of the source event's time), on which the
          callback result depends.
        - Only the destination links are changed. Event ``stage`` values
          are computed after this method (see the construction order in
          :class:`ActivityOnArrowModel`), so they describe the
          post-processed topology.
        - If no direct ``W -> Z`` activity exists, the group is not
          registered and the candidate is left untouched.
        """

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
                if a.duration[EXP] > maxa.duration[EXP]:
                    a.dst, maxa.dst = maxa.dst, a.dst
                    maxa = a

    #==========================================================================
    def _remove_duplicate_fields(self, wbs_data):
        """
        Remove fields from WBS data that are stored as separate activity
        attributes.

        Parameters
        ----------
        wbs_data : dict
            Complete WBS data for an activity

        Returns
        -------
        dict
            Shallow copy of ``wbs_data`` with the following keys removed
            (they are consumed during parsing and stored as separate
            attributes on the :class:`ArrowActivity` instance):
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

    #==========================================================================
    def _compute_time_params_pre(self):
        """
        Compute the preliminary (phase 1) time parameters and freeze the
        per-activity durations.

        This is the first of the two time-computation phases. It runs the
        preliminary (``_pre``) passes on the raw effort estimates and then
        freezes a single duration value per activity via
        :meth:`ArrowActivity.compute_final_duration`. The final passes and
        the time reserves are computed by :meth:`_compute_time_params`,
        which :meth:`__init__` calls after :meth:`_network_postprocess`.

        Raises
        ------
        RuntimeError
            Propagates :exc:`RuntimeError` from :meth:`_compute_target`,
            which is raised when a CPM invariant is violated beyond the
            accumulated error bound.

        Notes
        -----
        The phase consists of the following steps:

        1. Forward pass (``'early_pre'``) over the raw ``a.expected``
           effort estimate. The user's ``duration`` callback is invoked
           via the inner ``_duration_vec`` of :meth:`_compute_target`
           (once for each of ``EXP``, ``UPR``, and ``LWR``, all with the
           same base time).
        2. Initialisation of ``late_pre`` for every event from the
           maximum ``early_pre`` across the network, followed by the
           backward pass (``'late_pre'``).
        3. Freezing of the durations: for every activity,
           :meth:`ArrowActivity.compute_final_duration` derives a single
           ``duration`` value from the preliminary times. Because the
           same frozen value is used by both final passes (the backward
           pass applies :func:`_negative`), the start and end reserves
           are equal by construction.

        Numerical consistency of time parameters is enforced *during*
        the network traversal inside :meth:`_compute_target` by the
        local helpers ``_sanitize_act`` and ``_sanitize_evt``. Both the
        preliminary (``early_pre`` / ``late_pre``) and the final
        (``early`` / ``late``) passes are covered, so this method does
        not perform a separate post-hoc clamping pass after the
        forward/backward passes complete.

        Violations within the accumulated error bound are silently
        clamped; violations beyond it raise :exc:`RuntimeError` from
        inside :meth:`_compute_target` and propagate through this
        method.

        This method is called from :meth:`__init__` before
        :meth:`_network_postprocess`, which relies on the frozen
        :attr:`ArrowActivity.duration` values produced here.
        """
        self._compute_target('early_pre')

        # Set late times starting from project completion
        late_pre = np.zeros((5,), dtype=float)
        for e in self._events:
            if e.early_pre[EXP] > late_pre[EXP]:
                late_pre = e.early_pre.copy()

        for e in self._events:
            e.late_pre = late_pre.copy()

        self._compute_target('late_pre')

        for a in self._activities:
            a.compute_final_duration()


    def _compute_time_params(self):
        """
        Compute the final (phase 2) time parameters and the time reserves.

        This is the second of the two time-computation phases. It
        re-traverses the network using the durations frozen by
        :meth:`_compute_time_params_pre` and then derives the reserves.
        The user's ``duration`` callback is **not** invoked here.

        Raises
        ------
        RuntimeError
            If any activity or event has a negative expected time
            reserve, indicating a programming error or corrupted network.
            Also propagates :exc:`RuntimeError` from
            :meth:`_compute_target`, which is raised when a CPM invariant
            is violated beyond the accumulated error bound.

        Notes
        -----
        The phase consists of the following steps:

        1. Forward pass (``'early'``) using the frozen
           :attr:`ArrowActivity.duration` as the delta.
        2. Initialisation of ``late`` for every event from the maximum
           ``early`` across the network, followed by the backward pass
           (``'late'``) using :func:`_negative` of the same frozen
           duration.
        3. Time reserves. Reserves with ``reserve[EXP] < reserve[ERR]``
           (including slightly negative reserves within the error bound)
           are rounded to zero: the ``LWR`` and ``UPR`` components of the
           reserve vector are shifted by ``-reserve[EXP]`` and
           ``reserve[EXP]`` is set to ``0.0``. This keeps the reserve
           triple consistent (``LWR <= 0 <= UPR``) while preserving the
           scenario range. Reserves with ``reserve[EXP] <
           -reserve[ERR]`` raise :exc:`RuntimeError`.

        For activities, the start and end reserves are computed
        separately (because resource availability may be time-dependent)
        and the most certain one is selected via :func:`_choice`.

        This method is called from :meth:`__init__` after
        :meth:`_network_postprocess`, so it works on the post-processed
        topology and recomputes the event times for it.
        """
        self._compute_target('early')

        # Set late times starting from project completion
        late = np.zeros((5,), dtype=float)
        for e in self._events:
            if e.early[EXP] > late[EXP]:
                late = e.early.copy()

        for e in self._events:
            e.late = late.copy()

        self._compute_target('late')

        _VALS = [EXP, UPR, LWR]
        _OTHR = [VAR, ERR]

        # Compute reserves
        for e in self._events:
            e.reserve[_VALS] = e.late[_VALS] - e.early[_VALS]
            e.reserve[_OTHR] = e.late[_OTHR] + e.early[_OTHR]

            # Check for programming errors
            # Invariant holds only for expected values
            if e.reserve[EXP] < -e.reserve[ERR]:
                raise RuntimeError(f"Event {e.id} has negative expected time reserve ({e.reserve[EXP]})")

            # Round off insignificant values
            if e.reserve[EXP] < e.reserve[ERR]:
                e.reserve[UPR] -= e.reserve[EXP]
                e.reserve[LWR] -= e.reserve[EXP]
                e.reserve[EXP] = 0.0

        for a in self._activities:
            # Compute start and end reserve values separately
            # These values may differ due to resource availability time dependence
            start_res = np.zeros((5,), dtype=float)
            start_res[_VALS] = a.late_start[_VALS] - a.early_start[_VALS]
            start_res[_OTHR] = a.late_start[_OTHR] + a.early_start[_OTHR]

            end_res = np.zeros((5,), dtype=float)
            end_res[_VALS] = a.late_end[_VALS] - a.early_end[_VALS]
            end_res[_OTHR] = a.late_end[_OTHR] + a.early_end[_OTHR]

            # Choose minimum reserve value
            a.reserve = _choice(start_res, end_res, start_res[EXP] - end_res[EXP])

            # Check for programming errors
            # Invariant holds only for expected values
            if a.reserve[EXP] < -a.reserve[ERR]:
                raise RuntimeError(f"Activity {a.letter} has negative expected time reserve ({a.reserve[EXP]})")

            # Round off insignificant values
            if a.reserve[EXP] < a.reserve[ERR]:
                a.reserve[UPR] -= a.reserve[EXP]
                a.reserve[LWR] -= a.reserve[EXP]
                a.reserve[EXP] = 0.0


    #==========================================================================
    def _duration_vec(self, effort, activity, base_time, target):
        """
        Compute a duration component vector with variance
        propagation for PERT analysis.

        Used by the preliminary (``_pre``) passes. The callback is
        invoked three times -- for ``effort[EXP]``, ``effort[UPR]``,
        and ``effort[LWR]`` -- all with the **same** ``base_time``
        (``base_time[EXP]``). Using a single base time is what
        guarantees ``dur[LWR] <= dur[EXP] <= dur[UPR]``.

        Parameters
        ----------
        effort : numpy.ndarray
            Resource effort component array
            ``[EXP, VAR, LWR, UPR, ERR]``
        activity : ArrowActivity
            Activity object for context
        base_time : numpy.ndarray
            Base time component array
            ``[EXP, VAR, LWR, UPR, ERR]``
        target : str
            Target scenario identifier

        Returns
        -------
        numpy.ndarray
            Duration component array
            ``[EXP, VAR, LWR, UPR, ERR]``

        Notes
        -----
        - If the model uses the default duration callback, the
          effort array is returned as a copy.
        - Otherwise, the callback is invoked for ``effort[EXP]``,
          ``effort[UPR]``, and ``effort[LWR]``. The error bound is
          set to ``EPS * max(|dur[UPR]|, |dur[LWR]|,
          |dur[UPR] - dur[LWR]|)``.
        - For PERT models (``self.is_pert`` is true) with non-zero
          effort variance, and only when
          ``sqrt(effort[VAR]) > effort[ERR]`` **and**
          ``effort[VAR] > sqrt(EPS) * gamma_nom``, an MPERT-style
          mode correction is applied: if ``dur[EXP]`` is too close
          to ``dur[LWR]`` it is pushed inside by ``2 * dur[ERR]``;
          if it is too close to ``dur[UPR]``, ``dur[UPR]`` is grown
          so that the mean stays strictly inside; then ``dur[VAR]``
          is recomputed as
          ``(dur[EXP] - dur[LWR]) * (dur[UPR] - dur[EXP]) /
          (gamma + 3.0)`` (an algebraically equivalent form of the
          beta-distribution variance formula, given
          ``alpha = mu * (gamma + 2)`` and
          ``beta = (1 - mu) * (gamma + 2)``, where
          ``mu = (dur[EXP] - dur[LWR]) / (dur[UPR] - dur[LWR])``).
          Otherwise the activity is treated as deterministic and
          ``dur[UPR] = dur[LWR] = dur[EXP]``.
        - If the activity is deterministic, ``dur[UPR]`` and
          ``dur[LWR]`` are collapsed to ``dur[EXP]``.
        """
        # Optimize for default duration function
        if _default_duration == self._duration:
            return effort.copy()

        dur = np.zeros((5,), dtype=float)

        # Compute duration value and error bound
        # effort[EXP] is float: positive for forward pass, negative for backward pass
        dur[EXP] = self._duration(effort[EXP], activity, base_time[EXP], target)
        # Yes, we use one time base here to guarantee that dur[LWR] <= dur[EXP] <= dur[UPR]
        dur[UPR] = self._duration(effort[UPR], activity, base_time[EXP], target)
        dur[LWR] = self._duration(effort[LWR], activity, base_time[EXP], target)

        # Error bound
        dur[ERR] = EPS * max(abs(dur[UPR]),  abs(dur[LWR]),  abs(dur[UPR] - dur[LWR]))

        gamma_nom = (effort[EXP] - effort[LWR]) * (effort[UPR] - effort[EXP])

        if (np.sqrt(effort[VAR]) <= effort[ERR] or
            effort[VAR] <= np.sqrt(EPS) * gamma_nom or
            not self.is_pert):
            # Deterministic or fake activity
            # VAR is zero already
            dur[UPR] = dur[EXP]
            dur[LWR] = dur[EXP]
            return dur

        # Model is PERT and activity is not deterministic.
        # Make sure that mode is in (dur[LWR], dur[UPR]),
        # use MPERT math here
        gamma = gamma_nom / effort[VAR] - 3.0

        # Condition for mode
        if gamma < 6.0 * EPS:
            gamma = 6.0 * EPS

        # Now values corrections
        thr = dur[LWR] + (dur[UPR] - dur[LWR]) / (gamma + 2.0)
        if dur[EXP] <= thr:
            dur[EXP] = thr + 2.0 * dur[ERR]

        if dur[EXP] >= dur[UPR] - (dur[UPR] - dur[LWR]) / (gamma + 2.0):
            dur[UPR] = ((gamma + 2.0) * dur[EXP] - dur[LWR]) / (gamma + 1.0) + 2.0 * dur[ERR]

        dur[VAR] = (dur[EXP] - dur[LWR]) * (dur[UPR] - dur[EXP]) / (gamma + 3.0)

        return dur


    #==========================================================================
    def _compute_target(self, target=None):
        """
        Compute CPM parameters for events and activities.

        This method performs topological sorting and computes either
        preliminary times (``_pre`` passes), final times, or stage
        numbers.

        Parameters
        ----------
        target : str
            What to compute. One of:

            - ``'stage'`` -- topological stage of each event
            - ``'early_pre'`` -- preliminary early times (forward pass)
            - ``'late_pre'`` -- preliminary late times (backward pass)
            - ``'early'`` -- final early times (forward pass)
            - ``'late'`` -- final late times (backward pass)

        Raises
        ------
        ValueError
            If ``target`` parameter is invalid.
        RuntimeError
            If the network does not have exactly one starting event
            (i.e., the project does not have exactly one source event),
            or if it contains cycles not detected earlier.

            In addition, during the ``'early_pre'``, ``'late_pre'``,
            ``'early'`` and ``'late'`` passes the following CPM
            invariants are checked for every event and activity; a
            violation that exceeds the accumulated error bound raises
            :exc:`RuntimeError`:

            - On early passes (``'early_pre'`` / ``'early'``):

              - ``early_start[LWR] >= 0`` (clamped, not raised);
              - ``early_end[c] >= early_start[c]`` for
                ``c in {EXP, UPR, LWR}`` (raised on violation beyond
                the error bound);
              - ``early_start[EXP]`` and ``early_end[EXP]`` must lie
                within ``[LWR, UPR]`` (clamped within the error bound,
                raised otherwise).

            - On late passes (``'late_pre'`` / ``'late'``):

              - ``late_start[EXP] >= early_start[EXP]``;
              - ``late_end[EXP] >= late_start[EXP]``;
              - ``late_start[EXP]`` and ``late_end[EXP]`` must lie
                within ``[LWR, UPR]`` (clamped within the error bound,
                raised otherwise).

            - On both late passes, for events: ``late[EXP] >=
              early[EXP]``.
            - On the final ``'early'`` / ``'late'`` passes, for events:
              ``early[EXP]`` / ``late[EXP]`` must lie within
              ``[LWR, UPR]`` (clamped within the error bound, raised
              otherwise).

            Violations within the accumulated error bound are silently
            clamped. The corresponding messages identify the offending
            event / activity and the specific invariant. The ``_pre``
            passes check the same invariants against the ``_pre`` time
            attributes.

        Notes
        -----
        The computation uses different strategies for stage calculation
        vs time parameter calculation. The table below summarizes the
        delta and duration source for every target:

        - ``'stage'``: delta ``1`` per activity; no duration callback.
        - ``'early_pre'`` / ``'late_pre'``: delta is ``a.expected``
          (time vector ``[EXP, VAR, LWR, UPR, ERR]``); the model's
          ``duration`` callback is invoked via the inner
          ``_duration_vec`` for each of the ``EXP``, ``UPR``, and
          ``LWR`` components (all with the same ``base_time[EXP]``).
        - ``'early'`` / ``'late'``: delta is ``a.expected`` for shape
          but the effective per-activity duration is the frozen
          ``a.duration`` (``'early'``) or :func:`_negative` of it
          (``'late'``). The user's callback is **not** invoked again.

        The algorithm maintains a count of unresolved dependencies for
        each event (``n_dep``). Events with zero dependencies are
        pushed to the processing queue. When an activity is processed,
        the destination event's dependency counter is decremented; when
        it reaches zero, the event is added to the queue.

        Numerical consistency of time parameters is enforced *in place*
        during the ``'early_pre'``, ``'late_pre'``, ``'early'`` and
        ``'late'`` traversals by two local helpers:

        - ``_sanitize_act(a, target)`` -- called after each activity
          update. Enforces the activity invariants listed above.
        - ``_sanitize_evt(e, target)`` -- called after each event is
          processed. Enforces the event invariants listed above.

        Correcting values during the traversal keeps downstream
        calculations consistent and prevents small numerical
        inconsistencies from propagating through the rest of the
        network. As a consequence, callers do not need to run a
        separate post-processing pass to sanitize early/late values.

        Because these helpers only react to the ``'early_pre'``,
        ``'late_pre'``, ``'early'`` and ``'late'`` targets, the
        ``'stage'`` pass is unaffected by them.
        """

        #======================================================================
        if 'stage' == target:
            act_base      = None
            act_new       = None
            act_next      = 'dst'
            fwd           = '_out_activities'
            rev           = '_in_activities'
            choice        = max
            delta         = lambda a: 1
            process_delta = lambda d, a, b, t: d

        elif 'early_pre' == target:
            act_base      = 'early_start'
            act_new       = 'early_end'
            act_next      = 'dst'
            fwd           = '_out_activities'
            rev           = '_in_activities'
            choice        = _choice_early
            delta         = _delta_early
            process_delta = self._duration_vec

        elif 'late_pre' == target:
            act_base      = 'late_end'
            act_new       = 'late_start'
            act_next      = 'src'
            fwd           = '_in_activities'
            rev           = '_out_activities'
            choice        = _choice_late
            delta         = _delta_late
            process_delta = self._duration_vec

        elif 'early' == target:
            act_base      = 'early_start'
            act_new       = 'early_end'
            act_next      = 'dst'
            fwd           = '_out_activities'
            rev           = '_in_activities'
            choice        = _choice_early
            delta         = _delta_early
            process_delta = lambda d, a, b, t: a.duration

        elif 'late' == target:
            act_base      = 'late_end'
            act_new       = 'late_start'
            act_next      = 'src'
            fwd           = '_in_activities'
            rev           = '_out_activities'
            choice        = _choice_late
            delta         = _delta_late
            process_delta = lambda d, a, b, t: _negative(a.duration)

        else:
            raise ValueError(f"Unknown 'target' value: {target}")

        # Initialize activity parameters and processing function
        if target != 'stage':
            for a in self._activities:
                setattr(a, act_base, np.zeros((5,), dtype=float))
                setattr(a, act_new,  np.zeros((5,), dtype=float))

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
                    _sanitize_act(a, target)

                next_evt = getattr(a, act_next)

                setattr(next_evt, target, choice(getattr(next_evt, target), new_val))

                n_dep[next_evt] -= 1

                if 0 >= n_dep[next_evt]:
                    evt.append(next_evt)

            _sanitize_evt(e, target)

            i += 1
            if i >= len(evt):
                break

    #==========================================================================
    def _add_event(self, i):
        """
        Add a new event to the network.

        Parameters
        ----------
        i : int
            Unique event identifier (positive integer).
        """
        self._events.append(Event(i, self))

    #==========================================================================
    def _add_activity(self, wbs_id, letter, src_id, dst_id, expected, data, is_dummy=False):
        """
        Add a new activity to the network.

        Parameters
        ----------
        wbs_id : int
            Work Breakdown Structure identifier (``0`` for dummy
            activities)
        letter : str
            Activity letter / code for visualization
        src_id : int
            Source event ID (1-based index into ``self._events``)
        dst_id : int
            Destination event ID (1-based index into ``self._events``)
        expected : numpy.ndarray or None
            Resource effort component array
            ``[EXP, VAR, LWR, UPR, ERR]``, or ``None`` for dummy
            activities
        data : dict
            WBS data excluding fields stored as separate attributes
        is_dummy : bool, default=False
            Whether this is a dummy activity

        Raises
        ------
        TypeError
            If arguments have wrong types.
        """
        if not isinstance(wbs_id, int):
            raise TypeError(f"wbs_id must be int, got {type(wbs_id)}")

        if not isinstance(src_id, int):
            raise TypeError(f"src_id must be int, got {type(src_id)}")

        if not isinstance(dst_id, int):
            raise TypeError(f"dst_id must be int, got {type(dst_id)}")

        if not isinstance(letter, str):
            raise TypeError(f"letter must be str, got {type(letter)}")

        if not isinstance(data, dict):
            raise TypeError(f"data must be dict, got {type(data)}")

        act = ArrowActivity(self.next_act, wbs_id, letter, self,
                        self._events[src_id - 1], self._events[dst_id - 1],
                        expected, data, is_dummy=is_dummy)
        self._activities.append(act)
        self.next_act += 1

    #==========================================================================
    def __repr__(self):
        """String representation of the network model (events +
        activities)."""
        _repr = 'Events:{\n'
        for e in self._events:
            _repr += '        ' + str(e) + '\n'
        _repr += '}\n'

        _repr += 'Activities:{\n'
        for a in self._activities:
            _repr += '        ' + str(a) + '\n'
        _repr += '}\n'

        return _repr

    #==========================================================================
    def to_dict(self):
        """
        Convert the network model to a dictionary representation.

        Returns
        -------
        dict
            Dictionary with structure::

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
        computed CPM/PERT fields (see :meth:`ArrowActivity.to_dict` and
        :meth:`Event.to_dict`).
        """
        activities_data = [activity.to_dict() for activity in self._activities]
        events_data = [event.to_dict() for event in self._events]

        return {
            'activities': activities_data,
            'events': events_data
        }

    #==========================================================================
    def to_dataframe(self):
        """
        Convert the network model to pandas DataFrames.

        Returns
        -------
        tuple
            ``(activities_df, events_df)`` -- pandas DataFrames for
            activities and events.

        Notes
        -----
        Both DataFrames expand custom user data fields from the
        corresponding ``data`` attributes (``ArrowActivity.data`` and
        ``Event.data``) into separate columns for easy analysis.

        Activities' missing values are replaced with empty strings for
        object-typed columns (where possible, subject to the installed
        pandas version).
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

    #==========================================================================
    def viz(self, output_path=None, group_by_stage=True, get_style=_default_style):
        """
        Create a Graphviz visualization of the CPM/PERT network.

        Parameters
        ----------
        output_path : str, optional
            Output path for saving the visualization file. The rendering
            format is determined by the file extension:

            - ``.png`` -- Portable Network Graphics (default if no
              extension is given)
            - ``.svg`` -- Scalable Vector Graphics
            - ``.pdf`` -- Portable Document Format

            If ``None``, the graph is not rendered to a file (only the
            ``Digraph`` object is returned).
        group_by_stage : bool, default=True
            If ``True``, events are grouped into "layers" by their
            topological order (stage). If ``False``, events are placed
            freely by the Graphviz default layout.
        get_style : callable, default=_default_style
            Callback that computes styling attributes for a graph
            element. Signature: ``get_style(element) -> dict``, where
            ``element`` is either an :class:`Event` or an
            :class:`ArrowActivity`. The returned dictionary must contain
            keys ``color``, ``penwidth`` and ``fontsize``; the key
            ``weight`` is used for activity edges; the key ``tooltip``
            is used for hover tooltips (rendered only in SVG output).

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

        - Event nodes with early/late times and reserves (``EXP``
          component only)
        - Activity edges with duration and reserve information (``EXP``
          component only)

        The visualization uses the following coding:

        - Critical paths: red (``#ff0000``), penwidth=4, fontsize=16,
          weight=3
        - Sub-critical paths: orange (``#ffa000``), penwidth=3,
          fontsize=16, weight=2
        - Non-critical paths: black (``#000000``), penwidth=2,
          fontsize=14, weight=1
        - Dashed arrows: dummy activities

        Activity labels are embedded directly on edges through an
        intermediate label node; the label itself is drawn as a
        light-gray rounded box on the edge.
        The label font size is fixed at 12 for all activity labels.

        The activity duration shown in the label is the
        :attr:`ArrowActivity.duration` attribute -- the same frozen value
        that is used during the final forward pass and reported by
        :meth:`ArrowActivity.to_dict`; the backward pass applies
        :func:`_negative` locally inside
        :func:`ActivityOnArrowModel._compute_target`.

        When ``group_by_stage=True`` (the default), the topological order
        of events is preserved by clustering events with the same stage
        together, improving readability for large networks.

        Tooltips provided by ``get_style`` are rendered only in SVG
        output.
        """
        if not callable(get_style):
            raise TypeError(f"Parameter get_style must be callable, got {type(get_style)}")

        dot = graphviz.Digraph(node_attr={'shape': 'record', 'style': 'rounded'})
        dot.graph_attr['rankdir'] = 'LR'
        dot.graph_attr['splines'] = 'polyline'

        #======================================================================
        def _label_event(e):
            """Format event node label (early/late times, reserve)."""
            style = get_style(e)
            # Simple record label without HTML tags
            return '{{%d |{%.1f|%.1f}| %.2f}}' % (
                e.id, e.early[EXP], e.late[EXP], e.reserve[EXP]
            ), style

        #======================================================================
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
                lbl = f"{a.letter}\\nt={a.duration[EXP]:.1f}\\nr={a.reserve[EXP]:.2f}"
            else:          # dummy activity
                lbl = f"{a.letter}\\nr={a.reserve[EXP]:.2f}"

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
    n_old = ActivityOnArrowModel(wbs, src_old, dst_old)
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
            mean, var, a, b, err = _calculate_action_time_params(test_case)
            print(f"Test case {i + 1}: {test_case} -> mean={mean:.3f}, variance={var:.3f}")
        except ValueError as e:
            print(f"Test case {i + 1} error: {e}")

    # Demonstration of the new data formats in activities
    print("\n=== Demonstration of new time formats in activities ===")
    for i, activity in enumerate(n_old._activities[:8]):  # Show first 8 activities
        if activity.wbs_id != 0:  # Skip dummy activities
            print(f"Activity {i + 1}: wbs_id={activity.wbs_id}, letter='{activity.letter}'")
            print(f"  Expected effort: {activity.expected[EXP]:.3f}, Variance: {activity.expected[VAR]:.3f}")
            print(f"  Actual duration: {activity.duration[EXP]:.3f}")
            if activity.data:
                print(f"  Data fields: {list(activity.data.keys())}")

    # Advanced example: Resource-aware scheduling
    print("\n=== Advanced Example: Resource-Aware Scheduling ===")

    def resource_aware_duration(effort, activity, base_time, target):
        team = activity.data.get('team_size', 1)
        prod = activity.data.get('productivity', 1.0)

        if effort == 0 or team * prod <= 0:
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
    model_resource = ActivityOnArrowModel(wbs_resource, links=links_resource, duration=resource_aware_duration)

    # Show resource-aware durations
    for activity in model_resource._activities:
        if activity.wbs_id != 0:
            print(f"Activity {activity.letter}: "
                  f"Effort={activity.expected[EXP]:.1f}h, "
                  f"Duration={activity.duration[EXP]:.1f} days, "
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
    n_new1 = ActivityOnArrowModel(wbs, links=links_format1)
    print("Successfully created model with format 1")

    # New format 2 (two columns)
    print("\n3. New format 2 (two columns):")
    links_format2 = [
        [1, 5], [2, 5], [3, 5], [2, 6], [3, 6], [3, 7], [4, 7],
        [1, 8], [6, 8], [7, 8], [5, 9], [6, 9], [7, 9], [3, 10],
        [6, 10], [7, 10], [6, 11], [8, 11], [9, 11], [7, 12],
        [8, 12], [9, 12], [10, 12]
    ]
    n_new2 = ActivityOnArrowModel(wbs, links=links_format2)
    print("Successfully created model with format 2")

    # New format 3 (dictionary)
    print("\n4. New format 3 (dictionary):")
    links_format3 = {
        'src': [1, 2, 3, 2, 3, 3, 4, 1, 6, 7, 5, 6, 7, 3, 6, 7, 6, 8, 9, 7, 8, 9, 10],
        'dst': [5, 5, 5, 6, 6, 7, 7, 8, 8, 8, 9, 9, 9, 10, 10, 10, 11, 11, 11, 12, 12, 12, 12]
    }
    n_new3 = ActivityOnArrowModel(wbs, links=links_format3)
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
