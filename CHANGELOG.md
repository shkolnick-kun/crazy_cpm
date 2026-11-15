# Changelog

## v2.0.0

**Release type:** major (breaking changes)

This release overhauls the internal data model: the previous 3-component
representation of time/effort values is replaced by a unified 5-component
representation that carries the optimistic and pessimistic scenario bounds
inline. As a result, the public class names and several API details have
changed. Existing code targeting v1.0.4 will need to be updated.

---

## Breaking changes

### Renamed classes

* **NetworkModel** renamed to **ActivityOnArrowModel**.
* **Activity** renamed to **ArrowActivity**.

Both old names are gone — there are no compatibility aliases. Update imports,
type hints, and isinstance checks accordingly.

### Component arrays: 3 elements to 5 elements

The layout of time, effort, and reserve arrays has changed.

**Before (v1.0.4):** shape (3,), layout [RES, VAR, ERR]

**After (v2.0.0):** shape (5,), layout [EXP, VAR, LWR, UPR, ERR]

New component semantics:

* **EXP** — expected (mean) value. This replaces the old RES index.
* **VAR** — variance (unchanged semantics).
* **LWR** — lower limit (optimistic scenario bound).
* **UPR** — upper limit (pessimistic scenario bound).
* **ERR** — error bound (unchanged semantics).

The old module-level constant **RES** has been removed. Use **EXP** instead.
New module-level constants **LWR** and **UPR** have been added.

Code like:

    activity.early_start[RES]

must become:

    activity.early_start[EXP]

and code that reads scenario values must switch to the new LWR / UPR slots.

### Removed activity scenario attributes

The following attributes on activities have been removed:

* optimistic (scalar attribute)
* pessimistic (scalar attribute)
* opt_start, opt_end, opt_dur
* pes_start, pes_end, pes_dur

Scenario values now live in the LWR / UPR slots of the corresponding
component arrays:

* Optimistic effort — **activity.expected[LWR]**
* Pessimistic effort — **activity.expected[UPR]**
* Optimistic start — **activity.early_start[LWR]**
* Pessimistic start — **activity.early_start[UPR]**
* Optimistic end — **activity.early_end[LWR]**
* Pessimistic end — **activity.early_end[UPR]**

The output of **to_dict()** is unchanged in field names, so DataFrame
consumers keep working: the fields optimistic, opt_start, opt_end,
pessimistic, pes_start, pes_end are still present in the dictionary, but
they are now sourced from the LWR / UPR components of the underlying arrays.

### Removed event scenario attributes

The preliminary scenario attributes **opt_pre** and **pes_pre** have been
removed from **Event**, along with their debug output in
**Event.to_dict()**. The final scenario values are still exposed via
**Event.early[LWR]** and **Event.early[UPR]** (and mirrored in the
optimistic / pessimistic fields of the output dictionary).

### compute_final_durations renamed and simplified

**Activity.compute_final_durations()** is now
**ArrowActivity.compute_final_duration()** (singular).

The method no longer performs the scenario-range clamp and the b-growth
adjustment that existed in v1.0.4. It simply:

1. Derives the forward and backward duration estimates from the preliminary
   times.
2. Selects the most certain estimate via the internal **choice** helper.
3. Stores the result in **self.duration**.

### _calculate_action_time_params return type

**Return value changed** from a 5-tuple of scalars
(mean, variance, optimistic, most_likely, pessimistic) to a single
**numpy.ndarray** of shape (5,) in the new [EXP, VAR, LWR, UPR, ERR]
layout.

In addition, the function now:

* Performs a deterministic check: if **sqrt(VAR) <= ERR**, the activity is
  forced to deterministic (LWR = UPR = EXP, VAR = 0).
* Applies an MPERT-style mode correction (gamma) to keep the mean strictly
  inside the (LWR, UPR) interval, so the beta fit stays well-posed.

### Duration callback target values

The **target** argument passed to the user-supplied duration callback no
longer takes the values opt_pre, pes_pre, optimistic, or pessimistic.
Remaining values are:

* early_pre
* late_pre
* early
* late
* None (during network post-processing)

If your callback branches on target, remove the removed cases.

### Duration callback invocation contract

The callback is still invoked only during the preliminary passes and during
network post-processing. Two important clarifications:

* **Preliminary passes:** the callback is invoked **three times per
  activity** — for the EXP, UPR, and LWR components of the effort estimate.
  All three calls receive the **same** base time (the EXP component of the
  event's base time). This guarantees the ordering
  **dur[LWR] <= dur[EXP] <= dur[UPR]**.
* **Final passes:** the callback is **not** invoked. The frozen
  **ArrowActivity.duration** value is used instead (the backward pass
  applies the new **negative** helper to it).

### viz() default for group_by_stage

The default value of **group_by_stage** in **ActivityOnArrowModel.viz()**
changed from **False** to **True**. Pass group_by_stage=False explicitly to
restore the old behavior.

---

## Added

* **Module-level constants LWR and UPR** for the scenario bounds of the
  component arrays.
* **negative(vec)** helper — returns the signed negation of a 5-component
  array. Used by the backward (late) pass: negates EXP, swaps and negates
  LWR / UPR, preserves VAR and ERR.
* **_unpack_components(components)** helper — validates and unpacks a
  5-component array.
* **Private beta-distribution routines** on component vectors:
  **_fit_beta**, **_calc_ppf**, **_calc_cdf**. These operate directly on
  the full 5-component array and are used internally by the model.
* **MPERT-style mode correction** in **_calculate_action_time_params** and
  in the internal duration pipeline: the mean is pushed strictly inside
  the (LWR, UPR) interval to keep the beta fit well-posed, and VAR is
  recomputed from the beta-distribution formula.
* **Scenario-aware duration computation** inside the preliminary pass: the
  LWR and UPR scenario durations are computed alongside the EXP duration,
  all with the same base time.
* **Scenario-aware sanitizers**: early passes now check the CPM ordering
  invariants (early_end >= early_start) separately for EXP, UPR, and LWR;
  the late passes check only EXP.

---

## Changed

* **Public API surface for component access.** Time / effort / reserve
  values are now uniformly 5-component arrays. Use the named component
  constants (EXP, VAR, LWR, UPR, ERR) for readability.
* **CPM invariants in the early passes** are checked for all three of
  EXP, UPR, and LWR (early_end[c] >= early_start[c]). A violation beyond
  the accumulated error bound raises RuntimeError; violations within the
  bound are clamped by shifting all three components together.
* **CPM invariants in the late passes** are checked only for the EXP
  component, because the signed backward-pass transform of the scenario
  bounds does not preserve the (LWR, UPR) ordering in the same way as it
  does for EXP.
* **Clamping of negative lower bounds.** When early[LWR] < 0 on the early
  passes (or early_pre[LWR] < 0 on the preliminary passes), the bound is
  clamped to zero by shifting all three components of the array upward by
  -LWR. This preserves the shape of the distribution rather than collapsing
  it. Previously (v1.0.4) only the RES component was clamped.
* **Reserve rounding.** The condition for rounding a reserve to zero
  changed from **abs(res[RES]) > res[ERR]** to **res[EXP] < res[ERR]** (no
  absolute value, plus explicit LWR / UPR shift). The LWR and UPR
  components are shifted by -res[EXP] to keep the reserve triple
  consistent after rounding. Negative reserves beyond the error bound still
  raise RuntimeError.
* **beta fit via method of moments.** The internal fitting routine now
  also returns (None, None) when the resulting shape parameters are
  non-positive, in addition to the previous deterministic cases.
* **Public fit_beta, calc_ppf, calc_cdf** are now thin wrappers over the
  private 5-component versions. Their signatures are unchanged (scalar
  arguments M, D, a, b, err), so existing callers continue to work, but
  the underlying implementation is shared with the model.
* **Event docstring and reserved fields.** The reserved field names of
  **Event.to_dict()** have changed to reflect the removal of opt_pre and
  pes_pre. Users relying on those field names in the output should switch
  to **early[LWR]** and **early[UPR]**.

---

## Removed

* Old constants **RES** (replaced by EXP).
* Activity attributes: **optimistic**, **pessimistic**, **opt_start**,
  **opt_end**, **opt_dur**, **pes_start**, **pes_end**, **pes_dur**.
* Event attributes: **opt_pre**, **pes_pre** (and their debug output).
* Preliminary (opt_pre / pes_pre) and final (optimistic / pessimistic)
  passes as separate traversal targets. Scenario propagation now happens
  inline via the LWR / UPR components.
* Scenario-range clamp and b-growth adjustment inside
  compute_final_duration (previously compute_final_durations).
* The old tuple-returning variant of _calculate_action_time_params.

---

## Notes for migrating from v1.0.4

1. Rename all references:

   * NetworkModel becomes ActivityOnArrowModel
   * Activity becomes ArrowActivity
   * RES becomes EXP
   * compute_final_durations becomes compute_final_duration

2. Replace attribute reads:

   * activity.optimistic becomes activity.expected[LWR]
   * activity.pessimistic becomes activity.expected[UPR]
   * activity.opt_start becomes activity.early_start[LWR]
   * activity.pes_start becomes activity.early_start[UPR]
   * event.optimistic becomes event.early[LWR]
   * event.pessimistic becomes event.early[UPR]

3. Update duration callbacks: remove any branches on target values
   opt_pre, pes_pre, optimistic, pessimistic. The callback may now be
   called up to three times per activity per preliminary pass — once per
   component (EXP, UPR, LWR) with the same base time. Make sure the
   callback is pure (no side effects) and consistent across the three
   calls.

4. Update scalar indexing: change arr[RES] to arr[EXP]. Multi-element
   initialization must now allocate length 5 (e.g. numpy.zeros(5) instead
   of numpy.zeros(3)).

5. Update visualization calls if you rely on the old default of
   group_by_stage=False: pass it explicitly to preserve the previous
   layout behavior.

---

## v1.0.4

Previous stable release. No changes to v1.0.4 in this entry — it is
listed here only as the baseline for the migration described above.
