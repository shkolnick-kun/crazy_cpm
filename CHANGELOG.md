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

If your callback branches on target, remove the removed cases. Note that
**early**, **late**, and **None** are no longer passed to the callback at
all — see the next section.

### Duration callback invocation contract

The callback is invoked **only during the preliminary passes**
(early_pre and late_pre). It is **not** invoked during the final passes
and **not** during network post-processing. Two important clarifications:

* **Preliminary passes:** in each preliminary pass the callback is invoked
  **three times per activity** — for the EXP, UPR, and LWR components of
  the effort estimate. Across both preliminary passes this is six calls
  per activity. All three calls within a pass receive the **same** base
  time (the EXP component of the event's base time). This guarantees the
  ordering **dur[LWR] <= dur[EXP] <= dur[UPR]**.
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
* **Scenario-aware sanitizers**: the early passes now enforce

  * early_start[LWR] >= 0 (clamped, not raised);
  * early_end[c] >= early_start[c] for c in {EXP, UPR, LWR}
    (raised beyond the error bound, clamped within it);
  * early_start[EXP] and early_end[EXP] lie within [LWR, UPR]
    (raised beyond the error bound, clamped within it).

  The late passes enforce

  * late_start[EXP] >= early_start[EXP];
  * late_end[EXP] >= late_start[EXP];
  * late_start[EXP] and late_end[EXP] lie within [LWR, UPR].

  Events additionally get early[EXP] inside [LWR, UPR] on the early pass
  and late[EXP] >= early[EXP] plus late[EXP] inside [LWR, UPR] on the
  late pass. The same invariants are checked against the _pre attributes
  on the preliminary passes.

---

## Changed

* **Public API surface for component access.** Time / effort / reserve
  values are now uniformly 5-component arrays. Use the named component
  constants (EXP, VAR, LWR, UPR, ERR) for readability.
* **CPM invariants in the early passes** cover:

  * early_start[LWR] >= 0 (clamped);
  * early_end[c] >= early_start[c] for c in {EXP, UPR, LWR};
  * early_start[EXP] and early_end[EXP] lie within [LWR, UPR].

  A violation beyond the accumulated error bound raises RuntimeError;
  violations within the bound are clamped by shifting all three
  components together. The EXP-in-[LWR,UPR] check is a new v2.0.0
  invariant.

* **CPM invariants in the late passes** cover:

  * late_start[EXP] >= early_start[EXP];
  * late_end[EXP] >= late_start[EXP];
  * late_start[EXP] and late_end[EXP] lie within [LWR, UPR].

  The scenario bounds (LWR, UPR) are *not* re-checked against the
  late_end >= late_start ordering, because the signed backward-pass
  transform (**negative**) swaps and negates them, so the LWR/UPR
  ordering is reversed by construction. Only the EXP component carries
  the CPM ordering.

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
* **Split of the time-parameter computation.** The monolithic
  **_compute_time_params** is now two methods:
  **_compute_time_params_pre** (preliminary passes plus the freezing of the
  per-activity durations via **ArrowActivity.compute_final_duration**) and
  **_compute_time_params** (final passes plus the time reserves). The
  triangle-group optimization **_network_postprocess** runs between them,
  so it compares already frozen **ArrowActivity.duration** values and the
  final passes re-traverse the post-processed topology.
* **Event stage is computed after the network post-processing.** The
  'stage' pass runs after **_network_postprocess**, so **Event.stage**
  describes the post-processed topology. As a consequence, the final
  **Event.id** numbering (assigned by sorting the events by stage) and the
  layers produced by **viz(group_by_stage=True)** may differ from previous
  builds. Note that **Event.stage** is zero on every event while the
  preliminary and final passes run, so a duration callback must not rely
  on it.
* **Event renumbering moved to the end of construction.** **Event.id** is
  reassigned only after the topology has been finalised by
  **_network_postprocess**. The IDs seen by a duration callback during the
  time passes are therefore the construction-time IDs, not the final ones;
  **_network_postprocess** itself also works with the construction-time
  IDs.

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
   opt_pre, pes_pre, optimistic, pessimistic. The callback will be called
   three times per activity in each preliminary pass — once per component
   (EXP, UPR, LWR) with the same base time. Across both preliminary
   passes this is six calls per activity. Make sure the callback is pure
   (no side effects) and consistent across the calls.

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
