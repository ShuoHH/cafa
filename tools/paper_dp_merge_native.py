"""Independent, measured compact20/Top-2 coarsening of frozen mainline cuts.

Only existing boundaries may disappear. Recommendations freeze before trials;
DP consumes certified same-run frame costs, never predictions of storage size.
This supplements historical boundaries; it is NOT new end-to-end throughput.
"""
from __future__ import annotations

from collections import Counter
import math
from pathlib import Path
import shutil
import time

import paper_mainline_native as native
from paper_segmentation_selected import _geometry_allowed
from partition_dp_coarsening import enumerate_union_intervals, coarsen_partition_dp


VERSION = "frozen_mainline_compact20_top2_dp_merge_v1"


def _positive_int(value, name):
    if type(value) is not int or value < 1:
        raise ValueError(name + " must be a positive integer")
    return value


def run_dp_merge(job, partition, out, exporter, fixed_exe, *, policy,
                 fixed_regions=None, timeout=300, value_budget_multiplier=None,
                 trial_limit=None, minimum_free_gib=1.0, execute=None,
                 service_factory=None, source_boundaries_historical=True):
    """Price every affordable consecutive union, then solve the measured DAG.

    Original segments and the whole-series edge are scheduled first. If either
    cannot be priced, its guarantee is explicitly unavailable, not invented.
    A native failure is fatal; declared unsupported actions are not failures.
    The DP may still run on a budget-limited graph, labelled as such. Original
    geometry, model weights, training folds and historical outputs never change.
    """
    if value_budget_multiplier is not None:
        _positive_int(value_budget_multiplier, "value budget multiplier")
    if trial_limit is not None:
        _positive_int(trial_limit, "trial limit")
    if type(source_boundaries_historical) is not bool:
        raise ValueError("source_boundaries_historical must be Boolean")
    if (isinstance(minimum_free_gib, bool) or not isinstance(minimum_free_gib, (int, float))
            or not math.isfinite(minimum_free_gib) or minimum_free_gib < 0):
        raise ValueError("minimum free GiB must be finite and nonnegative")
    began = time.perf_counter()
    raw = native._job(job)
    original = native._geometry(partition, job["values"])
    all_unions = enumerate_union_intervals(original)
    whole = (0, job["values"])
    fixed = None if fixed_regions is None else native._geometry(fixed_regions, job["values"])
    # The fixed-1024 reference never introduces extra DP boundaries.
    reference_regions = list(dict.fromkeys(original + [whole] + (fixed or [])))
    reference_set = set(reference_regions)
    order = reference_regions + [r for r in all_unions if r not in reference_set]
    out = native._fresh(out, raw, exporter, fixed_exe)
    protected = native._protected(raw, exporter, fixed_exe)
    native._json(out / "frozen_geometry.json", dict(
        version=VERSION, job=job, original_partition=original, union_regions=all_unions,
        fixed1024_reference=fixed, pricing_order=order, only_delete_existing_boundaries=True,
        original_mainline_unchanged=True, training_folds_unchanged=True,
        feature_count=20, top_k=2, active_actions=list(native.RETAINED_ACTIONS)))
    records, costs, rows, selections = [], {}, [], []
    union_set = set(all_unions)
    original_set = set(original)
    fixed_set = set(fixed or [])
    phase_totals = {}
    with native.single_thread():
        prediction_started = time.perf_counter()
        recommendations = native._recommend(policy, raw, order, 2, _geometry_allowed)
        prediction_seconds = time.perf_counter() - prediction_started
        predictions = [dict(start=s, length=n, ranked_actions=list(recommendations[s, n]),
            frozen_before_native=True, test_costs_used=False, test_support_status_used=False)
            for s, n in order]
        native._json(out / "predictions_before_native.json", predictions)
        # Each of two semantic actions covers exactly this region, even when
        # ALP emits prefix+tail frames. Four physical keys/region is a safe cap.
        # The default therefore funds the entire graph, not the old 128N cap.
        value_budget = (2 * sum(r[1] for r in order) if value_budget_multiplier is None
                        else value_budget_multiplier * job["values"])
        actual_trial_limit = 4 * len(order) if trial_limit is None else trial_limit
        service = native._service(job, out, exporter, fixed_exe, timeout, execute,
            policy=native._FrozenRecommendations(recommendations), service_factory=service_factory,
            value_budget=value_budget, trial_limit=actual_trial_limit)
        if not isinstance(service, native.Top2FlatPoolService):
            raise ValueError("DP merge requires the strict frozen-Top2 native service")
        all_keys = {key for r in order if recommendations[r] for key in service._needed(r)}
        planned_volume = sum(key[1] for key in all_keys)
        native._json(out / "pricing_plan.json", dict(
            possible_dp_edges=len(all_unions), reference_regions=len(reference_regions),
            unique_physical_trials_if_all=len(all_keys), physical_value_volume_if_all=planned_volume,
            value_budget=service.value_budget, trial_limit=service.trial_limit,
            auto_fund_full_graph=value_budget_multiplier is None and trial_limit is None,
            budget_sufficient_for_full_pricing=(planned_volume <= service.value_budget
                                              and len(all_keys) <= service.trial_limit),
            file_header_bytes=native.fmt.FILE_HEADER.size,
            edge_cost_includes_all_physical_frame_headers=True))
        for index, region in enumerate(order, 1):
            start, length = region
            phase = ("before_dp" if region in original_set else "whole" if region == whole
                     else "fixed1024" if region in fixed_set else "merge_candidates")
            counters_before = dict(service.counters)
            trial_start = len(service.trials)
            tick = time.perf_counter()
            record = dict(start=start, length=length, phase=phase, dp_edge=region in union_set,
                recommended_actions=list(recommendations[region]), status="pending", frame_bits=None)
            if not recommendations[region]:
                record.update(status="no_feasible_recommendation")
            elif shutil.disk_usage(out).free < minimum_free_gib * 1024**3:
                record.update(status="disk_skipped")
            else:
                fresh_keys = [key for key in service._needed(region) if key not in service.cache]
                if (service.counters["trial_count"] + len(fresh_keys) > service.trial_limit
                        or service.counters["trial_value_volume"] + sum(k[1] for k in fresh_keys) > service.value_budget):
                    record.update(status="budget_skipped")
                else:
                    service.best_many([region])
                    options = service.options_for(region)
                    failures = [t for t in service.trials[trial_start:] if t["status"] == "failed"]
                    if failures:
                        native._json(out / "failed_trials.json", failures)
                        native._json(out / "failed_pricing_region.json", dict(record,
                            status="native_failed", counters=dict(service.counters)))
                        raise RuntimeError("native trial failed; see " + str(out / "failed_trials.json"))
                    if any(o["status"] not in ("passed", "unsupported") for o in options):
                        raise ValueError("pricing left a recommendation pending or failed")
                    valid = [o for o in options if o["status"] == "passed"]
                    record["options"] = options
                    if not valid:
                        record.update(status="all_recommended_actions_unsupported")
                    else:
                        winner = min(valid, key=lambda o: o["frame_bytes"])
                        if (type(winner["frame_bytes"]) is not int or winner["frame_bytes"] <= 0
                                or winner.get("roundtrip_verified") is not True):
                            raise ValueError("DP edge lacks a certified positive actual frame size")
                        record.update(status="priced", frame_bits=8 * winner["frame_bytes"],
                            chosen_action=winner["codec"], physical_keys=winner["physical_keys"])
                        costs[region] = record["frame_bits"]
            record["wall_seconds"] = time.perf_counter() - tick
            record["added_trials"] = service.counters["trial_count"] - counters_before["trial_count"]
            record["added_trial_value_volume"] = (service.counters["trial_value_volume"]
                                                 - counters_before["trial_value_volume"])
            totals = phase_totals.setdefault(phase, dict(trial_count=0, trial_value_volume=0, wall_seconds=0.0))
            for key, field in (("trial_count", "added_trials"), ("trial_value_volume", "added_trial_value_volume"),
                               ("wall_seconds", "wall_seconds")):
                totals[key] += record[field]
            records.append(record)
            native._json(out / "progress.json", dict(version=VERSION, series=job["series"],
                priced_or_skipped_regions=index, scheduled_regions=len(order), latest=record,
                counters=dict(service.counters), status_counts=dict(Counter(r["status"] for r in records))))
            if index == 1 or index % 16 == 0 or index == len(order):
                print("  %s DP pricing %d/%d; trials=%d values=%d; last=%s" %
                    (job["series"], index, len(order), service.counters["trial_count"],
                     service.counters["trial_value_volume"], record["status"]), flush=True)
        pricing_finished = time.perf_counter()
        dp_records = [r for r in records if r["dp_edge"]]
        full_pricing = all(r["status"] in ("priced", "no_feasible_recommendation",
            "all_recommended_actions_unsupported") for r in dp_records)
        dp_costs = {r: costs[r] for r in all_unions if r in costs}
        dp_started = time.perf_counter()
        missing_original = [r for r in original if r not in dp_costs]
        dp = None if missing_original else coarsen_partition_dp(
            original, dp_costs, full_pricing_complete=full_pricing)
        dp_seconds = time.perf_counter() - dp_started
        if dp is not None:
            native._json(out / "dp_solution.json", dp)
        geometry = dict(before_dp=original, whole=[whole])
        if fixed is not None:
            geometry["fixed1024"] = fixed
        if dp is not None:
            geometry["after_dp"] = [tuple(r) for r in dp["partition"]]
        counters = dict(service.counters)
        for layout in ("before_dp", "whole", "fixed1024", "after_dp"):
            if layout == "fixed1024" and fixed is None:
                continue
            regions = geometry.get(layout)
            row = dict(dataset=job["dataset"], series=job["series"], sha256=job["sha256"],
                values=job["values"], layout=layout, status="unavailable", detail="",
                complete_file_bits=None, complete_file_roundtrip_verified=False,
                boundary_segments=len(regions) if regions is not None else None,
                feature_count=20, top_k=2, same_run_prices=True, full_pool_exhaustive=False)
            missing = [] if regions is None else [r for r in regions if r not in costs]
            if regions is None or missing:
                row["detail"] = ("original partition lacks certified prices; no fabricated DP path"
                    if regions is None else "unpriced/unsupported regions: " + repr(missing))
                rows.append(row)
                continue
            path = out / (layout + ".f64pool")
            info, chosen = native._assemble(service, regions, path, job["values"], recommendations)
            expected = 8 * native.fmt.FILE_HEADER.size + sum(costs[r] for r in regions)
            if info["complete_file_bits"] != expected:
                raise ValueError("DP stored bytes differ from certified edge sum plus one file header")
            proof, seconds = native._verify(path, raw, exporter, fixed_exe, timeout, execute)
            native._json(path.with_suffix(".verification.json"), proof)
            row.update(status="passed", complete_file_bits=expected, complete_file_roundtrip_verified=True,
                container_path=str(path), container_sha256=native.replay.sha256(path),
                decoded_sha256=proof["decoded_sha256"], verification_seconds=seconds)
            selections.extend(dict(layout=layout, **choice) for choice in chosen)
            rows.append(row)
            if service.counters != counters:
                raise ValueError("DP output assembly unexpectedly re-ran native encoding")
        by = {r["layout"]: r for r in rows}
        if dp is not None:
            after = by["after_dp"]
            if after["status"] != "passed" or after["complete_file_bits"] > by["before_dp"]["complete_file_bits"]:
                raise AssertionError("certified DP result became worse than its input layout")
            if by["whole"]["status"] == "passed" and after["complete_file_bits"] > by["whole"]["complete_file_bits"]:
                raise AssertionError("certified DP result became worse than its measured whole-series edge")
        trials = [dict(t) for t in service.trials]
        # Fixed1024 is only a diagnostic. Its shared cache may happen to pay
        # for a physical key later needed by DP; do not call that DP work free.
        # Report an independent unique-key charge for DP versus its input chain.
        attempted_dp_regions = {tuple((r["start"], r["length"])) for r in records
                                if r["dp_edge"] and r["status"] in
                                ("priced", "all_recommended_actions_unsupported")}
        baseline_keys = {key for r in original if r in attempted_dp_regions and recommendations[r]
                         for key in service._needed(r) if key in service.cache}
        dp_keys = {key for r in attempted_dp_regions if recommendations[r]
                   for key in service._needed(r) if key in service.cache}
        extra_dp_keys = dp_keys - baseline_keys
        logical_dp_work = dict(
            baseline_unique_physical_trials=len(baseline_keys),
            baseline_unique_physical_value_volume=sum(k[1] for k in baseline_keys),
            dp_total_unique_physical_trials=len(dp_keys),
            dp_total_unique_physical_value_volume=sum(k[1] for k in dp_keys),
            dp_additional_physical_trials=len(extra_dp_keys),
            dp_additional_physical_value_volume=sum(k[1] for k in extra_dp_keys),
            fixed_reference_cache_reuse_not_counted_as_free_dp_work=True)
        for trial in trials:
            if trial["status"] == "passed":
                native.frozen.assert_hash(trial["candidate_path"], trial["candidate_sha256"], "DP paid frame")
        native._check(protected)
        result = dict(version=VERSION, job=job, rows=rows, selections=selections,
            geometry=geometry, predictions=predictions, pricing_records=records,
            trials=trials, counters=counters, phase_totals=phase_totals,
            logical_dp_work=logical_dp_work,
            dp=dp, full_pricing_complete=full_pricing, possible_dp_edges=len(all_unions),
            priced_dp_edges=len(dp_costs), value_budget=service.value_budget, trial_limit=service.trial_limit,
            complete_pricing_physical_value_volume=planned_volume, missing_original_regions=missing_original,
            active_actions=list(native.RETAINED_ACTIONS), only_delete_existing_boundaries=True,
            prediction_seconds=prediction_seconds, pricing_wall_seconds=pricing_finished - began,
            dp_compute_seconds=dp_seconds, supplemental_wall_seconds=time.perf_counter() - began,
            protected=protected, same_run_frozen_frames=True, independently_decoded_complete_files=True,
            original_geometry_and_models_unchanged=True, hidden_fallback_used=False,
            source_boundaries_historical=source_boundaries_historical, end_to_end_workflow_timing=False,
            pure_codec_throughput=False, all40_raw_features_then_compact20_projection=True)
        native._json(out / "dp_merge_manifest.json", result)
        return result
