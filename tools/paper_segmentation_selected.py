"""Frozen segmentation comparisons using only compact20 Top-2 actions.

NeaTS or another external segmentation supplies geometry only. All geometries
are frozen and predicted together before the first native trial, then priced
using one strict Top-2 service. No full-pool search or hidden fallback occurs.
"""
from __future__ import annotations

from collections.abc import Mapping

import paper_mainline_native as native


VERSION = "frozen_segmentation_compact20_top2_same_price_v1"


def _geometry_allowed(region):
    """Declared action feasibility from length alone, never measured support."""
    _, length = region
    result = set(native.RETAINED_ACTIONS) - {"alp"}
    if length % 1024 == 0:
        result.add("alp")
    elif length > 1024:
        result.update("alp+tail:" + action for action in native.RETAINED_ACTIONS
                      if action in ("gorilla", "chimp", "chimp128", "tsxor", "rle+tsxor"))
    return result


def selected_run_layouts(job, layouts, out, exporter, fixed_exe, *, policy,
                         timeout=300, execute=None, service_factory=None):
    """Write each complete layout using one shared frozen compact20 Top-2.

    ``policy`` must be the caller-authenticated, loaded raw-only compact20 model
    for this held-out source. This adapter never trains or loads another model.
    It trials only these recommendations, not the retained-eight full pool.
    """
    raw = native._job(job)
    if not isinstance(layouts, Mapping) or not layouts:
        raise ValueError("require a nonempty mapping of named frozen layouts")
    if any(not isinstance(name, str) or not name.strip() for name in layouts):
        raise ValueError("layout names must be nonempty strings")
    geometry = {name: native._geometry(regions, job["values"])
                for name, regions in layouts.items()}
    union = sorted({region for regions in geometry.values() for region in regions})
    out = native._fresh(out, raw, exporter, fixed_exe)
    protected = native._protected(raw, exporter, fixed_exe)
    value_budget, trial_limit = 128 * job["values"], max(100000, 32 * len(union))
    native._json(out / "frozen_geometry_before_native.json", dict(
        job=job, geometry=geometry, geometry_union=union,
        frozen_before_native=True, feature_count=20, top_k=2,
        policy_mode="compact20_top2", full_pool_exhaustive=False,
        selector_training_used=False, boundary_search_used=False))
    rows, selections = [], []
    with native.single_thread():
        recommendations = native._recommend(policy, raw, union, 2, _geometry_allowed)
        predictions = [dict(start=start, length=length,
            ranked_actions=list(recommendations[start, length]),
            frozen_before_native=True, test_costs_used=False,
            test_support_status_used=False, feature_count=20, top_k=2,
            policy_mode="compact20_top2") for start, length in union]
        native._json(out / "predictions_before_native.json", predictions)
        nonempty_union = [region for region in union if recommendations[region]]
        service = native._service(job, out, exporter, fixed_exe, timeout, execute,
            policy=native._FrozenRecommendations(recommendations),
            service_factory=service_factory, value_budget=value_budget,
            trial_limit=trial_limit)
        if not isinstance(service, native.Top2FlatPoolService):
            raise ValueError("selected segmentation requires strict frozen-Top2 native service")
        if nonempty_union:
            service.best_many(nonempty_union)
        counters = dict(service.counters)
        action_rows = [dict(option, dataset=job["dataset"], series=job["series"],
            sha256=job["sha256"], start=start, length=length,
            action=option["codec"], options_scope="frozen_top2_only")
            for start, length in nonempty_union for option in service.options_for((start, length))]
        for serial, (layout, regions) in enumerate(geometry.items()):
            path = out / ("output_%03d.f64pool" % serial)
            empty_regions = [region for region in regions if not recommendations[region]]
            options = [option for region in regions if recommendations[region]
                       for option in service.options_for(region)]
            failed = sum(option["status"] == "failed" for option in options)
            pending = sum(option["status"] not in ("passed", "unsupported", "failed")
                          for option in options)
            complete = not empty_regions and failed == pending == 0
            row = dict(dataset=job["dataset"], series=job["series"],
                sha256=job["sha256"], values=job["values"], layout=layout,
                variant="frozen20", top_k=2, selector_used=True,
                feature_count=20, policy_mode="compact20_top2", status="unavailable", detail="", reason="",
                complete_file_bits=None, complete_file_roundtrip_verified=False,
                boundary_segments=len(regions), same_run_prices=True,
                recommended_actions_complete=complete, full_pool_exhaustive=False,
                empty_recommendation_region_count=len(empty_regions),
                empty_recommendation_regions=empty_regions,
                options_scope="frozen_top2_only", codec_throughput=False,
                failed_option_count=failed, pending_option_count=pending,
                unsupported_option_count=sum(option["status"] == "unsupported" for option in options))
            if empty_regions:
                row.update(reason="no_feasible_recommendation",
                    detail="frozen Top-2 has no geometry-feasible learned action for regions: " + repr(empty_regions))
                rows.append(row)
                continue
            try:
                info, chosen = native._assemble(service, regions, path, job["values"], recommendations)
                proof, seconds = native._verify(path, raw, exporter, fixed_exe, timeout, execute)
                row.update(status="passed" if complete else "verified_output_incomplete_recommendations",
                    detail="" if complete else "verified output; failed or unvisited Top-2 recommendations remain",
                    complete_file_bits=info["complete_file_bits"],
                    complete_file_roundtrip_verified=True, container_path=str(path),
                    container_sha256=native.replay.sha256(path),
                    decoded_sha256=proof["decoded_sha256"], verification_seconds=seconds)
                selections.extend(dict(layout=layout, variant="frozen20", top_k=2,
                                       **choice) for choice in chosen)
                native._json(path.with_suffix(".verification.json"), proof)
            except (OSError, ValueError, KeyError, TypeError) as error:
                row["detail"] = str(error)
            rows.append(row)
            if service.counters != counters:
                raise ValueError("same-price selected file assembly repeated paid native trials")
    for trial in service.trials:
        if trial["status"] == "passed":
            native.frozen.assert_hash(trial["candidate_path"], trial["candidate_sha256"],
                                      "frozen selected segmentation paid frame")
    native._check(protected)
    result = dict(version=VERSION, rows=rows, action_rows=action_rows,
        selections=selections, selection=selections, predictions=predictions,
        trials=service.trials, pool_trials=service.trials, protected=protected,
        geometry=geometry, geometry_union=union, counters=counters,
        native_region_union=nonempty_union,
        empty_recommendation_regions=[region for region in union if not recommendations[region]],
        active_actions=list(native.RETAINED_ACTIONS), value_budget=value_budget,
        trial_limit=trial_limit, feature_count=20, top_k=2,
        policy_mode="compact20_top2", selector_used=True,
        selector_training_used=False, boundary_search_used=False,
        same_run_price_cache=True, new_native_encoding=True,
        independently_decoded_complete_files=True, geometry_frozen_before_native=True,
        predictions_frozen_before_native=True, full_pool_exhaustive=False,
        hidden_fallback_used=False, throughput_measured=False)
    native._json(out / "segmentation_selected_manifest.json", result)
    return result
