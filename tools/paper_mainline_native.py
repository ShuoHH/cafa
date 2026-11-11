"""Measured retained-eight output quality and isolated cold-cache workflow work.

Quality uses one same-run exhaustive physical-price cache. Speed uses separate
fresh caches for every arm/repeat and is harness work, not pure codec throughput.
The public release uses the same fixed eight actions at every active stage.
Legacy fourteen-action entrypoints below are disabled, not silently relabelled.
"""
from __future__ import annotations

from contextlib import contextmanager
import copy
from dataclasses import asdict
import json
import math
import os
from pathlib import Path
import statistics
import time
from types import SimpleNamespace

from btr_search_ledger import BudgetedRegionView
from flat_pool_backend import FlatPoolService, FlatPoolEvaluator, DEFAULT_ACTIONS
import flat_pool_format as fmt
import frozen_btr_extension as frozen
import replay_frozen_files as replay
from selector_consistent_backend import RETAINED_ACTIONS, Top2FlatPoolService, ConsistentTop2Evaluator
from codec_pool import check_choices


VERSION = "mainline_retained8_native_measurement_v1"
THREAD_ENV = dict(CLASS_NUMBA_DISK_CACHE="0", PYTHONDONTWRITEBYTECODE="1", NUMBA_NUM_THREADS="1",
                  OMP_NUM_THREADS="1", OPENBLAS_NUM_THREADS="1", MKL_NUM_THREADS="1", BLIS_NUM_THREADS="1")


@contextmanager
def single_thread():
    previous = {k: os.environ.get(k) for k in THREAD_ENV}
    os.environ.update(THREAD_ENV)
    try:
        # Environment variables control fresh native/detector subprocesses;
        # threadpoolctl also limits libraries already imported in this process.
        from threadpoolctl import threadpool_limits
        with threadpool_limits(limits=1):
            yield
    finally:
        for key, value in previous.items():
            if value is None:
                os.environ.pop(key, None)
            else:
                os.environ[key] = value


def _json(path, value):
    Path(path).write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + "\n", encoding="utf-8")


def _job(job):
    required = ("dataset", "series", "sha256", "input", "values")
    if any(k not in job for k in required) or type(job["values"]) is not int or not 1 <= job["values"] <= 102400:
        raise ValueError("require complete raw job identity and 1..102400 values")
    replay.digest_string(job["sha256"], "raw SHA")
    raw = Path(job["input"]).resolve()
    if raw.stat().st_size != 8 * job["values"] or replay.sha256(raw) != job["sha256"]:
        raise ValueError("raw input identity/count differs")
    return raw


def _geometry(regions, n):
    regions = [tuple(r) for r in regions]
    cursor = 0
    for start, length in regions:
        if type(start) is not int or type(length) is not int or start != cursor or not 1 <= length <= n - cursor:
            raise ValueError("semantic layout must cover every original value exactly once")
        cursor += length
    if not regions or cursor != n:
        raise ValueError("semantic layout is empty/incomplete")
    return regions


def _fresh(out, raw, exporter, fixed_exe):
    out = Path(out).resolve()
    if out.exists():
        raise ValueError("fresh native measurement output required")
    for source in map(lambda p: Path(p).resolve(), (raw, exporter, fixed_exe)):
        if out == source or out in source.parents or source in out.parents:
            raise ValueError("new output overlaps preserved input/native tool")
    out.mkdir(parents=True)
    return out


def _protected(raw, exporter, fixed_exe):
    return {str(Path(p).resolve()): replay.sha256(p) for p in (raw, exporter, fixed_exe)}


def _check(protected):
    for path, digest in protected.items():
        frozen.assert_hash(path, digest, "native measurement preserved source")


def _service(job, out, exporter, fixed_exe, timeout, execute, *, actions=RETAINED_ACTIONS,
             policy=None, service_factory=None, value_budget=None, trial_limit=None):
    cls = service_factory or (Top2FlatPoolService if policy is not None else FlatPoolService)
    kwargs = dict(value_budget=value_budget or 32 * job["values"], trial_limit=trial_limit or 20000,
                  timeout=timeout, execute=execute or replay.execute_export, actions=tuple(actions))
    if policy is not None:
        kwargs["policy"] = policy
    return cls(job=dict(n=job["values"], raw=job["input"], key=(job["dataset"], job["series"], job["sha256"], VERSION)),
               folder=Path(out) / "native", exporter=exporter, btr=fixed_exe, **kwargs)


def _recommend(policy, raw, regions, top_k, allowed):
    if policy is None or top_k not in (1, 2):
        raise ValueError("selected output requires a loaded raw-only Top-1/2 policy")
    if callable(getattr(policy, "predict", None)):
        rows = policy.predict(raw, list(regions))
    elif callable(getattr(policy, "predict_many", None)):
        rows = policy.predict_many(list(regions))
    elif callable(getattr(policy, "actions_for", None)):
        rows = [policy.actions_for(r) for r in regions]
    else:
        raise ValueError("policy must implement predict(raw,regions), predict_many, or actions_for")
    if not isinstance(rows, (tuple, list)) or len(rows) != len(regions):
        raise ValueError("policy recommendation coverage differs from requested regions")
    result = {}
    for region, row in zip(regions, rows):
        if isinstance(row, (str, bytes)):
            raise ValueError("recommendations must be action sequences")
        actions = tuple(row)
        if (len(actions) > 2 or len(actions) != len(set(actions))
                or any(not isinstance(a, str) or a not in allowed(region) for a in actions)):
            raise ValueError("policy recommends duplicate, unknown or infeasible semantic action")
        result[region] = actions[:top_k]
    return result


class _FrozenRecommendations:
    def __init__(self, recommendations):
        self.recommendations = recommendations

    def actions_for(self, region):
        return self.recommendations[tuple(region)]


def _assemble(service, regions, path, n, recommendations=None):
    frames, choices = [], []
    for region in regions:
        options = service.options_for(region)
        by_action = {o["codec"]: o for o in options}
        actions = list(by_action) if recommendations is None else list(recommendations[region])
        valid = [by_action[a] for a in actions if a in by_action and by_action[a]["status"] == "passed"]
        if not valid:
            raise ValueError("no lossless measured action among declared recommendations for %r" % (region,))
        winner = min(valid, key=lambda o: o["frame_bytes"])
        own = [f for key in winner["physical_keys"] for f in service._frames(tuple(key))]
        if frozen.frame_cost(own) != winner["frame_bytes"]:
            raise ValueError("selected measured physical bytes differ from cached price")
        frames.extend(own)
        choices.append(dict(start=region[0], length=region[1], codec=winner["codec"],
            frame_bytes=winner["frame_bytes"], physical_keys=winner["physical_keys"],
            recommended_actions=actions, options_scope="same_run_full_pool" if recommendations is None else "frozen_recommendations"))
    check_choices(choices)
    info = fmt.write_container(path, frames, n)
    with Path(path).open("r+b") as stream:
        os.fsync(stream.fileno())
    if info["complete_file_bits"] != 8 * (fmt.FILE_HEADER.size + sum(c["frame_bytes"] for c in choices)):
        raise ValueError("complete stored-file accounting differs")
    return info, choices


def _verify(path, raw, exporter, fixed_exe, timeout, execute):
    began = time.perf_counter()
    proof = fmt.verify_file(path, raw, exporter, fixed_exe, timeout, execute or replay.execute_export)
    return proof, time.perf_counter() - began


def run_quality(job, layouts, out, exporter, fixed_exe, *, policies, timeout=300,
                execute=None, service_factory=None):
    """One exhaustive retained-eight cache; all predictions freeze BEFORE prices.

    ``policies`` maps (variant_name, top_k) to already loaded raw-only policies.
    Four oracle layouts and selected fixed/adaptive outputs share identical
    persisted physical prices. Every resulting complete file is decoded anew.
    """
    raw = _job(job)
    if set(layouts) != {"whole", "fixed1024", "raw_class", "adaptive"}:
        raise ValueError("quality requires whole/fixed1024/raw_class/adaptive frozen layouts")
    geometry = {name: _geometry(regions, job["values"]) for name, regions in layouts.items()}
    out = _fresh(out, raw, exporter, fixed_exe)
    protected = _protected(raw, exporter, fixed_exe)
    rows, selections, predictions = [], [], []
    with single_thread():
        service = _service(job, out, exporter, fixed_exe, timeout, execute, service_factory=service_factory,
                           value_budget=128 * job["values"], trial_limit=100000)
        selected_union = sorted(set(geometry["fixed1024"] + geometry["adaptive"]))
        frozen_policies = {}
        allowed = lambda r: {action for action, _ in service._action_keys(r)}
        for (variant, top_k), policy in policies.items():
            if not isinstance(variant, str) or not variant or top_k not in (1, 2):
                raise ValueError("quality policy keys must be variant-name/Top-1-or-2 pairs")
            recommendations = _recommend(policy, raw, selected_union, top_k, allowed)
            frozen_policies[variant, top_k] = recommendations
            predictions.extend(dict(variant=variant, top_k=top_k, start=s, length=n,
                ranked_actions=list(recommendations[s, n]), frozen_before_native=True,
                test_costs_used=False, test_support_status_used=False) for s, n in selected_union)
        _json(out / "predictions_before_native.json", predictions)
        union = sorted({r for regions in geometry.values() for r in regions})
        service.best_many(union)
        counters = dict(service.counters)
        action_rows = [dict(option, dataset=job["dataset"], series=job["series"], sha256=job["sha256"],
            start=s, length=n, action=option["codec"]) for s, n in union for option in service.options_for((s, n))]
        arms = [(name, "all", "all", None) for name in geometry]
        arms += [(layout, variant, top_k, recommendations)
                 for (variant, top_k), recommendations in frozen_policies.items() for layout in ("fixed1024", "adaptive")]
        for serial, (layout, variant, top_k, recommendations) in enumerate(arms):
            path = out / ("output_%03d.f64pool" % serial)
            layout_options = [o for region in geometry[layout] for o in service.options_for(region)]
            failed_options = sum(o["status"] == "failed" for o in layout_options)
            pending_options = sum(o["status"] not in ("passed", "unsupported", "failed") for o in layout_options)
            pool_complete = failed_options == pending_options == 0
            row = dict(dataset=job["dataset"], series=job["series"], sha256=job["sha256"], values=job["values"],
                layout=layout, variant=variant, top_k=top_k, status="unavailable", detail="",
                complete_file_bits=None, complete_file_roundtrip_verified=False,
                boundary_segments=len(geometry[layout]), same_run_prices=True, codec_throughput=False,
                failed_option_count=failed_options, pending_option_count=pending_options,
                unsupported_option_count=sum(o["status"] == "unsupported" for o in layout_options),
                pool_complete=pool_complete)
            try:
                info, chosen = _assemble(service, geometry[layout], path, job["values"], recommendations)
                proof, seconds = _verify(path, raw, exporter, fixed_exe, timeout, execute)
                row.update(status="passed" if recommendations is not None or pool_complete else "verified_output_incomplete_pool",
                    complete_file_bits=info["complete_file_bits"],
                    complete_file_roundtrip_verified=True, container_path=str(path), container_sha256=replay.sha256(path),
                    decoded_sha256=proof["decoded_sha256"], verification_seconds=seconds)
                selections.extend(dict(layout=layout, variant=variant, top_k=top_k, **c) for c in chosen)
                _json(path.with_suffix(".verification.json"), proof)
            except (OSError, ValueError, KeyError, TypeError) as error:
                row["detail"] = str(error)
            rows.append(row)
            if service.counters != counters:
                raise ValueError("assembling same-price quality outputs repeated native selection trials")
    for trial in service.trials:
        if trial["status"] == "passed":
            frozen.assert_hash(trial["candidate_path"], trial["candidate_sha256"], "same-run quality paid frame")
    _check(protected)
    result = dict(version=VERSION, rows=rows, action_rows=action_rows, predictions=predictions,
        selections=selections, pool_trials=service.trials, protected=protected, geometry=geometry, geometry_union=union,
        active_actions=list(RETAINED_ACTIONS), historical_quality_geometry_unchanged=True,
        same_run_price_cache=True, new_native_encoding=True, independently_decoded_complete_files=True,
        throughput_measured=False)
    _json(out / "quality_manifest.json", result)
    return result


def run_layout(job, regions, out, exporter, fixed_exe, *, policy=None, policy_factory=None,
               top_k=2, timeout=300, execute=None, service_factory=None):
    """One fresh-cache output; model loading must occur before this call.

    Timer includes fresh policy construction, raw features/inference, trials,
    per-trial validation, complete writing/fsync and harness overhead. Final
    independent complete-file verification is separately timed and excluded.
    """
    if top_k not in (1, 2, "all") or policy is not None and policy_factory is not None:
        raise ValueError("declare Top-1/2/all and exactly one selected-policy source")
    raw = Path(job["input"]).resolve()
    regions = _geometry(regions, job["values"])
    out = _fresh(out, raw, exporter, fixed_exe)
    protected = _protected(raw, exporter, fixed_exe)
    with single_thread():
        began = time.perf_counter()
        _job(job)
        actual_policy = policy_factory() if policy_factory is not None and top_k != "all" else policy
        if top_k == "all":
            service = _service(job, out, exporter, fixed_exe, timeout, execute, service_factory=service_factory)
            recommendations = None
        else:
            # Geometry-only action lookup never inspects measured support.
            def allowed(region):
                start, length = region
                result = set(RETAINED_ACTIONS) - {"alp"}
                if length % 1024 == 0:
                    result.add("alp")
                elif length > 1024:
                    result.update("alp+tail:" + a for a in RETAINED_ACTIONS if a in ("gorilla", "chimp", "chimp128", "tsxor", "rle+tsxor"))
                return result
            recommendations = _recommend(actual_policy, raw, regions, top_k, allowed)
            _json(out / "predictions_before_native.json", [dict(start=s, length=n,
                ranked_actions=list(recommendations[s, n]), frozen_before_native=True) for s, n in regions])
            service = _service(job, out, exporter, fixed_exe, timeout, execute,
                policy=_FrozenRecommendations(recommendations), service_factory=service_factory)
        if service.best_many(regions) is None:
            raise ValueError("declared native output policy could not encode every original region")
        counters = dict(service.counters)
        path = out / "final.f64pool"
        info, chosen = _assemble(service, regions, path, job["values"], recommendations)
        wall = time.perf_counter() - began
        proof, verify_seconds = _verify(path, raw, exporter, fixed_exe, timeout, execute)
        if counters != service.counters:
            raise ValueError("final persistence repeated paid native trials")
    _check(protected)
    row = dict(dataset=job["dataset"], series=job["series"], sha256=job["sha256"], values=job["values"],
        top_k=top_k, status="passed", complete_file_bits=info["complete_file_bits"],
        complete_file_roundtrip_verified=True, container_path=str(path), container_sha256=replay.sha256(path),
        wall_seconds=wall, final_verify_seconds=verify_seconds,
        timing_scope="cold-cache feature/inference+trial/per-trial verification+write/fsync harness wall; final complete decode excluded; model loading/training excluded",
        trial_count=counters["trial_count"], trial_value_volume=counters["trial_value_volume"],
        cold_native_cache=True, os_page_cache_flushed=False, pure_codec_throughput=False,
        geometry=regions, selections=chosen, pool_trials=service.trials,
        active_actions=list(RETAINED_ACTIONS), protected=protected,
        policy_counters=copy.deepcopy(getattr(actual_policy, "counters", {})) if actual_policy is not None else {})
    _json(out / "verification.json", proof)
    _json(out / "native_output_manifest.json", row)
    return row


def run_efficiency(job, regions, out, exporter, fixed_exe, *, policy_factory, repeats=3,
                   timeout=300, execute=None, service_factory=None):
    if type(repeats) is not int or repeats < 3:
        raise ValueError("efficiency requires at least three fresh-cache repeats per arm")
    out = _fresh(out, job["input"], exporter, fixed_exe)
    rows, policy_objects = [], []
    def fresh_policy():
        value = policy_factory()
        if any(value is p for p in policy_objects):
            raise ValueError("each efficiency arm/repeat requires a fresh policy feature/inference cache")
        if getattr(value, "predictions", []):
            raise ValueError("efficiency policy contains warm precomputed recommendations")
        policy_objects.append(value)
        return value
    for repeat in range(1, repeats + 1):
        arms = (1, 2, "all")
        offset = (repeat - 1) % len(arms)
        order = arms[offset:] + arms[:offset]
        for top_k in order:
            folder = out / ("repeat_%02d_%s" % (repeat, top_k))
            try:
                row = run_layout(job, regions, folder, exporter, fixed_exe,
                    policy_factory=fresh_policy, top_k=top_k, timeout=timeout,
                    execute=execute, service_factory=service_factory)
            except (OSError, ValueError, KeyError, TypeError) as error:
                row = dict(top_k=top_k, status="unavailable", detail=str(error),
                    complete_file_bits=None, wall_seconds=None, complete_file_roundtrip_verified=False)
            rows.append(dict(row, repeat=repeat))
    summaries = []
    for top_k in (1, 2, "all"):
        own = [r for r in rows if r["top_k"] == top_k]
        valid = [r for r in own if r["status"] == "passed" and r["complete_file_roundtrip_verified"]]
        complete = len(valid) == len(own) == repeats
        summaries.append(dict(top_k=top_k, declared_repeats=repeats, valid_repeats=len(valid),
            status="passed" if complete else "unavailable", median_wall_seconds=statistics.median(r["wall_seconds"] for r in valid) if complete else None,
            median_complete_file_bits=statistics.median(r["complete_file_bits"] for r in valid) if complete else None,
            cold_native_cache=True, model_loading_in_timing=False, pure_codec_throughput=False))
    _json(out / "efficiency_manifest.json", dict(version=VERSION, rows=rows, summaries=summaries))
    return rows, summaries


def fresh_original_boundary(entry, out, exporter, fixed_exe, *, protected, timeout=300,
                            class_timeout=1800, execute=None, detector_runner=None):
    """Fresh XOR-change detector plus the ORIGINAL fourteen-action repair.

    Historical manifests authenticate parameters/code, not prices or output
    boundaries. Only the method's XOR-change arm is rerun; unrelated historical
    numeric/control comparison arms are not part of this compression workflow.
    """
    raise RuntimeError("legacy fourteen-action template workflow is disabled in the strict-eight release; "
                       "use examples/run_adaptive_verified.py")
    from compression_boundary_repair import RepairConfig, repair_partition
    from compression_candidates import build_candidates
    from compression_gain_proxy import CompressionGainProxy
    import class_mainline_btr as mainline
    import paper_boundary_origin as origin
    import paper_xor_change_class as detection

    job = entry["job"]
    raw = _job(job)
    folder = Path(entry["source_run"]).resolve() / job["series"]
    source_paths = (folder / "series_manifest.json", folder / "frozen_initial_layouts.json")
    for path in source_paths:
        expected = protected.get(str(path.resolve()))
        if expected is None or replay.sha256(path) != expected:
            raise ValueError("fresh workflow lacks authenticated original boundary source: " + str(path))
    state, plan = [json.loads(path.read_text(encoding="utf-8")) for path in source_paths]
    if (state.get("version") != origin.VERSION or state.get("job") != job or plan.get("job") != job
            or state.get("complete") is not True or state.get("protected_files_unchanged") is not True
            or state.get("active_actions") != list(DEFAULT_ACTIONS)
            or plan.get("active_actions") != list(DEFAULT_ACTIONS)
            or state.get("selector_training_used") is not False or state.get("selector_prediction_used") is not False
            or state.get("original_full_btrblocks_in_pool") is not False
            or plan.get("layouts_frozen_before_native_trials") is not True
            or state.get("candidate_protocol") != plan.get("candidate_protocol")):
        raise ValueError("fresh workflow source is not original full14 XOR-change boundary repair")
    original_hashes = state["protected_sha256"]
    code = {}
    for path, digest in original_hashes.items():
        if Path(path).suffix == ".py":
            name = Path(path).name
            if name in code and code[name] != digest:
                raise ValueError("original boundary source contains conflicting code versions")
            code[name] = digest
    required_code = ("paper_xor_change_class.py", "class_xor_change.py", "segment_class.py",
                     "compression_candidates.py", "compression_boundary_repair.py", "compression_gain_proxy.py",
                     "class_mainline_btr.py", "flat_pool_backend.py", "flat_pool_format.py",
                     "class_native_evaluator.py", "btr_search_ledger.py")
    code_bindings = {}
    for name in required_code:
        path = Path(__file__).with_name(name).resolve()
        if name not in code or replay.sha256(path) != code[name]:
            raise ValueError("cannot authenticate unchanged original boundary implementation: " + name)
        code_bindings[str(path)] = code[name]
    for path in map(lambda p: Path(p).resolve(), (raw, exporter, fixed_exe)):
        if original_hashes.get(str(path)) != replay.sha256(path):
            raise ValueError("fresh boundary raw/native differs from authenticated original source")
    protocol = state["candidate_protocol"]
    if (protocol.get("mode") != "union" or protocol.get("grid") != 64
            or protocol.get("minimum") != 64 or protocol.get("random_seed") != 0
            or protocol.get("cap_enlarged") is not False):
        raise ValueError("original candidate protocol cannot be reproduced unchanged")
    cfg = RepairConfig(**state["repair_config"])
    cfg.validate()
    configuration = state["detector_configuration"]
    args = SimpleNamespace(class_history=configuration["n_timepoints"],
        class_subsequence=configuration["window_size"], class_jump=configuration["jump"],
        progress_every=100, class_timeout=class_timeout)
    if detection.detector_config(args) != configuration:
        raise ValueError("recorded ClaSS parameters differ from the unchanged detector interface")
    budget, trials = state["each_arm_value_budget"], state["each_arm_trial_limit"]
    if type(budget) is not int or budget <= 0 or type(trials) is not int or trials <= 0:
        raise ValueError("original boundary measurement budgets are invalid")
    out = _fresh(out, raw, exporter, fixed_exe)
    current_protected = _protected(raw, exporter, fixed_exe) | code_bindings
    with single_thread():
        began = time.perf_counter()
        _json(out / "job.json", job)
        detector = (detector_runner or detection.detector_stage)(job, out / "detect_xor_change", "xor_change", args)
        initial = [tuple(r) for r in detection.validate_detector_result(detector, job, "xor_change", configuration)]
        _geometry(initial, job["values"])
        values = mainline.read_float64(str(raw))
        cache = mainline.FeatureCache(values, (job["dataset"], job["series"], job["sha256"]))
        candidates = build_candidates(cache, initial, mode="union", grid=64, minimum=64,
            window=protocol["window"], maximum=protocol["maximum"], random_seed=0)
        _json(out / "fresh_initial_and_candidates.json", dict(initial=initial, candidates=candidates,
            detector_configuration=configuration, candidate_protocol=protocol, active_actions=list(DEFAULT_ACTIONS),
            frozen_before_native=True))
        service = _service(job, out, exporter, fixed_exe, timeout, execute, actions=DEFAULT_ACTIONS,
                           value_budget=budget, trial_limit=trials)
        view = BudgetedRegionView(service, budget, trials)
        evaluator = FlatPoolEvaluator(view)
        evaluator.evaluate(initial, phase="seed")
        initial_path = out / "initial.f64pool"
        initial_info, _ = _assemble(service, initial, initial_path, job["values"])
        initial_proof, _ = _verify(initial_path, raw, exporter, fixed_exe, timeout, execute)
        proxy = CompressionGainProxy(values)
        repair = repair_partition(initial, candidates, evaluator, config=cfg, tag="xor_change_repaired", proposal_scorer=proxy.gain)
        regions = [tuple(r) for r in repair["intervals"]]
        final_path = out / "final.f64pool"
        final_info, choices = _assemble(service, regions, final_path, job["values"])
        final_proof, _ = _verify(final_path, raw, exporter, fixed_exe, timeout, execute)
        origin.validate_repair(repair, initial, regions, initial_info["complete_file_bits"], final_info["complete_file_bits"])
        if any(t["status"] == "failed" for t in service.trials):
            raise ValueError("fresh original14 boundary trial coverage is incomplete; no full-pool result claimed")
        wall = time.perf_counter() - began
    _check(current_protected)
    result = dict(version=VERSION, regions=regions, initial_regions=initial, detector=detector,
        candidates=candidates, repair=repair, repair_config=asdict(cfg), boundary_actions=list(DEFAULT_ACTIONS),
        boundary_trial_count=view.counters["trial_count"], boundary_trial_value_volume=view.counters["trial_value_volume"],
        boundary_wall_seconds=wall, complete_file_bits=final_info["complete_file_bits"],
        complete_file_roundtrip_verified=True, initial_verification=initial_proof, final_verification=final_proof,
        selections=choices, pool_trials=service.trials, protected=current_protected,
        fresh_xor_change_detection=True, fresh_original_candidate_repair=True,
        historical_prices_used=False, selector_consistent_geometry_used=False,
        boundary_pool_is_original14=True, full_original_btrblocks_in_pool=False,
        other_historical_comparison_arms_rerun=False, numba_disk_cache_enabled=False)
    _json(out / "fresh_boundary_manifest.json", result)
    return result


def run_complete(entry, out, exporter, fixed_exe, *, protected, policy_factory,
                 timeout=300, class_timeout=1800, execute=None, detector_runner=None):
    """Fresh ORIGINAL14 boundary workflow followed by retained8 Top-2 output.

    This repeat's geometry/quality must never replace frozen historical quality
    rows. Fresh native sampling can change sizes and visited local search paths.
    Model loading/training are outside; construction/features/inference inside.
    """
    raise RuntimeError("legacy fourteen-action complete workflow is disabled in the strict-eight release; "
                       "use examples/run_adaptive_verified.py")
    job = entry["job"]
    out = _fresh(out, job["input"], exporter, fixed_exe)
    began = time.perf_counter()
    boundary = fresh_original_boundary(entry, out / "boundary", exporter, fixed_exe,
        protected=protected, timeout=timeout, class_timeout=class_timeout,
        execute=execute, detector_runner=detector_runner)
    output = run_layout(job, boundary["regions"], out / "selected_output", exporter, fixed_exe,
        policy_factory=policy_factory, top_k=2, timeout=timeout, execute=execute)
    row = dict(output, complete_wall_seconds=time.perf_counter() - began,
        complete_timing_scope="fresh XOR-change ClaSS/cold JIT + ORIGINAL14 candidate/exhaustive repair and seed/final verification + retained8 raw-feature/Top2 output + complete verification; model loading/training excluded",
        includes_fresh_xor_change=True, includes_original14_boundary_trials=True,
        boundary_pool=list(DEFAULT_ACTIONS), output_pool=list(RETAINED_ACTIONS),
        selector_guided_boundary_search=False, fresh_boundary=boundary,
        replaces_historical_quality_geometry=False, independent_final_dataset=False,
        pure_codec_throughput=False)
    _json(out / "complete_workflow_manifest.json", row)
    return row
