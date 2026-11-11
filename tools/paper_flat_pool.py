"""ClaSS local-gain experiment with explicit fixed Btr primitive pipelines.

This is a full measured-pool segmentation/coverage experiment, not learned
selector inference. Old code, old native builds and frozen baseline results
are never modified. No complete-BtrBlocks action, global output guard, alternate
seed or DP is introduced. After all original 11 sequences finish successfully,
the user's zero-final-use rule exports a smaller active configuration for later
runs; it never changes this experiment's frozen pool or deletes decoders.
"""
import argparse
import csv
from dataclasses import asdict
import json
from pathlib import Path
import subprocess
import sys
import time

import class_mainline_btr as mainline
from btr_search_ledger import BudgetedRegionView, fixed_partition
from compression_boundary_repair import RepairConfig, repair_partition
from compression_gain_proxy import CompressionGainProxy
import fixed_pool_usage as usage
import finalize_flat_pool as finalization
from flat_pool_backend import DEFAULT_ACTIONS, FlatPoolEvaluator, FlatPoolService, validate_actions
import flat_pool_format as fmt
import frozen_btr_extension as frozen
import paper_class_local_gain as prior
import paper_class_start_pair as pair
import paper_intro_motivation as intro
import replay_frozen_files as replay

VERSION = "class_flat_fixed_pool_v2"
REFERENCE, TREATMENT = "fixed1024_reference", "adaptive_flat_pool"
METHODS = (REFERENCE, TREATMENT)
CODE_FILES = tuple(dict.fromkeys(prior.CODE_FILES + (Path(__file__).name,
    "flat_pool_format.py", "flat_pool_backend.py", "fixed_pool_usage.py", "finalize_flat_pool.py")))
FIELDS = ("dataset", "series", "values", "sha256", "method", "status", "detail",
    "complete_file_bits", "complete_file_bytes", "complete_file_roundtrip_verified", "container_path",
    "container_sha256", "normal_initial_bits", "gain_vs_initial_pct", "gain_vs_fixed1024_pct",
    "detected_class_segments", "boundary_segments", "accepted_operations", "trial_count",
    "trial_value_volume", "termination", "budget_limited", "pool_native_failures",
    "pool_unsupported_trials", "normal_encoding_policy", "wall_seconds_not_codec_throughput")
COMPARISON_FIELDS = ("dataset", "series", "values", "sha256", "status", "fixed1024_bits",
    "adaptive_bits", "gain_vs_fixed1024_pct", "normal_initial_bits", "gain_vs_initial_pct",
    "boundary_segments", "accepted_operations", "pool_native_failures", "detail")


def catalog(actions):
    return [dict(codec=a, group="btr" if a in fmt.FIXED_ACTIONS else "alp" if a == "alp" else "stream")
            for a in actions]


def comparison(rows, job):
    result = {k: job[k] for k in ("dataset", "series", "values", "sha256")}
    by = {r["method"]: r for r in rows}
    a, b = by.get(REFERENCE, {}), by.get(TREATMENT, {})
    result.update(status="unavailable", detail="; ".join(r.get("detail", "") for r in (a,b) if r.get("detail")))
    if all(r.get("status") == "passed" and r.get("complete_file_roundtrip_verified") is True for r in (a,b)):
        result.update(status="passed", fixed1024_bits=a["complete_file_bits"], adaptive_bits=b["complete_file_bits"],
            gain_vs_fixed1024_pct=100*(a["complete_file_bits"]-b["complete_file_bits"])/a["complete_file_bits"])
        result.update({k: b[k] for k in ("normal_initial_bits", "gain_vs_initial_pct", "boundary_segments",
                                       "accepted_operations", "pool_native_failures")})
    return result


def protected_sources():
    return {str(Path(__file__).with_name(n).resolve()): replay.sha256(Path(__file__).with_name(n))
            for n in CODE_FILES}


def load_pool_config(path):
    """Read an explicitly chosen post-run pool without modifying that artifact."""
    state = frozen.read_json(Path(path).resolve())
    if (state.get("status") != "finalized"
            or state.get("selection_scope") != "actual_adaptive_final_selections_only"
            or state.get("schema_version") != 1 or state.get("policy") != finalization.POLICY
            or state.get("batch_complete") is not True or state.get("batch_status") != "passed"
            or any(state.get(k) != 11 for k in ("declared_series_count", "observed_series_count", "required_original_series_count"))
            or state.get("reasons") != [] or state.get("pruning_applied_to_new_configuration_only") is not True):
        raise ValueError("pool-config is not an all-11-series finalized active pool")
    digest = state.get("usage_payload_canonical_sha256")
    identities, counts = state.get("source_identities"), state.get("counts")
    if (not isinstance(digest, str) or not finalization.HASH_PATTERN.fullmatch(digest)
            or not isinstance(identities, list) or len(identities) != 11
            or not isinstance(counts, list)):
        raise ValueError("pool-config is missing complete frozen usage/source evidence")
    names = []
    total_values = 0
    for identity in identities:
        if (not isinstance(identity, dict) or not isinstance(identity.get("series"), str) or not identity["series"]
                or type(identity.get("values")) is not int or identity["values"] <= 0
                or not isinstance(identity.get("sha256"), str) or not finalization.HASH_PATTERN.fullmatch(identity["sha256"])):
            raise ValueError("pool-config has an invalid original source identity")
        names.append(identity["series"])
        total_values += identity["values"]
    if len(set(names)) != 11:
        raise ValueError("pool-config original identities are not eleven unique sequences")
    actions = validate_actions(state.get("active_actions", []))
    original = validate_actions(state.get("original_actions", []))
    removed = state.get("removed_actions")
    if (not isinstance(removed, list) or len(set(removed)) != len(removed)
            or set(actions) & set(removed) or set(actions) | set(removed) != set(original)):
        raise ValueError("pool-config active/removed lists do not partition the original pool")
    if len(counts) != len(original) or any(not isinstance(c, dict) for c in counts):
        raise ValueError("pool-config needs one usage count row for every original candidate")
    by = {c.get("codec"): c for c in counts}
    if set(by) != set(original):
        raise ValueError("pool-config usage counts do not cover the original candidates")
    for codec, count in by.items():
        for field in ("direct_segments", "direct_values", "direct_frame_bytes", "required_alp_tail_segments", "required_alp_tail_values"):
            if type(count.get(field)) is not int or count[field] < 0:
                raise ValueError("pool-config has unknown or invalid final usage counts")
        used = count["direct_segments"] > 0 or count["required_alp_tail_segments"] > 0
        if used != (codec in actions):
            raise ValueError("pool-config membership contradicts actual final usage/dependencies")
    if sum(c["direct_values"] for c in counts) != total_values:
        raise ValueError("pool-config usage values do not cover all original inputs")
    return actions


def original_declared_jobs(config_path):
    """Use the full original declaration, never the --series filtered job list."""
    path = Path(config_path).resolve()
    declared = frozen.read_json(path)["datasets"]
    rows = frozen.read_json(path.parent/"baselines/data_manifest.json")["datasets"]
    known = {r["series"]: r for r in rows}
    names = [r["series"] for r in declared]
    if len(known) != len(rows) or len(set(names)) != len(names) or set(names) != set(known):
        raise ValueError("original config and certified data manifest have different declarations")
    return [{k: known[name][k] for k in ("series", "values", "sha256")} for name in names]


def run_series(job, out, exporter, fixed_exe, args, *, execute=replay.execute_export,
               class_runner=pair.class_stage, backend_factory=FlatPoolService):
    actions = validate_actions(args.pool_actions)
    out, raw = Path(out).resolve(), Path(job["input"]).resolve()
    if out.exists():
        raise ValueError("per-series output must be new")
    protected = protected_sources()
    for path in map(Path, (raw, exporter, fixed_exe)):
        protected[str(path.resolve())] = replay.sha256(path)
    if any(intro.overlap(out, Path(p)) for p in protected):
        raise ValueError("new output overlaps a protected source")
    if replay.sha256(raw) != job["sha256"] or raw.stat().st_size != 8*job["values"]:
        raise ValueError("original full input identity differs")
    values = mainline.read_float64(str(raw))
    out.mkdir(parents=True)
    rows, layouts, backend = [], [], None
    budget = args.value_budget_multiplier*len(values)
    manifest = out/"series_manifest.json"
    state = dict(version=VERSION, status="running", complete=False, job=job, results=rows,
        protected_sha256=protected, active_actions=list(actions), predeclared_methods=list(METHODS),
        normal_encoding_policy="measured_flat_fixed_pipeline_pool", selector_used=False,
        selector_training_used=False, learned_selector_not_integrated_in_this_diagnostic=True,
        complete_btrblocks_in_pool=False, btr_adaptive_selection=False,
        normal_parent_cached=True, class_proposes_but_does_not_authorize_boundaries=True,
        no_external_baseline_fallback=True, no_separate_whole_file_prescan=True,
        each_arm_value_budget=budget, each_arm_trial_limit=args.trial_limit,
        independent_virtual_work_including_cached_attempts=True,
        pool_frozen_before_any_input_trial=True, automatic_pool_pruning=False,
        container_version=fmt.MAGIC.decode(), temporary_decoded_slices_retained=False,
        timing_scope="harness work and verification, NOT codec throughput or fast-selector inference")

    def save():
        mainline.stream_json(manifest, state)
        frozen.write_csv(out/"summary.csv", rows, FIELDS)
        frozen.write_csv(out/"comparison.csv", [comparison(rows, job)], COMPARISON_FIELDS)

    def measured(command, log, timeout):
        before = time.perf_counter()
        record = dict(command=list(map(str, command)), log=str(log), event="start")
        with (out/"native_commands.jsonl").open("a", encoding="utf-8") as stream:
            stream.write(json.dumps(record)+"\n")
        try:
            rc = execute(command, Path(log), timeout)
            record.update(event="finish", exit_code=rc)
            return rc
        except BaseException as error:
            record.update(event="error", detail=str(error))
            raise
        finally:
            record["wall_seconds"] = time.perf_counter()-before
            with (out/"native_commands.jsonl").open("a", encoding="utf-8") as stream:
                stream.write(json.dumps(record)+"\n")

    def persist(evaluator, regions, path):
        counters = (dict(evaluator.view.counters), dict(backend.counters))
        segments = evaluator.native_segments(regions)
        info = fmt.write_container(path, [f for item in segments for f in item["frames"]], job["values"])
        proof = fmt.verify_file(path, raw, exporter, fixed_exe, args.timeout, measured)
        if info["complete_file_bits"] != 8*fmt.FILE_HEADER.size + sum(evaluator.costs[r] for r in regions):
            raise ValueError("actual final file size differs from measured frame prices")
        if counters != (dict(evaluator.view.counters), dict(backend.counters)):
            raise ValueError("persistence repeated encoding or selection")
        proof_path = path.with_suffix(".verification.json")
        mainline.stream_json(proof_path, proof)
        for p in (path, proof_path):
            protected[str(p)] = replay.sha256(p)
        return info

    save()
    try:
        class_dir = Path(class_runner(job, out, args)).resolve()
        identity = (job["dataset"], job["series"], job["sha256"])
        candidates, candidate_info = mainline.class_candidates(class_dir, values, identity, protected, args.max_candidates)
        with (class_dir/"boundaries.csv").open(newline="", encoding="utf-8-sig") as source:
            detected = [(int(r["start"]), int(r["length"])) for r in csv.DictReader(source)]
        candidates = prior.retain_class_candidates(candidates, detected, len(values))
        candidate_path = out/"common_candidates.csv"
        frozen.write_csv(candidate_path, candidates, prior.CANDIDATE_FIELDS)
        protected[str(candidate_path)] = replay.sha256(candidate_path)
        config = RepairConfig(max_rounds=args.max_rounds, max_proposals_per_round=args.max_proposals,
                              max_split_candidates=args.max_split_candidates)
        config.validate()
        state.update(candidate_protocol=candidate_info, repair_config=asdict(config), class_source=str(class_dir),
            class_seconds_historical_when_reused=out not in class_dir.parents,
            detected_class_segments=len(detected), accepted_initial_segments=1)
        backend = backend_factory(dict(n=len(values), raw=raw, key=identity+(VERSION,)), out/"native",
            exporter, fixed_exe, args.timeout, value_budget=2*budget, trial_limit=2*args.trial_limit,
            execute=measured, actions=actions)
        for method in METHODS:
            initial = fixed_partition(len(values), 1024) if method == REFERENCE else [(0,len(values))]
            folder = out/method
            folder.mkdir()
            view = BudgetedRegionView(backend, budget, args.trial_limit)
            evaluator = FlatPoolEvaluator(view)
            row = dict(job, method=method, status="failed", detail="", complete_file_bits=None,
                complete_file_roundtrip_verified=False, normal_encoding_policy="measured_flat_fixed_pipeline_pool",
                detected_class_segments=len(detected))
            layout = dict(method="fixed1024" if method == REFERENCE else "adaptive", status="failed",
                sha256=job["sha256"], complete_file_roundtrip_verified=False, segments=[])
            rows.append(row)
            layouts.append(layout)
            before = time.perf_counter()
            repair = None
            print(job["series"]+"/"+method+": start", flush=True)
            try:
                if method == TREATMENT:
                    repair = repair_partition(initial, candidates, evaluator, config=config, tag=method,
                        proposal_scorer=CompressionGainProxy(values).gain)
                    final = [tuple(r) for r in repair["intervals"]]
                    if (repair["telescoping_ok"] is not True or repair["estimates_used_for_acceptance"] is not False
                            or any(op["gain_bits"] <= 0 or op["old_bits"]-op["new_bits"] != op["gain_bits"]
                                   for op in repair["operations"])):
                        raise ValueError("repair accepted an edit without real net savings")
                    mainline.stream_json(folder/"repair.json", repair)
                    protected[str(folder/"repair.json")] = replay.sha256(folder/"repair.json")
                else:
                    evaluator.evaluate(initial, phase="seed")
                    final = initial
                initial_info = persist(evaluator, initial, folder/"initial.f64pool")
                final_info = persist(evaluator, final, folder/"final.f64pool")
                if final_info["complete_file_bits"] > initial_info["complete_file_bits"]:
                    raise ValueError("accepted edits increased complete stored size")
                if repair is not None and initial_info["complete_file_bits"]-final_info["complete_file_bits"] != sum(op["gain_bits"] for op in repair["operations"]):
                    raise ValueError("persisted complete savings do not telescope")
                frozen.write_csv(folder/"boundaries.csv", [dict(start=s,length=n,codec=evaluator.choices[(s,n)]["codec"])
                    for s,n in final], ("start","length","codec"))
                protected[str(folder/"boundaries.csv")] = replay.sha256(folder/"boundaries.csv")
                row.update(status="passed", complete_file_bits=final_info["complete_file_bits"],
                    complete_file_bytes=final_info["complete_file_bytes"], complete_file_roundtrip_verified=True,
                    container_path=str(folder/"final.f64pool"), container_sha256=replay.sha256(folder/"final.f64pool"),
                    normal_initial_bits=initial_info["complete_file_bits"],
                    gain_vs_initial_pct=100*(initial_info["complete_file_bits"]-final_info["complete_file_bits"])/initial_info["complete_file_bits"],
                    boundary_segments=len(final), accepted_operations=len(repair["operations"]) if repair else 0,
                    termination=repair["termination"] if repair else "reference_only_no_search",
                    budget_limited=repair["budget_limited"] if repair else False)
                layout.update(status="passed", complete_file_bytes=final_info["complete_file_bytes"],
                    complete_file_roundtrip_verified=True, container_overhead_bytes=fmt.FILE_HEADER.size,
                    segments=evaluator.usage_segments(final))
            except (OSError, ValueError, KeyError, TypeError) as error:
                row.update(status="failed", detail=str(error), complete_file_bits=None,
                    complete_file_bytes=None, complete_file_roundtrip_verified=False)
                layout.update(status="failed", complete_file_roundtrip_verified=False, segments=[])
            finally:
                failures = [backend.cache[k] for k in view.charged_keys if k in backend.cache]
                row.update(trial_count=view.counters["trial_count"], trial_value_volume=view.counters["trial_value_volume"],
                    pool_native_failures=sum(t["status"] == "failed" for t in failures),
                    pool_unsupported_trials=sum(t["status"] == "unsupported" for t in failures),
                    wall_seconds_not_codec_throughput=time.perf_counter()-before)
                ledger = dict(counters=dict(view.counters), charged_keys=sorted(view.charged_keys),
                    value_budget=budget, trial_limit=args.trial_limit, active_actions=list(actions))
                mainline.stream_json(folder/"ledger.json", ledger)
                protected[str(folder/"ledger.json")] = replay.sha256(folder/"ledger.json")
                if row["trial_count"] > args.trial_limit or row["trial_value_volume"] > budget:
                    raise ValueError("arm exceeded declared work budget")
            if row["status"] == rows[0]["status"] == "passed":
                row["gain_vs_fixed1024_pct"] = 100*(rows[0]["complete_file_bits"]-row["complete_file_bits"])/rows[0]["complete_file_bits"]
            intro.check_protected(protected)
            save()
            print(job["series"], method, row["status"], row.get("complete_file_bits"), row["detail"], flush=True)
        usage_input = dict(schema_version=1, size_scope="complete_container_bytes", codec_size_scope="complete_frame_bytes",
            candidates=catalog(actions), series=[dict(series=job["series"], sha256=job["sha256"],
                values=job["values"], layouts=layouts, trials=backend.trials)])
        # Validate statistics before publishing a supposedly successful child.
        usage.analyze(usage_input)
        mainline.stream_json(out/"usage_input.json", usage_input)
        protected[str(out/"usage_input.json")] = replay.sha256(out/"usage_input.json")
        for trial in backend.trials:
            if trial["status"] == "passed":
                protected[trial["candidate_path"]] = trial["candidate_sha256"]
        intro.check_protected(protected)
        all_passed = all(r["status"] == "passed" for r in rows)
        state.update(complete=True, status=("passed" if all_passed and not backend.counters["failed_trials"]
            else "completed_with_unavailable_cases" if all_passed else "completed_with_failures"),
                     protected_files_unchanged=True, native_counters=dict(backend.counters))
    except BaseException as error:
        state.update(status="failed", complete=False, detail=str(error))
        for row in rows:
            row.update(status="invalidated_failed_run", complete_file_bits=None, complete_file_bytes=None,
                       complete_file_roundtrip_verified=False)
        raise
    finally:
        if backend is not None:
            mainline.stream_json(out/"native_trials.json", backend.trials)
        save()
    return rows


def collect(job, folder, rc):
    folder = Path(folder).resolve()
    state = frozen.read_json(folder/"series_manifest.json")
    if (rc not in (0,1) or state.get("version") != VERSION or state.get("complete") is not True
            or state.get("complete_btrblocks_in_pool") is not False
            or state.get("btr_adaptive_selection") is not False
            or state.get("protected_files_unchanged") is not True):
        raise ValueError("worker lacks complete protected flat-pool proof")
    intro.check_protected(state["protected_sha256"])
    rows = state["results"]
    if [r["method"] for r in rows] != list(METHODS):
        raise ValueError("worker result methods differ")
    for row in rows:
        if any(row[k] != job[k] for k in ("dataset","series","values","sha256")):
            raise ValueError("worker source identity differs")
        if row["status"] != "passed":
            if row.get("complete_file_bits") is not None:
                raise ValueError("failed row has an eligible size")
            continue
        path = Path(row["container_path"]).resolve()
        info = fmt.inspect_container(path)
        if (folder not in path.parents or replay.sha256(path) != row["container_sha256"]
                or info["complete_file_bits"] != row["complete_file_bits"] or info["values"] != job["values"]
                or row["complete_file_roundtrip_verified"] is not True):
            raise ValueError("worker saved-file size/hash/count differs")
        proof = frozen.read_json(path.with_suffix(".verification.json"))
        if (proof.get("complete_file_roundtrip_verified") is not True
                or proof.get("decoded_from_persisted_file_only") is not True
                or proof.get("source_sha256") != job["sha256"]
                or proof.get("decoded_sha256") != job["sha256"]
                or proof.get("decoded_values") != job["values"]
                or proof.get("container_sha256") != row["container_sha256"]
                or proof.get("complete_file_bits") != row["complete_file_bits"]):
            raise ValueError("worker independent decoding proof differs")
        ledger = frozen.read_json(folder/row["method"]/"ledger.json")
        if (len(ledger["charged_keys"]) != row["trial_count"]
                or sum(k[1] for k in ledger["charged_keys"]) != row["trial_value_volume"]
                or row["trial_count"] > ledger["trial_limit"] or row["trial_value_volume"] > ledger["value_budget"]
                or "btrblocks" in ledger["active_actions"]
                or any(k[2] == "btrblocks" for k in ledger["charged_keys"])):
            raise ValueError("worker pool/budget proof differs")
    return rows, frozen.read_json(folder/"usage_input.json"), state["protected_sha256"]


def batch(args, *, executor=pair.execute_child):
    actions = validate_actions(args.pool_actions)
    args.btr_exe = args.fixed_btr_exe
    jobs, protected, exporter, fixed_exe = intro.preflight(args)
    protected.update(protected_sources())
    if getattr(args, "pool_config", None):
        config = Path(args.pool_config).resolve()
        if intro.overlap(Path(args.out_dir).resolve(), config):
            raise ValueError("output overlaps finalized pool configuration")
        protected[str(config)] = replay.sha256(config)
        if load_pool_config(config) != actions:
            raise ValueError("finalized pool configuration changed after loading")
    out = Path(args.out_dir).resolve()
    if args.class_source_root and intro.overlap(out, Path(args.class_source_root).resolve()):
        raise ValueError("output overlaps ClaSS reuse root")
    out.mkdir(parents=True, exist_ok=True)
    (out/"logs").mkdir()
    rows, children, series_usage = [], [], []
    state = dict(version=VERSION, status="running", complete=False, configuration=vars(args), jobs=jobs,
        results=rows, children=children, active_actions=list(actions), protected_sha256=protected,
        sequential_fresh_workers=True, automatic_pool_pruning=True,
        pruning_timing="after_all_original_11_series_only; no mid-run pool changes",
        pruning_scope="actual_adaptive_final_selections_only; retain selected ALP tail dependencies",
        pruning_changes="new active configuration only; implementations/decoders/results preserved",
        original_baselines_unchanged=True)

    def save():
        mainline.stream_json(out/"manifest.json", state)
        frozen.write_csv(out/"summary.csv", rows, FIELDS)
        frozen.write_csv(out/"comparison.csv", [comparison([r for r in rows if r["series"] == j["series"]],j) for j in jobs], COMPARISON_FIELDS)
        frozen.write_csv(out/"coverage.csv", [dict(series=j["series"], method=m,
            status=next((r["status"] for r in rows if r["series"] == j["series"] and r["method"] == m), "pending"),
            detail=next((r["detail"] for r in rows if r["series"] == j["series"] and r["method"] == m), ""))
            for j in jobs for m in METHODS], ("series","method","status","detail"))

    save()
    try:
        for index, job in enumerate(jobs, 1):
            folder, log = out/job["series"], out/"logs"/(job["series"]+".log")
            command = [sys.executable,"-u",str(Path(__file__).resolve()),"--worker",
                "--source-config",args.source_config,"--data-repo",args.data_repo,"--out-dir",str(folder),
                "--series",job["series"],"--exporter",str(exporter),"--fixed-btr-exe",str(fixed_exe)]
            for name in ("value_budget_multiplier","trial_limit","max_rounds","max_proposals","max_split_candidates",
                         "max_candidates","timeout","class_history","class_subsequence","class_jump","class_timeout"):
                command.extend(("--"+name.replace("_","-"),str(getattr(args,name))))
            if args.class_source_root:
                command.extend(("--class-source-root",args.class_source_root))
            command.extend(("--pool-actions",*actions))
            child = dict(series=job["series"], command=command, log=str(log), status="running")
            children.append(child)
            save()
            print("===== %s/%s %s; fixed-pool; log=%s =====" % (index,len(jobs),job["series"],log),flush=True)
            try:
                rc = executor(command,log)
                result, recorded, child_protected = collect(job,folder,rc)
                if [v["codec"] for v in recorded["candidates"]] != list(actions):
                    raise ValueError("worker changed its predeclared pool")
                series_usage.extend(recorded["series"])
                if any(p in protected and protected[p] != digest for p,digest in child_protected.items()):
                    raise ValueError("worker changed a previously protected input/tool/source")
                protected.update(child_protected)
                rows.extend(result)
                valid_layouts = all(r["status"] == "passed" for r in result)
                child.update(exit_code=rc,status=("passed" if valid_layouts and not any(r.get("pool_native_failures",0) for r in result)
                    else "completed_with_unavailable_cases" if valid_layouts else "completed_with_failures"))
            except (OSError,ValueError,KeyError,TypeError) as error:
                child.update(status="failed",detail=str(error))
                rows.extend(dict(job,method=m,status="failed",detail=str(error),complete_file_bits=None,
                    complete_file_roundtrip_verified=False) for m in METHODS)
                series_usage.append(dict(series=job["series"],sha256=job["sha256"],values=job["values"],trials=[],
                    layouts=[dict(method=m,status="failed",sha256=job["sha256"],segments=[],
                                  complete_file_roundtrip_verified=False) for m in ("fixed1024","adaptive")]))
            intro.check_protected(protected)
            payload = dict(schema_version=1,size_scope="complete_container_bytes",codec_size_scope="complete_frame_bytes",
                           candidates=catalog(actions),series=series_usage)
            mainline.stream_json(out/"usage_input.json",payload)
            usage.write_report(usage.analyze(payload),out/"pool_usage")
            save()
            print("===== completed %s: %s =====" % (job["series"],child["status"]),flush=True)
        all_passed = all(r["status"] == "passed" for r in rows)
        state.update(complete=True,status=("passed" if all_passed and not any(r.get("pool_native_failures",0) for r in rows)
            else "completed_with_unavailable_cases" if all_passed else "completed_with_failures"),
                     protected_files_unchanged=True)
        try:
            result = finalization.finalize(payload, original_declared_jobs(args.source_config),
                batch_complete=state["complete"], batch_status=state["status"])
            result["usage_input_file_sha256"] = replay.sha256(out/"usage_input.json")
            result["source_config_sha256"] = replay.sha256(Path(args.source_config))
            result["source_data_manifest_sha256"] = replay.sha256(Path(args.source_config).parent/"baselines/data_manifest.json")
            finalization.write_finalization(result, out/"pool_finalization")
            state["pool_finalization"] = dict(status=result["status"],
                path=str(out/"pool_finalization/finalized_pool.json"),
                sha256=replay.sha256(out/"pool_finalization/finalized_pool.json"),
                active_actions=result["active_actions"], removed_actions=result["removed_actions"])
        except (OSError, ValueError, TypeError, KeyError) as error:
            # A report/config write failure must not invalidate independently
            # decoded files or label unmeasured candidates as removed.
            state["pool_finalization"] = dict(status="error", detail=str(error),
                path="", active_actions=list(actions), removed_actions=[])
            state["status"] = "completed_with_pool_finalization_error"
        intro.check_protected(protected)
    except BaseException as error:
        state.update(status="failed",complete=False,detail=str(error))
        if replay.unchanged(protected):
            for row in rows:
                row.update(status="protected_input_changed",complete_file_bits=None,complete_file_roundtrip_verified=False)
        raise
    finally:
        save()
    print("Comparison: "+str(out/"comparison.csv"),flush=True)
    print("Pool usage/removal diagnostics: "+str(out/"pool_usage"),flush=True)
    if "pool_finalization" in state:
        result = state["pool_finalization"]
        print("Post-run active pool: %s; removed=%s; config=%s" %
            (result["status"], ",".join(result["removed_actions"]) or "none", result["path"]),flush=True)
    return int(state["status"] != "passed")


def main():
    parser = intro.parser()
    parser.description = __doc__
    parser.add_argument("--worker",action="store_true")
    parser.add_argument("--fixed-btr-exe",required=True)
    parser.add_argument("--class-source-root")
    pool_args = parser.add_mutually_exclusive_group()
    pool_args.add_argument("--pool-actions",nargs="+")
    pool_args.add_argument("--pool-config",help="Explicit finalized_pool.json from a completed all-11-series run")
    for name,value in dict(class_history=4096,class_subsequence=32,class_jump=5,class_timeout=1800).items():
        parser.add_argument("--"+name.replace("_","-"),type=int,default=value)
    args = parser.parse_args()
    args.btr_exe = args.fixed_btr_exe
    try:
        args.pool_actions = list(load_pool_config(args.pool_config) if args.pool_config else
                                 validate_actions(args.pool_actions or DEFAULT_ACTIONS))
        validate_actions(args.pool_actions)
        if min(args.class_history,args.class_subsequence,args.class_jump,args.class_timeout) <= 0 or args.class_history <= 10*args.class_subsequence:
            raise ValueError("invalid ClaSS configuration")
        if args.worker:
            jobs,_,exporter,fixed_exe = intro.preflight(args)
            if len(jobs) != 1:
                raise ValueError("worker must declare one original series")
            result = run_series(jobs[0],args.out_dir,exporter,fixed_exe,args)
            return int(any(r["status"] != "passed" or r.get("pool_native_failures",0) for r in result))
        return batch(args)
    except (OSError,ValueError,KeyError,TypeError,subprocess.SubprocessError) as error:
        parser.exit(2,VERSION+": "+str(error)+"\n")


if __name__ == "__main__":
    raise SystemExit(main())
