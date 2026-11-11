"""ClaSS proposes cuts; the same declared encoding policy prices parent/children.

There is no dedicated whole-input exhaustive prepass, external BtrBlocks
guarantee, alternate search seed, or final output fallback. The normal parent
encoding is obtained inside repair_partition and reused for every local gain
comparison. Full-pool measurement is available only as an explicit C1
diagnostic; declared codec actions are an explicit non-learned policy.
"""
from dataclasses import asdict
import csv
import json
from pathlib import Path
import subprocess
import sys
import time

import paper_intro_motivation as intro
import paper_class_start_pair as pair
import class_mainline_btr as mainline
from btr_search_ledger import BudgetedRegionView, fixed_partition
from class_native_evaluator import NativeRegionEvaluator
from compression_boundary_repair import RepairConfig, repair_partition
from compression_gain_proxy import CompressionGainProxy
from external_file_format import FILE_HEADER, inspect_container
from local_codec_policy import DeclaredActionService, validate_declared_actions
import frozen_btr_extension as frozen
import replay_frozen_files as replay

VERSION = "paper_class_local_gain_v2"
REFERENCE, TREATMENT = "fixed1024_reference", "class_local_gain"
METHODS = (REFERENCE, TREATMENT)
DIAGNOSTIC_POLICY = "diagnostic_full_feasible_pool"
DECLARED_POLICY = "declared_codec_actions"
class_stage = pair.class_stage
execute_child = pair.execute_child
CANDIDATE_FIELDS = ("position", "score", "sources", "strongest_group", "state_contrast")
FIELDS = tuple(dict.fromkeys(intro.FIELDS + ("normal_encoding_policy", "declared_codec_actions",
    "detected_class_segments", "initial_segments", "initial_bits", "normal_initial_bits",
    "gain_vs_initial_pct", "gain_vs_fixed1024_pct", "initial_container_path", "initial_container_sha256",
    "initial_file_roundtrip_verified", "termination", "budget_limited", "candidate_sha256",
    "virtual_value_budget", "virtual_trial_limit", "infeasible_policy_proposals",
    "wall_seconds_not_comparable_throughput")))
COMPARISON_FIELDS = ("dataset", "series", "values", "sha256", "status", "fixed1024_bits",
    "normal_initial_bits", "adaptive_bits", "gain_vs_fixed1024_pct", "gain_vs_initial_pct",
    "normal_encoding_policy", "detected_class_segments", "initial_segments", "boundary_segments",
    "accepted_operations", "termination", "budget_limited", "infeasible_policy_proposals", "detail")
CODE_FILES = tuple(dict.fromkeys(pair.CODE_FILES + (Path(__file__).name,
    "local_codec_policy.py", "btr_selected_costs.py")))


def encoding_policy(args):
    """No policy/model is guessed when the caller has not declared one."""
    diagnostic = getattr(args, "diagnostic_full_pool", False)
    actions = getattr(args, "codec_actions", None)
    if type(diagnostic) is not bool or diagnostic == bool(actions):
        raise ValueError("Declare exactly one of --diagnostic-full-pool or --codec-actions")
    if diagnostic:
        return DIAGNOSTIC_POLICY, ()
    return DECLARED_POLICY, validate_declared_actions(tuple(actions))


class CachedNativeRegionEvaluator(NativeRegionEvaluator):
    """Do not re-enter encoding/selection for a frozen semantic region."""

    def __init__(self, view):
        super().__init__(view)
        self.infeasible_regions = set()

    def policy_infeasible(self, intervals):
        actions_for = getattr(self.service, "actions_for", None)
        if actions_for is None:
            return []
        regions = [r for r in self._intervals(intervals) if not actions_for(r)]
        self.infeasible_regions.update(regions)
        return regions

    def can_evaluate(self, intervals):
        regions = [r for r in self._intervals(intervals) if r not in self.costs]
        if self.policy_infeasible(regions):
            return False
        return not regions or super().can_evaluate(regions)

    def evaluate(self, intervals, *, phase="trial"):
        if phase not in ("seed", "trial"):
            raise ValueError("measurement phase must be seed or trial")
        regions = self._intervals(intervals)
        missing = [r for r in regions if r not in self.costs]
        if self.policy_infeasible(missing):
            raise ValueError("No declared codec action is feasible for this requested region; no substitution or fallback")
        if missing:
            super().evaluate(missing, phase=phase)
        return {region: self.costs[region] for region in regions}


def mark_infeasible_policy_proposals(repair, evaluator):
    """Distinguish an unencodable child layout from an exhausted work quota."""
    records = repair["proposal_records"]
    for record in records:
        if (record["status"] == "budget_skipped"
                and evaluator.policy_infeasible(record["new_intervals"])):
            record.update(status="infeasible_codec_policy",
                detail="A child region has no feasible declared action; no codec substitution, encoding, or charge")
    infeasible = sum(r["status"] == "infeasible_codec_policy" for r in records)
    for row in repair["round_records"]:
        row["budget_skipped"] = sum(r["status"] == "budget_skipped" and r["round"] == row["round"]
                                    for r in records)
        row["infeasible_policy_proposals"] = sum(r["status"] == "infeasible_codec_policy"
            and r["round"] == row["round"] for r in records)
    repair["infeasible_policy_proposals"] = infeasible
    repair["budget_exhausted"] = repair["budget_limited"] = any(r["status"] == "budget_skipped" for r in records)
    if infeasible and repair["termination"] == "no_affordable_proposals":
        repair["termination_detail"] = "No admitted proposal: feasibility under the declared codec policy may restrict the search"
    return repair


def retain_class_candidates(candidates, detected, n):
    """The capped extra pool may not silently discard authenticated anchors."""
    by = {int(row["position"]): dict(row) for row in candidates}
    positions = [int(start) for start, _ in detected if start]
    for position in positions:
        if not 0 < position < n or position % 64:
            raise ValueError("Authenticated ClaSS cut is not an internal grid boundary")
        item = by.setdefault(position, dict(position=position, score=1., sources="",
            strongest_group="", state_contrast=0.))
        sources = set(filter(None, str(item.get("sources", "")).split(";")))
        sources.add("class")
        item.update(score=max(1., float(item.get("score", 0.))), sources=";".join(sorted(sources)))
    return [by[p] for p in sorted(by)]


def comparison(rows, job):
    by = {r["method"]: r for r in rows}
    reference, treatment = by.get(REFERENCE, {}), by.get(TREATMENT, {})
    valid = all(r.get("status") == "passed" and r.get("complete_file_roundtrip_verified") is True
                for r in (reference, treatment))
    result = dict.fromkeys(COMPARISON_FIELDS)
    result.update({k: job[k] for k in ("dataset", "series", "values", "sha256")})
    result.update(status="passed" if valid else "unavailable",
        detail="; ".join(r.get("detail", "") for r in (reference, treatment) if r.get("detail")))
    if valid:
        if (reference.get("normal_encoding_policy") != treatment.get("normal_encoding_policy")
                or reference.get("declared_codec_actions") != treatment.get("declared_codec_actions")):
            raise ValueError("Reference and treatment use different encoding policies")
        fixed, final = reference["complete_file_bits"], treatment["complete_file_bits"]
        result.update(fixed1024_bits=fixed, adaptive_bits=final,
            gain_vs_fixed1024_pct=100*(fixed-final)/fixed)
        for name in ("normal_initial_bits", "gain_vs_initial_pct", "normal_encoding_policy",
                "detected_class_segments", "initial_segments", "boundary_segments",
                "accepted_operations", "termination", "budget_limited", "infeasible_policy_proposals"):
            result[name] = treatment.get(name)
    return result


def validate_local_gain_row(row, n, repair=None):
    """Only local measured monotonicity is claimed, never Btr dominance."""
    if row["method"] != TREATMENT or row["status"] != "passed":
        return
    initial, final = row["initial_bits"], row["complete_file_bits"]
    if (row.get("initial_segments") != 1 or row.get("normal_initial_bits") != initial
            or row.get("initial_file_roundtrip_verified") is not True or final > initial):
        raise ValueError("Local-gain result violates its normal-parent cost proof")
    policy = row.get("normal_encoding_policy")
    actions = row.get("declared_codec_actions")
    if policy == DIAGNOSTIC_POLICY:
        if actions != []:
            raise ValueError("Diagnostic policy must not masquerade as declared actions")
    elif policy != DECLARED_POLICY or list(validate_declared_actions(tuple(actions or ()))) != actions:
        raise ValueError("Local-gain result lacks its explicit normal encoding policy")
    if repair is not None:
        operations = repair["operations"]
        if (repair.get("initial_intervals") not in ([(0, n)], [[0, n]])
                or repair["initial_bits"] + FILE_HEADER.size*8 != initial
                or repair["total_bits"] + FILE_HEADER.size*8 != final
                or repair.get("telescoping_ok") is not True
                or repair.get("estimates_used_for_acceptance") is not False
                or len(operations) != row["accepted_operations"]
                or repair.get("infeasible_policy_proposals") != row.get("infeasible_policy_proposals")
                or repair.get("budget_limited") != row.get("budget_limited")
                or sum(op["gain_bits"] for op in operations) != initial-final):
            raise ValueError("Local measured repair proof differs from the persisted file")
        records = repair["proposal_records"]
        if (repair["infeasible_policy_proposals"] != sum(r["status"] == "infeasible_codec_policy" for r in records)
                or repair["budget_limited"] != any(r["status"] == "budget_skipped" for r in records)
                or repair.get("budget_exhausted") != repair["budget_limited"]):
            raise ValueError("Policy-infeasible proposals and budget refusals must be counted separately")
        for op in operations:
            if (op.get("status") != "accepted" or op["gain_bits"] <= 0
                    or op["old_bits"]-op["new_bits"] != op["gain_bits"]):
                raise ValueError("A boundary edit was accepted without strictly positive actual net gain")


def run_series(job, out, exporter, btr, args, *, execute=replay.execute_export,
               backend_factory=None, class_runner=class_stage):
    policy, actions = encoding_policy(args)
    out, raw = Path(out).resolve(), Path(job["input"]).resolve()
    if out.exists():
        raise ValueError("Fresh per-series output required")
    protected = {str(Path(p).resolve()): replay.sha256(p) for p in (raw, exporter, btr)}
    for name in CODE_FILES:
        path = Path(__file__).with_name(name).resolve()
        protected[str(path)] = replay.sha256(path)
    if any(intro.overlap(out, Path(p)) for p in protected):
        raise ValueError("Output overlaps protected input/code/native files")
    if protected[str(raw)] != job["sha256"]:
        raise ValueError("Raw input identity changed")
    values = mainline.read_float64(str(raw))
    if len(values) != job["values"]:
        raise ValueError("Raw input count changed")
    out.mkdir(parents=True)
    rows, backend = [], None
    budget = args.value_budget_multiplier*len(values)
    manifest = out/"series_manifest.json"
    state = dict(version=VERSION, complete=False, status="running", job=job, results=rows,
        protected_sha256=protected, predeclared_methods=list(METHODS),
        normal_encoding_policy=policy, declared_codec_actions=list(actions),
        initial_source="one uncut current region; ClaSS proposes candidates only",
        common_candidate_source="original ClaSS + compression-feature union; every ClaSS anchor retained",
        fixed1024_is_reference_not_seed=True, same_normal_policy_for_parent_children_reference=True,
        no_dedicated_whole_input_exhaustive_prepass=True, no_external_baseline_output_fallback=True,
        detected_class_layout_is_not_accepted_layout=True, normal_parent_encoding_reused=True,
        measured_net_gain_required_for_every_accepted_edit=True,
        each_arm_value_budget=budget, each_arm_trial_limit=args.trial_limit,
        normal_parent_measurement_charged_inside_own_budget=True, warm_cache_keys_still_charged=True,
        same_run_immutable_native_prices=True, historical_prices_loaded=False,
        selector_prediction_used=False, selector_training_used=False, dp_used=False,
        complete_trained_framework=False, full_pool_segmentation_diagnostic=policy == DIAGNOSTIC_POLICY,
        guarantee_scope="strictly positive actual local edit gain relative to the normal current encoding only",
        external_baseline_dominance_claimed=False,
        timing_scope="shared-cache/verification wall time, NOT codec or end-to-end throughput")

    def save():
        mainline.stream_json(manifest, state)
        frozen.write_csv(out/"summary.csv", rows, FIELDS)
        frozen.write_csv(out/"comparison.csv", [comparison(rows, job)], COMPARISON_FIELDS)

    def measured(command, log, timeout):
        began = time.perf_counter()
        with (out/"native_commands.jsonl").open("a", encoding="utf-8") as stream:
            stream.write(json.dumps(dict(event="start", command=list(map(str, command)), log=str(log)))+"\n")
        record = dict(event="finish", log=str(log))
        try:
            record["exit_code"] = execute(command, Path(log), timeout)
            return record["exit_code"]
        except BaseException as error:
            record.update(event="error", detail=str(error))
            raise
        finally:
            record["wall_seconds"] = time.perf_counter()-began
            with (out/"native_commands.jsonl").open("a", encoding="utf-8") as stream:
                stream.write(json.dumps(record)+"\n")

    def persist(evaluator, regions, path):
        counters_before = dict(evaluator.view.counters), dict(backend.counters)
        info = mainline.write_verified(path, evaluator.native_segments(regions), raw,
            exporter, btr, args.timeout, measured)
        expected = FILE_HEADER.size*8 + sum(evaluator.costs[r] for r in regions)
        if info["complete_file_bits"] != expected:
            raise ValueError("Complete-file bytes differ from normal-policy paid prices")
        if counters_before != (dict(evaluator.view.counters), dict(backend.counters)):
            raise ValueError("Reopening normal paid bytes unexpectedly changed encoding work")
        protected[str(path)] = replay.sha256(path)
        return info["complete_file_bits"], protected[str(path)]

    save()
    try:
        class_dir = Path(class_runner(job, out, args)).resolve()
        source = (job["dataset"], job["series"], job["sha256"])
        candidates, candidate_info = mainline.class_candidates(class_dir, values, source, protected, args.max_candidates)
        with (class_dir/"boundaries.csv").open(newline="", encoding="utf-8-sig") as stream:
            detected = [(int(r["start"]), int(r["length"])) for r in csv.DictReader(stream)]
        original_positions = {int(r["position"]) for r in candidates}
        candidates = retain_class_candidates(candidates, detected, len(values))
        class_positions = [s for s, _ in detected if s]
        if not set(class_positions) <= {int(r["position"]) for r in candidates}:
            raise ValueError("An authenticated ClaSS candidate was silently dropped")
        candidate_info.update(original_capped_union_count=len(original_positions),
            maximum_candidates_applies_to="union extras only; authenticated ClaSS anchors cannot be removed",
            restored_class_anchor_count=len(set(class_positions)-original_positions),
            candidate_count=len(candidates), all_detected_class_cuts_retained=True)
        pool_path, layout_path = out/"common_candidates.csv", out/"initial_layout.json"
        frozen.write_csv(pool_path, candidates, CANDIDATE_FIELDS)
        mainline.stream_json(layout_path, dict(detected_class_layout=detected,
            detected_class_positions=class_positions, accepted_initial_layout=[(0, len(values))],
            class_source=str(class_dir), class_is_candidate_source_only=True))
        for path in (pool_path, layout_path):
            protected[str(path)] = replay.sha256(path)
        candidate_hash = protected[str(pool_path)]
        config = RepairConfig(max_rounds=args.max_rounds, max_proposals_per_round=args.max_proposals,
            max_split_candidates=args.max_split_candidates)
        config.validate()
        state.update(candidate_protocol=candidate_info, candidate_sha256=candidate_hash,
            repair_config=asdict(config), class_source=str(class_dir), detected_class_segments=len(detected),
            detected_class_positions=class_positions, initial_accepted_segments=1,
            class_seconds_are_historical_when_reused=out not in class_dir.parents)
        factory = backend_factory or (mainline.DiskCosts if policy == DIAGNOSTIC_POLICY else DeclaredActionService)
        policy_kwargs = {} if policy == DIAGNOSTIC_POLICY else dict(declared_actions=actions)
        backend = factory(dict(n=len(values), raw=raw, key=source+(VERSION,)),
            out/"native", exporter, btr, args.timeout, value_budget=2*budget,
            trial_limit=2*args.trial_limit, execute=measured, **policy_kwargs)
        for method in METHODS:
            regions = fixed_partition(len(values), 1024) if method == REFERENCE else [(0, len(values))]
            folder = out/method
            folder.mkdir()
            row = dict(job, method=method, status="failed", detail="", source_manifest=str(manifest),
                complete_file_bits=None, complete_file_roundtrip_verified=False,
                initial_file_roundtrip_verified=False, initial_segments=len(regions), detected_class_segments=len(detected),
                candidate_sha256=candidate_hash, virtual_value_budget=budget, virtual_trial_limit=args.trial_limit,
                normal_encoding_policy=policy, declared_codec_actions=list(actions))
            rows.append(row)
            view = BudgetedRegionView(backend, budget, args.trial_limit)
            evaluator = CachedNativeRegionEvaluator(view)
            if policy == DECLARED_POLICY:
                evaluator.cost_source = "actual_native_declared_codec_actions_stored_frame_bytes"
            began = time.perf_counter()
            print(job["series"]+"/"+method+": start; policy="+policy, flush=True)
            try:
                repair, final_regions = None, regions
                if evaluator.policy_infeasible(regions):
                    raise ValueError("Normal initial region cannot be encoded by the declared actions; no substitution or fallback")
                if method == TREATMENT:
                    # The repair algorithm obtains the normal parent's cost once.
                    # No all-codec/baseline prepass is performed outside this call.
                    repair = repair_partition(regions, candidates, evaluator, config=config,
                        tag=method, proposal_scorer=CompressionGainProxy(values).gain)
                    mark_infeasible_policy_proposals(repair, evaluator)
                    repair_path = folder/"repair.json"
                    mainline.stream_json(repair_path, repair)
                    protected[str(repair_path)] = replay.sha256(repair_path)
                    final_regions = [tuple(item) for item in repair["intervals"]]
                else:
                    evaluator.evaluate(regions, phase="seed")
                # This is an audit copy of already-priced immutable bytes, not
                # another encoding or an output candidate/fallback decision.
                initial = folder/"initial.f64blk"
                initial_bits, digest = persist(evaluator, regions, initial)
                row.update(initial_bits=initial_bits, normal_initial_bits=initial_bits,
                    initial_container_path=str(initial), initial_container_sha256=digest,
                    initial_file_roundtrip_verified=True)
                final = folder/"final.f64blk"
                bits, digest = persist(evaluator, final_regions, final)
                boundary_path = folder/"boundaries.csv"
                frozen.write_csv(boundary_path, [dict(start=s, length=n,
                    codec=evaluator.choices[(s, n)]["codec"]) for s, n in final_regions], ("start", "length", "codec"))
                protected[str(boundary_path)] = replay.sha256(boundary_path)
                row.update(status="passed", complete_file_bits=bits, complete_file_roundtrip_verified=True,
                    container_path=str(final), container_sha256=digest, boundary_segments=len(final_regions),
                    accepted_operations=len(repair["operations"]) if repair else 0,
                    gain_vs_initial_pct=100*(initial_bits-bits)/initial_bits,
                    termination=repair["termination"] if repair else "reference_only_no_search",
                    budget_limited=repair["budget_limited"] if repair else False,
                    infeasible_policy_proposals=repair["infeasible_policy_proposals"] if repair else 0)
                validate_local_gain_row(row, len(values), repair)
            except (OSError, ValueError, KeyError, TypeError) as error:
                row.update(status="failed", detail=str(error), complete_file_bits=None,
                    complete_file_roundtrip_verified=False)
            finally:
                row.update(trial_count=view.counters["trial_count"], trial_value_volume=view.counters["trial_value_volume"],
                    wall_seconds_not_comparable_throughput=time.perf_counter()-began)
                ledger_path = folder/"ledger.json"
                mainline.stream_json(ledger_path, dict(counters=view.counters, value_budget=budget,
                    trial_limit=args.trial_limit, candidate_sha256=candidate_hash, charged_keys=sorted(view.charged_keys),
                    normal_encoding_policy=policy, declared_codec_actions=list(actions),
                    dedicated_whole_input_prepass=False, external_baseline_fallback=False,
                    cached_normal_parent_reused=True))
                protected[str(ledger_path)] = replay.sha256(ledger_path)
            if row["trial_count"] > args.trial_limit or row["trial_value_volume"] > budget:
                raise ValueError("Independent arm exceeded budget")
            if row["status"] == rows[0]["status"] == "passed":
                row["gain_vs_fixed1024_pct"] = 100*(rows[0]["complete_file_bits"]-row["complete_file_bits"])/rows[0]["complete_file_bits"]
            intro.check_protected(protected)
            save()
            print(job["series"], method, row["status"], "initial=", row.get("normal_initial_bits"),
                "final=", row.get("complete_file_bits"), row["detail"], flush=True)
        state.update(complete=True, status="passed" if all(r["status"] == "passed" for r in rows) else "completed_with_failures",
            protected_files_unchanged=True, native_counters=dict(backend.counters))
    except BaseException as error:
        state.update(status="failed", detail=str(error), protected_files_unchanged=not replay.unchanged(protected))
        for row in rows:
            row.update(status="invalidated_failed_run", complete_file_bits=None, complete_file_roundtrip_verified=False)
        raise
    finally:
        if backend is not None:
            mainline.stream_json(out/"native_trials.json", backend.trials)
        save()
    return rows


def collect(job, folder, rc):
    """Recheck identities, normal-policy ledger, local gains and stored files."""
    folder = Path(folder).resolve()
    try:
        state = frozen.read_json(folder/"series_manifest.json")
        if (rc not in (0, 1) or state.get("version") != VERSION or state.get("complete") is not True
                or state.get("protected_files_unchanged") is not True
                or state.get("predeclared_methods") != list(METHODS)
                or state.get("status") not in ("passed", "completed_with_failures")
                or state.get("detected_class_layout_is_not_accepted_layout") is not True
                or state.get("same_normal_policy_for_parent_children_reference") is not True
                or state.get("no_dedicated_whole_input_exhaustive_prepass") is not True
                or state.get("no_external_baseline_output_fallback") is not True
                or state.get("normal_parent_encoding_reused") is not True
                or state.get("external_baseline_dominance_claimed") is not False):
            raise ValueError("Child lacks protected normal-policy local-gain proof")
        policy, actions = state["normal_encoding_policy"], state["declared_codec_actions"]
        if policy == DIAGNOSTIC_POLICY:
            if actions != [] or state.get("full_pool_segmentation_diagnostic") is not True:
                raise ValueError("Diagnostic encoding policy is inconsistent")
        elif (policy != DECLARED_POLICY or list(validate_declared_actions(tuple(actions))) != actions
                or state.get("full_pool_segmentation_diagnostic") is not False):
            raise ValueError("Child has no explicit valid normal encoding policy")
        intro.check_protected(state["protected_sha256"])
        rows = state["results"]
        if [r["method"] for r in rows] != list(METHODS):
            raise ValueError("Unexpected child methods")
        for row in rows:
            if any(row.get(k) != job[k] for k in ("dataset", "series", "sha256", "values")):
                raise ValueError("Child identity mismatch")
            if row.get("normal_encoding_policy") != policy or row.get("declared_codec_actions") != actions:
                raise ValueError("A child arm used a different normal encoding policy")
            if row["status"] != "passed":
                if row.get("complete_file_bits") is not None:
                    raise ValueError("Failed child has a usable size")
                continue

            def verify(prefix, bits_field, flag):
                if row.get(flag) is not True:
                    raise ValueError("Child lacks persisted-file bitwise proof")
                path = Path(row[prefix+"container_path"]).resolve()
                if folder not in path.parents:
                    raise ValueError("Child output escapes worker folder")
                info = inspect_container(path)
                if (replay.sha256(path) != row[prefix+"container_sha256"] or info["values"] != job["values"]
                        or info["complete_file_bits"] != row[bits_field]
                        or state["protected_sha256"].get(str(path)) != row[prefix+"container_sha256"]):
                    raise ValueError("Child persisted size/hash/count/protection mismatch")
            verify("initial_", "initial_bits", "initial_file_roundtrip_verified")
            verify("", "complete_file_bits", "complete_file_roundtrip_verified")
            if row["normal_initial_bits"] != row["initial_bits"]:
                raise ValueError("Child normal initial cost differs from its paid bytes")
            ledger_path = folder/row["method"]/"ledger.json"
            if state["protected_sha256"].get(str(ledger_path)) != replay.sha256(ledger_path):
                raise ValueError("Ledger proof is not protected")
            ledger = frozen.read_json(ledger_path)
            if (len(ledger["charged_keys"]) != row["trial_count"]
                    or sum(k[1] for k in ledger["charged_keys"]) != row["trial_value_volume"]
                    or row["trial_count"] > ledger["trial_limit"]
                    or row["trial_value_volume"] > ledger["value_budget"]
                    or ledger.get("normal_encoding_policy") != policy
                    or ledger.get("declared_codec_actions") != actions
                    or ledger.get("dedicated_whole_input_prepass") is not False
                    or ledger.get("external_baseline_fallback") is not False
                    or ledger.get("cached_normal_parent_reused") is not True):
                raise ValueError("Child independent normal-policy budget proof differs from paid keys")
            if row["method"] == TREATMENT:
                repair_path = folder/TREATMENT/"repair.json"
                if state["protected_sha256"].get(str(repair_path)) != replay.sha256(repair_path):
                    raise ValueError("Repair proof is not protected")
                validate_local_gain_row(row, job["values"], frozen.read_json(repair_path))
        return rows
    except (OSError, ValueError, KeyError, TypeError) as error:
        return [dict(job, method=m, status="failed", detail="exit=%s; %s; inspect per-series log" % (rc, error),
            complete_file_bits=None, complete_file_roundtrip_verified=False, initial_file_roundtrip_verified=False,
            source_manifest=str(folder/"series_manifest.json")) for m in METHODS]


def batch(args, *, executor=execute_child):
    policy, actions = encoding_policy(args)
    jobs, protected, exporter, btr = intro.preflight(args)
    own = Path(__file__).resolve()
    for name in CODE_FILES:
        path = Path(__file__).with_name(name).resolve()
        protected[str(path)] = replay.sha256(path)
    out = Path(args.out_dir).resolve()
    if args.class_source_root and intro.overlap(out, Path(args.class_source_root).resolve()):
        raise ValueError("Output overlaps ClaSS source root")
    out.mkdir(parents=True, exist_ok=True)
    (out/"logs").mkdir()
    state = dict(version=VERSION, status="running", complete=False, jobs=jobs, results=[], children=[],
        configuration=vars(args), protected_sha256=protected, predeclared_methods=list(METHODS),
        normal_encoding_policy=policy, declared_codec_actions=list(actions),
        sequential_fresh_child_processes=True, no_best_arm_selected=True,
        fixed1024_is_reference_not_seed=True, class_is_candidate_source_only=True,
        no_dedicated_whole_input_exhaustive_prepass=True, no_external_baseline_output_fallback=True,
        external_baseline_dominance_claimed=False,
        timing_scope="NOT comparable compressor throughput")

    def save():
        mainline.stream_json(out/"manifest.json", state)
        frozen.write_csv(out/"summary.csv", state["results"], FIELDS)
        frozen.write_csv(out/"comparison.csv", [comparison(
            [r for r in state["results"] if r["series"] == j["series"]], j) for j in jobs], COMPARISON_FIELDS)
        frozen.write_csv(out/"coverage.csv", [dict(series=j["series"], method=m,
            status=next((r["status"] for r in state["results"] if r["series"] == j["series"] and r["method"] == m), "pending"),
            detail=next((r.get("detail", "") for r in state["results"] if r["series"] == j["series"] and r["method"] == m), ""))
            for j in jobs for m in METHODS], ("series", "method", "status", "detail"))
    save()
    try:
        for index, job in enumerate(jobs, 1):
            folder, log = out/job["series"], out/"logs"/(job["series"]+".log")
            cmd = [sys.executable, "-u", str(own), "--worker", "--source-config", str(Path(args.source_config).resolve()),
                "--data-repo", args.data_repo, "--out-dir", str(folder), "--series", job["series"],
                "--exporter", str(exporter), "--btr-exe", str(btr)]
            if policy == DIAGNOSTIC_POLICY:
                cmd.append("--diagnostic-full-pool")
            else:
                cmd.extend(("--codec-actions", *actions))
            for name in ("value_budget_multiplier", "trial_limit", "max_rounds", "max_proposals", "max_split_candidates",
                    "max_candidates", "timeout", "class_history", "class_subsequence", "class_jump", "class_timeout"):
                cmd.extend(("--"+name.replace("_", "-"), str(getattr(args, name))))
            for name in ("native_manifest", "class_source_root"):
                if getattr(args, name):
                    cmd.extend(("--"+name.replace("_", "-"), str(Path(getattr(args, name)).resolve())))
            child = dict(series=job["series"], command=cmd, log=str(log), status="running")
            state["children"].append(child)
            save()
            print("===== %s/%s %s; policy=%s; log=%s =====" % (index, len(jobs), job["series"], policy, log), flush=True)
            try:
                rc = executor(cmd, log)
            except OSError as error:
                rc, child["detail"] = 127, str(error)
            rows = collect(job, folder, rc)
            if any(r.get("status") == "passed" and (r.get("normal_encoding_policy") != policy
                    or r.get("declared_codec_actions") != list(actions)) for r in rows):
                raise ValueError("Child policy differs from parent predeclared policy")
            state["results"].extend(rows)
            child.update(exit_code=rc, status="passed" if all(r["status"] == "passed" for r in rows) else "completed_with_failures")
            intro.check_protected(protected)
            if rc in (0, 1) and any(r.get("status") == "passed" for r in rows):
                proof = frozen.read_json(folder/"series_manifest.json")
                for path, digest in proof["protected_sha256"].items():
                    if path in protected and protected[path] != digest:
                        raise ValueError("Child source identity differs from prior protected input")
                    protected[path] = digest
                intro.check_protected(protected)
            save()
            print("===== completed %s: %s =====" % (job["series"], child["status"]), flush=True)
        intro.check_protected(protected)
        state.update(complete=True, protected_files_unchanged=True,
            status="passed" if all(r["status"] == "passed" for r in state["results"]) else "completed_with_failures")
    except BaseException as error:
        state.update(status="failed", complete=False, detail=str(error))
        if replay.unchanged(protected):
            for row in state["results"]:
                row.update(status="protected_input_changed", complete_file_bits=None, complete_file_roundtrip_verified=False)
        raise
    finally:
        save()
    print("Comparison: "+str(out/"comparison.csv"), flush=True)
    print("Coverage: "+str(out/"coverage.csv"), flush=True)
    return int(state["status"] != "passed")


def main():
    parser = intro.parser()
    parser.description = __doc__
    parser.add_argument("--worker", action="store_true")
    parser.add_argument("--class-source-root")
    modes = parser.add_mutually_exclusive_group(required=True)
    modes.add_argument("--diagnostic-full-pool", action="store_true",
        help="Explicit full-codec C1 diagnostic, not an online prediction policy")
    modes.add_argument("--codec-actions", nargs="+",
        help="One or two predeclared codec actions; not a learned selector and no automatic fallback")
    for name, default in dict(class_history=4096, class_subsequence=32, class_jump=5, class_timeout=1800).items():
        parser.add_argument("--"+name.replace("_", "-"), type=int, default=default)
    args = parser.parse_args()
    if (min(args.class_history, args.class_subsequence, args.class_jump, args.class_timeout) <= 0
            or args.class_history <= 10*args.class_subsequence):
        parser.error("Invalid real ClaSS configuration")
    try:
        encoding_policy(args)
        if args.worker:
            jobs, _, exporter, btr = intro.preflight(args)
            if len(jobs) != 1:
                raise ValueError("Worker requires exactly one declared series")
            rows = run_series(jobs[0], Path(args.out_dir), exporter, btr, args)
            return int(any(r["status"] != "passed" for r in rows))
        return batch(args)
    except (OSError, ValueError, KeyError, TypeError, subprocess.SubprocessError) as error:
        parser.exit(2, VERSION+": "+str(error)+"\n")


if __name__ == "__main__":
    raise SystemExit(main())
