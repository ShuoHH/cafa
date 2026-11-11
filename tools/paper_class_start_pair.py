"""ClaSS detected seed + unchanged cost repair versus strict fixed1024 only.

No numeric seed, whole seed, best-arm fallback, selector or historical prices.
Native work is CPU-only and independently charged for both predeclared arms.
"""
from dataclasses import asdict
import csv
import json
import os
from pathlib import Path
import subprocess
import sys
import time

import paper_intro_motivation as intro
import class_mainline_btr as mainline
from btr_search_ledger import BudgetedRegionView, fixed_partition
from class_native_evaluator import NativeRegionEvaluator
from compression_boundary_repair import RepairConfig, repair_partition
from compression_gain_proxy import CompressionGainProxy
from external_file_format import FILE_HEADER, inspect_container
import frozen_btr_extension as frozen
import replay_frozen_files as replay

VERSION = "paper_class_start_pair_v1"
REFERENCE, TREATMENT = "fixed1024_reference", "class_start"
METHODS = (REFERENCE, TREATMENT)
FIELDS = intro.FIELDS + ("initial_segments", "initial_bits", "gain_vs_initial_pct",
    "gain_vs_fixed1024_pct", "initial_container_path", "initial_container_sha256",
    "initial_file_roundtrip_verified", "termination", "budget_limited", "candidate_sha256",
    "virtual_value_budget", "virtual_trial_limit", "wall_seconds_not_comparable_throughput")
COMPARISON_FIELDS = ("dataset", "series", "values", "sha256", "status", "fixed1024_bits",
    "initial_class_bits", "adaptive_bits", "gain_vs_fixed1024_pct", "gain_vs_initial_pct",
    "initial_segments", "boundary_segments", "accepted_operations", "detail")
CODE_FILES = tuple(dict.fromkeys(intro.RUNTIME_FILES + (Path(__file__).name,)))


def class_stage(job, out, args):
    """Reuse exactly one complete authenticated layout or really run ClaSS."""
    candidates = []
    if args.class_source_root:
        root = Path(args.class_source_root).resolve()
        candidates = [root / job["series"] / "class", root / job["series"]]
    ready = []
    for folder in candidates:
        flags = [(folder / name).is_file() for name in ("boundaries.csv", "segmentation_manifest.json")]
        if any(flags) and not all(flags):
            raise ValueError("Incomplete reused ClaSS proof; refusing fallback: " + str(folder))
        if all(flags): ready.append(folder.resolve())
    if len(ready) > 1:
        raise ValueError("Ambiguous ClaSS source: both supported directory layouts exist")
    if ready:
        directory = ready[0]
        if intro.overlap(out, directory): raise ValueError("Output overlaps reused ClaSS source")
    else:
        directory = out / "class"
        command = [sys.executable, "-u", str(Path(__file__).with_name("segment_class.py")), job["input"],
            "--dataset", job["dataset"], "--series", job["series"], "--out-dir", str(directory),
            "--n-timepoints", str(args.class_history), "--window-size", str(args.class_subsequence),
            "--jump", str(args.class_jump), "--snap-grid", "64", "--min-segment", "64"]
        mainline.stream_json(out / "class_command.json", dict(command=command, real_class=True))
        print(job["series"] + ": fresh real ClaSS (no alternate detector fallback)", flush=True)
        with (out / "class.log").open("w", encoding="utf-8") as log:
            result = subprocess.run(command, stdout=log, stderr=subprocess.STDOUT, timeout=args.class_timeout)
        if result.returncode:
            raise ValueError("Real ClaSS failed, exit=%s; inspect class.log; no fallback" % result.returncode)
    state = frozen.read_json(directory / "segmentation_manifest.json")
    wanted = dict(n_timepoints=args.class_history, window_size=args.class_subsequence, jump=args.class_jump)
    if any(state.get("requested_configuration", {}).get(k) != v for k, v in wanted.items()):
        raise ValueError("ClaSS configuration differs from declared settings")
    return directory


def comparison(rows, job):
    by = {r["method"]: r for r in rows}
    a, b = by.get(REFERENCE, {}), by.get(TREATMENT, {})
    valid = all(r.get("status") == "passed" and r.get("complete_file_roundtrip_verified") is True
                for r in (a, b))
    result = {k: job[k] for k in ("dataset", "series", "values", "sha256")}
    result.update(status="passed" if valid else "unavailable",
        detail="; ".join(r.get("detail", "") for r in (a, b) if r.get("detail")))
    for field in COMPARISON_FIELDS:
        if field not in result: result[field] = None
    if valid:
        x, y = a["complete_file_bits"], b["complete_file_bits"]
        result.update(fixed1024_bits=x, initial_class_bits=b["initial_bits"], adaptive_bits=y,
            gain_vs_fixed1024_pct=100*(x-y)/x, gain_vs_initial_pct=b["gain_vs_initial_pct"],
            initial_segments=b["initial_segments"], boundary_segments=b["boundary_segments"],
            accepted_operations=b["accepted_operations"])
    return result


def run_series(job, out, exporter, btr, args, *, execute=replay.execute_export,
               backend_factory=mainline.DiskCosts, class_runner=class_stage):
    """Same class_start orchestration as initial_boundary_sources_v1, one arm."""
    out, raw = Path(out).resolve(), Path(job["input"]).resolve()
    if out.exists(): raise ValueError("Fresh per-series output required")
    protected = {str(Path(p).resolve()): replay.sha256(p) for p in (raw, exporter, btr)}
    for name in CODE_FILES:
        path = Path(__file__).with_name(name).resolve()
        protected[str(path)] = replay.sha256(path)
    if any(intro.overlap(out, Path(p)) for p in protected):
        raise ValueError("Output overlaps protected input/code/native files")
    if protected[str(raw)] != job["sha256"]: raise ValueError("Raw input identity changed")
    values = mainline.read_float64(str(raw))
    if len(values) != job["values"]: raise ValueError("Raw input count changed")
    out.mkdir(parents=True)
    rows, backend = [], None
    budget = args.value_budget_multiplier * len(values)
    manifest = out / "series_manifest.json"
    state = dict(version=VERSION, complete=False, status="running", job=job, results=rows,
        protected_sha256=protected, predeclared_methods=list(METHODS),
        initial_source="authenticated real ClaSS detected full-input layout",
        common_candidate_source="unchanged original ClaSS + compression-feature union",
        fixed1024_is_reference_not_seed=True, no_alternate_seed_or_output_fallback=True,
        each_arm_value_budget=budget, each_arm_trial_limit=args.trial_limit,
        seed_measurement_charged_inside_own_budget=True, warm_cache_keys_still_charged=True,
        same_run_immutable_native_prices=True, historical_prices_loaded=False,
        selector_prediction_used=False, selector_training_used=False, dp_used=False,
        complete_trained_framework=False, full_pool_segmentation_diagnostic=True,
        timing_scope="shared-cache/verification wall time, NOT codec or end-to-end throughput")

    def save():
        mainline.stream_json(manifest, state)
        frozen.write_csv(out / "summary.csv", rows, FIELDS)
        frozen.write_csv(out / "comparison.csv", [comparison(rows, job)], COMPARISON_FIELDS)

    def measured(command, log, timeout):
        began = time.perf_counter()
        with (out / "native_commands.jsonl").open("a", encoding="utf-8") as stream:
            stream.write(json.dumps(dict(event="start", command=list(map(str, command)), log=str(log))) + "\n")
        record = dict(event="finish", log=str(log))
        try:
            record["exit_code"] = execute(command, Path(log), timeout)
            return record["exit_code"]
        except BaseException as error:
            record.update(event="error", detail=str(error))
            raise
        finally:
            record["wall_seconds"] = time.perf_counter()-began
            with (out / "native_commands.jsonl").open("a", encoding="utf-8") as stream:
                stream.write(json.dumps(record) + "\n")

    def persist(evaluator, regions, path):
        info = mainline.write_verified(path, evaluator.native_segments(regions), raw,
            exporter, btr, args.timeout, measured)
        if info["complete_file_bits"] != FILE_HEADER.size*8 + sum(evaluator.costs[r] for r in regions):
            raise ValueError("Complete-file bytes differ from paid prices")
        protected[str(path)] = replay.sha256(path)
        return info["complete_file_bits"], protected[str(path)]

    save()
    try:
        class_dir = Path(class_runner(job, out, args)).resolve()
        source = (job["dataset"], job["series"], job["sha256"])
        candidates, info = mainline.class_candidates(class_dir, values, source, protected, args.max_candidates)
        # Use the authenticated table itself, not a fixed-grid or numeric stand-in.
        with (class_dir / "boundaries.csv").open(newline="", encoding="utf-8-sig") as stream:
            seed = [(int(r["start"]), int(r["length"])) for r in csv.DictReader(stream)]
        pool_path = out / "common_candidates.csv"
        frozen.write_csv(pool_path, candidates, ("position", "score", "sources", "strongest_group", "state_contrast"))
        mainline.stream_json(out / "initial_layout.json", dict(class_start=seed, class_source=str(class_dir)))
        for path in (pool_path, out / "initial_layout.json"): protected[str(path)] = replay.sha256(path)
        candidate_hash = protected[str(pool_path)]
        config = RepairConfig(max_rounds=args.max_rounds, max_proposals_per_round=args.max_proposals,
            max_split_candidates=args.max_split_candidates)
        config.validate()
        state.update(candidate_protocol=info, candidate_sha256=candidate_hash, repair_config=asdict(config),
            class_source=str(class_dir), class_seconds_are_historical_when_reused=out not in class_dir.parents)
        backend = backend_factory(dict(n=len(values), raw=raw, key=source+(VERSION,)),
            out / "native", exporter, btr, args.timeout, value_budget=2*budget,
            trial_limit=2*args.trial_limit, execute=measured)
        for method in METHODS:
            regions = fixed_partition(len(values), 1024) if method == REFERENCE else seed
            folder = out / method
            folder.mkdir()
            row = dict(job, method=method, status="failed", detail="", source_manifest=str(manifest),
                complete_file_bits=None, complete_file_roundtrip_verified=False,
                initial_file_roundtrip_verified=False, initial_segments=len(regions),
                candidate_sha256=candidate_hash, virtual_value_budget=budget, virtual_trial_limit=args.trial_limit)
            rows.append(row)
            view = BudgetedRegionView(backend, budget, args.trial_limit)
            evaluator = NativeRegionEvaluator(view)
            began = time.perf_counter()
            print(job["series"] + "/" + method + ": start; segments=" + str(len(regions)), flush=True)
            try:
                evaluator.evaluate(regions, phase="seed")
                initial = folder / "initial.f64blk"
                initial_bits, digest = persist(evaluator, regions, initial)
                row.update(initial_bits=initial_bits, initial_container_path=str(initial),
                    initial_container_sha256=digest, initial_file_roundtrip_verified=True)
                repair = None
                final_regions = regions
                if method == TREATMENT:
                    repair = repair_partition(regions, candidates, evaluator, config=config,
                        tag=method, proposal_scorer=CompressionGainProxy(values).gain)
                    mainline.stream_json(folder / "repair.json", repair)
                    if (repair["initial_bits"]+FILE_HEADER.size*8 != initial_bits
                            or repair["total_bits"] > repair["initial_bits"] or not repair["telescoping_ok"]):
                        raise ValueError("Repair violated measured seed-relative accounting")
                    final_regions = [tuple(pair) for pair in repair["intervals"]]
                final = folder / "final.f64blk"
                bits, digest = persist(evaluator, final_regions, final)
                if repair and bits != repair["total_bits"]+FILE_HEADER.size*8:
                    raise ValueError("Repair cost differs from complete verified file")
                frozen.write_csv(folder / "boundaries.csv", [dict(start=s, length=n,
                    codec=evaluator.choices[(s, n)]["codec"]) for s, n in final_regions], ("start", "length", "codec"))
                row.update(status="passed", complete_file_bits=bits, complete_file_roundtrip_verified=True,
                    container_path=str(final), container_sha256=digest, boundary_segments=len(final_regions),
                    accepted_operations=len(repair["operations"]) if repair else 0,
                    gain_vs_initial_pct=100*(initial_bits-bits)/initial_bits,
                    termination=repair["termination"] if repair else "reference_only_no_search",
                    budget_limited=repair["budget_limited"] if repair else False)
            except (OSError, ValueError, KeyError, TypeError) as error:
                row.update(detail=str(error), complete_file_bits=None, complete_file_roundtrip_verified=False)
            finally:
                row.update(trial_count=view.counters["trial_count"], trial_value_volume=view.counters["trial_value_volume"],
                    wall_seconds_not_comparable_throughput=time.perf_counter()-began)
                mainline.stream_json(folder / "ledger.json", dict(counters=view.counters, value_budget=budget,
                    trial_limit=args.trial_limit, candidate_sha256=candidate_hash, charged_keys=sorted(view.charged_keys)))
            if row["trial_count"] > args.trial_limit or row["trial_value_volume"] > budget:
                raise ValueError("Independent arm exceeded budget")
            if row["status"] == rows[0]["status"] == "passed":
                row["gain_vs_fixed1024_pct"] = 100*(rows[0]["complete_file_bits"]-row["complete_file_bits"])/rows[0]["complete_file_bits"]
            intro.check_protected(protected)
            save()
            print(job["series"], method, row["status"], "initial=", row.get("initial_bits"),
                "final=", row.get("complete_file_bits"), row["detail"], flush=True)
        state.update(complete=True, status="passed" if all(r["status"] == "passed" for r in rows) else "completed_with_failures",
            protected_files_unchanged=True, native_counters=dict(backend.counters))
    except BaseException as error:
        state.update(status="failed", detail=str(error), protected_files_unchanged=not replay.unchanged(protected))
        for row in rows:
            row.update(status="invalidated_failed_run", complete_file_bits=None, complete_file_roundtrip_verified=False)
        raise
    finally:
        if backend is not None: mainline.stream_json(out / "native_trials.json", backend.trials)
        save()
    return rows


def execute_child(command, log):
    env = dict(os.environ)
    env.update({name: "1" for name in ("NUMBA_NUM_THREADS", "OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS",
        "MKL_NUM_THREADS", "BLIS_NUM_THREADS")})
    env.update(CLASS_NUMBA_DISK_CACHE="0", PYTHONDONTWRITEBYTECODE="1")
    with Path(log).open("w", encoding="utf-8") as stream:
        return subprocess.run(command, stdout=stream, stderr=subprocess.STDOUT, env=env).returncode


def collect(job, folder, rc):
    """Accept only complete child proofs and still-existing certified files."""
    try:
        state = frozen.read_json(folder / "series_manifest.json")
        if (rc not in (0, 1) or state.get("version") != VERSION or state.get("complete") is not True
                or state.get("protected_files_unchanged") is not True
                or state.get("predeclared_methods") != list(METHODS)
                or state.get("status") not in ("passed", "completed_with_failures")):
            raise ValueError("Child lacks complete protected success/failure proof")
        intro.check_protected(state["protected_sha256"])
        rows = state["results"]
        if [r["method"] for r in rows] != list(METHODS): raise ValueError("Unexpected child methods")
        for row in rows:
            if any(row.get(k) != job[k] for k in ("dataset", "series", "sha256", "values")):
                raise ValueError("Child identity mismatch")
            if row["status"] != "passed":
                if row.get("complete_file_bits") is not None: raise ValueError("Failed child has usable size")
                continue
            for prefix, field in (("initial_", "initial_bits"), ("", "complete_file_bits")):
                if row.get(prefix+"file_roundtrip_verified" if prefix else "complete_file_roundtrip_verified") is not True:
                    raise ValueError("Child lacks initial/final bitwise proof")
                path = Path(row[prefix+"container_path"]).resolve()
                if folder.resolve() not in path.parents: raise ValueError("Child output escapes worker folder")
                info = inspect_container(path)
                if (replay.sha256(path) != row[prefix+"container_sha256"] or info["values"] != job["values"]
                        or info["complete_file_bits"] != row[field]):
                    raise ValueError("Child persisted size/hash/count mismatch")
        return rows
    except (OSError, ValueError, KeyError, TypeError) as error:
        return [dict(job, method=m, status="failed", detail="exit=%s; %s; inspect per-series log" % (rc, error),
            complete_file_bits=None, complete_file_roundtrip_verified=False, initial_file_roundtrip_verified=False,
            source_manifest=str(folder / "series_manifest.json")) for m in METHODS]


def batch(args, *, executor=execute_child):
    jobs, protected, exporter, btr = intro.preflight(args)
    own = Path(__file__).resolve()
    protected[str(own)] = replay.sha256(own)
    out = Path(args.out_dir).resolve()
    if args.class_source_root and intro.overlap(out, Path(args.class_source_root).resolve()):
        raise ValueError("Output overlaps ClaSS source root")
    out.mkdir(parents=True, exist_ok=True)
    (out / "logs").mkdir()
    state = dict(version=VERSION, status="running", complete=False, jobs=jobs, results=[], children=[],
        configuration=vars(args), protected_sha256=protected, predeclared_methods=list(METHODS),
        sequential_fresh_child_processes=True, no_best_arm_selected=True, fixed1024_is_reference_not_seed=True,
        timing_scope="NOT comparable compressor throughput")

    def save():
        mainline.stream_json(out / "manifest.json", state)
        frozen.write_csv(out / "summary.csv", state["results"], FIELDS)
        pairs = [comparison([r for r in state["results"] if r["series"] == j["series"]], j) for j in jobs]
        frozen.write_csv(out / "comparison.csv", pairs, COMPARISON_FIELDS)
        frozen.write_csv(out / "coverage.csv", [dict(series=j["series"], method=m,
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
            for name in ("value_budget_multiplier", "trial_limit", "max_rounds", "max_proposals", "max_split_candidates",
                    "max_candidates", "timeout", "class_history", "class_subsequence", "class_jump", "class_timeout"):
                cmd.extend(("--"+name.replace("_", "-"), str(getattr(args, name))))
            for name in ("native_manifest", "class_source_root"):
                if getattr(args, name): cmd.extend(("--"+name.replace("_", "-"), str(Path(getattr(args, name)).resolve())))
            child = dict(series=job["series"], command=cmd, log=str(log), status="running")
            state["children"].append(child)
            save()
            print("===== %s/%s %s; log=%s =====" % (index, len(jobs), job["series"], log), flush=True)
            try: rc = executor(cmd, log)
            except OSError as error:
                rc, child["detail"] = 127, str(error)
            rows = collect(job, folder, rc)
            state["results"].extend(rows)
            child.update(exit_code=rc, status="passed" if all(r["status"] == "passed" for r in rows) else "completed_with_failures")
            intro.check_protected(protected)
            # Keep already-used detector proofs and worker sources protected
            # through the END of the full batch, not only until child exit.
            if rc in (0, 1) and any(r.get("status") == "passed" for r in rows):
                proof = frozen.read_json(folder / "series_manifest.json")
                for path, digest in proof["protected_sha256"].items():
                    if path in protected and protected[path] != digest:
                        raise ValueError("Child source identity differs from prior protected input")
                    protected[path] = digest
                intro.check_protected(protected)
            for row in rows:
                for key in ("container_path", "initial_container_path"):
                    if row.get(key) and Path(row[key]).is_file(): protected[row[key]] = replay.sha256(row[key])
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
    finally: save()
    print("Comparison: " + str(out / "comparison.csv"), flush=True)
    print("Coverage: " + str(out / "coverage.csv"), flush=True)
    return int(state["status"] != "passed")


def main():
    parser = intro.parser()
    parser.description = __doc__
    parser.add_argument("--worker", action="store_true")
    parser.add_argument("--class-source-root")
    for name, default in dict(class_history=4096, class_subsequence=32, class_jump=5, class_timeout=1800).items():
        parser.add_argument("--"+name.replace("_", "-"), type=int, default=default)
    args = parser.parse_args()
    if min(args.class_history, args.class_subsequence, args.class_jump, args.class_timeout) <= 0 or args.class_history <= 10*args.class_subsequence:
        parser.error("Invalid real ClaSS configuration")
    try:
        if args.worker:
            jobs, _, exporter, btr = intro.preflight(args)
            if len(jobs) != 1: raise ValueError("Worker requires exactly one declared series")
            rows = run_series(jobs[0], Path(args.out_dir), exporter, btr, args)
            return int(any(r["status"] != "passed" for r in rows))
        return batch(args)
    except (OSError, ValueError, KeyError, TypeError, subprocess.SubprocessError) as error:
        parser.exit(2, VERSION + ": " + str(error) + "\n")


if __name__ == "__main__": raise SystemExit(main())
