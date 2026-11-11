"""Two-panel motivation experiment, isolated from the ClaSS mainline.

A: whole-series codec choice vs strict fixed1024 per-block choice.
B: numeric-only detected layout vs compression-aware repair of THAT layout.
All four use the same feasible full codec pool and one immutable native cache.
No RF, ClaSS, DP, historical trial price, or result-dependent series selection.
This is an exhaustive diagnostic, NOT a fast deployed selector experiment.
"""
import argparse
from dataclasses import asdict
import json
import math
import os
from pathlib import Path
import re
import time
import traceback

import class_mainline_btr as mainline
from class_native_evaluator import NativeRegionEvaluator
from btr_search_ledger import BudgetedRegionView, fixed_partition
from compression_boundary_repair import RepairConfig, repair_partition
from compression_candidates import build_candidates
from compression_gain_proxy import CompressionGainProxy
from external_file_format import FILE_HEADER
import frozen_btr_extension as frozen
import replay_frozen_files as replay
from motivation_numeric_changes import detect


VERSION = "paper_intro_motivation_v1"
METHODS = ("global_best", "fixed1024_best", "numeric_change_best", "compression_aware_best")
SAFE = re.compile(r"[A-Za-z0-9][A-Za-z0-9_.+-]*", re.ASCII)
FIELDS = ("dataset", "series", "values", "sha256", "method", "status", "detail",
    "complete_file_bits", "complete_file_roundtrip_verified", "container_path", "container_sha256",
    "boundary_segments", "trial_value_volume", "trial_count", "accepted_operations", "source_manifest")
RUNTIME_FILES = ("paper_intro_motivation.py", "motivation_numeric_changes.py") + mainline.CODE_FILES


def overlap(a, b):
    return a == b or a in b.parents or b in a.parents


def check_protected(protected):
    changed = replay.unchanged(protected)
    if changed:
        raise ValueError("Protected inputs changed: " + repr(changed))


def candidate_union(values, source, numeric, maximum):
    """Keep numeric origins explicit; never relabel them as genuine ClaSS."""
    cache = mainline.FeatureCache(values, source)
    compression = build_candidates(cache, [(0, len(values))], mode="compression",
        maximum=maximum, window=256, grid=64, minimum=64)
    quota = maximum // 2
    ranked_numeric = sorted(numeric, key=lambda r: (-r["score"], r["position"]))[:quota]
    ranked_compression = sorted(compression, key=lambda r: (-r["score"], r["position"]))[:maximum - quota]
    selected = {}
    for item in ranked_numeric + ranked_compression:
        item = dict(item)
        if item["position"] not in selected:
            selected[item["position"]] = item
        else:
            old = selected[item["position"]]
            old["score"] = max(old["score"], item["score"])
            old["sources"] = ";".join(sorted(set(old["sources"].split(";")) | set(item["sources"].split(";"))))
    # Empty sources stay empty: never fill with arbitrary uniform boundaries.
    return sorted(selected.values(), key=lambda r: r["position"]), dict(
        numeric_retained=len(ranked_numeric), compression_retained=len(ranked_compression),
        union_count=len(selected), feature_values_scanned=cache.values_scanned,
        feature_seconds=cache.seconds, actual_class_used=False)


def run_series(job, out, exporter, btr, args, *, execute=replay.execute_export,
               backend_factory=mainline.DiskCosts, detector=detect):
    raw = Path(job["input"]).resolve()
    out = Path(out).resolve()
    if out.exists():
        raise ValueError("Per-series output already exists")
    source = (job["dataset"], job["series"], job["sha256"])
    protected = {str(p): replay.sha256(p) for p in (raw, exporter, btr)}
    for name in dict.fromkeys(RUNTIME_FILES):
        path = Path(__file__).with_name(name).resolve()
        protected[str(path)] = replay.sha256(path)
    for path in map(Path, protected):
        if overlap(out, path):
            raise ValueError("Output overlaps protected input/runtime")
    if protected[str(raw)] != job["sha256"]:
        raise ValueError("Raw input differs from preflight identity")
    values = mainline.read_float64(str(raw))
    if len(values) != job["values"]:
        raise ValueError("Raw input count changed")
    detection = detector(values, window=args.window, grid=64, minimum=64,
        threshold=args.numeric_threshold, max_boundaries=args.max_numeric_boundaries)
    numeric = [tuple(pair) for pair in detection["intervals"]]
    if (not numeric or numeric[0][0] != 0 or sum(n for _, n in numeric) != len(values)
            or any(n < 64 or s != sum(k for _, k in numeric[:i]) for i, (s, n) in enumerate(numeric))):
        raise ValueError("Numeric detector did not produce complete legal regions")
    candidates, candidate_state = candidate_union(values, source, detection["candidates"], args.max_candidates)
    config = RepairConfig(max_rounds=args.max_rounds, max_proposals_per_round=args.max_proposals,
        max_split_candidates=args.max_split_candidates)
    config.validate()
    proxy = CompressionGainProxy(values)
    out.mkdir(parents=True)
    state = dict(version=VERSION, status="running", identity=dict(zip(mainline.IDENTITY, source)),
        values=len(values), results=[], protected_sha256=protected, detector=detection,
        candidate_protocol=candidate_state, repair_config=asdict(config),
        actual_class_used=False, selector_training_used=False, selector_prediction_used=False, dp_used=False,
        same_codec_pool=True, same_run_immutable_native_prices=True,
        right_comparison_scope="numeric segmentation versus compression-feature candidates PLUS actual-cost repair; not feature-only causal ablation",
        treatment_initial_partition="numeric_change_best only; no whole/fixed output fallback or best-seed selection",
        budget_policy="each arm independently pays unique attempted codec/value keys; same ceiling, NOT equal work or speed",
        each_arm_value_budget=args.value_budget_multiplier * len(values), each_arm_trial_limit=args.trial_limit,
        global_choice_tail_policy="ALP aligned prefix may use an explicit streaming-tail composite action",
        no_historical_price_import=True, no_mainline_files_modified=True)
    manifest_path = out / "series_manifest.json"
    mainline.stream_json(out / "numeric_detector.json", detection)
    frozen.write_csv(out / "compression_candidates.csv", candidates,
        ("position", "score", "sources", "strongest_group", "state_contrast"))
    rows = state["results"]
    service = backend_factory(dict(n=len(values), raw=raw, key=source + (VERSION,)),
        out / "native", exporter, btr, args.timeout,
        value_budget=4 * args.value_budget_multiplier * len(values),
        trial_limit=4 * args.trial_limit, execute=execute)

    def save():
        mainline.stream_json(manifest_path, state)
        frozen.write_csv(out / "summary.csv", rows, FIELDS)

    try:
        for method in METHODS:
            row = dict(job, method=method, status="failed", detail="", complete_file_bits=None,
                complete_file_roundtrip_verified=False, source_manifest=str(manifest_path))
            rows.append(row)
            print(job["series"] + "/" + method + ": start", flush=True)
            view = BudgetedRegionView(service, args.value_budget_multiplier * len(values), args.trial_limit)
            evaluator = NativeRegionEvaluator(view)
            try:
                regions = ([(0, len(values))] if method == "global_best" else
                    fixed_partition(len(values), 1024) if method == "fixed1024_best" else numeric)
                if not evaluator.can_evaluate(regions):
                    raise ValueError("Declared cap cannot price this complete initial layout")
                evaluator.evaluate(regions, phase="seed")
                repair = None
                if method == "compression_aware_best":
                    repair = repair_partition(numeric, candidates, evaluator, config=config,
                        tag="numeric_seed_compression_repair", proposal_scorer=proxy.gain)
                    if not repair["telescoping_ok"] or repair["total_bits"] > repair["initial_bits"]:
                        raise ValueError("Repair violated its measured seed-relative guarantee")
                    regions = repair["intervals"]
                    mainline.stream_json(out / "repair.json", repair)
                file = out / (method + ".f64blk")
                info = mainline.write_verified(file, evaluator.native_segments(regions), raw,
                    exporter, btr, args.timeout, execute)
                if repair and info["complete_file_bits"] != repair["total_bits"] + FILE_HEADER.size * 8:
                    raise ValueError("Repair costs disagree with complete stored file")
                frozen.write_csv(out / (method + "_boundaries.csv"),
                    [dict(start=s, length=n, codec=evaluator.choices[(s, n)]["codec"]) for s, n in regions],
                    ("start", "length", "codec"))
                row.update(status="passed", complete_file_bits=info["complete_file_bits"],
                    complete_file_roundtrip_verified=True, container_path=str(file), container_sha256=replay.sha256(file),
                    boundary_segments=len(regions), accepted_operations=len(repair["operations"]) if repair else 0,
                    trial_value_volume=view.counters["trial_value_volume"], trial_count=view.counters["trial_count"])
            except (OSError, ValueError, KeyError, TypeError) as error:
                row.update(status="failed", detail=str(error))
            finally:
                row.update(trial_value_volume=view.counters["trial_value_volume"],
                    trial_count=view.counters["trial_count"])
            check_protected(protected)
            print(job["series"] + "/" + method + ": " + row["status"] + " bits=" + str(row.get("complete_file_bits")), flush=True)
            save()
        state.update(status="passed" if all(r["status"] == "passed" for r in rows) else "completed_with_failures",
            protected_files_unchanged=True, native_counters=dict(service.counters))
        mainline.stream_json(out / "native_trials.json", service.trials)
    except BaseException as error:
        changed = replay.unchanged(protected)
        state.update(status="failed", detail=str(error), protected_files_unchanged=not changed)
        if changed:
            for row in rows:
                row.update(status="protected_input_changed", complete_file_bits=None,
                    complete_file_roundtrip_verified=False)
        raise
    finally:
        save()
    return rows


def native_paths(args):
    header = {}
    if args.native_manifest:
        path = Path(args.native_manifest).resolve()
        with path.open("rb") as stream:
            text = stream.read(1024 * 1024).decode("utf-8-sig", errors="replace")
        for key in ("exporter", "btr_exe"):
            matches = re.findall(r'(?m)^ {2}"' + key + r'"\s*:\s*("(?:[^"\\]|\\.)*")\s*(?:,|$)', text)
            if len(matches) > 1:
                raise ValueError("Ambiguous native executable: " + key)
            if matches:
                value = Path(json.loads(matches[0]))
                header[key] = value if value.is_absolute() else path.parent / value
    repo = Path(args.data_repo).resolve()
    candidates = dict(exporter=repo / "experiment_results/framed_plan_build_v1/FrozenPlanFile",
        btr_exe=repo / "experiment_results/external_native_build_cache_v7_files/btrblocks/BtrBlocksSmoke")
    result = []
    for key, default in candidates.items():
        file = Path(getattr(args, key) or header.get(key) or default).resolve()
        if not file.is_file() or file.suffix.lower() == ".exe" or (os.name != "nt" and not os.access(file, os.X_OK)):
            raise ValueError("Missing existing Linux native executable: " + str(file))
        result.append(file)
    return tuple(result)


def preflight(args):
    for name in ("value_budget_multiplier", "trial_limit", "max_rounds", "max_proposals",
                 "max_split_candidates", "max_candidates", "window", "max_numeric_boundaries"):
        if type(getattr(args, name)) is not int or getattr(args, name) < 1:
            raise ValueError(name + " must be a positive integer")
    if args.max_candidates < 2 or args.window < 64 or args.window % 64:
        raise ValueError("max-candidates >=2 and window a multiple of64 >=64 required")
    if not math.isfinite(args.numeric_threshold) or args.numeric_threshold <= 0:
        raise ValueError("numeric-threshold must be finite and positive")
    if not math.isfinite(args.timeout) or args.timeout <= 0:
        raise ValueError("timeout must be finite and positive")
    config_path = Path(args.source_config).resolve()
    state = frozen.read_json(config_path)
    original = state.get("datasets")
    if not isinstance(original, list) or not original:
        raise ValueError("Source config needs its original datasets list")
    names = [r["series"] for r in original]
    if len(names) != len(set(names)) or any(not SAFE.fullmatch(name) for name in names):
        raise ValueError("Unsafe or duplicate declared series")
    if args.series and (len(set(args.series)) != len(args.series) or not set(args.series) <= set(names)):
        raise ValueError("Explicit series must be unique names from original source config")
    # Preserve original declaration order, never order by compression outcome.
    source_rows = [r for r in original if not args.series or r["series"] in args.series]
    out = Path(args.out_dir).resolve()
    if out.exists() and (not out.is_dir() or any(p.name != "run.log" or not p.is_file() or p.is_symlink() for p in out.iterdir())):
        raise ValueError("Use fresh output; only redirected run.log is allowed")
    if overlap(out, config_path.parent) or overlap(out, Path(__file__).resolve().parents[1]):
        raise ValueError("Output overlaps old preparation run or code")
    exporter, btr = native_paths(args)
    protected = {str(p): replay.sha256(p) for p in (config_path, exporter, btr)}
    for name in dict.fromkeys(RUNTIME_FILES):
        runtime = Path(__file__).with_name(name).resolve()
        protected[str(runtime)] = replay.sha256(runtime)
    if args.native_manifest:
        native_manifest = Path(args.native_manifest).resolve()
        protected[str(native_manifest)] = replay.sha256(native_manifest)
    prior = config_path.parent / "baselines/data_manifest.json"
    known = None
    if prior.is_file():
        protected[str(prior)] = replay.sha256(prior)
        known_rows = frozen.read_json(prior).get("datasets", [])
        known = {r["series"]: r for r in known_rows}
        if len(known) != len(known_rows):
            raise ValueError("Duplicate old data-manifest identity")
    jobs = []
    for row in source_rows:
        raw = Path(row["input"]).resolve()
        values = mainline.read_float64(str(raw))
        if not 1024 <= len(values) <= 102400:
            raise ValueError("Require every original finite f64 input in1024..102400; no truncation")
        if row["series"] != raw.stem or row.get("dataset", raw.parent.name) != raw.parent.name:
            raise ValueError("Input path does not match declared dataset/series")
        job = dict(series=row["series"], dataset=raw.parent.name, family=row.get("family", raw.parent.name),
            input=str(raw), values=len(values), sha256=replay.sha256(raw))
        if known is not None and (job["series"] not in known or any(
                job[k] != known[job["series"]][k] for k in ("sha256", "values"))):
            raise ValueError("Input differs from prior certified baseline preparation")
        protected[str(raw)] = job["sha256"]
        jobs.append(job)
    for file in map(Path, protected):
        if overlap(out, file):
            raise ValueError("Output overlaps protected source")
    check_protected(protected)
    return jobs, protected, exporter, btr


def batch(args, *, execute=replay.execute_export, backend_factory=mainline.DiskCosts):
    jobs, protected, exporter, btr = preflight(args)
    out = Path(args.out_dir).resolve()
    out.mkdir(parents=True, exist_ok=True)
    state = dict(version=VERSION, status="running", complete=False, jobs=jobs, results=[],
        source_config=str(Path(args.source_config).resolve()), protected_sha256=protected,
        configuration=vars(args), historical_costs_loaded=False, actual_class_used=False,
        experiment_scope="motivation, exhaustive codec choice; not the finalized ClaSS framework")
    def save():
        mainline.stream_json(out / "experiment_manifest.json", state)
        frozen.write_csv(out / "summary.csv", state["results"], FIELDS)
    save()
    for i, job in enumerate(jobs):
        print("===== %d/%d %s =====" % (i + 1, len(jobs), job["series"]), flush=True)
        try:
            rows = run_series(job, out / job["series"], exporter, btr, args,
                execute=execute, backend_factory=backend_factory)
        except (OSError, ValueError, KeyError, TypeError) as error:
            print(traceback.format_exc(), flush=True)
            rows = [dict(job, method=method, status="failed", detail=str(error),
                complete_file_roundtrip_verified=False) for method in METHODS]
        state["results"].extend(rows)
        try:
            check_protected(protected)
        except ValueError:
            for row in state["results"]:
                row.update(status="protected_input_changed", complete_file_bits=None,
                    complete_file_roundtrip_verified=False)
            state.update(status="protected_input_changed", complete=False)
            save()
            raise
        save()
    state.update(complete=True, protected_files_unchanged=True,
        status="passed" if all(r["status"] == "passed" for r in state["results"]) else "completed_with_failures")
    save()
    print("Motivation complete: " + str(out / "summary.csv"), flush=True)
    return int(state["status"] != "passed")


def parser():
    p = argparse.ArgumentParser(description=__doc__)
    for name in ("source-config", "data-repo", "out-dir"):
        p.add_argument("--" + name, required=True)
    p.add_argument("--native-manifest")
    p.add_argument("--exporter")
    p.add_argument("--btr-exe")
    p.add_argument("--series", nargs="+")
    for key, default in dict(value_budget_multiplier=128, trial_limit=8192, max_rounds=12,
            max_proposals=64, max_split_candidates=6, max_candidates=256, window=256,
            max_numeric_boundaries=64).items():
        p.add_argument("--" + key.replace("_", "-"), type=int, default=default)
    p.add_argument("--numeric-threshold", type=float, default=1.)
    p.add_argument("--timeout", type=float, default=300.)
    return p


def main():
    p = parser()
    try:
        return batch(p.parse_args())
    except (OSError, ValueError, KeyError, TypeError) as error:
        p.exit(2, "paper_intro_motivation: " + str(error) + "\n")


if __name__ == "__main__":
    raise SystemExit(main())
