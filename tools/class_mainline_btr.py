"""Original ClaSS+feature-union boundary repair, with same-pool fixed1024 control.

This restores the ORIGINAL candidate builder and repair_partition, not the
btr_search_space planner. It isolates contribution C1: all feasible codecs on
visited regions are actually tried, including BtrBlocks; no RF is trained or
used. Actual retained native bytes authorize changes, never proxy estimates.
"""

import argparse
import csv
from dataclasses import asdict
import json
import math
from pathlib import Path
import time

import numpy as np

import btr_boundary_costs as costs
import btr_mixed_file as bridge
from btr_search_ledger import BudgetedRegionView
from class_native_evaluator import NativeRegionEvaluator
from compression_boundary_repair import RepairConfig, repair_partition
from compression_candidates import build_candidates
from compression_gain_proxy import CompressionGainProxy
from external_file_format import FILE_HEADER
import frozen_btr_extension as frozen
import profile_codec_features as ordinary
import profile_residue_features as residue
import replay_frozen_files as replay
from segment_class import read_float64


VERSION = "class_mainline_btr_c1_v1"
IDENTITY = ("dataset", "series", "sha256")
CODE_FILES = ("class_mainline_btr.py", "class_native_evaluator.py", "compression_boundary_repair.py",
    "compression_candidates.py", "compression_gain_proxy.py", "segment_class.py", "profile_codec_features.py",
    "profile_residue_features.py", "btr_search_ledger.py", "btr_boundary_search.py", "btr_boundary_costs.py", "btr_mixed_file.py",
    "external_file_format.py", "frozen_btr_extension.py", "replay_frozen_files.py")


def stream_json(path, value):
    """Bounded serialization memory; the whole JSON text is never constructed."""
    path = Path(path)
    pending = path.with_name(path.name + ".partial")
    with pending.open("w", encoding="utf-8") as stream:
        json.dump(value, stream, indent=2, allow_nan=False)
        stream.write("\n")
    pending.replace(path)


class DiskCosts(costs.NativeRegionService):
    def _run_trial(self, key):
        super()._run_trial(key)
        self.cache[key].pop("frames", None)


class FeatureCache:
    """Same extractor calls as original RawFeatureCache, without sklearn imports."""
    def __init__(self, values, source):
        self.values, self.source, self.rows = values, source, {}
        self.seconds, self.values_scanned = 0., 0

    def row(self, start, length):
        key = (int(start), int(length))
        if key not in self.rows:
            if start < 0 or length < 2 or start + length > len(self.values):
                raise ValueError("invalid raw feature region")
            began = time.perf_counter()
            part = self.values[start:start + length]
            row = dict(zip(IDENTITY, self.source), start=start, length=length,
                       source_kind="class_segment", segment_length=length, segment_log2_length=math.log2(length))
            row.update(ordinary.one_window(part, part.view("<u8")))
            row.update(residue.one_window(part))
            self.rows[key] = row
            self.seconds += time.perf_counter() - began
            self.values_scanned += length
        return self.rows[key]


def class_candidates(folder, values, source, protected, maximum=256):
    folder = Path(folder).resolve()
    manifest_path, table = folder / "segmentation_manifest.json", folder / "boundaries.csv"
    for path in (manifest_path, table):
        protected[str(path)] = replay.sha256(path)
    state = frozen.read_json(manifest_path)
    with table.open(newline="", encoding="utf-8-sig") as stream:
        rows = list(csv.DictReader(stream))
    intervals, cursor = [], 0
    for row in rows:
        if tuple(row.get(k) for k in IDENTITY) != source:
            raise ValueError("ClaSS boundaries identity differs from original raw input")
        start, length = int(row["start"]), int(row["length"])
        if start != cursor or length < 64 or start % 64:
            raise ValueError("ClaSS boundaries must cover the full input on the original grid")
        intervals.append((start, length))
        cursor += length
    if not intervals or cursor != len(values):
        raise ValueError("ClaSS boundaries omit or duplicate raw values")
    if (state.get("adapter_version") != "class_boundaries_v1"
            or state.get("implementation") != "claspy.streaming.segmentation.StreamingClaSPSegmentation"
            or state.get("identity") != dict(zip(IDENTITY, source))
            or state.get("values") != len(values) or state.get("input_file_values") != len(values)
            or state.get("max_values") != 0 or state.get("dtype") != "<f8"
            or state.get("original_values_transformed") is not False
            or state.get("codec_costs_or_labels_used") is not False
            or state.get("dependency_versions", {}).get("claspy") != "0.2.8"
            or state.get("snap_grid") != 64 or state.get("min_segment") != 64
            or state.get("segments") != len(intervals)
            or state.get("final_boundaries") != [s for s, _ in intervals] + [len(values)]):
        raise ValueError("require real full-input raw-only ClaSS v0.2.8 provenance, not a grid fallback")
    cache = FeatureCache(values, source)
    candidates = build_candidates(cache, intervals, mode="union", grid=64,
                                  minimum=64, window=256, maximum=maximum, random_seed=0)
    return candidates, dict(candidate_mode="union", class_execution=True, class_detector_modified=False,
        class_segments=len(intervals), candidate_count=len(candidates), maximum_candidates=maximum,
        state_window=256, feature_values_scanned=cache.values_scanned, feature_seconds=cache.seconds,
        class_configuration=state.get("requested_configuration"), class_seconds=state.get("segmentation_seconds"),
        no_grid_or_compression_only_fallback=True)


def fixed_partitions(n):
    strict = [(s, min(1024, n - s)) for s in range(0, n, 1024)]
    seed = list(strict)
    # Preserve the ORIGINAL repair's semantic-tail policy; the primary fixed
    # control remains strict1024 with an explicit tail, including a single value.
    if len(seed) > 1 and seed[-1][1] < 64:
        start, length = seed[-2]
        seed[-2:] = [(start, length + seed[-1][1])]
    return strict, seed


def write_verified(path, segments, raw, exporter, btr, timeout, execute):
    n = Path(raw).stat().st_size // 8
    info = bridge.write_mixed(path, [f for seg in segments for f in seg["frames"]], n)
    frozen.decode_exact(path, raw, path.parent / (path.stem + "_decode"), exporter, btr, timeout, execute)
    expected = (FILE_HEADER.size + sum(frozen.frame_cost(s["frames"]) for s in segments)) * 8
    if expected != info["complete_file_bits"]:
        raise ValueError("complete persisted bytes differ from priced physical frames")
    return info


def run(args, execute=replay.execute_export, backend_factory=DiskCosts):
    began = time.perf_counter()
    raw, exporter, btr = (Path(p).resolve() for p in (args.input, args.exporter, args.btr_exe))
    out = Path(args.out_dir).resolve()
    if out.exists() and (not out.is_dir() or any(p.name != "run.log" for p in out.iterdir())):
        raise ValueError("choose a fresh method output; old experiments are never overwritten")
    protected = {str(p): replay.sha256(p) for p in (raw, exporter, btr)}
    for name in CODE_FILES:
        path = Path(__file__).with_name(name).resolve()
        protected[str(path)] = replay.sha256(path)
    for path in map(Path, protected):
        if replay.inside(path, out) or replay.inside(out, path.parent):
            raise ValueError("output overlaps protected raw/code/native files")
    values = read_float64(str(raw))
    if not 1024 <= len(values) <= 102400:
        raise ValueError("require 1024..102400 complete finite float64 values")
    source = (raw.parent.name, raw.stem, protected[str(raw)])
    if (args.dataset is not None and args.dataset != source[0]) or (args.series is not None and args.series != source[1]):
        raise ValueError("explicit identity differs from input parent/stem")
    for name in ("value_budget_multiplier", "trial_limit", "max_rounds", "max_proposals", "max_split_candidates", "max_candidates"):
        if type(getattr(args, name)) is not int or getattr(args, name) < 1:
            raise ValueError(name + " must be a positive integer")
    if isinstance(args.timeout, bool) or not math.isfinite(args.timeout) or args.timeout <= 0:
        raise ValueError("timeout must be positive and finite")
    candidates, candidate_state = class_candidates(args.class_dir, values, source, protected, args.max_candidates)
    for path in map(Path, protected):
        if replay.inside(path, out) or replay.inside(out, path.parent):
            raise ValueError("output overlaps the frozen ClaSS stage")
    out.mkdir(parents=True, exist_ok=True)
    frozen.write_csv(out / "candidates.csv", candidates,
                     ("position", "score", "sources", "strongest_group", "state_contrast"))
    stream_json(out / "candidate_manifest.json", candidate_state)
    config = RepairConfig(max_rounds=args.max_rounds, max_proposals_per_round=args.max_proposals,
                          max_split_candidates=args.max_split_candidates)
    config.validate()
    proxy = CompressionGainProxy(values)  # exact ORIGINAL ranking configuration
    state = dict(version=VERSION, status="running", identity=dict(zip(IDENTITY, source)), values=len(values),
        original_candidate_builder="compression_candidates.build_candidates(mode=union)",
        original_repair="compression_boundary_repair.repair_partition", repair_config=asdict(config),
        candidate_protocol=candidate_state, proxy_config=proxy.config, primary_reference="same_pool_fixed1024",
        no_dp=True, selector_training_used=False, selector_used=False, c1_full_pool_trial_diagnostic=True,
        complete_file_framing=True, original_repair_and_candidates_unchanged=True,
        codec_pool=list(costs.STREAM_CODECS) + ["alp", "btrblocks"],
        total_series_value_budget=args.value_budget_multiplier * len(values), trial_limit=args.trial_limit,
        budget_policy="original shared total; common seeds charged first; remaining work quota divided by remaining seeds",
        fixed1024_tail_policy="strict blocks and explicit complete short tail",
        repair_seed_tail_policy="original: fold final <64 values into preceding semantic region",
        proxy_is_only_ranking_not_acceptance=True, protected_file_sha256=protected,
        command_journal=str(out / "native_commands.jsonl"), command_record_count=0)
    stream_json(out / "method_manifest.json", state)

    def measured(command, log, timeout):
        record = dict(event="start", command=list(map(str, command)), log_path=str(log))
        with (out / "native_commands.jsonl").open("a", encoding="utf-8") as journal:
            journal.write(json.dumps(record) + "\n")
        state["command_record_count"] += 1
        started = time.perf_counter()
        try:
            code = execute(command, Path(log), timeout)
        except BaseException as error:
            result = dict(event="error", log_path=str(log), detail=str(error))
            raise
        else:
            result = dict(event="finish", log_path=str(log), exit_code=code)
            return code
        finally:
            result["wall_seconds"] = time.perf_counter() - started
            with (out / "native_commands.jsonl").open("a", encoding="utf-8") as journal:
                journal.write(json.dumps(result) + "\n")

    try:
        job = dict(n=len(values), raw=raw, key=source + ("class_union_net_gain",))
        service = backend_factory(job, out / "native", exporter, btr, args.timeout,
            value_budget=args.value_budget_multiplier * len(values), trial_limit=args.trial_limit, execute=measured)
        view = BudgetedRegionView(service, args.value_budget_multiplier * len(values), args.trial_limit)
        evaluator = NativeRegionEvaluator(view)
        strict_fixed, seed_fixed = fixed_partitions(len(values))
        seeds = (("fixed1024", seed_fixed), ("whole", [(0, len(values))]))
        initial_union = sorted(set(strict_fixed + seed_fixed + [(0, len(values))]))
        if not evaluator.can_evaluate(initial_union):
            raise ValueError("budget cannot price original complete seeds plus strict fixed1024 control")
        print("ClaSS+feature union frozen: {} candidates, {} ClaSS segments; original four-operation repair".format(
            len(candidates), candidate_state["class_segments"]), flush=True)
        evaluator.evaluate(initial_union, phase="seed")
        controls = {}
        for name, regions in (("fixed1024", strict_fixed), ("whole", [(0, len(values))])):
            controls[name] = write_verified(out / (name + ".f64blk"), evaluator.native_segments(regions),
                                           raw, exporter, btr, args.timeout, measured)
        histories, seed_rows = [], []
        for index, (tag, partition) in enumerate(seeds):
            evaluator.set_seed_quota(evaluator.remaining_edges // (len(seeds) - index),
                                     evaluator.remaining_values // (len(seeds) - index))
            print("start ORIGINAL ClaSS-union repair: " + tag, flush=True)
            result = repair_partition(partition, candidates, evaluator, config=config,
                                      tag=tag, proposal_scorer=proxy.gain)
            info = write_verified(out / (tag + "_repaired.f64blk"), evaluator.native_segments(result["intervals"]),
                                  raw, exporter, btr, args.timeout, measured)
            if info["complete_file_bits"] != result["total_bits"] + FILE_HEADER.size * 8:
                raise ValueError("original repair reported a different cost from actual native file")
            if result["total_bits"] > result["initial_bits"] or not result["telescoping_ok"]:
                raise ValueError("original repair lost monotonic measured-byte guarantee")
            stream_json(out / (tag + "_repair.json"), result)
            seed_rows.append(dict(seed=tag, initial_bits=result["initial_bits"] + FILE_HEADER.size * 8,
                bits=info["complete_file_bits"], boundary_segments=len(result["intervals"]),
                accepted_operations=len(result["operations"]), termination=result["termination"],
                budget_limited=result["budget_limited"], **view.counters))
            histories.append(result)
            state.update(last_completed_seed=tag, counters=dict(view.counters))
            stream_json(out / "method_manifest.json", state)
            print("  {}: {} -> {} bits; accepted={} termination={}".format(tag, seed_rows[-1]["initial_bits"],
                info["complete_file_bits"], len(result["operations"]), result["termination"]), flush=True)
        chosen = min(histories, key=lambda r: (r["total_bits"], len(r["intervals"]), r["tag"]))
        segments = evaluator.native_segments(chosen["intervals"])
        final = write_verified(out / "adaptive.f64blk", segments, raw, exporter, btr, args.timeout, measured)
        fixed = controls["fixed1024"]["complete_file_bits"]
        changed = replay.unchanged(protected)
        if changed:
            raise ValueError("protected inputs changed: " + str(changed))
        row = dict(dataset=source[0], series=source[1], method="class_union_net_gain", status="passed", values=len(values),
            fixed1024_bits=fixed, adaptive_bits=final["complete_file_bits"],
            gain_vs_fixed1024_pct=100 * (fixed - final["complete_file_bits"]) / fixed,
            whole_bits=controls["whole"]["complete_file_bits"], chosen_seed=chosen["tag"],
            boundary_segments=len(chosen["intervals"]), accepted_operations=len(chosen["operations"]),
            **{kind + "_count": sum(o["kind"] == kind for o in chosen["operations"])
               for kind in ("split", "merge", "move", "pair_split")},
            class_segments=candidate_state["class_segments"], candidate_count=len(candidates),
            measured_unique_edge_value_volume=view.counters["trial_value_volume"],
            native_trial_failures=service.counters["failed_trials"], complete_file_roundtrip_verified=True,
            c1_full_pool_trial_diagnostic=True, selector_used=False, dp_used=False,
            adaptive_path=str(out / "adaptive.f64blk"), adaptive_sha256=replay.sha256(out / "adaptive.f64blk"))
        frozen.write_csv(out / "seed_results.csv", seed_rows, list(seed_rows[0]))
        frozen.write_csv(out / "frozen_boundaries.csv", [dict(start=s, length=n) for s, n in chosen["intervals"]], ("start", "length"))
        stream_json(out / "native_trials.json", service.trials)
        state.update(status="passed", protected_files_unchanged=True, result=row, wall_seconds=time.perf_counter() - began,
                     native_counters=dict(service.counters), command_journal_sha256=replay.sha256(out / "native_commands.jsonl"))
        stream_json(out / "method_manifest.json", state)
        frozen.write_csv(out / "comparison.csv", [row], list(row))
        print("{}: fixed1024={} adaptive={} gain={:.3f}%".format(source[1], fixed, row["adaptive_bits"], row["gain_vs_fixed1024_pct"]), flush=True)
        return 0
    except BaseException as error:
        state.update(status="failed", detail=str(error), wall_seconds=time.perf_counter() - began)
        stream_json(out / "method_manifest.json", state)
        raise


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for key in ("input", "class-dir", "out-dir", "exporter", "btr-exe"):
        parser.add_argument("--" + key, required=True)
    parser.add_argument("--dataset")
    parser.add_argument("--series")
    parser.add_argument("--value-budget-multiplier", type=int, default=128)
    parser.add_argument("--trial-limit", type=int, default=8192)
    parser.add_argument("--max-rounds", type=int, default=12)
    parser.add_argument("--max-proposals", type=int, default=64)
    parser.add_argument("--max-split-candidates", type=int, default=6)
    parser.add_argument("--max-candidates", type=int, default=256)
    parser.add_argument("--timeout", type=float, default=300)
    args = parser.parse_args()
    try:
        return run(args)
    except (ValueError, OSError, KeyError, TypeError) as error:
        parser.exit(2, "class_mainline_btr: " + str(error) + "\n")


if __name__ == "__main__":
    raise SystemExit(main())
