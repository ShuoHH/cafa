"""Measured BtrBlocks extension of already persisted, frozen codec plans.

This is a fixed-boundary codec-pool diagnostic, not a prediction-only method or
a new segmentation result. Each logical interval retains its frozen old codec
or the exact BtrBlocks bytes actually measured for that interval. The original
native BtrBlocks file and original plan file are explicit final fallbacks.
No boundary search, selector training, DP, or historical best-arm selection is
performed. Actual trial/decode work, including unsuccessful trials, is logged.
"""

import argparse
import csv
import json
import math
from pathlib import Path
import shutil
import time

import btr_mixed_file as bridge
import external_file_format
import replay_frozen_files as replay
from external_file_format import FILE_HEADER, FRAME_HEADER, inspect_container


DEFAULT_SERIES = ("202005_SPlite2", "geolife_074_lon", "household_global_active_power_7day",
                  "BT", "BW", "BM", "Kelmarsh_1_power")
DEFAULT_METHODS = ("tsec__k2", "tsec+decimal+residue__k2")
SUMMARY_FIELDS = ("dataset", "series", "method", "status", "detail", "values", "source_sha256",
                  "original_bits", "btr_original_bits", "frozen_btr_only_bits", "frozen_mixed_bits",
                  "guarded_bits", "gain_btr_only_vs_original_pct", "gain_mixed_vs_original_pct",
                  "gain_mixed_vs_btr_pct", "gain_guarded_vs_btr_pct", "selected_output_arm",
                  "logical_segments", "selected_btr_segments", "physical_frames", "btr_trial_failures",
                  "unique_encoded_values_this_job", "unique_trial_count_this_job", "wall_seconds",
                  "original_plan_path", "original_plan_sha256", "original_container_path",
                  "original_container_sha256", "btr_original_path", "btr_original_sha256",
                  "frozen_btr_only_path", "frozen_btr_only_sha256", "frozen_mixed_path",
                  "frozen_mixed_sha256", "guarded_path", "guarded_sha256", "segment_decisions_path",
                  "complete_file_roundtrip_verified", "boundary_search", "selector_training",
                  "codec_trial_selection", "fixed_boundary_measured_trial_diagnostic")
SIZE_FIELDS = tuple(name for name in SUMMARY_FIELDS if name.endswith("_bits") or name.startswith("gain_"))
SEGMENT_FIELDS = ("dataset", "series", "method", "start", "length", "frozen_codec", "old_frame_bits",
                  "btr_frame_bits", "btr_physical_frames", "trial_status", "trial_detail", "trial_cached",
                  "chosen", "chosen_frame_bits", "candidate_path", "candidate_sha256")


def read_json(path):
    value = json.loads(Path(path).read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError("JSON manifest/report must be an object: " + str(path))
    return value


def write_json(path, value):
    Path(path).write_text(json.dumps(value, indent=2), encoding="utf-8")


def write_csv(path, rows, fields):
    with Path(path).open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def exact_bytes(first, second):
    """Compare persisted bytes (including signed zero and NaN bit patterns)."""
    first, second = Path(first), Path(second)
    if first.stat().st_size != second.stat().st_size:
        return False
    with first.open("rb") as left, second.open("rb") as right:
        while True:
            chunk = left.read(1024 * 1024)
            if chunk != right.read(1024 * 1024):
                return False
            if not chunk:
                return True


def frame_cost(frames):
    return sum(FRAME_HEADER.size + len(frame["payload"]) for frame in frames)


def choose_interval(old_frame, candidate_frames):
    """Keep the frozen old bytes on ties; charge every physical frame header."""
    old = frame_cost([old_frame])
    if candidate_frames is None or frame_cost(candidate_frames) >= old:
        return "original", [dict(old_frame)]
    return "btrblocks", [dict(frame, codec_id=25) for frame in candidate_frames]


def choose_guard(original, mixed, baseline):
    """Actual files, with stable preference for original, mixed, then fallback."""
    alternatives = [("original", Path(original)), ("frozen_original_or_btr", Path(mixed)),
                    ("original_btrblocks_fallback", Path(baseline))]
    return min(alternatives, key=lambda pair: pair[1].stat().st_size)


def assert_hash(path, expected, label):
    expected = replay.digest_string(expected, label)
    actual = replay.sha256(path)
    if actual != expected:
        raise ValueError(label + " changed or disagrees with provenance: " + str(path))
    return actual


def source_jobs(args):
    """Validate every selected historical artifact before any new trial runs."""
    night = Path(args.source_night_dir).resolve()
    external, batch, inputs = replay.load_batch(night / "external")
    replay_root = night / "replay"
    replay_path = replay_root / "replay_manifest.json"
    state = read_json(replay_path)
    if (state.get("version") != "replay_frozen_files_v1" or state.get("batch_finished") is not True
            or state.get("protected_files_unchanged") is not True
            or Path(state.get("external_batch", "")).resolve() != external
            or not isinstance(state.get("results"), list)):
        raise ValueError("require a finished, unchanged frozen replay matching this external batch")
    if (not isinstance(state.get("protected_file_sha256_before"), dict)
            or any(not isinstance(row, dict) for row in state["results"])
            or any(not isinstance(row, dict) for row in batch["results"])):
        raise ValueError("historical provenance/results must contain structured file hashes and result objects")
    protected = {}
    for path, expected in state.get("protected_file_sha256_before", {}).items():
        resolved = Path(path).resolve()
        assert_hash(resolved, expected, "replay protected artifact")
        protected[str(resolved)] = expected
    if not protected:
        raise ValueError("replay protected-file provenance is absent")
    if state.get("changed_protected_files"):
        raise ValueError("historical replay records changed protected artifacts")
    night_marker = night / "overnight_manifest.json"
    if night_marker.is_file():
        marker = read_json(night_marker)
        if marker.get("finished") is not True or marker.get("protected_files_unchanged") is not True:
            raise ValueError("night batch is unfinished or records changed source files")
        if not isinstance(marker.get("selected_original_sha256", {}), dict):
            raise ValueError("night selected source hashes must be an object")
        for path, expected in marker.get("selected_original_sha256", {}).items():
            resolved = Path(path).resolve()
            assert_hash(resolved, expected, "night selected input/plan")
            protected[str(resolved)] = expected
    methods, series = list(args.methods), list(args.series)
    if not methods or len(set(methods)) != len(methods) or any(name not in DEFAULT_METHODS for name in methods):
        raise ValueError("select only the predeclared tsec__k2 / tsec+decimal+residue__k2 arms")
    if not series or len(set(series)) != len(series):
        raise ValueError("series must be nonempty and unique")
    exporter = Path(args.exporter or state.get("exporter", "")).resolve()
    repo = Path(args.repo).resolve()
    btr = Path(args.btr_exe or repo / "experiment_results/external_native_build_cache_v7_files/btrblocks/BtrBlocksSmoke").resolve()
    if not exporter.is_file() or not btr.is_file():
        raise ValueError("existing FrozenPlanFile exporter and BtrBlocksSmoke executable are required; no build is performed")
    jobs = []
    for name in series:
        for method in methods:
            matches = [row for row in state["results"] if row.get("series") == name and row.get("method") == method]
            if len(matches) != 1:
                raise ValueError("need exactly one persisted source replay for " + name + "/" + method)
            row = matches[0]
            if row.get("status") != "passed" or row.get("complete_file_roundtrip_verified") is not True:
                raise ValueError("source replay did not pass: " + name + "/" + method)
            key = tuple(row.get(field) for field in ("dataset", "series", "source_sha256", "method"))
            plan = Path(row["original_plan_path"]).resolve()
            plan_hash = assert_hash(plan, row.get("original_plan_sha256"), "original frozen plan hash")
            groups = [(group_key, rows) for group_key, rows in replay.load_groups(plan) if group_key == key]
            if len(groups) != 1:
                raise ValueError("persisted replay identity differs from current frozen plan")
            ordered, n = replay.validate_plan(*groups[0])
            if replay.integer(row.get("n"), "source values") != n:
                raise ValueError("source replay count differs from frozen plan")
            raw = replay.matching_raw(key, n, batch, inputs)
            if Path(row.get("input_path", "")).resolve() != raw:
                raise ValueError("replay input path differs from external full raw input")
            paths = {}
            for path_key, hash_key in (("container_path", "container_sha256"), ("report_path", "report_sha256"),
                                        ("filtered_plan_path", "filtered_plan_sha256"), ("log_path", "log_sha256")):
                path = Path(row[path_key]).resolve()
                if not replay.inside(path, replay_root):
                    raise ValueError("persisted replay artifact escapes its replay directory")
                assert_hash(path, row.get(hash_key), "persisted replay " + path_key)
                paths[path_key] = path
                protected[str(path)] = row[hash_key]
            filtered_groups = replay.load_groups(paths["filtered_plan_path"])
            if len(filtered_groups) != 1 or filtered_groups[0][0] != key:
                raise ValueError("persisted filtered plan identity differs")
            filtered, _ = replay.validate_plan(*filtered_groups[0])
            canonical = lambda rows: [{k: str(r[k]) for k in replay.PLAN_FIELDS} for r in rows]
            if canonical(filtered) != canonical(ordered):
                raise ValueError("persisted filtered plan changed a frozen choice")
            old_info, old_hash = replay.validate_export(paths["report_path"], paths["container_path"], raw, key, ordered, n)
            replay.check_metrics(row, old_info)
            # Restrict the shared validator to BtrBlocks only; failed Camel/NeaTS
            # rows do not invalidate an otherwise complete historical batch.
            btr_batch = dict(batch, results=[r for r in batch["results"] if r.get("method") == "btrblocks"])
            baseline, rejected = replay.strongest_baseline(key, raw, n, external, btr_batch)
            eligible = [r for r in btr_batch["results"] if r.get("series") == name
                        and r.get("sha256") == key[2] and r.get("input") and Path(r["input"]).resolve() == raw]
            if baseline is None or len(eligible) != 1 or rejected:
                raise ValueError("need exactly one verified original BtrBlocks complete file for " + name)
            btr_info = inspect_container(baseline["path"])
            if btr_info["block_size"] != 64000:
                raise ValueError("original BtrBlocks must use the matched native 64000-point configuration")
            strongest, _ = replay.strongest_baseline(key, raw, n, external, batch)
            job = dict(key=key, rows=ordered, raw=raw, n=n, original=paths["container_path"],
                       original_hash=old_hash, original_info=old_info, plan=plan, plan_hash=plan_hash,
                       baseline=Path(baseline["path"]), baseline_hash=baseline["sha256"],
                       baseline_info=btr_info, strongest=strongest)
            jobs.append(job)
            protected[str(plan)] = plan_hash
    # Include every source batch/replay artifact and all inherited protected
    # paths. Do not place output beneath historical directories or raw data.
    for directory in (external, replay_root):
        for path in directory.rglob("*"):
            if path.is_file():
                protected[str(path.resolve())] = replay.sha256(path)
    for path in (replay_path, night_marker, exporter, btr, Path(__file__), Path(bridge.__file__),
                 Path(replay.__file__), Path(external_file_format.__file__)):
        if path.is_file():
            protected[str(path.resolve())] = replay.sha256(path)
    for path, identity in inputs.items():
        protected[str(path)] = identity["sha256"]
    return night, exporter, btr, jobs, protected


def decode_exact(container, expected, folder, exporter, btr, timeout, execute):
    """Standalone decode first; only the caller then compares original bytes."""
    output = folder.parent / (folder.name + ".raw")
    result = bridge.decode_file(container, output, folder, exporter, btr, timeout=timeout, execute=execute)
    if (result.get("status") != "decoded" or result.get("decoded_from_persisted_file_only") is not True
            or replay.integer(result.get("decoded_values"), "decoded values") != Path(expected).stat().st_size // 8
            or not exact_bytes(output, expected)):
        raise ValueError("standalone persisted-file decode is not bitwise equal: " + str(container))
    return dict(result, exact_source_comparison_passed=True, decoded_sha256=replay.sha256(output))


def parse_native_json(log):
    records = []
    for line in Path(log).read_text(encoding="utf-8", errors="replace").splitlines():
        try:
            item = json.loads(line)
            if isinstance(item, dict) and item.get("method") == "btrblocks":
                records.append(item)
        except (TypeError, ValueError):
            pass
    if len(records) != 1:
        raise ValueError("BtrBlocks must emit exactly one JSON result")
    return records[0]


def encode_trial(job, start, length, cache_dir, btr, exporter, exe_hash, timeout, execute):
    key = (job["key"][2], start, length, 64000, exe_hash)
    folder = cache_dir / ("{:06d}".format(len(list(cache_dir.iterdir()))))
    folder.mkdir()
    sliced, container, log = folder / "slice.double_bin", folder / "candidate.f64blk", folder / "encode.log"
    with job["raw"].open("rb") as stream:
        stream.seek(start * 8)
        data = stream.read(length * 8)
    if len(data) != length * 8:
        raise ValueError("raw interval no longer covers the frozen logical segment")
    with sliced.open("xb") as stream:
        stream.write(data)
    trial = dict(cache_key=key, source_sha256=key[0], start=start, length=length, block_size=64000,
                 executable_sha256=exe_hash, slice_path=str(sliced), slice_sha256=replay.sha256(sliced),
                 candidate_path=str(container), status="unavailable", detail="", encoded_values=length)
    command = [btr, sliced, "64000", "0", container]
    try:
        code = execute(command, log, timeout)
        trial["exit_code"] = code
        if code != 0:
            raise ValueError("native BtrBlocks trial exited " + str(code))
        result = parse_native_json(log)
        for name in ("values", "decoded_values"):
            if replay.integer(result.get(name), name) != length:
                raise ValueError("native BtrBlocks trial does not cover full interval")
        if (replay.integer(result.get("mismatch_count"), "mismatch count") != 0
                or result.get("strict_bitwise_equal") is not True
                or result.get("complete_file_roundtrip_verified") is not True):
            raise ValueError("native BtrBlocks trial failed its bitwise persisted-file round trip")
        info, frames = bridge.read_frames(container)
        if info["method_id"] != 4 or info["block_size"] != 64000 or info["values"] != length:
            raise ValueError("BtrBlocks trial container changed identity/block configuration/count")
        replay.check_metrics(result, info)
        trial["decode_report"] = decode_exact(container, sliced, folder / "decode", exporter, btr, timeout, execute)
        trial.update(status="passed", frames=frames, candidate_sha256=replay.sha256(container),
                     frame_bytes=frame_cost(frames), physical_frames=len(frames))
    except (OSError, ValueError, KeyError, TypeError) as error:
        trial.update(status="unavailable", detail=str(error))
    if replay.sha256(sliced) != trial["slice_sha256"]:
        raise ValueError("native trial changed its frozen input slice")
    write_json(folder / "trial_manifest.json", {k: v for k, v in trial.items() if k != "frames"})
    return key, trial


def run(args, execute=replay.execute_export):
    began = time.perf_counter()
    if not math.isfinite(args.timeout) or args.timeout <= 0:
        raise ValueError("timeout must be finite and positive")
    night, exporter, btr, jobs, protected = source_jobs(args)
    out = Path(args.out_dir).resolve()
    protected_paths = [Path(name) for name in protected]
    if (replay.inside(out, night) or replay.inside(night, out)
            or any(replay.inside(path, out) or replay.inside(out, path.parent) for path in protected_paths)):
        raise ValueError("out-dir cannot overlap any source/history/raw/tool artifact directory")
    if out.exists() and (not out.is_dir() or any(path.name != "run.log" or not path.is_file() for path in out.iterdir())):
        raise ValueError("out-dir must be fresh (only the wrapper's run.log may already exist)")
    out.mkdir(parents=True, exist_ok=True)
    cache_dir, group_dir = out / "trial_cache", out / "groups"
    cache_dir.mkdir()
    group_dir.mkdir()
    results, commands, cache, trials, decode_cache = [], [], {}, [], {}
    state = dict(version="frozen_btr_extension_v1", batch_finished=False, source_night_dir=str(night),
                 methods=list(args.methods), series=list(args.series), boundary_search=False,
                 selector_training=False, dp_used=False, codec_trial_selection=True,
                 fixed_boundary_measured_trial_diagnostic=True, end_to_end_prediction_only=False,
                 codec_choice_scope="exact_choice_between_frozen_original_and_btr",
                 original_arm_selection="explicit method list fixed before trials; never by source bits/gain",
                 native_btr_block_size=64000, btr_sampling_configuration_unchanged=True,
                 priced_payloads_reencoded=False, final_fallbacks=["original", "frozen_original_or_btr", "original_btrblocks"],
                 wall_time_excludes_historical_encoding=True, not_end_to_end_compression_throughput_benchmark=True,
                 original_btr_file_reused_without_reencoding=True,
                 trial_value_volume_scope="unique interval input values submitted/attempted, including failed trials; not observed completed native encodes",
                 no_new_segmentation_contribution_claimed=True, exporter=str(exporter), btr_exe=str(btr),
                 protected_file_sha256_before=protected, results=results, executed_commands=commands)
    frozen = []
    for job in jobs:
        frozen.append(dict(dataset=job["key"][0], series=job["key"][1], method=job["key"][3],
                           source_sha256=job["key"][2], input=str(job["raw"]), values=job["n"],
                           plan_path=str(job["plan"]), plan_sha256=job["plan_hash"], rows=job["rows"],
                           original_container_path=str(job["original"]), original_container_sha256=job["original_hash"],
                           btr_original_path=str(job["baseline"]), btr_original_sha256=job["baseline_hash"]))
    write_json(out / "frozen_inputs.json", dict(frozen_before_first_trial=True, jobs=frozen))

    def measured_execute(command, log, timeout):
        item = dict(command=[str(value) for value in command], log_path=str(log), started_seconds=time.perf_counter() - began)
        commands.append(item)
        start = time.perf_counter()
        try:
            code = execute(command, Path(log), timeout)
            item["exit_code"] = code
            return code
        finally:
            item["wall_seconds"] = time.perf_counter() - start

    def checkpoint(finished=False, changed=None):
        if changed:
            for row in results:
                row.update(status="protected_input_changed", detail="Run invalidated: " + ", ".join(changed),
                           complete_file_roundtrip_verified=False)
                for field in SIZE_FIELDS:
                    row.pop(field, None)
        state.update(batch_finished=finished, protected_files_unchanged=not changed,
                     changed_protected_files=changed or [], wall_seconds=time.perf_counter() - began,
                     unique_trial_count=len(trials), unique_encoded_value_volume=sum(t["encoded_values"] for t in trials),
                     all_checks_passed=finished and not changed and bool(results)
                     and all(row.get("status") in ("passed", "passed_with_unavailable_trials") for row in results))
        pending = out / "summary.csv.partial"
        write_csv(pending, results, SUMMARY_FIELDS)
        pending.replace(out / "summary.csv")
        for filename, content in (("extension_manifest.json", state), ("executed_commands.json", commands)):
            pending = out / (filename + ".partial")
            write_json(pending, content)
            pending.replace(out / filename)

    checkpoint()
    for number, job in enumerate(jobs, 1):
        started, initial_trials = time.perf_counter(), len(trials)
        key = job["key"]
        row = dict(dataset=key[0], series=key[1], method=key[3], values=job["n"], source_sha256=key[2],
                   status="pending", detail="", boundary_search=False, selector_training=False,
                   codec_trial_selection=True, fixed_boundary_measured_trial_diagnostic=True,
                   complete_file_roundtrip_verified=False, original_plan_path=str(job["plan"]),
                   original_plan_sha256=job["plan_hash"], original_container_path=str(job["original"]),
                   original_container_sha256=job["original_hash"], btr_original_path=str(job["baseline"]),
                   btr_original_sha256=job["baseline_hash"])
        results.append(row)
        folder = group_dir / ("{:04d}".format(number))
        folder.mkdir()
        print("===== {}/{}: {} / {} (fixed boundaries) =====".format(number, len(jobs), key[1], key[3]), flush=True)
        checkpoint()
        try:
            changes = replay.unchanged(protected)
            if changes:
                raise ValueError("protected inputs changed before codec trials")
            for path, tag in ((job["original"], "original"), (job["baseline"], "btr_original")):
                token = (str(path), replay.sha256(path), key[2])
                if token not in decode_cache:
                    decode_cache[token] = decode_exact(path, job["raw"], folder / ("decode_" + tag), exporter, btr, args.timeout, measured_execute)
            original_info, old_frames = bridge.read_frames(job["original"])
            if len(job["rows"]) != len(old_frames):
                raise ValueError("original physical frames differ from frozen logical segment count")
            mixed_frames, btr_frames, decisions, failures, selected = [], [], [], 0, 0
            for segment_number, (plan_row, old_frame) in enumerate(zip(job["rows"], old_frames), 1):
                start, length = int(plan_row["start"]), int(plan_row["length"])
                if old_frame["start"] != start or old_frame["count"] != length or old_frame["codec_id"] != replay.CODEC_IDS[plan_row["codec"]]:
                    raise ValueError("persisted old frame differs from frozen logical interval")
                token = (key[2], start, length, 64000, protected[str(btr)])
                was_cached = token in cache
                if not was_cached:
                    token, trial = encode_trial(job, start, length, cache_dir, btr, exporter, protected[str(btr)], args.timeout, measured_execute)
                    cache[token] = trial
                    trials.append(trial)
                trial = cache[token]
                candidate = trial.get("frames") if trial["status"] == "passed" else None
                if candidate is None:
                    failures += 1
                else:
                    assert_hash(trial["candidate_path"], trial["candidate_sha256"], "cached priced BtrBlocks payload")
                    btr_frames.extend(dict(frame, codec_id=25, start=start + frame["start"]) for frame in candidate)
                chosen, frames = choose_interval(old_frame, candidate)
                selected += chosen == "btrblocks"
                if chosen == "btrblocks":
                    mixed_frames.extend(dict(frame, start=start + frame["start"]) for frame in frames)
                else:
                    mixed_frames.extend(frames)
                decisions.append(dict(dataset=key[0], series=key[1], method=key[3], start=start, length=length,
                                      frozen_codec=plan_row["codec"], old_frame_bits=frame_cost([old_frame]) * 8,
                                      btr_frame_bits=frame_cost(candidate) * 8 if candidate is not None else "",
                                      btr_physical_frames=len(candidate) if candidate is not None else "",
                                      trial_status=trial["status"], trial_detail=trial["detail"], trial_cached=was_cached,
                                      chosen=chosen, chosen_frame_bits=frame_cost(frames) * 8,
                                      candidate_path=trial["candidate_path"], candidate_sha256=trial.get("candidate_sha256", "")))
                if segment_number == 1 or segment_number % 10 == 0 or segment_number == len(job["rows"]):
                    print("  intervals {}/{}; Btr selected={}, unavailable={}".format(
                        segment_number, len(job["rows"]), selected, failures), flush=True)
            decisions_path = folder / "segment_decisions.csv"
            write_csv(decisions_path, decisions, SEGMENT_FIELDS)
            row.update(segment_decisions_path=str(decisions_path), logical_segments=len(job["rows"]),
                       selected_btr_segments=selected, btr_trial_failures=failures,
                       original_bits=original_info["complete_file_bits"], btr_original_bits=job["baseline_info"]["complete_file_bits"])
            mixed = folder / "frozen_original_or_btr.f64blk"
            mixed_info = bridge.write_mixed(mixed, mixed_frames, job["n"])
            if mixed_info["complete_file_bytes"] > original_info["complete_file_bytes"]:
                raise ValueError("fixed-interval measured-choice size guarantee failed")
            row["mixed_decode_report"] = decode_exact(mixed, job["raw"], folder / "decode_mixed", exporter, btr, args.timeout, measured_execute)
            row.update(frozen_mixed_path=str(mixed), frozen_mixed_sha256=replay.sha256(mixed), frozen_mixed_bits=mixed_info["complete_file_bits"])
            if not failures:
                only = folder / "frozen_btr_only.f64blk"
                only_info = bridge.write_mixed(only, btr_frames, job["n"])
                row["btr_only_decode_report"] = decode_exact(only, job["raw"], folder / "decode_btr_only", exporter, btr, args.timeout, measured_execute)
                if mixed_info["complete_file_bytes"] > only_info["complete_file_bytes"]:
                    raise ValueError("mixed measured-choice size exceeds fixed BtrBlocks-only arm")
                row.update(frozen_btr_only_path=str(only), frozen_btr_only_sha256=replay.sha256(only), frozen_btr_only_bits=only_info["complete_file_bits"],
                           gain_btr_only_vs_original_pct=100 * (1 - only_info["complete_file_bits"] / row["original_bits"]))
            selected_arm, selected_path = choose_guard(job["original"], mixed, job["baseline"])
            guarded = folder / "chosen.f64blk"
            with selected_path.open("rb") as source, guarded.open("xb") as target:
                shutil.copyfileobj(source, target)
            if replay.sha256(guarded) != replay.sha256(selected_path):
                raise ValueError("final guard did not copy selected persisted bytes unchanged")
            row["guard_decode_report"] = decode_exact(guarded, job["raw"], folder / "decode_guarded", exporter, btr, args.timeout, measured_execute)
            guard_info, _ = bridge.read_frames(guarded)
            if guard_info["complete_file_bits"] > min(row["original_bits"], row["btr_original_bits"], row["frozen_mixed_bits"]):
                raise ValueError("actual-file fallback guarantee failed")
            row.update(status="passed_with_unavailable_trials" if failures else "passed",
                       detail="Some BtrBlocks interval trials were unavailable; old bytes retained; Btr-only size not published." if failures else "",
                       guarded_path=str(guarded), guarded_sha256=replay.sha256(guarded), guarded_bits=guard_info["complete_file_bits"],
                       selected_output_arm=selected_arm, physical_frames=guard_info["frames"], complete_file_roundtrip_verified=True,
                       gain_mixed_vs_original_pct=100 * (1 - row["frozen_mixed_bits"] / row["original_bits"]),
                       gain_mixed_vs_btr_pct=100 * (1 - row["frozen_mixed_bits"] / row["btr_original_bits"]),
                       gain_guarded_vs_btr_pct=100 * (1 - guard_info["complete_file_bits"] / row["btr_original_bits"]))
        except (OSError, ValueError, KeyError, TypeError, csv.Error) as error:
            row.update(status="failed", detail=str(error), complete_file_roundtrip_verified=False)
            for field in SIZE_FIELDS:
                row.pop(field, None)
        finally:
            row.update(wall_seconds=time.perf_counter() - started, unique_trial_count_this_job=len(trials) - initial_trials,
                       unique_encoded_values_this_job=sum(t["encoded_values"] for t in trials[initial_trials:]))
            changed = replay.unchanged(protected)
            checkpoint(changed=changed)
            print("{} / {}: {}{}".format(key[1], key[3], row["status"], ": " + row["detail"] if row["detail"] else ""), flush=True)
        if changed:
            break
    changed = replay.unchanged(protected)
    state["protected_file_sha256_after"] = {name: replay.sha256(name) if Path(name).is_file() else None for name in protected}
    state["trial_failures"] = [{k: v for k, v in t.items() if k != "frames"} for t in trials if t["status"] != "passed"]
    checkpoint(finished=True, changed=changed)
    print("Fixed-boundary diagnostic summary: " + str(out / "summary.csv"), flush=True)
    return 0 if state["all_checks_passed"] else 1


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-night-dir", required=True)
    parser.add_argument("--out-dir", required=True)
    parser.add_argument("--repo", default=str(Path(__file__).resolve().parents[1]))
    parser.add_argument("--series", nargs="+", default=list(DEFAULT_SERIES))
    parser.add_argument("--methods", nargs="+", default=list(DEFAULT_METHODS))
    parser.add_argument("--exporter")
    parser.add_argument("--btr-exe")
    parser.add_argument("--timeout", type=float, default=300)
    args = parser.parse_args()
    try:
        return run(args)
    except (OSError, ValueError, KeyError, TypeError, csv.Error) as error:
        parser.exit(2, "frozen_btr_extension: " + str(error) + "\n")


if __name__ == "__main__":
    raise SystemExit(main())
