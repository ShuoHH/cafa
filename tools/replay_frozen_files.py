"""Replay existing, full-series frozen plans into actual F64BLK01 files.

This runner does not optimize, search, train, or transfer a historical payload
objective guarantee to complete-file sizes. Comparisons are restricted to the
strongest verified full-series Elf/Camel/BtrBlocks/exact-decimal-NeaTS file in
one finished complete-file batch. Only available, passed artifacts are eligible.
"""

import argparse
import csv
import hashlib
import json
import math
import os
from pathlib import Path
import re
import signal
import subprocess
import time

from external_file_format import FILE_HEADER, FRAME_HEADER, inspect_container


IDENTITY = ("dataset", "series", "sha256", "method")
PLAN_FIELDS = IDENTITY + ("start", "length", "codec")
CODEC_IDS = dict(zip(("gorilla", "chimp", "chimp128", "tsxor", "rle+gorilla",
                     "rle+chimp", "rle+chimp128", "rle+tsxor", "alp"), range(16, 25)))
METRICS = ("complete_file_bytes", "complete_file_bits", "codec_storage_bytes", "framing_bytes")
BASELINE_METHOD_IDS = {"elf": 1, "camel": 2, "btrblocks": 4, "neats": 5,
                       "neats_exact_decimal": 5}
SCOPE = ("strongest verified available full-series complete F64BLK01 file within "
         "Elf/Camel/BtrBlocks/exact-decimal-NeaTS; failed/unsupported/unbuilt methods excluded")
FIELDS = ("dataset", "series", "source_sha256", "n", "method", "status", "detail",
          "original_plan_path", "original_plan_sha256", "input_path", "old_payload32_bits",
          "complete_file_bits", "complete_file_bytes", "codec_storage_bytes", "framing_bytes",
          "blocks", "baseline_method", "baseline_complete_file_bits", "gain_vs_baseline",
          "gain_percent_vs_baseline", "baseline_scope", "baseline_container_path",
          "baseline_container_sha256", "filtered_plan_path", "filtered_plan_sha256",
          "container_path", "container_sha256", "report_path", "report_sha256", "log_path",
          "log_sha256", "complete_file_roundtrip_verified", "replayed_frozen_payload_objective",
          "historical_payload_guarantee_transferred", "search_or_training_performed")


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for data in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(data)
    return digest.hexdigest()


def integer(value, name):
    """No truncation of floats, booleans, signed strings or scientific notation."""
    if isinstance(value, bool) or not re.fullmatch(r"[0-9]+", str(value)):
        raise ValueError(name + " must be an unsigned integer")
    return int(value)


def digest_string(value, name):
    if not isinstance(value, str) or not re.fullmatch(r"[0-9a-fA-F]{64}", value):
        raise ValueError(name + " must be a SHA-256 hex digest")
    return value.lower()


def inside(path, directory):
    return path == directory or directory in path.parents


def discover_plans(results_root):
    """Only authoritative paths, never recursive diagnostic/cache CSV matches."""
    root = Path(results_root).resolve()
    if not root.is_dir():
        raise ValueError("results root does not exist: " + str(root))
    patterns = ("frozen_plans.csv", "frozen_plan.csv", "*/boundary/method/frozen_plans.csv",
                "*/selector/*/frozen_plan.csv", "boundary/method/frozen_plans.csv",
                "selector/*/frozen_plan.csv")
    return sorted({p.resolve() for pattern in patterns for p in root.glob(pattern) if p.is_file()})


def load_groups(path):
    with path.open(newline="", encoding="utf-8-sig") as stream:
        reader = csv.DictReader(stream)
        if not set(PLAN_FIELDS) <= set(reader.fieldnames or ()):
            raise ValueError("frozen plan is missing required columns: " + ", ".join(PLAN_FIELDS))
        groups = {}
        for number, row in enumerate(reader, 2):
            if None in row or any(row.get(name) is None for name in PLAN_FIELDS):
                raise ValueError("malformed CSV row " + str(number))
            key = tuple(row[name] for name in IDENTITY)
            groups.setdefault(key, []).append(dict(row))
        if not groups:
            raise ValueError("frozen plan has no rows")
        return list(groups.items())


def validate_plan(key, rows):
    if any(not isinstance(value, str) or not value or value != value.strip() for value in key):
        raise ValueError("dataset, series, sha256 and method must be nonempty exact identifiers")
    digest_string(key[2], "plan sha256")
    ordered = sorted(rows, key=lambda row: integer(row["start"], "start"))
    cursor = 0
    for row in ordered:
        start, length = integer(row["start"], "start"), integer(row["length"], "length")
        if start != cursor or length == 0:
            raise ValueError("plan must have positive contiguous lengths starting at zero")
        if row["codec"] not in CODEC_IDS:
            raise ValueError("unsupported frozen codec: " + row["codec"])
        cursor += length
    return ordered, cursor


def load_batch(directory):
    root = Path(directory).resolve()
    manifest_path = root / "batch_manifest.json"
    batch = json.loads(manifest_path.read_text(encoding="utf-8"))
    if not isinstance(batch, dict):
        raise ValueError("external manifest must be a JSON object")
    if batch.get("version") != "external_baseline_batch_v5":
        raise ValueError("require actual complete-file external_baseline_batch_v5; v4 is not accepted")
    if batch.get("batch_finished") is not True or batch.get("inputs_unchanged") is not True:
        raise ValueError("external batch must be finished with unchanged inputs")
    if integer(batch.get("max_values"), "external max_values") != 0:
        raise ValueError("prefix external reports are not accepted; require max_values=0")
    if not isinstance(batch.get("results"), list) or not isinstance(batch.get("input_sha256"), dict):
        raise ValueError("external batch must contain results and input_sha256")
    inputs = {}
    for name, expected in batch["input_sha256"].items():
        raw = Path(name).resolve()
        digest = digest_string(expected, "external input hash")
        if not raw.is_file() or raw.stat().st_size == 0 or raw.stat().st_size % 8:
            raise ValueError("external raw input is absent or not full float64 data: " + str(raw))
        if sha256(raw) != digest:
            raise ValueError("external raw input changed: " + str(raw))
        inputs[raw] = dict(sha256=digest, n=raw.stat().st_size // 8)
    if not inputs:
        raise ValueError("external batch has no raw inputs")
    return root, batch, inputs


def matching_raw(key, n, batch, inputs):
    digest, series = key[2].lower(), key[1]
    matches = set()
    for row in batch["results"]:
        if not isinstance(row, dict) or row.get("series") != series or row.get("sha256") != digest:
            continue
        if not row.get("input"):
            continue
        raw = Path(row["input"]).resolve()
        if raw in inputs and inputs[raw] == dict(sha256=digest, n=n):
            matches.add(raw)
    if len(matches) != 1:
        raise ValueError("plan must match exactly one external full raw path + SHA-256 + series + count")
    return next(iter(matches))


def check_metrics(report, info):
    for name in METRICS + ("blocks",):
        if integer(report.get(name), name) != info[name]:
            raise ValueError(name + " disagrees with the actual persisted container")


def strongest_baseline(key, raw, n, root, batch):
    candidates, rejected = [], []
    for index, row in enumerate(batch["results"]):
        if (not isinstance(row, dict) or row.get("method") not in BASELINE_METHOD_IDS
                or row.get("series") != key[1] or row.get("sha256") != key[2].lower()
                or not row.get("input") or Path(row["input"]).resolve() != raw):
            continue
        try:
            if row.get("status") != "passed" or row.get("complete_file_roundtrip_verified") is not True:
                raise ValueError("not a genuinely passed complete-file round trip")
            for name in ("tested_values", "decoded_values"):
                if integer(row.get(name), name) != n:
                    raise ValueError("external " + name + " does not cover the full series")
            if integer(row.get("mismatch_count"), "mismatch_count") != 0:
                raise ValueError("external decoded mismatch")
            if row.get("container_version") != "F64BLK01":
                raise ValueError("external container is not F64BLK01")
            container = Path(row["container_path"]).resolve()
            if not inside(container, root) or not container.is_file():
                raise ValueError("external container must be an existing artifact inside its batch")
            actual_hash = sha256(container)
            if actual_hash != digest_string(row.get("container_sha256"), "container_sha256"):
                raise ValueError("external container hash mismatch")
            info = inspect_container(container)
            if info["values"] != n or info["method_id"] != BASELINE_METHOD_IDS[row["method"]]:
                raise ValueError("external container identity/full coverage mismatch")
            check_metrics(row, info)
            candidates.append(dict(method=row["method"], bits=info["complete_file_bits"],
                                   path=str(container), sha256=actual_hash))
        except (OSError, ValueError, KeyError, TypeError) as error:
            rejected.append(dict(result_index=index, method=row["method"], reason=str(error)))
    return (min(candidates, key=lambda item: (item["bits"], item["method"])) if candidates else None), rejected


def execute_export(command, log, timeout):
    """Capture the exporter and terminate its process group on timeout."""
    with log.open("x", encoding="utf-8") as stream:
        stream.write("command: " + json.dumps([str(item) for item in command]) + "\n")
        stream.flush()
        options = dict(start_new_session=True) if os.name == "posix" else dict(
            creationflags=subprocess.CREATE_NEW_PROCESS_GROUP)
        process = subprocess.Popen([str(item) for item in command], stdout=stream,
                                   stderr=subprocess.STDOUT, **options)
        try:
            return process.wait(timeout=timeout)
        except subprocess.TimeoutExpired:
            if os.name == "posix":
                try:
                    os.killpg(process.pid, signal.SIGKILL)
                except ProcessLookupError:
                    pass
            else:
                try:
                    subprocess.run(["taskkill", "/PID", str(process.pid), "/T", "/F"],
                                   stdout=stream, stderr=subprocess.STDOUT, check=False, timeout=15)
                except (OSError, subprocess.TimeoutExpired):
                    pass
            if process.poll() is None:
                process.kill()
            try:
                process.wait(timeout=15)
            except subprocess.TimeoutExpired:
                pass
            stream.write("\nExporter timed out; process group stopped.\n")
            return 124


def validate_export(report_path, container, raw, key, rows, n):
    report = json.loads(report_path.read_text(encoding="utf-8"))
    if not isinstance(report, dict):
        raise ValueError("native report must be a JSON object")
    if report.get("status") != "passed" or report.get("complete_file_roundtrip_verified") is not True:
        raise ValueError("native report does not verify the persisted complete-file round trip")
    for name in ("values", "decoded_values"):
        if integer(report.get(name), name) != n:
            raise ValueError("native " + name + " does not cover full source")
    if "tested_values" in report and integer(report["tested_values"], "tested_values") != n:
        raise ValueError("native tested_values does not cover full source")
    if integer(report.get("mismatch_count"), "mismatch_count") != 0:
        raise ValueError("native decoded data mismatches source")
    for name, expected in (("input", str(raw)), ("container_path", str(container))):
        if name in report and Path(report[name]).resolve() != Path(expected):
            raise ValueError("native " + name + " differs from expected path")
    if "method" in report and report["method"] != key[3]:
        raise ValueError("native method differs from frozen method")
    for name in ("sha256", "input_sha256", "source_sha256"):
        if name in report and report[name] != key[2].lower():
            raise ValueError("native " + name + " differs from frozen source hash")
    info = inspect_container(container)
    if info["method_id"] != 3 or info["values"] != n or info["blocks"] != len(rows):
        raise ValueError("native container does not cover exactly the frozen plan")
    check_metrics(report, info)
    # The envelope inspector does not decode; the exporter's persisted-file
    # verification supplies losslessness. This additional check protects choices.
    with container.open("rb") as stream:
        stream.seek(FILE_HEADER.size)
        for row in rows:
            codec, count, length = FRAME_HEADER.unpack(stream.read(FRAME_HEADER.size))
            if codec != CODEC_IDS[row["codec"]] or count != int(row["length"]):
                raise ValueError("persisted frame changed a frozen codec or boundary")
            stream.seek(length, 1)
    actual_hash = sha256(container)
    if "container_sha256" in report and report["container_sha256"] != actual_hash:
        raise ValueError("native container_sha256 differs from actual artifact")
    return info, actual_hash


def unchanged(protected):
    failures = []
    for name, expected in protected.items():
        try:
            if sha256(name) != expected:
                failures.append(name)
        except OSError:
            failures.append(name)
    return failures


def run(args, execute=execute_export):
    began = time.perf_counter()
    timeout = args.timeout
    if not math.isfinite(timeout) or timeout <= 0:
        raise ValueError("timeout must be finite and positive")
    external, batch, inputs = load_batch(args.external_batch)
    exporter, out = Path(args.exporter).resolve(), Path(args.out_dir).resolve()
    if not exporter.is_file():
        raise ValueError("exporter executable not found: " + str(exporter))
    plans = [Path(name).resolve() for name in (args.plan or [])]
    if args.results_root:
        plans.extend(discover_plans(args.results_root))
    plans = list(dict.fromkeys(plans))
    if not plans or any(not path.is_file() for path in plans):
        raise ValueError("select existing frozen plans with --plan or --results-root")
    protected_paths = set(inputs) | set(plans) | {exporter}
    protected_paths.update(path.resolve() for path in external.rglob("*") if path.is_file())
    if (out.exists() or inside(out, external) or inside(external, out)
            or any(inside(path, out) or inside(out, path.parent) for path in set(plans))
            or any(inside(path, out) or path == out for path in protected_paths)):
        raise ValueError("out-dir must be fresh and cannot overlap raw, plan, exporter or external artifacts")
    protected = {str(path): sha256(path) for path in sorted(protected_paths)}
    out.mkdir(parents=True, exist_ok=False)
    jobs = out / "groups"
    jobs.mkdir()
    results = []
    manifest = dict(version="replay_frozen_files_v1", batch_finished=False,
                    external_batch=str(external), exporter=str(exporter), timeout=timeout,
                    original_plan_paths=[str(path) for path in plans],
                    baseline_scope=SCOPE, strongest_within_elf_camel_only=False,
                    strongest_within_verified_available_external_pool_only=True,
                    eligible_external_method_ids=BASELINE_METHOD_IDS,
                    not_strongest_across_all_published_or_historical_methods=True,
                    replayed_frozen_payload_objective=True, search_or_training_performed=False,
                    historical_payload_guarantee_transferred=False,
                    objective_note="Frozen payload-objective choices replayed unchanged; no C1/payload32 guarantee applies to complete-file metrics.",
                    protected_file_sha256_before=protected, results=results)

    def checkpoint(finished=False, changed=None):
        if changed:
            # A later mutation invalidates earlier rows too: comparison.csv must
            # not retain apparently valid gains from a compromised run.
            for result in results:
                if result.get("complete_file_roundtrip_verified") is True or result.get("status") in (
                        "passed", "passed_no_eligible_external_baseline"):
                    result.update(status="protected_input_changed",
                                  detail="Run invalidated because protected files changed: " + ", ".join(changed),
                                  complete_file_roundtrip_verified=False)
                for name in METRICS + ("blocks", "gain_vs_baseline", "gain_percent_vs_baseline"):
                    result.pop(name, None)
        manifest.update(batch_finished=finished, protected_files_unchanged=not changed,
                        changed_protected_files=changed or [], wall_seconds=time.perf_counter() - began,
                        all_checks_passed=finished and not changed and bool(results)
                        and all(row["status"] == "passed" for row in results))
        csv_pending = out / "comparison.csv.partial"
        with csv_pending.open("w", newline="", encoding="utf-8") as stream:
            writer = csv.DictWriter(stream, fieldnames=FIELDS, extrasaction="ignore")
            writer.writeheader()
            writer.writerows(results)
        csv_pending.replace(out / "comparison.csv")
        pending = out / "replay_manifest.json.partial"
        pending.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
        pending.replace(out / "replay_manifest.json")

    print("Frozen-file replay output: " + str(out), flush=True)
    checkpoint()
    stop = False
    for plan in plans:
        try:
            groups = load_groups(plan)
        except (OSError, ValueError, csv.Error) as error:
            results.append(dict(status="plan_invalid", detail=str(error), original_plan_path=str(plan),
                                original_plan_sha256=protected[str(plan)]))
            checkpoint()
            continue
        for key, source_rows in groups:
            print("Replay {} / {} [{}]".format(key[0], key[1], key[3]), flush=True)
            row = dict(zip(("dataset", "series", "source_sha256", "method"), key))
            row.update(status="pending", detail="", original_plan_path=str(plan),
                       original_plan_sha256=protected[str(plan)], baseline_scope=SCOPE,
                       old_payload32_bits="", replayed_frozen_payload_objective=True,
                       historical_payload_guarantee_transferred=False, search_or_training_performed=False,
                       complete_file_roundtrip_verified=False)
            results.append(row)
            checkpoint()
            try:
                ordered, n = validate_plan(key, source_rows)
                row["n"] = n
                raw = matching_raw(key, n, batch, inputs)
                row["input_path"] = str(raw)
                # No historical payload-cost table is opened or used for ranking.
                if all(item.get("old_payload32_bits", "") for item in ordered):
                    row["old_payload32_bits"] = sum(integer(item["old_payload32_bits"],
                        "old_payload32_bits") for item in ordered)
                baseline, rejected = strongest_baseline(key, raw, n, external, batch)
                row["rejected_external_baselines"] = rejected
                if baseline:
                    row.update(baseline_method=baseline["method"], baseline_complete_file_bits=baseline["bits"],
                               baseline_container_path=baseline["path"], baseline_container_sha256=baseline["sha256"])
                folder = jobs / ("{:04d}".format(len(results)))
                folder.mkdir()
                filtered, container, report, log = (folder / name for name in
                    ("frozen_plan.csv", "replayed.f64blk", "report.json", "export.log"))
                with filtered.open("x", newline="", encoding="utf-8") as stream:
                    writer = csv.DictWriter(stream, fieldnames=PLAN_FIELDS, extrasaction="ignore")
                    writer.writeheader()
                    writer.writerows(ordered)
                row.update(filtered_plan_path=str(filtered), filtered_plan_sha256=sha256(filtered),
                           container_path=str(container), report_path=str(report), log_path=str(log))
                changes = unchanged(protected)
                if changes:
                    raise ValueError("protected source changed before export: " + ", ".join(changes))
                code = execute([exporter, raw, filtered, key[3], container, report], log, timeout)
                row["exporter_exit_code"] = code
                if code != 0:
                    raise ValueError("native exporter failed with exit code " + str(code))
                info, artifact_hash = validate_export(report, container, raw, key, ordered, n)
                if sha256(filtered) != row["filtered_plan_sha256"]:
                    raise ValueError("exporter changed filtered frozen plan")
                row.update({name: info[name] for name in METRICS + ("blocks",)})
                row.update(container_sha256=artifact_hash, complete_file_roundtrip_verified=True)
                if baseline:
                    gain = 1.0 - info["complete_file_bits"] / baseline["bits"]
                    row.update(status="passed", gain_vs_baseline=gain, gain_percent_vs_baseline=100.0 * gain)
                else:
                    row.update(status="passed_no_eligible_external_baseline",
                               detail="Actual native file verified; no qualifying available external full-file baseline, so no gain is claimed.")
            except (OSError, ValueError, KeyError, TypeError, csv.Error) as error:
                row.update(status="replay_failed", detail=str(error), complete_file_roundtrip_verified=False)
                for name in METRICS + ("blocks", "gain_vs_baseline", "gain_percent_vs_baseline"):
                    row.pop(name, None)
            finally:
                for path_key, hash_key in (("container_path", "container_sha256"),
                                           ("report_path", "report_sha256"), ("log_path", "log_sha256")):
                    artifact = row.get(path_key)
                    if artifact and Path(artifact).is_file():
                        row[hash_key] = sha256(artifact)
                changes = unchanged(protected)
                if changes:
                    row.update(status="protected_input_changed", detail="Protected files changed: " + ", ".join(changes),
                               complete_file_roundtrip_verified=False)
                    for name in METRICS + ("blocks", "gain_vs_baseline", "gain_percent_vs_baseline"):
                        row.pop(name, None)
                    stop = True
                checkpoint(changed=changes)
                print("  " + row["status"] + (": " + row["detail"] if row["detail"] else ""), flush=True)
            if stop:
                break
        if stop:
            break
    changes = unchanged(protected)
    manifest["protected_file_sha256_after"] = {name: sha256(name) if Path(name).is_file() else None
                                               for name in protected}
    checkpoint(finished=True, changed=changes)
    print("Comparison: " + str(out / "comparison.csv"), flush=True)
    print("Manifest: " + str(out / "replay_manifest.json"), flush=True)
    return 0 if manifest["all_checks_passed"] else 1


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--external-batch", required=True)
    parser.add_argument("--exporter", required=True)
    parser.add_argument("--plan", action="append", default=[])
    parser.add_argument("--results-root")
    parser.add_argument("--out-dir", required=True)
    parser.add_argument("--timeout", type=float, default=300)
    args = parser.parse_args()
    try:
        return run(args)
    except (OSError, ValueError, TypeError, KeyError, csv.Error) as error:
        parser.exit(2, "replay_frozen_files: " + str(error) + "\n")


if __name__ == "__main__":
    raise SystemExit(main())
