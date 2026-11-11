#!/usr/bin/env python3
"""Finalize only the new active-pool configuration after all 11 original inputs.

The user-requested rule is actual FINAL ADAPTIVE usage, not oracle ties, native
trial wins, or fixed1024 usage. An explicit ALP-tail action also retains the
stream codec required to decode that tail. Implementations and old results are
never removed. This module does not modify its input or start another search.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
from pathlib import Path
import re
from typing import Any

import fixed_pool_usage as usage


ORIGINAL_SERIES_COUNT = 11
HASH_PATTERN = re.compile(r"^[0-9a-fA-F]{64}$")
POLICY = "actual_final_adaptive_selection_plus_explicit_alp_tail_dependencies"


def _canonical_bytes(payload: Any) -> bytes:
    return json.dumps(payload, sort_keys=True, separators=(",", ":"),
                      ensure_ascii=False, allow_nan=False).encode("utf-8")


def _identity(row: Any, where: str) -> dict:
    if not isinstance(row, dict):
        raise ValueError(f"{where}: expected identity object")
    name, n, digest = row.get("series"), row.get("values"), row.get("sha256")
    if not isinstance(name, str) or not name:
        raise ValueError(f"{where}: series must be a nonempty string")
    if type(n) is not int or n <= 0:
        raise ValueError(f"{where}: values must be a positive integer")
    if not isinstance(digest, str) or not HASH_PATTERN.fullmatch(digest):
        raise ValueError(f"{where}: expected complete raw SHA-256")
    return {"series": name, "values": n, "sha256": digest.lower()}


def _original_actions(payload: Any) -> list[str] | None:
    if not isinstance(payload, dict) or not isinstance(payload.get("candidates"), list):
        return None
    actions = [r.get("codec") if isinstance(r, dict) else None for r in payload["candidates"]]
    if not actions or any(not isinstance(a, str) or not a for a in actions) or len(set(actions)) != len(actions):
        return None
    return actions


def finalize(payload: dict, declared_jobs: list[dict], *, batch_complete: bool,
             batch_status: str) -> dict:
    """Return a guarded, JSON-serializable new-pool configuration/report.

    ``declared_jobs`` MUST be the original unfiltered eleven identities, each
    containing series, values and raw sha256. Passing a filtered SERIES_LIST is
    not accepted. The payload's canonical JSON hash freezes this exact evidence
    independently of whitespace. A caller with the actual input file should
    additionally retain its byte SHA-256 (the CLI does this automatically).

    A blocked finalization retains the original action list, removes nothing,
    and never publishes an active_pool.json via write_finalization().
    """
    original_actions = _original_actions(payload)
    result = {
        "schema_version": 1, "status": "not_finalized", "policy": POLICY,
        "selection_scope": "actual_adaptive_final_selections_only",
        "active_actions": original_actions, "removed_actions": [], "original_actions": original_actions,
        "pruning_applied_to_new_configuration_only": False,
        "implementations_deleted": False, "old_results_modified": False,
        "batch_complete": batch_complete, "batch_status": batch_status,
        "required_original_series_count": ORIGINAL_SERIES_COUNT,
        "declared_series_count": None, "observed_series_count": None,
        "usage_payload_canonical_sha256": None,
        "usage_payload_hash_scope": "SHA256 of UTF-8 canonical JSON, sorted keys, compact separators, ensure_ascii=False, allow_nan=False",
        "source_identities": [], "counts": [], "reasons": [],
        "scope": {
            "selection": "actual adaptive final semantic segments only",
            "fixed1024": "must complete successfully; its selections do not preserve candidates",
            "ties": "only the recorded chosen action counts, not every tied option",
            "alp": "alp+tail:X counts as direct alp usage and retains dependency X",
            "generalization": "empirical configuration for this completed run, not proof of global uselessness",
            "future_experiments": "freeze this new pool before an independent rerun; do not relabel old full-pool results",
        },
    }

    def block(reason: str) -> None:
        result["reasons"].append(reason)

    try:
        result["usage_payload_canonical_sha256"] = hashlib.sha256(_canonical_bytes(payload)).hexdigest()
    except (TypeError, ValueError, OverflowError) as error:
        block("input_not_finite_json: " + str(error))
        return result
    if type(batch_complete) is not bool or batch_complete is not True:
        block("batch_not_complete")
    if batch_status != "passed":
        block("batch_status_not_passed")
    declared = []
    try:
        if not isinstance(declared_jobs, list):
            raise ValueError("declared_jobs must be the original identity list")
        declared = [_identity(row, f"declared_jobs[{i}]") for i, row in enumerate(declared_jobs)]
        result["declared_series_count"] = len(declared)
        if len(declared) != ORIGINAL_SERIES_COUNT:
            block("original_eleven_identity_list_required")
        if len({row["series"] for row in declared}) != len(declared):
            block("duplicate_declared_series")
        result["source_identities"] = declared
    except (TypeError, ValueError) as error:
        block("invalid_declared_jobs: " + str(error))
    try:
        report = usage.analyze(payload)
    except (usage.UsageError, TypeError, ValueError, KeyError) as error:
        block("usage_validation_failed: " + str(error))
        return result
    observed = [_identity(row, "usage series") for row in payload["series"]]
    result["observed_series_count"] = len(observed)
    if ({row["series"]: row for row in declared} != {row["series"]: row for row in observed}
            or len(declared) != len(observed)):
        block("observed_series_do_not_exactly_match_original_declared_identities")
    if len(observed) != ORIGINAL_SERIES_COUNT:
        block("all_original_eleven_series_must_finish")
    if report["passed_layout_count"] != 2 * ORIGINAL_SERIES_COUNT:
        block("all_twenty_two_verified_layouts_required")
    for entry in payload["series"]:
        methods = {layout["method"]: layout for layout in entry["layouts"]}
        if set(methods) != {"fixed1024", "adaptive"}:
            block("both_and_only_fixed1024_and_adaptive_required: " + entry["series"])
            continue
        if any(layout["status"] != "passed" or layout.get("complete_file_roundtrip_verified") is not True
               for layout in methods.values()):
            block("unavailable_or_unverified_layout: " + entry["series"])
        if not entry["trials"]:
            block("native_trial_evidence_missing: " + entry["series"])
        if any(trial["status"] in {"failed", "pending"} for trial in entry["trials"]):
            block("failed_or_pending_native_trial: " + entry["series"])
    if any(row["status"] in {"incomplete_coverage", "layout_unavailable"} for row in report["layout_removals"]):
        block("incomplete_or_unavailable_frozen_layout_options")

    counts = {codec: dict(codec=codec, direct_segments=0, direct_values=0, direct_frame_bytes=0,
                          required_alp_tail_segments=0, required_alp_tail_values=0)
              for codec in original_actions}
    if not result["reasons"]:
        # All preceding guards passed. Select only the actual adaptive layout,
        # never minima from options/trials and never the fixed1024 layout.
        for entry in payload["series"]:
            adaptive = next(layout for layout in entry["layouts"] if layout["method"] == "adaptive")
            for region in adaptive["segments"]:
                action, length = region["codec"], region["length"]
                codec = "alp" if action.startswith("alp+tail:") else action
                counts[codec]["direct_segments"] += 1
                counts[codec]["direct_values"] += length
                counts[codec]["direct_frame_bytes"] += region["frame_bytes"]
                if action.startswith("alp+tail:"):
                    tail = action[len("alp+tail:"):]
                    counts[tail]["required_alp_tail_segments"] += 1
                    counts[tail]["required_alp_tail_values"] += length % 1024
        active, removed = [], []
        for codec in original_actions:
            count = counts[codec]
            if count["direct_segments"]:
                count["keep_reason"] = "direct_adaptive_final_selection"
                active.append(codec)
            elif count["required_alp_tail_segments"]:
                count["keep_reason"] = "required_explicit_alp_tail_dependency"
                active.append(codec)
            else:
                count["keep_reason"] = "removed_zero_adaptive_final_usage_and_no_dependency"
                removed.append(codec)
        result.update(status="finalized", active_actions=active, removed_actions=removed,
                      pruning_applied_to_new_configuration_only=True)
    else:
        # These zeros are not presented as measured selections in a blocked run.
        for count in counts.values():
            for field in ("direct_segments", "direct_values", "direct_frame_bytes",
                          "required_alp_tail_segments", "required_alp_tail_values"):
                count[field] = None
            count["keep_reason"] = "not_finalized_original_pool_unchanged"
    result["counts"] = list(counts.values())
    return result


def write_finalization(result: dict, out_dir: str | Path) -> None:
    """Write exclusive new reports; atomically publish the main JSON LAST.

    A failure before publication leaves no finalized_pool.json for a future
    launcher to consume. Partial files are never configuration inputs.
    """
    report_text = json.dumps(result, indent=2, ensure_ascii=False, allow_nan=False) + "\n"
    config_text = None
    if result.get("status") == "finalized":
        config = {key: result[key] for key in ("schema_version", "status", "policy", "active_actions",
                  "removed_actions", "usage_payload_canonical_sha256", "source_identities", "scope")}
        config_text = json.dumps(config, indent=2, ensure_ascii=False, allow_nan=False) + "\n"
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=False)
    fields = ("codec", "direct_segments", "direct_values", "direct_frame_bytes", "required_alp_tail_segments",
              "required_alp_tail_values", "keep_reason")
    with (out / "removal_report.csv").open("x", newline="", encoding="utf-8-sig") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(result.get("counts", []))
    if config_text is not None:
        with (out / ".active_pool.json.partial").open("x", encoding="utf-8") as handle:
            handle.write(config_text)
    with (out / ".finalized_pool.json.partial").open("x", encoding="utf-8") as handle:
        handle.write(report_text)
    if config_text is not None:
        (out / ".active_pool.json.partial").rename(out / "active_pool.json")
    # The supported future POOL_CONFIG input is this last, atomically renamed
    # report, never one of the .partial files or a half-written JSON document.
    (out / ".finalized_pool.json.partial").rename(out / "finalized_pool.json")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("input", type=Path, help="final usage_input.json")
    parser.add_argument("--declared-jobs", type=Path, required=True, help="original eleven normalized identities as a JSON list")
    parser.add_argument("--batch-manifest", type=Path, required=True)
    parser.add_argument("--out-dir", type=Path, required=True)
    args = parser.parse_args(argv)
    try:
        raw_input = args.input.read_bytes()
        raw_jobs = args.declared_jobs.read_bytes()
        raw_manifest = args.batch_manifest.read_bytes()
        manifest = json.loads(raw_manifest)
        result = finalize(json.loads(raw_input), json.loads(raw_jobs), batch_complete=manifest.get("complete"),
                          batch_status=manifest.get("status"))
        result["evidence_files"] = {name: {"path": str(path.resolve()), "sha256": hashlib.sha256(data).hexdigest()}
                                    for name, path, data in (("usage_input", args.input, raw_input),
                                                             ("declared_jobs", args.declared_jobs, raw_jobs),
                                                             ("batch_manifest", args.batch_manifest, raw_manifest))}
        write_finalization(result, args.out_dir)
    except (OSError, ValueError, TypeError, KeyError) as error:
        parser.exit(2, "finalize_flat_pool: " + str(error) + "\n")
    print("New-pool finalization: " + result["status"])
    print("Removed actions: " + ", ".join(result["removed_actions"]))
    print("Implementations and old results were not changed")
    return int(result["status"] != "finalized")


if __name__ == "__main__":
    raise SystemExit(main())
