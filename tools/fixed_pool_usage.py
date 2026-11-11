#!/usr/bin/env python3
"""Describe a fixed codec pool without changing it or claiming global redundancy.

Input schema v1 is intentionally independent of previous experiment manifests.
Sizes are complete frame bytes, not codec payload bytes. A layout is verified as
its container overhead plus contiguous, input-hash-bound verified frames.

Removal diagnostics KEEP THE VISITED LAYOUT FIXED. They reselect from already
measured options, and compare against reselection with the complete active pool.
They do not rerun boundary search, train a selector, or authorize pruning.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import re
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any


STATUSES = {"passed", "unsupported", "failed", "pending"}
SHA256 = re.compile(r"^[0-9a-fA-F]{64}$")
GROUPS = {"stream": "old_stream", "old_stream": "old_stream", "alp": "alp", "btr": "btr"}
CAVEATS = [
    "Zero final selections means unused in these recorded layouts, not globally useless.",
    "Native winners are minima among recorded verified trials for the same region, not unvisited regions.",
    "Removal diagnostics keep visited boundaries fixed and reselect recorded options; they do not rerun search.",
    "Failed, pending, or missing options are unknown, never zero bytes or proof of infeasibility.",
    "Empirical no loss applies only to these hash-bound frozen layouts and this declared active pool.",
    "Usage statistics alone do not establish universal redundancy; any reduced pool needs a separately declared subsequent run.",
    "Frame bytes exclude the once-per-container overhead; layout and removal totals include it.",
    "This reporter validates declared hash/roundtrip evidence; the experiment producer, not this reporter, decodes native files.",
]


class UsageError(ValueError):
    """The usage input is inconsistent or cannot support a certified size."""


def _integer(value: Any, where: str, *, positive: bool = False) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < (1 if positive else 0):
        raise UsageError(f"{where}: expected {'positive' if positive else 'nonnegative'} integer")
    return value


def _sha(value: Any, where: str, expected: str | None = None) -> str:
    if not isinstance(value, str) or not SHA256.fullmatch(value):
        raise UsageError(f"{where}: expected a full SHA-256 hex string")
    value = value.lower()
    if expected is not None and value != expected:
        raise UsageError(f"{where}: raw input hash differs from series hash")
    return value


def _list(value: Any, where: str) -> list:
    if not isinstance(value, list):
        raise UsageError(f"{where}: expected list")
    return value


def _dict(value: Any, where: str) -> dict:
    if not isinstance(value, dict):
        raise UsageError(f"{where}: expected object")
    return value


def _canonical(codec: Any, catalog: dict[str, str]) -> str:
    if not isinstance(codec, str):
        raise UsageError("codec: expected a string")
    if codec in catalog:
        return codec
    if codec.startswith("alp+tail:") and "alp" in catalog:
        tail = codec[len("alp+tail:"):]
        if tail in catalog and catalog[tail] == "old_stream":
            return "alp"
    raise UsageError(f"codec {codec!r}: not a declared active candidate or explicit ALP-tail action")


def _expected_actions(catalog: dict[str, str], length: int) -> set[str]:
    expected = set(catalog) - {"alp"}
    if "alp" in catalog:
        streams = [c for c, group in catalog.items() if group == "old_stream"]
        if length >= 1024 and length % 1024 and streams:
            expected.update("alp+tail:" + stream for stream in streams)
        else:
            # Below one vector, or without a declared stream for a nonaligned
            # tail, the explicit ordinary ALP option must be unsupported.
            expected.add("alp")
    return expected


def _validate_alp_geometry(option: dict, length: int, where: str) -> None:
    if option["status"] != "passed":
        return
    if option["codec"] == "alp" and (length < 1024 or length % 1024):
        raise UsageError(f"{where}: ordinary ALP cannot pass a short or nonaligned region")
    if option["codec"].startswith("alp+tail:") and (length < 1024 or length % 1024 == 0):
        raise UsageError(f"{where}: ALP-tail requires a full-vector prefix and a nonempty tail")


def _option(item: Any, catalog: dict[str, str], raw_sha: str, where: str) -> dict:
    item = _dict(item, where)
    canonical = _canonical(item.get("codec"), catalog)
    status = item.get("status")
    if not isinstance(status, str) or status not in STATUSES:
        raise UsageError(f"{where}.status: unknown status {status!r}")
    if item.get("size_scope", "complete_frame_bytes") != "complete_frame_bytes":
        raise UsageError(f"{where}: option scope is not complete frame bytes")
    _sha(item.get("sha256"), f"{where}.sha256", raw_sha)
    result = {"codec": item["codec"], "canonical": canonical, "status": status, "frame_bytes": None}
    if status == "passed":
        if item.get("roundtrip_verified") is not True:
            raise UsageError(f"{where}: passed frame is not explicitly roundtrip verified")
        result["frame_bytes"] = _integer(item.get("frame_bytes"), f"{where}.frame_bytes", positive=True)
    elif item.get("frame_bytes") is not None:
        # An invalid native size must never enter a size competition.
        raise UsageError(f"{where}: non-passed frame must have null/absent frame_bytes")
    return result


def _removal(layout: dict, removed: set[str], *, remove_name: str, remove_type: str) -> dict:
    row = {
        "series": layout["series"], "sha256": layout["sha256"], "method": layout["method"],
        "remove_type": remove_type, "removed": remove_name,
        "stored_complete_file_bytes": layout.get("complete_file_bytes"),
        "all_pool_reselected_bytes": None, "without_reselected_bytes": None,
        "delta_bytes": None, "incomplete_regions": 0, "no_alternative_regions": 0,
    }
    if layout["status"] != "passed":
        row["status"] = "layout_unavailable"
        return row
    all_cost = layout["container_overhead_bytes"]
    retained_cost = layout["container_overhead_bytes"]
    for segment in layout["segments"]:
        options = segment["options"]
        unknown = not segment["coverage_complete"] or any(o["status"] in {"failed", "pending"} for o in options)
        if unknown:
            row["incomplete_regions"] += 1
        passed = [o for o in options if o["status"] == "passed"]
        remaining = [o for o in passed if o["canonical"] not in removed]
        # The selected action is required to be represented among passed options.
        all_cost += min(o["frame_bytes"] for o in passed)
        if remaining:
            retained_cost += min(o["frame_bytes"] for o in remaining)
        elif not unknown:
            row["no_alternative_regions"] += 1
    if row["incomplete_regions"]:
        row["status"] = "incomplete_coverage"
    elif row["no_alternative_regions"]:
        row["status"] = "no_feasible_alternative"
        row["all_pool_reselected_bytes"] = all_cost
    else:
        row["all_pool_reselected_bytes"] = all_cost
        row["without_reselected_bytes"] = retained_cost
        row["delta_bytes"] = retained_cost - all_cost
        row["status"] = "empirical_no_loss_on_frozen_layout" if row["delta_bytes"] == 0 else "observed_size_loss"
    return row


def analyze(payload: dict) -> dict:
    """Validate schema v1 and return usage, observed winners and frozen removals.

    Every trial/option sha256 binds the source sequence, not the compressed file.
    Container and payload SHA values, if supplied separately, are not substituted
    for this source identity. This tool validates declared verification evidence;
    it does not itself decode native files.
    """
    payload = _dict(payload, "root")
    if isinstance(payload.get("schema_version"), bool) or payload.get("schema_version") != 1:
        raise UsageError("schema_version must be 1")
    if payload.get("size_scope") != "complete_container_bytes":
        raise UsageError("size_scope must be complete_container_bytes")
    if payload.get("codec_size_scope") != "complete_frame_bytes":
        raise UsageError("codec_size_scope must be complete_frame_bytes")
    catalog = {}
    for item in _list(payload.get("candidates"), "candidates"):
        item = _dict(item, "candidate")
        codec, group = item.get("codec"), item.get("group")
        if not isinstance(codec, str) or not codec or codec in catalog or codec.startswith("alp+tail:"):
            raise UsageError(f"invalid or duplicate base codec {codec!r}")
        if not isinstance(group, str) or group not in GROUPS:
            raise UsageError(f"candidate {codec}: group must be stream/old_stream, alp, or btr")
        catalog[codec] = GROUPS[group]
    if not catalog:
        raise UsageError("candidates must not be empty")
    counters = {c: Counter() for c in catalog}
    selected = {c: defaultdict(Counter) for c in catalog}
    layouts = []
    series_names = set()
    layout_methods = set()
    winner_regions = []
    for entry in _list(payload.get("series"), "series"):
        entry = _dict(entry, "series item")
        name = entry.get("series")
        if not isinstance(name, str) or not name or name in series_names:
            raise UsageError(f"invalid or duplicate series {name!r}")
        series_names.add(name)
        raw_sha = _sha(entry.get("sha256"), f"{name}.sha256")
        n = _integer(entry.get("values"), f"{name}.values", positive=True)
        by_region = defaultdict(list)
        trial_keys = set()
        for index, item in enumerate(_list(entry.get("trials"), f"{name}.trials")):
            where = f"{name}.trials[{index}]"
            trial = _option(item, catalog, raw_sha, where)
            start = _integer(item.get("start"), f"{where}.start")
            length = _integer(item.get("length"), f"{where}.length", positive=True)
            if start + length > n:
                raise UsageError(f"{where}: region exceeds input length")
            _validate_alp_geometry(trial, length, where)
            key = start, length, trial["codec"]
            if key in trial_keys:
                raise UsageError(f"{where}: duplicate physical trial for same region and action")
            trial_keys.add(key)
            count = counters[trial["canonical"]]
            count[f"trial_{trial['status']}"] += 1
            attempted = item.get("attempted", trial["status"] != "pending")
            if not isinstance(attempted, bool):
                raise UsageError(f"{where}.attempted: expected boolean")
            if trial["status"] == "passed" and attempted is not True:
                raise UsageError(f"{where}: passed native trial cannot claim no attempt")
            if attempted:
                count["trial_attempted"] += 1
            by_region[(start, length)].append(trial)
        for (start, length), trials in by_region.items():
            passed = [o for o in trials if o["status"] == "passed"]
            if not passed:
                continue
            minimum = min(o["frame_bytes"] for o in passed)
            actions = [o["codec"] for o in passed if o["frame_bytes"] == minimum]
            winning_bases = sorted({_canonical(c, catalog) for c in actions})
            for codec in winning_bases:
                counters[codec]["observed_region_min_wins"] += 1
                counters[codec]["observed_region_sole_min_wins" if len(winning_bases) == 1 else "observed_region_tied_min_wins"] += 1
            winner_regions.append({"series": name, "start": start, "length": length,
                                   "frame_bytes": minimum, "winning_candidates": winning_bases,
                                   "winning_actions": actions,
                                   "coverage_complete": _expected_actions(catalog, length).issubset({o["codec"] for o in trials})
                                   and all(o["status"] not in {"pending", "failed"} for o in trials)})
        seen_methods = set()
        for item in _list(entry.get("layouts"), f"{name}.layouts"):
            item = _dict(item, f"{name}.layout")
            method, status = item.get("method"), item.get("status")
            if not isinstance(method, str) or not method or method in seen_methods:
                raise UsageError(f"{name}: layout methods must be unique nonempty strings")
            seen_methods.add(method)
            layout_methods.add(method)
            if not isinstance(status, str) or status not in STATUSES:
                raise UsageError(f"{name}/{method}: unknown layout status")
            if item.get("size_scope", "complete_container_bytes") != "complete_container_bytes":
                raise UsageError(f"{name}/{method}: layout scope is not complete container bytes")
            _sha(item.get("sha256"), f"{name}/{method}.sha256", raw_sha)
            layout = {"series": name, "sha256": raw_sha, "method": method, "status": status}
            if status != "passed":
                if item.get("complete_file_bytes") is not None:
                    raise UsageError(f"{name}/{method}: unavailable layout must not claim valid bytes")
                layouts.append(layout)
                continue
            if item.get("complete_file_roundtrip_verified") is not True:
                raise UsageError(f"{name}/{method}: passed layout must be roundtrip verified")
            total = _integer(item.get("complete_file_bytes"), f"{name}/{method}.complete_file_bytes", positive=True)
            overhead = _integer(item.get("container_overhead_bytes"), f"{name}/{method}.container_overhead_bytes")
            segments = []
            cursor, stored = 0, overhead
            for index, region in enumerate(_list(item.get("segments"), f"{name}/{method}.segments")):
                where = f"{name}/{method}.segments[{index}]"
                region = _dict(region, where)
                start = _integer(region.get("start"), f"{where}.start")
                length = _integer(region.get("length"), f"{where}.length", positive=True)
                if start != cursor or start + length > n:
                    raise UsageError(f"{where}: layout must be contiguous, nonoverlapping and cover input")
                cursor += length
                codec = region.get("codec")
                canonical = _canonical(codec, catalog)
                frame = _integer(region.get("frame_bytes"), f"{where}.frame_bytes", positive=True)
                options = []
                action_names = set()
                for oi, option in enumerate(_list(region.get("options", []), f"{where}.options")):
                    parsed = _option(option, catalog, raw_sha, f"{where}.options[{oi}]")
                    _validate_alp_geometry(parsed, length, where)
                    if parsed["codec"] in action_names:
                        raise UsageError(f"{where}: duplicate option action")
                    action_names.add(parsed["codec"])
                    options.append(parsed)
                matched = [o for o in options if o["codec"] == codec and o["status"] == "passed" and o["frame_bytes"] == frame]
                if not matched:
                    raise UsageError(f"{where}: selected frame lacks matching verified passed option")
                complete_flag = region.get("options_complete")
                if not isinstance(complete_flag, bool):
                    raise UsageError(f"{where}.options_complete must be boolean")
                segments.append({"start": start, "length": length, "codec": codec,
                                 "canonical": canonical, "frame_bytes": frame, "options": options,
                                 "coverage_complete": complete_flag and _expected_actions(catalog, length).issubset(action_names)})
                stored += frame
                selected[canonical][method]["segments"] += 1
                selected[canonical][method]["values"] += length
                selected[canonical][method]["frame_bytes"] += frame
            if cursor != n or stored != total:
                raise UsageError(f"{name}/{method}: complete bytes or covered values disagree with segments")
            layout.update(complete_file_bytes=total, container_overhead_bytes=overhead, segments=segments)
            layouts.append(layout)
    removals = [_removal(layout, {codec}, remove_name=codec, remove_type="candidate")
                for layout in layouts for codec in catalog]
    groups = {g: {c for c in catalog if catalog[c] == g} for g in sorted(set(catalog.values()))}
    group_removals = [_removal(layout, codecs, remove_name=group, remove_type="group")
                      for layout in layouts for group, codecs in groups.items()]
    candidates, suggestions = [], []
    passed_layouts = sum(layout["status"] == "passed" for layout in layouts)
    for codec, group in catalog.items():
        row = {"codec": codec, "group": group}
        for key in ("trial_attempted", "trial_passed", "trial_unsupported", "trial_failed", "trial_pending",
                    "observed_region_min_wins", "observed_region_sole_min_wins", "observed_region_tied_min_wins"):
            row[key] = counters[codec][key]
        for measure in ("segments", "values", "frame_bytes"):
            row[f"selected_{measure}"] = sum(s[measure] for s in selected[codec].values())
        row["selected_by_method"] = {method: {m: selected[codec][method][m] for m in ("segments", "values", "frame_bytes")}
                                     for method in sorted(layout_methods)}
        rows = [r for r in removals if r["removed"] == codec]
        complete = rows and all(r["status"] in {"empirical_no_loss_on_frozen_layout", "observed_size_loss"} for r in rows)
        no_loss = bool(complete and all(r["delta_bytes"] == 0 for r in rows))
        unused = passed_layouts > 0 and row["selected_segments"] == 0
        suggestion = {
            "codec": codec, "group": group, "unused_on_this_run": unused,
            "empirical_no_loss_on_frozen_layouts": no_loss,
            "removal_diagnostic_complete": bool(complete),
            "evaluated_layouts": len(rows), "passed_layouts": passed_layouts,
            "incomplete_layouts": sum(r["status"] == "incomplete_coverage" for r in rows),
            "unavailable_layouts": sum(r["status"] == "layout_unavailable" for r in rows),
            "no_alternative_layouts": sum(r["status"] == "no_feasible_alternative" for r in rows),
            "observed_loss_layouts": sum(r["status"] == "observed_size_loss" for r in rows),
            "total_delta_bytes": sum(r["delta_bytes"] for r in rows) if complete else None,
            "recommendation": "candidate_for_independent_small_pool_validation" if unused and no_loss
                              else "keep_or_investigate_before_pruning",
            "native_observed_min_wins": row["observed_region_min_wins"],
            "scope": "this_run_frozen_visited_layouts_only; not_global_redundancy; no_automatic_pruning",
        }
        candidates.append(row)
        suggestions.append(suggestion)
    group_usage = []
    for group, codecs in groups.items():
        rows = [r for r in candidates if r["codec"] in codecs]
        summary = {"group": group, "candidates": sorted(codecs),
                   **{key: sum(r[key] for r in rows) for key in rows[0]
                      if isinstance(rows[0][key], int) and not key.startswith("observed_region_")}}
        # Group wins count each region once, even when two members of the same
        # group tie. They are not a sum of per-candidate winner counts.
        summary.update(observed_region_min_wins=0, observed_region_sole_min_wins=0, observed_region_tied_min_wins=0)
        for winner in winner_regions:
            winning_groups = {catalog[c] for c in winner["winning_candidates"]}
            if group in winning_groups:
                summary["observed_region_min_wins"] += 1
                summary["observed_region_sole_min_wins" if len(winning_groups) == 1 else "observed_region_tied_min_wins"] += 1
        group_usage.append(summary)
    return {"schema_version": 1, "size_scope": "complete_container_bytes", "codec_size_scope": "complete_frame_bytes",
            "series_count": len(series_names), "layout_count": len(layouts), "passed_layout_count": passed_layouts,
            "candidates": candidates, "group_usage": group_usage, "layout_removals": removals,
            "group_removals": group_removals, "suggestions": suggestions,
            "native_region_winners": winner_regions, "caveats": CAVEATS}


def _csv(path: Path, rows: list[dict]) -> None:
    fields = list(dict.fromkeys(key for row in rows for key in row))
    with path.open("w", newline="", encoding="utf-8-sig") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            writer.writerow({key: json.dumps(value, ensure_ascii=False, sort_keys=True) if isinstance(value, (dict, list)) else value
                             for key, value in row.items()})


def write_report(report: dict, out_dir: str | Path) -> None:
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    (out / "report.json").write_text(json.dumps(report, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    for key, filename in (("candidates", "candidate_usage.csv"), ("suggestions", "suggestions.csv"),
                          ("layout_removals", "candidate_removal.csv"), ("group_usage", "group_usage.csv"),
                          ("group_removals", "group_removal.csv"), ("native_region_winners", "native_region_winners.csv")):
        _csv(out / filename, report[key])
    (out / "scope.txt").write_text("\n".join(report["caveats"]) + "\n", encoding="utf-8")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("input", type=Path, help="schema-v1 usage_input.json")
    parser.add_argument("--out-dir", type=Path, required=True)
    args = parser.parse_args(argv)
    try:
        data = args.input.read_bytes()
        report = analyze(json.loads(data))
        report["usage_input_sha256"] = hashlib.sha256(data).hexdigest()
        write_report(report, args.out_dir)
    except (UsageError, OSError, json.JSONDecodeError) as exc:
        parser.exit(2, f"fixed_pool_usage: {exc}\n")
    print(f"Verified layouts: {report['passed_layout_count']}/{report['layout_count']}")
    print(f"Active candidates: {len(report['candidates'])}; no pool was changed")
    print(f"Usage and frozen-layout diagnostics: {args.out_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
