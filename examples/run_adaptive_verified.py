"""Run fresh adaptive boundaries, measured repair, frozen Top-2 DP, and decode.

This is a small reproducible source-code example, not a replay of the paper's
historical inputs, trained folds, or timings. Native dependencies are external.
No historical boundary template or compression-price cache is consulted.
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
import shutil
import sys
from types import SimpleNamespace


def _cgroup_memory_snapshot() -> dict:
    cgroup = Path("/sys/fs/cgroup")
    maximum, current = cgroup / "memory.max", cgroup / "memory.current"
    if not (maximum.is_file() and current.is_file()):
        return {"available": False}
    limit_text = maximum.read_text(encoding="ascii").strip()
    if limit_text == "max":
        return {"available": False, "unlimited": True}
    limit, used = int(limit_text), int(current.read_text(encoding="ascii"))
    return dict(available=True, cgroup_memory_max=limit,
                cgroup_memory_current=used, cgroup_headroom_bytes=limit-used)


# A resumed process may start just above the 512 MiB floor. Check its original
# admission BEFORE importing NumPy/Numba/ClaSPy; later import overhead is not a
# reason to discard an authenticated completed boundary. No cache is purged.
_RESUMING_CLI = any(a == "--resume-boundary" or a.startswith("--resume-boundary=")
                    for a in sys.argv[1:]) and not any(a in ("-h", "--help") for a in sys.argv[1:])
EARLY_RESUME_MEMORY = _cgroup_memory_snapshot() if _RESUMING_CLI else None
if (EARLY_RESUME_MEMORY is not None and EARLY_RESUME_MEMORY["available"]
        and EARLY_RESUME_MEMORY["cgroup_headroom_bytes"] < 512 * 1024**2):
    raise SystemExit("cgroup memory headroom below the original 512 MiB admission floor")


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "tools"))

from btr_search_ledger import BudgetedRegionView  # noqa: E402
from compression_boundary_repair import RepairConfig, repair_partition  # noqa: E402
from compression_candidates import build_candidates  # noqa: E402
from compression_gain_proxy import CompressionGainProxy  # noqa: E402
from flat_pool_backend import DEFAULT_ACTIONS, FlatPoolEvaluator  # noqa: E402
from codec_pool import check_choices, check_trial_ledger  # noqa: E402
from strict8_boundary_workflow import (  # noqa: E402
    StrictEightBoundaryService, assert_boundary_choices, bank_cache,
)
import flat_pool_format as fmt  # noqa: E402
import class_mainline_btr as mainline  # noqa: E402
import paper_boundary_origin as origin  # noqa: E402
import paper_dp_merge_native as dp_native  # noqa: E402
import paper_mainline_native as native  # noqa: E402
import paper_xor_change_class as detection  # noqa: E402
from shared_compression_features import FeatureBank  # noqa: E402
from shared_feature_policy import FactoryPolicy  # noqa: E402
from run_verified_compression import checked_model, sha256  # noqa: E402


DEFAULT_INPUT = ROOT / "examples" / "data" / "synthetic_8192.f64le"
VERSION = "cacf_public_strict8_shared_adaptive_example_v2"


def resource_guard(output_parent: Path, minimum_free_gib: float,
                   minimum_headroom_mib: float,
                   *, memory_snapshot: dict | None = None) -> dict:
    """Admission floor only; this does not bound future native peak memory."""
    disk_free = shutil.disk_usage(output_parent).free
    if disk_free < minimum_free_gib * 1024**3:
        raise RuntimeError("free output disk is below the requested safety floor")
    result = {"disk_free_bytes": disk_free, "minimum_free_gib": minimum_free_gib,
              "minimum_headroom_mib": minimum_headroom_mib,
              "peak_memory_guaranteed": False, "cache_purge_used": False}
    memory = _cgroup_memory_snapshot() if memory_snapshot is None else memory_snapshot
    result.update(memory)
    if (memory["available"]
            and memory["cgroup_headroom_bytes"] < minimum_headroom_mib * 1024**2):
        raise RuntimeError("cgroup memory headroom is below the requested safety floor")
    return result


def run_boundary(job: dict, folder: Path, exporter: Path, fixed_exe: Path,
                 args: argparse.Namespace, *, bank: FeatureBank) -> dict:
    """Fresh XOR-change ClaSS seed + original measured candidate/repair code."""
    protected = native._protected(Path(job["input"]), exporter, fixed_exe)
    folder.mkdir()
    mainline.stream_json(folder / "job.json", job)
    detector_args = SimpleNamespace(class_history=args.class_history,
                                    class_subsequence=32, class_jump=5,
                                    progress_every=1000,
                                    class_timeout=args.class_timeout)
    config = detection.detector_config(detector_args)
    with native.single_thread():
        detector = detection.detector_stage(job, folder / "detect_xor_change",
                                            "xor_change", detector_args)
    seed = native._geometry(detection.validate_detector_result(
        detector, job, "xor_change", config), job["values"])
    # Use the SAME authenticated immutable bank later consumed by the selector.
    cache = bank_cache(bank, job)
    values = bank.values
    candidates = build_candidates(cache, seed, mode="union", grid=64, minimum=64,
                                  window=256, maximum=256, random_seed=0)
    mainline.stream_json(folder / "initial_and_candidates.json", dict(
        version=VERSION, initial=seed, candidates=candidates,
        detector_configuration=config, frozen_before_native_trials=True))
    budget = args.boundary_value_budget_multiplier * job["values"]
    service = native._service(job, folder, exporter, fixed_exe, args.timeout, None,
                              actions=DEFAULT_ACTIONS, value_budget=budget,
                              trial_limit=args.boundary_trial_limit,
                              service_factory=StrictEightBoundaryService)
    view = BudgetedRegionView(service, budget, args.boundary_trial_limit)
    evaluator = FlatPoolEvaluator(view)
    with native.single_thread():
        evaluator.evaluate(seed, phase="seed")
        initial_path = folder / "initial.f64pool"
        initial_info, initial_choices = native._assemble(
            service, seed, initial_path, job["values"])
        assert_boundary_choices(service, initial_choices)
        initial_proof, _ = native._verify(
            initial_path, Path(job["input"]), exporter, fixed_exe, args.timeout, None)
        repair = repair_partition(
            seed, candidates, evaluator,
            config=RepairConfig(max_rounds=12, max_proposals_per_round=64,
                                max_split_candidates=6),
            tag="xor_change_repaired", proposal_scorer=CompressionGainProxy(values).gain)
        repaired = native._geometry(repair["intervals"], job["values"])
        repaired_path = folder / "repaired.f64pool"
        repaired_info, repaired_choices = native._assemble(
            service, repaired, repaired_path, job["values"])
        assert_boundary_choices(service, repaired_choices)
        repaired_proof, _ = native._verify(
            repaired_path, Path(job["input"]), exporter, fixed_exe, args.timeout, None)
    origin.validate_repair(repair, seed, repaired,
                           initial_info["complete_file_bits"],
                           repaired_info["complete_file_bits"])
    if any(trial["status"] == "failed" for trial in service.trials):
        raise RuntimeError("a full-pool native boundary trial failed; see boundary/native")
    if (view.counters["trial_count"] != service.counters["trial_count"]
            or view.counters["trial_value_volume"] != service.counters["trial_value_volume"]):
        raise ValueError("boundary trials are not fully charged to the declared budget")
    native._check(protected)
    bank.assert_unchanged()
    mainline.stream_json(folder / "native_trials.json", service.trials)
    result = dict(version=VERSION, job=job,
                  exporter_sha256=protected[str(exporter.resolve())],
                  fixed_exe_sha256=protected[str(fixed_exe.resolve())],
                  source_boundaries_historical=False,
                  fresh_xor_change_detection=True, candidate_mode="union",
                  boundary_actions=list(DEFAULT_ACTIONS),
                  boundary_pool_is_original14=False,
                  strict8_at_all_stages=True,
                  shared_feature_bank_for_boundary_and_selection=True,
                  selector_or_training_used_for_boundary=False,
                  detector=detector, initial_regions=seed, repaired_regions=repaired,
                  candidates=candidates, repair=repair,
                  initial_complete_file_bits=initial_info["complete_file_bits"],
                  repaired_complete_file_bits=repaired_info["complete_file_bits"],
                  initial_container_path=str(initial_path),
                  initial_container_sha256=sha256(initial_path),
                  repaired_container_path=str(repaired_path),
                  repaired_container_sha256=sha256(repaired_path),
                  initial_verification=initial_proof,
                  repaired_verification=repaired_proof,
                  initial_choices=initial_choices, repaired_choices=repaired_choices,
                  initial_actual_physical_codec_counts=initial_info["actual_physical_codec_counts"],
                  repaired_actual_physical_codec_counts=repaired_info["actual_physical_codec_counts"],
                  actual_frame_ids_restricted8=True,
                  boundary_trial_count=view.counters["trial_count"],
                  boundary_trial_value_volume=view.counters["trial_value_volume"],
                  boundary_value_budget=budget,
                  boundary_trial_limit=args.boundary_trial_limit,
                  boundary_native_failures=sum(t["status"] == "failed" for t in service.trials),
                  boundary_unsupported_trials=sum(t["status"] == "unsupported" for t in service.trials),
                  feature_candidate_values_scanned=cache.values_scanned)
    mainline.stream_json(folder / "boundary_manifest.json", result)
    return result


def resume_boundary(job: dict, source: Path, output: Path,
                    exporter: Path, fixed_exe: Path, timeout: int) -> dict:
    """Authenticate a prior *fresh* boundary without writing to its folder.

    The source may be a boundary directory or its boundary_manifest.json. A
    new, independently decoded raw file is retained only under ``output``.
    This does not reuse historical trial prices for DP: DP measures anew.
    """
    manifest_path = (source / "boundary_manifest.json" if source.is_dir() else source)
    manifest_path = manifest_path.resolve(strict=True)
    if manifest_path.name != "boundary_manifest.json" or not manifest_path.is_file():
        raise ValueError("--resume-boundary must identify boundary_manifest.json")
    source_folder = manifest_path.parent
    if output == source_folder or source_folder in output.parents or output in source_folder.parents:
        raise ValueError("new output must not overlap the prior boundary directory")
    before = {"manifest": sha256(manifest_path)}
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if (manifest.get("version") != VERSION
            or manifest.get("source_boundaries_historical") is not False
            or manifest.get("fresh_xor_change_detection") is not True
            or manifest.get("selector_or_training_used_for_boundary") is not False
            or manifest.get("boundary_actions") != list(DEFAULT_ACTIONS)
            or manifest.get("strict8_at_all_stages") is not True
            or manifest.get("shared_feature_bank_for_boundary_and_selection") is not True
            or manifest.get("boundary_native_failures") != 0):
        raise ValueError("prior boundary is not a completed fresh strict-eight shared-bank result")
    detector = manifest.get("detector", {})
    old_job = manifest.get("job")
    if ((old_job is not None and
         (old_job.get("sha256") != job["sha256"] or old_job.get("values") != job["values"]))
            or detector.get("status") != "passed"
            or detector.get("input_sha256") != job["sha256"]
            or detector.get("values") != job["values"]):
        raise ValueError("prior boundary input SHA/count differs from current input")
    if (manifest.get("exporter_sha256") not in (None, sha256(exporter))
            or manifest.get("fixed_exe_sha256") not in (None, sha256(fixed_exe))):
        raise ValueError("prior boundary native executable identity differs")
    initial = native._geometry(manifest["initial_regions"], job["values"])
    repaired = native._geometry(manifest["repaired_regions"], job["values"])
    files = {}
    for stem in ("initial", "repaired"):
        candidate = Path(manifest[stem + "_container_path"]).resolve(strict=True)
        if (candidate.parent != source_folder or candidate.name != stem + ".f64pool"
                or not candidate.is_file()):
            raise ValueError("prior boundary container escaped its source directory")
        digest = sha256(candidate)
        bits = candidate.stat().st_size * 8
        if (digest != manifest[stem + "_container_sha256"]
                or bits != manifest[stem + "_complete_file_bits"]):
            raise ValueError("prior boundary stored bytes differ from manifest")
        # Check the actual persisted frame IDs, not merely its manifest's pool.
        info = fmt.inspect_container(candidate)
        if info["values"] != job["values"]:
            raise ValueError("prior boundary persisted value count differs")
        if (manifest[stem + "_verification"].get("decoded_sha256") != job["sha256"]
                or manifest[stem + "_verification"].get("complete_file_roundtrip_verified") is not True):
            raise ValueError("prior boundary verification claim is incomplete")
        files[stem] = candidate
        before[stem] = digest
    check_choices(manifest.get("initial_choices", ()))
    check_choices(manifest.get("repaired_choices", ()))
    origin.validate_repair(manifest["repair"], initial, repaired,
                           manifest["initial_complete_file_bits"],
                           manifest["repaired_complete_file_bits"])
    trials_path = source_folder / "native_trials.json"
    before["trials"] = sha256(trials_path)
    trials = json.loads(trials_path.read_text(encoding="utf-8"))
    if (not isinstance(trials, list)
            or len(trials) != manifest.get("boundary_trial_count")
            or any(not isinstance(t, dict) or t.get("status") not in ("passed", "unsupported")
                   for t in trials)):
        raise ValueError("prior boundary native-trial ledger differs or contains failures")
    check_trial_ledger(trials)
    del trials
    decoded = output / "resumed_boundary_decoded.f64le"
    proof = fmt.decode_file(files["repaired"], decoded, exporter, fixed_exe,
                            timeout=timeout)
    if (proof.get("decoded_sha256") != job["sha256"]
            or decoded.stat().st_size != 8 * job["values"]
            or sha256(Path(job["input"])) != job["sha256"]
            or sha256(manifest_path) != before["manifest"]
            or any(sha256(files[stem]) != before[stem] for stem in files)
            or sha256(trials_path) != before["trials"]):
        raise ValueError("prior boundary changed or fresh independent decode failed")
    result = dict(version=VERSION, source_boundary_manifest=str(manifest_path),
                  source_boundary_manifest_sha256=before["manifest"],
                  source_boundaries_historical=False,
                  source_native_identity_recorded=(manifest.get("exporter_sha256") is not None
                                                   and manifest.get("fixed_exe_sha256") is not None),
                  initial_regions=initial, repaired_regions=repaired,
                  repaired_complete_file_bits=manifest["repaired_complete_file_bits"],
                  repaired_container_sha256=before["repaired"],
                  independently_decoded_again=True,
                  independent_decoded_sha256=proof["decoded_sha256"],
                  boundary_native_failures=0, boundary_trial_count=manifest["boundary_trial_count"])
    mainline.stream_json(output / "resume_boundary_verification.json", result)
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT,
                        help="headerless little-endian binary64 series (at least 4096 values by default)")
    parser.add_argument("--out", type=Path, required=True,
                        help="fresh output directory; existing paths are never overwritten")
    parser.add_argument("--exporter", type=Path, required=True)
    parser.add_argument("--fixed-btr-exe", type=Path, required=True)
    parser.add_argument("--model", type=Path,
                        help="trusted compact20 joblib; supply together with --model-metadata")
    parser.add_argument("--model-metadata", type=Path,
                        help="trusted compact20 JSON schema and expected model SHA-256")
    parser.add_argument("--dataset", help="original source dataset identity; pair with --series")
    parser.add_argument("--series", help="original source series identity; pair with --dataset")
    parser.add_argument("--resume-boundary", type=Path,
                        help="completed boundary directory or boundary_manifest.json from a prior run")
    parser.add_argument("--class-history", type=int, default=4096)
    parser.add_argument("--class-timeout", type=int, default=1800)
    parser.add_argument("--timeout", type=int, default=300)
    parser.add_argument("--boundary-value-budget-multiplier", type=int, default=128)
    parser.add_argument("--boundary-trial-limit", type=int, default=8192)
    parser.add_argument("--dp-value-budget-multiplier", type=int, default=128)
    parser.add_argument("--dp-trial-limit", type=int, default=8192)
    parser.add_argument("--paper-full-pricing", action="store_true",
                        help="explicitly fund every frozen Top-2 DP edge, matching the paper protocol; "
                             "does not change the boundary-stage budget")
    parser.add_argument("--minimum-free-gib", type=float, default=2.0)
    parser.add_argument("--minimum-headroom-mib", type=float, default=512.0)
    args = parser.parse_args()
    if args.paper_full_pricing and any(
            item == option or item.startswith(option + "=")
            for item in sys.argv[1:]
            for option in ("--dp-value-budget-multiplier", "--dp-trial-limit")):
        raise SystemExit("--paper-full-pricing cannot be combined with capped DP budget options")
    if (args.model is None) != (args.model_metadata is None):
        raise SystemExit("--model and --model-metadata must be supplied together")
    if (args.dataset is None) != (args.series is None) or args.dataset == "" or args.series == "":
        raise SystemExit("--dataset and --series must be nonempty and supplied together")

    raw = args.input.resolve(strict=True)
    exporter = args.exporter.resolve(strict=True)
    fixed_exe = args.fixed_btr_exe.resolve(strict=True)
    out = args.out.resolve()
    if not all(p.is_file() for p in (raw, exporter, fixed_exe)):
        raise SystemExit("input and both native executables must be regular files")
    count, remainder = divmod(raw.stat().st_size, 8)
    if (remainder or not 1 <= count <= 102400 or
            not 32 * 10 < args.class_history <= count):
        raise SystemExit("require 1..102400 binary64 values and 320 < class-history <= values")
    for name in ("class_timeout", "timeout", "boundary_value_budget_multiplier",
                 "boundary_trial_limit", "dp_value_budget_multiplier", "dp_trial_limit"):
        if getattr(args, name) < 1:
            raise SystemExit(name + " must be positive")
    if (args.boundary_value_budget_multiplier > 128
            or args.boundary_trial_limit > 8192
            or args.dp_value_budget_multiplier > 128
            or args.dp_trial_limit > 8192):
        raise SystemExit("this example caps each native stage at 128N values and 8192 trials")
    if (not math.isfinite(args.minimum_free_gib) or args.minimum_free_gib < 2
            or not math.isfinite(args.minimum_headroom_mib)
            or args.minimum_headroom_mib < 512):
        raise SystemExit("resource floors must be at least 2 GiB disk and 512 MiB cgroup headroom")
    if out.exists() or not out.parent.is_dir():
        raise SystemExit("output must be a fresh path inside an existing directory")
    if args.resume_boundary is not None:
        source = args.resume_boundary.resolve(strict=True)
        source_folder = source if source.is_dir() else source.parent
        if out == source_folder or source_folder in out.parents or out in source_folder.parents:
            raise SystemExit("new output must not overlap the prior boundary directory")

    admission = resource_guard(
        out.parent, args.minimum_free_gib, args.minimum_headroom_mib,
        memory_snapshot=EARLY_RESUME_MEMORY if args.resume_boundary is not None else None)
    admission["memory_checked_before_runtime_imports"] = args.resume_boundary is not None
    input_sha = sha256(raw)
    identity = (args.dataset or "CACF_external_demo",
                args.series or "sample_" + input_sha[:12], input_sha)
    job = dict(dataset=identity[0], series=identity[1], sha256=input_sha,
               input=str(raw), values=count)
    out = native._fresh(out, raw, exporter, fixed_exe)
    mainline.stream_json(out / "admission.json", admission)
    model, metadata = checked_model(args.model, args.model_metadata)
    bank = FeatureBank(raw, identity)
    if args.resume_boundary is None:
        boundary = run_boundary(job, out / "boundary", exporter, fixed_exe, args, bank=bank)
        resource_guard(out, args.minimum_free_gib, args.minimum_headroom_mib)
    else:
        boundary = resume_boundary(job, args.resume_boundary, out,
                                   exporter, fixed_exe, args.timeout)

    policy = FactoryPolicy(job, model, metadata, bank, top_k=2)
    if args.resume_boundary is None:
        resource_guard(out, args.minimum_free_gib, args.minimum_headroom_mib)
    dp = dp_native.run_dp_merge(
        job, boundary["repaired_regions"], out / "dp", exporter, fixed_exe,
        policy=policy, timeout=args.timeout,
        value_budget_multiplier=None if args.paper_full_pricing else args.dp_value_budget_multiplier,
        trial_limit=None if args.paper_full_pricing else args.dp_trial_limit,
        minimum_free_gib=args.minimum_free_gib,
        source_boundaries_historical=False)
    if args.paper_full_pricing and not dp["full_pricing_complete"]:
        raise RuntimeError("paper protocol requires complete DP pricing; incomplete evidence is preserved")
    rows = {row["layout"]: row for row in dp["rows"]}
    final = rows.get("after_dp", {})
    if final.get("status") != "passed":
        raise RuntimeError("DP output unavailable; inspect dp/dp_merge_manifest.json; no fallback claimed")
    stored = Path(final["container_path"])
    decoded = out / "decoded.f64le"
    proof = fmt.decode_file(stored, decoded, exporter, fixed_exe, timeout=args.timeout)
    if (not final["complete_file_roundtrip_verified"]
            or final["complete_file_bits"] != stored.stat().st_size * 8
            or proof["decoded_sha256"] != input_sha
            or decoded.stat().st_size != raw.stat().st_size
            or sha256(raw) != input_sha
            or sha256(Path(metadata["model_path"])) != metadata["model_sha256"]):
        raise ValueError("stored-file or independent bitwise verification failed")
    report = dict(version=VERSION, scope="fresh adaptive compression + independent decode",
                  historical_replay=False, paper_timing_claim=False,
                  paper_byte_identity_claim=False,
                  active_actions=list(DEFAULT_ACTIONS), strict8_at_all_stages=True,
                  shared_feature_bank_for_boundary_and_selection=True,
                  boundary_resumed=args.resume_boundary is not None,
                  boundary_source_manifest=boundary.get("source_boundary_manifest"),
                  boundary_source_manifest_sha256=boundary.get("source_boundary_manifest_sha256"),
                  boundary_independently_decoded_on_resume=boundary.get("independently_decoded_again", False),
                  boundary_source_native_identity_recorded=boundary.get(
                      "source_native_identity_recorded", args.resume_boundary is None),
                  input=str(raw), values=count, input_sha256=input_sha,
                  boundary_initial_segments=len(boundary["initial_regions"]),
                  boundary_repaired_segments=len(boundary["repaired_regions"]),
                  boundary_repaired_bits=boundary["repaired_complete_file_bits"],
                  final_segments=final["boundary_segments"],
                  final_complete_file_bits=final["complete_file_bits"],
                  dp_full_pricing_complete=dp["full_pricing_complete"],
                  dp_pricing_mode="paper_full_graph" if args.paper_full_pricing else "capped_demo",
                  dp_budget_limited=not dp["full_pricing_complete"],
                  dp_priced_edges=dp["priced_dp_edges"],
                  dp_possible_edges=dp["possible_dp_edges"],
                  selected_codecs=[x["codec"] for x in dp["selections"]
                                   if x["layout"] == "after_dp"],
                  stored_file=str(stored), stored_sha256=final["container_sha256"],
                  decoded_file=str(decoded), decoded_sha256=proof["decoded_sha256"],
                  bitwise_identical=True, frozen_model_sha256=metadata["model_sha256"],
                  resource_admission=admission)
    with (out / "example_report.json").open("x", encoding="utf-8") as stream:
        json.dump(report, stream, ensure_ascii=False, indent=2)
        stream.write("\n")
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
