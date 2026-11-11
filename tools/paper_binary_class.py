"""Detector-only comparison: numeric ClaSS, IEEE64 Hamming ClaSS and union.

The author F1/profile/validation is retained. Only the binary arm's nearest
neighbour metric changes. Detected layouts are compressed without any repair,
feature candidates, trained selector, seed competition or pool pruning.
"""
import argparse
import csv
import json
import math
from pathlib import Path
import subprocess
import sys
import time

from btr_search_ledger import BudgetedRegionView, fixed_partition
import class_mainline_btr as mainline
from flat_pool_backend import DEFAULT_ACTIONS, STREAM_CODECS, FlatPoolEvaluator, FlatPoolService
import flat_pool_format as fmt
import frozen_btr_extension as frozen
import paper_flat_pool as base
import paper_intro_motivation as intro
import replay_frozen_files as replay
import segment_class as raw_class

VERSION = "binary_class_detector_comparison_v1"
REFERENCE, NUMERIC, BINARY, UNION = "fixed1024_reference", "numeric_class", "binary_class", "union_class"
METHODS = (REFERENCE, NUMERIC, BINARY, UNION)
CODE_FILES = tuple(dict.fromkeys(base.CODE_FILES + (Path(__file__).name, "class_bit_hamming.py")))
FIELDS = ("dataset", "series", "values", "sha256", "method", "status", "detail", "complete_file_bits",
    "complete_file_roundtrip_verified", "container_path", "container_sha256", "boundary_segments",
    "detector_seconds_including_jit", "accepted_operations", "trial_value_volume", "trial_count",
    "pool_native_failures", "pool_unsupported_trials")
COMPARISON_FIELDS = ("dataset", "series", "values", "sha256", "control", "treatment", "status",
    "control_bits", "treatment_bits", "reduction_pct", "detail")
CONTRASTS = ((REFERENCE, NUMERIC), (REFERENCE, BINARY), (REFERENCE, UNION), (NUMERIC, BINARY), (NUMERIC, UNION))


def protected_sources():
    return {str(Path(__file__).with_name(n).resolve()): replay.sha256(Path(__file__).with_name(n)) for n in CODE_FILES}


def detector_config(args):
    if (type(args.class_history) is not int or type(args.class_subsequence) is not int
            or args.class_subsequence < 2 or args.class_history <= 10*args.class_subsequence
            or type(args.class_jump) is not int or args.class_jump < 1):
        raise ValueError("require history > 10*subsequence >=20 and jump >=1")
    return dict(n_timepoints=args.class_history, n_warmup=args.class_history, window_size=args.class_subsequence,
        k_neighbours=3, distance="znormed_euclidean_distance", score="f1", jump=args.class_jump,
        validation="significance_test", threshold="default", excl_radius=5)


def union_regions(left, right, n):
    # Both inputs have already been snapped by the same declared rule. No
    # score ranking, point cap, cheap proxy or codec cost enters this union.
    def points(regions):
        cursor, found = 0, []
        for start, length in regions:
            if type(start) is not int or type(length) is not int or start != cursor or length < 1:
                raise ValueError("invalid detected interval geometry")
            found.append(start)
            cursor += length
        if cursor != n or not found:
            raise ValueError("detected layout must cover all original values")
        return found[1:]
    return raw_class.normalize_boundaries(sorted(set(points(left)+points(right))), n, 64, 64)


def validate_detector_result(state, job, mode, config):
    metric="binary64_subsequence_hamming" if mode=="binary" else "znormed_euclidean_distance"
    resolved=state.get("resolved_configuration",{})
    if (state.get("version")!=VERSION or state.get("complete") is not True or state.get("status")!="passed"
            or state.get("mode")!=mode or state.get("input_sha256")!=job["sha256"]
            or state.get("values")!=job["values"] or state.get("requested_configuration")!=config
            or state.get("codec_costs_or_labels_used") is not False or state.get("actual_metric")!=metric
            or state.get("source_unchanged") is not True or state.get("input_values_transformed") is not False
            or state.get("binary_distance_uses_exact_bit_view")!=(mode=="binary")
            or state.get("dependency_versions",{}).get("claspy")!=raw_class.SUPPORTED_CLASPY
            or state.get("snap_grid")!=64 or state.get("min_segment")!=64
            or state.get("terminal_policy")!="no synthetic flush; append exact original end"
            or resolved.get("distance")!=metric or resolved.get("threshold")!=1e-50
            or any(resolved.get(k)!=v for k,v in config.items() if k not in ("distance","threshold"))):
        raise ValueError("incomplete/wrong detector provenance")
    events=state.get("events")
    if not isinstance(events,list) or any(not isinstance(e,dict) or type(e.get("change_point")) is not int
            or type(e.get("detected_after_values")) is not int
            or not 0<e["change_point"]<=e["detected_after_values"]<=job["values"] for e in events):
        raise ValueError("invalid global detection events")
    expected=raw_class.normalize_boundaries([e["change_point"] for e in events],job["values"],64,64)
    if [list(r) for r in expected]!=state.get("intervals"):
        # JSON converts tuples to lists; accept an in-memory actual adapter too.
        if [list(r) for r in expected]!=[list(r) for r in state.get("intervals",[])]:
            raise ValueError("detector geometry differs from recorded events")
    if not math.isfinite(state["segmentation_seconds"]) or state["segmentation_seconds"]<0:
        raise ValueError("invalid detection time")
    return expected


def comparisons(rows, job):
    by, results = {r["method"]: r for r in rows}, []
    for control, treatment in CONTRASTS:
        a, b = by.get(control, {}), by.get(treatment, {})
        result = dict((k, job[k]) for k in ("dataset", "series", "values", "sha256"))
        result.update(control=control, treatment=treatment, status="unavailable", control_bits=None,
            treatment_bits=None, reduction_pct=None, detail="; ".join(r.get("detail", "") for r in (a,b) if r.get("detail")))
        if all(r.get("status")=="passed" and r.get("complete_file_roundtrip_verified") is True
               and r.get("pool_native_failures")==0 for r in (a,b)):
            x, y = a["complete_file_bits"], b["complete_file_bits"]
            result.update(status="passed", control_bits=x, treatment_bits=y, reduction_pct=100*(x-y)/x)
        results.append(result)
    return results


def validate_choices(choices, job):
    for choice in choices:
        length=choice["length"]
        expected=set(DEFAULT_ACTIONS)-{"alp"}
        if length>=1024 and length%1024:
            expected.update("alp+tail:"+a for a in STREAM_CODECS)
        else: expected.add("alp")
        options=choice["options"]
        if (choice.get("options_complete") is not True or len(options)!=len(expected)
                or {o["codec"] for o in options}!=expected):
            raise ValueError("missing/duplicate exhaustive codec option")
        for option in options:
            if option.get("sha256")!=job["sha256"] or option["status"] not in ("passed","unsupported","failed"):
                raise ValueError("codec option lacks source-bound completed status")
            if option["status"]=="passed":
                if (option.get("roundtrip_verified") is not True or type(option.get("frame_bytes")) is not int
                        or option["frame_bytes"]<=0):
                    raise ValueError("codec option lacks verified full frame bytes")
            elif option.get("frame_bytes") is not None:
                raise ValueError("invalid codec option has a usable size")
        feasible=[o for o in options if o["status"]=="passed"]
        if not feasible or not any(o["codec"]==choice["codec"] and o["frame_bytes"]==choice["frame_bytes"] for o in feasible):
            raise ValueError("selected codec has no matching verified option")
        if choice["frame_bytes"]!=min(o["frame_bytes"] for o in feasible):
            raise ValueError("final choice is not the full-pool measured minimum")


def detect(job, out, mode, args):
    """Fresh process per detector; no warm kernel inherited from the other arm."""
    out = Path(out).resolve()
    if out.exists():
        raise ValueError("detector output must be fresh")
    config = detector_config(args)
    values = raw_class.read_float64(job["input"])
    if len(values)!=job["values"] or replay.sha256(Path(job["input"]))!=job["sha256"]:
        raise ValueError("detector input identity differs")
    if len(values)<config["n_warmup"]:
        raise ValueError("complete input is shorter than the declared detector warmup; do not pad it")
    if mode == "numeric":
        events, resolved, elapsed, versions = raw_class.segment_values(values, config, args.progress_every)
        metric = config["distance"]
    elif mode == "binary":
        from class_bit_hamming import segment_bit_events
        events, resolved, elapsed, versions = segment_bit_events(values, config, args.progress_every)
        metric = "binary64_subsequence_hamming"
    else:
        raise ValueError("unknown detector")
    intervals = raw_class.normalize_boundaries([e["change_point"] for e in events],len(values),64,64)
    state = dict(version=VERSION, complete=True, status="passed", mode=mode, actual_metric=metric,
        input_sha256=job["sha256"], values=len(values), source_unchanged=True, requested_configuration=config,
        resolved_configuration=resolved, dependency_versions=versions, events=events,
        intervals=intervals, segmentation_seconds=elapsed, codec_costs_or_labels_used=False,
        snap_grid=64, min_segment=64, terminal_policy="no synthetic flush; append exact original end",
        input_values_transformed=False, binary_distance_uses_exact_bit_view=(mode=="binary"))
    if replay.sha256(Path(job["input"]))!=job["sha256"]:
        raise ValueError("detector input changed during processing")
    out.mkdir(parents=True)
    frozen.write_csv(out/"raw_change_points.csv",events,("change_point","detected_after_values"))
    frozen.write_csv(out/"boundaries.csv",[dict(start=s,length=n) for s,n in intervals],("start","length"))
    mainline.stream_json(out/"detector_manifest.json",state)
    print("%s: %s segments, %.2fs including JIT"%(mode,len(intervals),elapsed),flush=True)
    return state


def detector_stage(job, folder, mode, args):
    command = [sys.executable,"-u",str(Path(__file__).resolve()),"--detect-mode",mode,
        "--job-file",str(Path(folder).parent/"job.json"),"--out-dir",str(folder),
        "--class-history",str(args.class_history),"--class-subsequence",str(args.class_subsequence),
        "--class-jump",str(args.class_jump),"--progress-every",str(args.progress_every)]
    mainline.stream_json(Path(folder).parent/(mode+"_command.json"),dict(command=command))
    began = time.perf_counter()
    with Path(folder).with_suffix(".log").open("w",encoding="utf-8") as log:
        process = subprocess.run(command,stdout=log,stderr=subprocess.STDOUT,timeout=args.class_timeout)
    if process.returncode:
        raise ValueError("%s detector exit=%s; see %s.log"%(mode,process.returncode,folder))
    state = frozen.read_json(Path(folder)/"detector_manifest.json")
    validate_detector_result(state,job,mode,detector_config(args))
    state["fresh_subprocess_wall_seconds"] = time.perf_counter()-began
    return state


def run_series(job, out, exporter, fixed_exe, args, *, execute=replay.execute_export,
               detector_runner=detector_stage, backend_factory=FlatPoolService):
    out, raw = Path(out).resolve(), Path(job["input"]).resolve()
    if out.exists(): raise ValueError("per-series output must be new")
    config = detector_config(args)
    protected = protected_sources()
    protected.update({str(Path(p).resolve()):replay.sha256(Path(p)) for p in (raw,exporter,fixed_exe)})
    if any(intro.overlap(out,Path(p)) for p in protected): raise ValueError("output overlaps protected source")
    if protected[str(raw)]!=job["sha256"] or raw.stat().st_size!=8*job["values"]:
        raise ValueError("input identity differs")
    out.mkdir(parents=True)
    mainline.stream_json(out/"job.json",job)
    protected[str(out/"job.json")]=replay.sha256(out/"job.json")
    rows, detectors, backend = [], {}, None
    budget = args.value_budget_multiplier*job["values"]
    state = dict(version=VERSION,job=job,complete=False,status="running",results=rows,detectors=detectors,
        active_actions=list(DEFAULT_ACTIONS),predeclared_methods=list(METHODS),configuration=config,
        boundary_search_used=False,feature_candidate_builder_used=False,selector_training_used=False,
        selector_prediction_used=False,pool_pruning_used=False,original_full_btrblocks_in_pool=False,
        protected_sha256=protected,each_arm_value_budget=budget,each_arm_trial_limit=args.trial_limit,
        union_compute_policy="both detector costs charged; not equal compute to one detector",
        detection_timing_scope="fresh subprocess per detector; includes own JIT; no throughput claim",
        encoding_timing_scope="shared physical cache; independent virtual charges; not throughput",
        binary_interpretation="Hamming similarity of aligned raw IEEE64 subsequences; not a compression-cost model")

    def save():
        mainline.stream_json(out/"series_manifest.json",state)
        frozen.write_csv(out/"summary.csv",rows,FIELDS)
        frozen.write_csv(out/"comparison.csv",comparisons(rows,job),COMPARISON_FIELDS)

    def measured(command, log, timeout):
        record = dict(event="start",command=list(map(str,command)),log=str(log))
        with (out/"native_commands.jsonl").open("a",encoding="utf-8") as stream:
            stream.write(json.dumps(record)+"\n")
        began = time.perf_counter()
        try:
            rc = execute(command,Path(log),timeout)
            record.update(event="finish",exit_code=rc)
            return rc
        finally:
            record["wall_seconds"] = time.perf_counter()-began
            with (out/"native_commands.jsonl").open("a",encoding="utf-8") as stream:
                stream.write(json.dumps(record)+"\n")

    save()
    try:
        for mode in ("numeric","binary"):
            print(job["series"]+"/"+mode+": fresh detection",flush=True)
            try:
                result = detector_runner(job,out/("detect_"+mode),mode,args)
                validate_detector_result(result,job,mode,config)
                detectors[mode]=result
                for path in (out/("detect_"+mode)).rglob("*"):
                    if path.is_file(): protected[str(path.resolve())]=replay.sha256(path)
            except (OSError,ValueError,KeyError,TypeError,subprocess.SubprocessError) as error:
                detectors[mode]=dict(status="failed",detail=str(error))
            save()
        layouts = {REFERENCE:fixed_partition(job["values"],1024)}
        for mode,method in (("numeric",NUMERIC),("binary",BINARY)):
            if detectors[mode].get("status")=="passed": layouts[method]=[tuple(r) for r in detectors[mode]["intervals"]]
        if NUMERIC in layouts and BINARY in layouts:
            layouts[UNION]=union_regions(layouts[NUMERIC],layouts[BINARY],job["values"])
            raw_points = {s for s,_ in layouts[NUMERIC][1:]}
            bit_points = {s for s,_ in layouts[BINARY][1:]}
            state["boundary_overlap"]=dict(numeric=len(raw_points),binary=len(bit_points),
                shared=len(raw_points&bit_points),numeric_only=len(raw_points-bit_points),binary_only=len(bit_points-raw_points))
        plan=out/"frozen_layouts.json"
        mainline.stream_json(plan,dict(version=VERSION,job=job,active_actions=list(DEFAULT_ACTIONS),
            layouts=layouts,detector_config=config,boundary_repair_used=False))
        protected[str(plan.resolve())]=replay.sha256(plan)
        state["frozen_layouts_sha256"]=protected[str(plan.resolve())]
        save()
        backend=backend_factory(dict(n=job["values"],raw=raw,key=(job["dataset"],job["series"],job["sha256"],VERSION)),
            out/"native",exporter,fixed_exe,args.timeout,value_budget=4*budget,
            trial_limit=4*args.trial_limit,execute=measured,actions=DEFAULT_ACTIONS)
        for method in METHODS:
            folder=out/method
            folder.mkdir()
            row=dict(job,method=method,status="failed",detail="",complete_file_bits=None,
                complete_file_roundtrip_verified=False,accepted_operations=0,pool_native_failures=0)
            rows.append(row)
            view=BudgetedRegionView(backend,budget,args.trial_limit)
            evaluator=FlatPoolEvaluator(view)
            try:
                if method not in layouts:
                    raise ValueError("required detector failed; no fallback: "+"; ".join(
                        m+": "+d.get("detail","unavailable") for m,d in detectors.items() if d.get("status")!="passed"))
                regions=layouts[method]
                row["boundary_segments"]=len(regions)
                required = () if method==REFERENCE else ("numeric",) if method==NUMERIC else ("binary",) if method==BINARY else ("numeric","binary")
                row["detector_seconds_including_jit"]=sum(detectors[m]["segmentation_seconds"] for m in required)
                evaluator.evaluate(regions,phase="seed")
                selected=evaluator.usage_segments(regions)
                validate_choices(selected,job)
                before=(dict(view.counters),dict(backend.counters))
                frames=[f for segment in evaluator.native_segments(regions) for f in segment["frames"]]
                final=folder/"final.f64pool"
                info=fmt.write_container(final,frames,job["values"])
                proof=fmt.verify_file(final,raw,exporter,fixed_exe,args.timeout,measured)
                if info["complete_file_bits"]!=8*fmt.FILE_HEADER.size+sum(evaluator.costs[r] for r in regions):
                    raise ValueError("full stored cost differs from actual frame costs")
                if before!=(dict(view.counters),dict(backend.counters)):
                    raise ValueError("persistence repeated encoding trials")
                mainline.stream_json(folder/"verification.json",proof)
                mainline.stream_json(folder/"codec_choices.json",selected)
                frozen.write_csv(folder/"boundaries.csv",[dict(start=s,length=n,codec=evaluator.choices[(s,n)]["codec"])
                    for s,n in regions],("start","length","codec"))
                for path in (final,folder/"verification.json",folder/"codec_choices.json",folder/"boundaries.csv"):
                    protected[str(path.resolve())]=replay.sha256(path)
                row.update(status="passed",complete_file_bits=info["complete_file_bits"],
                    complete_file_roundtrip_verified=True,container_path=str(final),container_sha256=replay.sha256(final))
            except (OSError,ValueError,KeyError,TypeError) as error:
                row.update(detail=str(error),complete_file_bits=None,complete_file_roundtrip_verified=False)
            finally:
                trials=[backend.cache[k] for k in view.charged_keys if k in backend.cache]
                row.update(trial_count=view.counters["trial_count"],trial_value_volume=view.counters["trial_value_volume"],
                    pool_native_failures=sum(t["status"]=="failed" for t in trials),
                    pool_unsupported_trials=sum(t["status"]=="unsupported" for t in trials))
                mainline.stream_json(folder/"ledger.json",dict(counters=dict(view.counters),charged_keys=sorted(view.charged_keys),
                    value_budget=budget,trial_limit=args.trial_limit,active_actions=list(DEFAULT_ACTIONS)))
                protected[str((folder/"ledger.json").resolve())]=replay.sha256(folder/"ledger.json")
            intro.check_protected(protected)
            save()
            print(job["series"],method,row["status"],row.get("complete_file_bits"),row["detail"],flush=True)
        state.update(complete=True,protected_files_unchanged=True,status="passed" if all(r["status"]=="passed"
            and r["pool_native_failures"]==0 for r in rows) else "completed_with_unavailable_cases")
    except BaseException as error:
        state.update(status="failed",detail=str(error),complete=False)
        for row in rows: row.update(status="invalidated_failed_run",complete_file_bits=None,complete_file_roundtrip_verified=False)
        raise
    finally:
        if backend is not None:
            mainline.stream_json(out/"native_trials.json",backend.trials)
            protected[str((out/"native_trials.json").resolve())]=replay.sha256(out/"native_trials.json")
        save()
    return rows


def collect(job,folder,rc):
    folder=Path(folder).resolve()
    state=frozen.read_json(folder/"series_manifest.json")
    if (rc not in (0,1) or state.get("version")!=VERSION or state.get("complete") is not True
            or state.get("protected_files_unchanged") is not True or state.get("job")!=job
            or state.get("predeclared_methods")!=list(METHODS) or state.get("active_actions")!=list(DEFAULT_ACTIONS)
            or any(state.get(k) is not False for k in ("boundary_search_used","selector_training_used",
                "selector_prediction_used","pool_pruning_used","feature_candidate_builder_used","original_full_btrblocks_in_pool"))):
        raise ValueError("worker did not finish the declared detector-only comparison")
    intro.check_protected(state["protected_sha256"])
    plan_path=folder/"frozen_layouts.json"
    if replay.sha256(plan_path)!=state["frozen_layouts_sha256"]:
        raise ValueError("pre-trial frozen layouts changed")
    plan=frozen.read_json(plan_path)
    if (plan.get("version")!=VERSION or plan.get("job")!=job or plan.get("active_actions")!=list(DEFAULT_ACTIONS)
            or plan.get("boundary_repair_used") is not False):
        raise ValueError("wrong frozen layout provenance")
    expected_layouts={REFERENCE:fixed_partition(job["values"],1024)}
    for mode,method in (("numeric",NUMERIC),("binary",BINARY)):
        detector=state["detectors"][mode]
        if detector.get("status")=="passed":
            expected_layouts[method]=validate_detector_result(detector,job,mode,state["configuration"])
    if NUMERIC in expected_layouts and BINARY in expected_layouts:
        expected_layouts[UNION]=union_regions(expected_layouts[NUMERIC],expected_layouts[BINARY],job["values"])
    if plan["layouts"]!=json.loads(json.dumps(expected_layouts)):
        raise ValueError("pre-trial layout differs from detector events")
    rows=state["results"]
    if [r["method"] for r in rows]!=list(METHODS): raise ValueError("missing/duplicate worker methods")
    trials=json.loads((folder/"native_trials.json").read_text(encoding="utf-8"))
    if not isinstance(trials,list): raise ValueError("physical trials must be a list")
    by_trial={(t["start"],t["length"],t["codec"]):t for t in trials}
    if len(by_trial)!=len(trials): raise ValueError("duplicate physical trial")
    for row in rows:
        if any(row[k]!=job[k] for k in ("dataset","series","values","sha256")) or row["accepted_operations"]!=0:
            raise ValueError("worker identity or search protocol differs")
        if row["status"]=="passed":
            method=row["method"]
            sub=folder/method
            path=Path(row["container_path"]).resolve()
            proof=frozen.read_json(sub/"verification.json")
            choices=json.loads((sub/"codec_choices.json").read_text(encoding="utf-8"))
            if not isinstance(choices,list): raise ValueError("codec choices must be a complete segment list")
            ledger=frozen.read_json(sub/"ledger.json")
            expected=expected_layouts.get(method)
            with (sub/"boundaries.csv").open(newline="",encoding="utf-8-sig") as stream:
                table=list(csv.DictReader(stream))
            if (expected is None or [(r["start"],r["length"]) for r in choices]!=expected
                    or [(int(r["start"]),int(r["length"])) for r in table]!=expected
                    or [r["codec"] for r in choices]!=[r["codec"] for r in table]
                    or row["boundary_segments"]!=len(expected)):
                raise ValueError("saved codec choice geometry differs from frozen layout")
            validate_choices(choices,job)
            if (row.get("complete_file_roundtrip_verified") is not True or replay.sha256(path)!=row["container_sha256"]
                    or path!=sub/"final.f64pool" or path.stat().st_size*8!=row["complete_file_bits"]
                    or fmt.inspect_container(path)["values"]!=job["values"]
                    or 8*(fmt.FILE_HEADER.size+sum(s["frame_bytes"] for s in choices))!=row["complete_file_bits"]
                    or proof.get("status")!="passed" or proof.get("complete_file_roundtrip_verified") is not True
                    or proof.get("decoded_from_persisted_file_only") is not True
                    or proof.get("source_sha256")!=job["sha256"] or proof.get("decoded_sha256")!=job["sha256"]
                    or proof.get("decoded_values")!=job["values"] or proof.get("container_sha256")!=row["container_sha256"]
                    or proof.get("complete_file_bits")!=row["complete_file_bits"]):
                raise ValueError("worker saved-file proof differs")
            if (ledger["active_actions"]!=list(DEFAULT_ACTIONS) or row["trial_count"]!=len(ledger["charged_keys"])
                    or row["trial_value_volume"]!=sum(k[1] for k in ledger["charged_keys"])
                    or row["trial_count"]>ledger["trial_limit"] or row["trial_value_volume"]>ledger["value_budget"]
                    or ledger["value_budget"]!=state["each_arm_value_budget"]
                    or ledger["trial_limit"]!=state["each_arm_trial_limit"]
                    or any(k[2] not in DEFAULT_ACTIONS for k in ledger["charged_keys"])):
                raise ValueError("worker work ledger differs from declared full-pool budget")
            charged=[by_trial.get(tuple(k)) for k in ledger["charged_keys"]]
            if (any(t is None for t in charged) or len({tuple(k) for k in ledger["charged_keys"]})!=len(charged)
                    or row["pool_native_failures"]!=sum(t["status"]=="failed" for t in charged)
                    or row["pool_unsupported_trials"]!=sum(t["status"]=="unsupported" for t in charged)):
                raise ValueError("worker failure/coverage counts differ from actual charged trials")
        elif row.get("complete_file_bits") is not None: raise ValueError("failed worker row has usable size")
    return rows,state["protected_sha256"]


def batch(args, *, executor=None):
    detector_config(args)
    args.btr_exe=args.fixed_btr_exe
    if not (Path(args.source_config).resolve().parent/"baselines/data_manifest.json").is_file():
        raise ValueError("preserved certified baselines/data_manifest.json is required")
    jobs,protected,exporter,fixed_exe=intro.preflight(args)
    if any(j["values"]<args.class_history for j in jobs):
        raise ValueError("all complete inputs must cover the declared detector warmup; no padding/truncation")
    protected.update(protected_sources())
    out=Path(args.out_dir).resolve()
    out.mkdir(parents=True,exist_ok=True)
    (out/"logs").mkdir()
    rows,children=[],[]
    state=dict(version=VERSION,complete=False,status="running",jobs=jobs,results=rows,children=children,
        protected_sha256=protected,configuration=vars(args),active_actions=list(DEFAULT_ACTIONS),
        boundary_search_used=False,selector_training_used=False,pool_pruning_used=False)
    def save():
        mainline.stream_json(out/"manifest.json",state)
        frozen.write_csv(out/"summary.csv",rows,FIELDS)
        frozen.write_csv(out/"comparison.csv",[c for job in jobs for c in comparisons(
            [r for r in rows if r["series"]==job["series"]],job)],COMPARISON_FIELDS)
    save()
    for index,job in enumerate(jobs,1):
        folder,log=out/job["series"],out/"logs"/(job["series"]+".log")
        cmd=[sys.executable,"-u",str(Path(__file__).resolve()),"--worker","--source-config",args.source_config,
            "--data-repo",args.data_repo,"--out-dir",str(folder),"--series",job["series"],
            "--exporter",str(exporter),"--fixed-btr-exe",str(fixed_exe)]
        for name in ("class_history","class_subsequence","class_jump","progress_every","class_timeout",
                     "value_budget_multiplier","trial_limit","timeout"):
            cmd.extend(["--"+name.replace("_","-"),str(getattr(args,name))])
        child=dict(series=job["series"],status="running",command=cmd,log=str(log))
        children.append(child)
        save()
        print("===== %s/%s %s; fresh numeric/binary detection, no repair ====="%(index,len(jobs),job["series"]),flush=True)
        try:
            if executor is None:
                with log.open("w",encoding="utf-8") as stream:
                    rc=subprocess.run(cmd,stdout=stream,stderr=subprocess.STDOUT,timeout=args.worker_timeout).returncode
            else: rc=executor(cmd,log)
            result,proof=collect(job,folder,rc)
            for path,digest in proof.items():
                if path in protected and protected[path]!=digest: raise ValueError("worker changed protected input")
            protected.update(proof)
            rows.extend(result)
            child.update(exit_code=rc,status="passed" if all(r["status"]=="passed" and not r["pool_native_failures"]
                for r in result) else "completed_with_unavailable_cases")
        except (OSError,ValueError,KeyError,TypeError,subprocess.SubprocessError) as error:
            child.update(status="failed",detail=str(error))
            rows.extend(dict(job,method=m,status="failed",detail=str(error),complete_file_bits=None,
                complete_file_roundtrip_verified=False,accepted_operations=0) for m in METHODS)
        try: intro.check_protected(protected)
        except ValueError:
            for row in rows: row.update(status="protected_input_changed",complete_file_bits=None,complete_file_roundtrip_verified=False)
            state.update(status="failed",complete=False)
            save()
            raise
        save()
        print("===== completed %s: %s ====="%(job["series"],child["status"]),flush=True)
    state.update(complete=True,protected_files_unchanged=True,status="passed" if all(c["status"]=="passed" for c in children)
        else "completed_with_unavailable_cases")
    save()
    print("Comparison: "+str(out/"comparison.csv"),flush=True)
    return int(state["status"]!="passed")


def parser():
    p=argparse.ArgumentParser(description=__doc__)
    for name in ("source-config","data-repo","exporter","fixed-btr-exe","job-file"):
        p.add_argument("--"+name)
    p.add_argument("--out-dir",required=True)
    p.add_argument("--series",nargs="+")
    p.add_argument("--worker",action="store_true")
    p.add_argument("--detect-mode",choices=("numeric","binary"))
    for name,value in (("class-history",4096),("class-subsequence",32),("class-jump",5),("progress-every",1000),
                       ("value-budget-multiplier",128),("trial-limit",8192)):
        p.add_argument("--"+name,type=int,default=value)
    for name,value in (("timeout",300),("class-timeout",1800),("worker-timeout",7200)):
        p.add_argument("--"+name,type=float,default=value)
    p.set_defaults(max_rounds=12,max_proposals=64,max_split_candidates=6,max_candidates=256,
        window=256,max_numeric_boundaries=64,numeric_threshold=1.,native_manifest=None)
    return p


def main():
    cli=parser()
    args=cli.parse_args()
    try:
        detector_config(args)
        if args.progress_every<0 or any(not math.isfinite(getattr(args,k)) or getattr(args,k)<=0
                for k in ("timeout","class_timeout","worker_timeout")):
            raise ValueError("positive finite timeouts and nonnegative progress-every required")
        if args.detect_mode:
            if not args.job_file or args.worker: raise ValueError("detector requires job-file, not worker")
            detect(frozen.read_json(args.job_file),args.out_dir,args.detect_mode,args)
            return 0
        if args.job_file: raise ValueError("job-file is private to the detector subprocess")
        if any(not getattr(args,k) for k in ("source_config","data_repo","exporter","fixed_btr_exe")):
            raise ValueError("source-config, data-repo, exporter and fixed-btr-exe are required")
        args.btr_exe=args.fixed_btr_exe
        if args.worker:
            jobs,_,exporter,fixed_exe=intro.preflight(args)
            if len(jobs)!=1: raise ValueError("worker requires one series")
            rows=run_series(jobs[0],args.out_dir,exporter,fixed_exe,args)
            return int(any(r["status"]!="passed" or r["pool_native_failures"] for r in rows))
        return batch(args)
    except (OSError,ValueError,KeyError,TypeError,ImportError,subprocess.SubprocessError) as error:
        cli.exit(2,VERSION+": "+str(error)+"\n")


if __name__=="__main__":
    raise SystemExit(main())
