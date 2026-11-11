"""Same-run initial-detector versus original compression-repair diagnostic.

A is untouched numeric ClaSS, B repairs that exact same seed, C repairs an
adjacent-XOR-change seed. Every arm pays the same full14 native codec policy.
Boundary-coordinate origins are evidence, not per-boundary causal gains.
"""
import argparse
import csv
from dataclasses import asdict
import json
import math
from pathlib import Path
import subprocess
import sys
import time

from btr_search_ledger import BudgetedRegionView, fixed_partition
from compression_boundary_repair import RepairConfig, repair_partition
from compression_candidates import build_candidates
from compression_gain_proxy import CompressionGainProxy
from flat_pool_backend import DEFAULT_ACTIONS, FlatPoolEvaluator, FlatPoolService
import boundary_origin_report as origin
import class_mainline_btr as mainline
import flat_pool_format as fmt
import frozen_btr_extension as frozen
import paper_intro_motivation as intro
import paper_xor_change_class as detection
import replay_frozen_files as replay

VERSION = "boundary_origin_comparison_v1"
REFERENCE, RAW, REPAIRED, CHANGE = ("fixed1024_reference", "numeric_class_raw",
                                 "numeric_class_repaired", "xor_change_repaired")
METHODS = (REFERENCE, RAW, REPAIRED, CHANGE)
DETECTORS = ("numeric", "xor_change")
METHOD_DETECTOR = {RAW:"numeric", REPAIRED:"numeric", CHANGE:"xor_change"}
REPAIRED_METHODS = (REPAIRED, CHANGE)
CONTRASTS = ((REFERENCE,RAW),(REFERENCE,REPAIRED),(REFERENCE,CHANGE),
             (RAW,REPAIRED),(RAW,CHANGE),(REPAIRED,CHANGE))
CODE_FILES = tuple(dict.fromkeys(detection.CODE_FILES + (Path(__file__).name,
    "boundary_origin_report.py","compression_boundary_repair.py","compression_candidates.py",
    "compression_gain_proxy.py")))
FIELDS = tuple(dict.fromkeys(detection.FIELDS + ("initial_bits","initial_file_roundtrip_verified",
    "initial_container_path","initial_container_sha256","initial_segments","gain_vs_initial_pct",
    "candidate_count","candidate_sha256","termination","budget_limited","detector_time_reused_from",
    "seed_boundary_acceptance_individually_proven")))
COMPARISON_FIELDS = detection.COMPARISON_FIELDS
CANDIDATE_FIELDS = ("position","score","sources","strongest_group","state_contrast")
ORIGIN_COUNTS = ("initial_segments","initial_detector_boundaries","final_segments","final_boundaries",
    "retained_exact_seed_boundaries","untouched_surviving_seed_boundaries","reintroduced_seed_boundaries",
    "new_final_boundaries","removed_seed_boundaries","accepted_operations","total_accepted_gain_bits",
    "detector_only","compression_only","both","nearby_expansion","repair_generated",
    "final_last_introduced_by_split","final_last_introduced_by_pair_split","final_last_introduced_by_merge",
    "final_last_introduced_by_move","category_count_sum")


def protected_sources():
    return {str(Path(__file__).with_name(n).resolve()): replay.sha256(Path(__file__).with_name(n)) for n in CODE_FILES}


def repair_config(args):
    result = RepairConfig(max_rounds=args.max_rounds,max_proposals_per_round=args.max_proposals,
                          max_split_candidates=args.max_split_candidates)
    result.validate()
    if type(args.max_candidates) is not int or args.max_candidates<1 or type(args.window) is not int or args.window<64:
        raise ValueError("positive candidate cap and state window >=64 required")
    return result


def comparisons(rows,job):
    by, result = {r["method"]:r for r in rows}, []
    for control,treatment in CONTRASTS:
        a,b = by.get(control,{}),by.get(treatment,{})
        row = {k:job[k] for k in ("dataset","series","values","sha256")}
        row.update(control=control,treatment=treatment,status="unavailable",control_bits=None,
                   treatment_bits=None,reduction_pct=None,
                   detail="; ".join(r.get("detail","") for r in (a,b) if r.get("detail")))
        if all(r.get("status")=="passed" and r.get("complete_file_roundtrip_verified") is True
               and r.get("pool_native_failures")==0 for r in (a,b)):
            x,y = a["complete_file_bits"],b["complete_file_bits"]
            row.update(status="passed",control_bits=x,treatment_bits=y,reduction_pct=100*(x-y)/x)
        result.append(row)
    return result


def csv_rows(rows):
    return [{k:json.dumps(v,sort_keys=True,separators=(",",":")) if isinstance(v,(list,dict)) else v
             for k,v in row.items()} for row in rows]


def write_rows(path,rows):
    rows = csv_rows(rows)
    fields = tuple(dict.fromkeys(k for row in rows for k in row)) or ("series","method","status")
    frozen.write_csv(path,rows,fields)


def origin_tables(jobs,rows,reports):
    boundaries,operations,summaries = [],[],[]
    by = {(r["series"],r["method"]):r for r in rows}
    for job in jobs:
        for method in METHODS:
            row = by.get((job["series"],method),{})
            context = {k:job[k] for k in ("dataset","series","values","sha256")}
            context.update(method=method,status="not_applicable_reference" if method==REFERENCE else "unavailable")
            report = reports.get(job["series"],{}).get(method)
            if report is None:
                summaries.append(dict(context,**{k:None for k in ORIGIN_COUNTS},detail=row.get("detail","pending or unavailable")))
                continue
            context["status"] = "passed" if row.get("pool_native_failures")==0 else "verified_layout_incomplete_pool"
            summaries.append(dict(context,**report["summary"]))
            boundaries.extend(dict(context,**item) for item in report["boundary_rows"])
            operations.extend(dict(context,**item) for item in report["operation_rows"])
    return boundaries,operations,summaries


def write_origins(folder,jobs,rows,reports,complete):
    boundaries,operations,summaries = origin_tables(jobs,rows,reports)
    mainline.stream_json(Path(folder)/"origin_report.json",dict(version=VERSION,complete=complete,reports=reports,
        scope="semantic final internal boundaries, not physical codec frames or causal gain attribution"))
    for name,data in (("final_boundary_origins.csv",boundaries),("operation_origins.csv",operations),
                      ("origin_summary.csv",summaries)):
        write_rows(Path(folder)/name,data)


def persist(evaluator,regions,folder,job,raw,exporter,fixed_exe,args,execute,protected,backend,initial=False):
    stem = "initial" if initial else "final"
    choices = evaluator.usage_segments(regions)
    detection.validate_choices(choices,job)
    before = (dict(evaluator.view.counters),dict(backend.counters))
    path = folder/(stem+".f64pool")
    frames = [f for item in evaluator.native_segments(regions) for f in item["frames"]]
    info = fmt.write_container(path,frames,job["values"])
    proof = fmt.verify_file(path,raw,exporter,fixed_exe,args.timeout,execute)
    if (info["complete_file_bits"]!=8*(fmt.FILE_HEADER.size+sum(c["frame_bytes"] for c in choices))
            or before!=(dict(evaluator.view.counters),dict(backend.counters))):
        raise ValueError("persistence changed paid prices or repeated encoding")
    prefix = "initial_" if initial else ""
    proof_path,choices_path = folder/(prefix+"verification.json"),folder/(prefix+"codec_choices.json")
    table = folder/(prefix+"boundaries.csv")
    mainline.stream_json(proof_path,proof)
    mainline.stream_json(choices_path,choices)
    frozen.write_csv(table,[dict(start=s,length=n,codec=evaluator.choices[(s,n)]["codec"]) for s,n in regions],
                     ("start","length","codec"))
    for file in (path,proof_path,choices_path,table): protected[str(file.resolve())]=replay.sha256(file)
    return dict(bits=info["complete_file_bits"],path=str(path),sha256=replay.sha256(path))


def validate_repair(repair,initial,final,initial_bits,final_bits):
    if (repair.get("initial_intervals")!=[list(r) for r in initial]
            and repair.get("initial_intervals")!=list(initial)):
        raise ValueError("repair changed the initial seed")
    if ([tuple(r) for r in repair["intervals"]]!=final or repair.get("telescoping_ok") is not True
            or repair.get("dp_used") is not False or repair.get("estimates_used_for_acceptance") is not False
            or repair["initial_bits"]+8*fmt.FILE_HEADER.size!=initial_bits
            or repair["total_bits"]+8*fmt.FILE_HEADER.size!=final_bits or final_bits>initial_bits
            or repair["initial_bits"]-repair["total_bits"]!=sum(o["gain_bits"] for o in repair["operations"])
            or any(o.get("status")!="accepted" or o["gain_bits"]<=0
                   or o["old_bits"]-o["new_bits"]!=o["gain_bits"] for o in repair["operations"])):
        raise ValueError("repair violated measured seed-relative positive-gain accounting")


def run_series(job,out,exporter,fixed_exe,args,*,execute=replay.execute_export,
               detector_runner=detection.detector_stage,backend_factory=FlatPoolService):
    out,raw = Path(out).resolve(),Path(job["input"]).resolve()
    if out.exists(): raise ValueError("fresh per-series output required")
    config,cfg = detection.detector_config(args),repair_config(args)
    protected = protected_sources()
    for path in (raw,Path(exporter).resolve(),Path(fixed_exe).resolve()): protected[str(path)]=replay.sha256(path)
    if any(intro.overlap(out,Path(p)) for p in protected): raise ValueError("output overlaps protected input")
    values = mainline.read_float64(str(raw))
    if protected[str(raw)]!=job["sha256"] or len(values)!=job["values"]: raise ValueError("original input identity differs")
    out.mkdir(parents=True)
    mainline.stream_json(out/"job.json",job)
    protected[str(out/"job.json")]=replay.sha256(out/"job.json")
    rows,detectors,reports,backend = [],{}, {},None
    budget = args.value_budget_multiplier*job["values"]
    state = dict(version=VERSION,job=job,complete=False,status="running",results=rows,detectors=detectors,
        protected_sha256=protected,active_actions=list(DEFAULT_ACTIONS),predeclared_methods=list(METHODS),
        detector_configuration=config,detector_provenance_version=detection.VERSION,repair_config=asdict(cfg),
        candidate_protocol=dict(builder="compression_candidates.build_candidates",mode="union",grid=64,minimum=64,
                                window=args.window,maximum=args.max_candidates,random_seed=0,cap_enlarged=False),
        boundary_search_used=True,selector_prediction_used=False,selector_training_used=False,pool_pruning_used=False,
        original_full_btrblocks_in_pool=False,whole_fallback_used=False,best_seed_competition_used=False,
        each_arm_value_budget=budget,each_arm_trial_limit=args.trial_limit,
        numeric_detection_shared_same_run_between_A_and_B=True,initial_boundaries_individually_accepted=False,
        timing_scope="shared numeric detection/cache; independent virtual charges; NOT compression throughput",
        guarantee="accepted edits improve their own seed only; inherited seed cuts are not individually gain-approved")

    def save():
        mainline.stream_json(out/"series_manifest.json",state)
        frozen.write_csv(out/"summary.csv",rows,FIELDS)
        frozen.write_csv(out/"comparison.csv",comparisons(rows,job),COMPARISON_FIELDS)
        write_origins(out,[job],rows,{job["series"]:reports},state["complete"])

    def measured(command,log,timeout):
        record = dict(event="start",command=list(map(str,command)),log=str(log))
        with (out/"native_commands.jsonl").open("a",encoding="utf-8") as stream: stream.write(json.dumps(record)+"\n")
        began = time.perf_counter()
        try:
            result = execute(command,Path(log),timeout)
            record.update(event="finish",exit_code=result)
            return result
        finally:
            record["wall_seconds"]=time.perf_counter()-began
            with (out/"native_commands.jsonl").open("a",encoding="utf-8") as stream: stream.write(json.dumps(record)+"\n")

    save()
    try:
        for mode in DETECTORS:
            try:
                result = detector_runner(job,out/("detect_"+mode),mode,args)
                detection.validate_detector_result(result,job,mode,config)
                detectors[mode]=result
                for path in (out/("detect_"+mode)).rglob("*"):
                    if path.is_file(): protected[str(path.resolve())]=replay.sha256(path)
            except (OSError,ValueError,KeyError,TypeError,ImportError,subprocess.SubprocessError) as error:
                detectors[mode]=dict(status="failed",detail=str(error))
            save()
        initial = {REFERENCE:fixed_partition(job["values"],1024)}
        candidates = {method:[] for method in METHODS}
        cache = mainline.FeatureCache(values,(job["dataset"],job["series"],job["sha256"]))
        for method,mode in METHOD_DETECTOR.items():
            if detectors[mode].get("status")=="passed":
                initial[method]=[tuple(r) for r in detectors[mode]["intervals"]]
                if method in REPAIRED_METHODS:
                    candidates[method]=build_candidates(cache,initial[method],mode="union",grid=64,minimum=64,
                        window=args.window,maximum=args.max_candidates,random_seed=0)
        for method in METHODS:
            folder=out/method
            folder.mkdir()
            frozen.write_csv(folder/"candidates.csv",candidates[method],CANDIDATE_FIELDS)
            protected[str(folder/"candidates.csv")]=replay.sha256(folder/"candidates.csv")
        plan=out/"frozen_initial_layouts.json"
        mainline.stream_json(plan,dict(version=VERSION,job=job,initial=initial,candidates=candidates,
            detector_configuration=config,candidate_protocol=state["candidate_protocol"],active_actions=list(DEFAULT_ACTIONS),
            layouts_frozen_before_native_trials=True))
        protected[str(plan)]=replay.sha256(plan)
        state["frozen_initial_layouts_sha256"]=protected[str(plan)]
        state["feature_candidate_values_scanned"]=cache.values_scanned
        save()
        proxy=CompressionGainProxy(values)
        backend=backend_factory(dict(n=job["values"],raw=raw,key=(job["dataset"],job["series"],job["sha256"],VERSION)),
            out/"native",exporter,fixed_exe,args.timeout,value_budget=4*budget,trial_limit=4*args.trial_limit,
            execute=measured,actions=DEFAULT_ACTIONS)
        for method in METHODS:
            folder=out/method
            row=dict(job,method=method,status="failed",detail="",complete_file_bits=None,
                complete_file_roundtrip_verified=False,initial_file_roundtrip_verified=False,accepted_operations=0,
                pool_native_failures=0,candidate_count=len(candidates[method]),
                candidate_sha256=protected[str(folder/"candidates.csv")],seed_boundary_acceptance_individually_proven=False)
            rows.append(row)
            view=BudgetedRegionView(backend,budget,args.trial_limit)
            evaluator=FlatPoolEvaluator(view)
            try:
                if method not in initial: raise ValueError("required detector failed; no fallback")
                regions=initial[method]
                row["initial_segments"]=len(regions)
                mode=METHOD_DETECTOR.get(method)
                row["detector_seconds_including_jit"]=detectors[mode]["segmentation_seconds"] if mode else 0
                row["detector_time_reused_from"]="same-run detect_numeric shared with A" if method==REPAIRED else ""
                evaluator.evaluate(regions,phase="seed")
                seed=persist(evaluator,regions,folder,job,raw,exporter,fixed_exe,args,measured,protected,backend,True)
                row.update(initial_bits=seed["bits"],initial_container_path=seed["path"],
                    initial_container_sha256=seed["sha256"],initial_file_roundtrip_verified=True)
                repair=None
                final_regions=list(regions)
                if method in REPAIRED_METHODS:
                    repair=repair_partition(regions,candidates[method],evaluator,config=cfg,tag=method,proposal_scorer=proxy.gain)
                    mainline.stream_json(folder/"repair.json",repair)
                    protected[str(folder/"repair.json")]=replay.sha256(folder/"repair.json")
                    final_regions=[tuple(r) for r in repair["intervals"]]
                final=persist(evaluator,final_regions,folder,job,raw,exporter,fixed_exe,args,measured,protected,backend)
                if repair is not None: validate_repair(repair,regions,final_regions,seed["bits"],final["bits"])
                elif seed!=dict(bits=final["bits"],path=seed["path"],sha256=final["sha256"]):
                    raise ValueError("untouched layout changed its priced bytes")
                if method!=REFERENCE:
                    report=origin.build_origin_report(regions,final_regions,candidates[method],
                        repair["operations"] if repair else [],job["values"],mode+"_class")
                    reports[method]=report
                    mainline.stream_json(folder/"origin_report.json",report)
                    protected[str(folder/"origin_report.json")]=replay.sha256(folder/"origin_report.json")
                row.update(status="passed",complete_file_bits=final["bits"],container_path=final["path"],
                    container_sha256=final["sha256"],complete_file_roundtrip_verified=True,boundary_segments=len(final_regions),
                    accepted_operations=len(repair["operations"]) if repair else 0,
                    gain_vs_initial_pct=100*(seed["bits"]-final["bits"])/seed["bits"],
                    termination=repair["termination"] if repair else "no_repair",budget_limited=repair["budget_limited"] if repair else False)
            except (OSError,ValueError,KeyError,TypeError) as error:
                row.update(detail=str(error),complete_file_bits=None,complete_file_roundtrip_verified=False)
                reports.pop(method,None)
            finally:
                trials=[backend.cache[k] for k in view.charged_keys if k in backend.cache]
                row.update(trial_count=view.counters["trial_count"],trial_value_volume=view.counters["trial_value_volume"],
                    pool_native_failures=sum(t["status"]=="failed" for t in trials),
                    pool_unsupported_trials=sum(t["status"]=="unsupported" for t in trials))
                mainline.stream_json(folder/"ledger.json",dict(counters=dict(view.counters),charged_keys=sorted(view.charged_keys),
                    value_budget=budget,trial_limit=args.trial_limit,active_actions=list(DEFAULT_ACTIONS)))
                protected[str(folder/"ledger.json")]=replay.sha256(folder/"ledger.json")
            intro.check_protected(protected)
            save()
            print(job["series"],method,row["status"],row.get("initial_bits"),row.get("complete_file_bits"),row["detail"],flush=True)
        by={r["method"]:r for r in rows}
        if all(by[m]["status"]=="passed" for m in (RAW,REPAIRED)):
            if (by[RAW]["complete_file_bits"]!=by[REPAIRED]["initial_bits"]
                    or by[RAW]["container_sha256"]!=by[REPAIRED]["initial_container_sha256"]):
                raise ValueError("A final and B initial are not exactly the same same-run bytes")
        state.update(complete=True,protected_files_unchanged=True,status="passed" if all(r["status"]=="passed"
            and r["pool_native_failures"]==0 for r in rows) else "completed_with_unavailable_cases")
    except BaseException as error:
        state.update(complete=False,status="failed",detail=str(error))
        reports.clear()
        for row in rows: row.update(status="invalidated_failed_run",complete_file_bits=None,complete_file_roundtrip_verified=False)
        raise
    finally:
        if backend is not None:
            mainline.stream_json(out/"native_trials.json",backend.trials)
            protected[str(out/"native_trials.json")]=replay.sha256(out/"native_trials.json")
            for trial in backend.trials:
                if trial["status"]=="passed": protected[trial["candidate_path"]]=trial["candidate_sha256"]
        save()
        if state["complete"]:
            for name in ("origin_report.json","origin_summary.csv","final_boundary_origins.csv","operation_origins.csv"):
                protected[str(out/name)]=replay.sha256(out/name)
            mainline.stream_json(out/"series_manifest.json",state)
    return rows


def check_csv(path,rows):
    rows=csv_rows(rows)
    fields=tuple(dict.fromkeys(k for r in rows for k in r)) or ("series","method","status")
    with Path(path).open(newline="",encoding="utf-8-sig") as stream:
        reader=csv.DictReader(stream)
        actual=list(reader)
        if reader.fieldnames!=list(fields): raise ValueError("origin CSV columns differ from verified report")
    expected=[{k:"" if row.get(k) is None else str(row.get(k,"")) for k in fields} for row in rows]
    if actual!=expected: raise ValueError("origin CSV differs from verified semantic report")


def validate_saved(folder,row,job,expected,initial=False):
    prefix="initial_" if initial else ""
    key="initial_container_path" if initial else "container_path"
    sha="initial_container_sha256" if initial else "container_sha256"
    bits="initial_bits" if initial else "complete_file_bits"
    path=Path(row[key]).resolve()
    proof=frozen.read_json(folder/(prefix+"verification.json"))
    choices=json.loads((folder/(prefix+"codec_choices.json")).read_text(encoding="utf-8"))
    if not isinstance(choices,list): raise ValueError("codec choices must be list")
    with (folder/(prefix+"boundaries.csv")).open(newline="",encoding="utf-8-sig") as stream:
        table=list(csv.DictReader(stream))
    if ([(s["start"],s["length"]) for s in choices]!=expected
            or [(int(s["start"]),int(s["length"])) for s in table]!=expected
            or [s["codec"] for s in choices]!=[s["codec"] for s in table]):
        raise ValueError("persisted semantic geometry differs from recorded layout")
    detection.validate_choices(choices,job)
    info=fmt.inspect_container(path)
    if (path!=folder/("initial.f64pool" if initial else "final.f64pool")
            or replay.sha256(path)!=row[sha] or info["values"]!=job["values"]
            or info["complete_file_bits"]!=row[bits]
            or 8*(fmt.FILE_HEADER.size+sum(c["frame_bytes"] for c in choices))!=row[bits]
            or row.get("initial_file_roundtrip_verified" if initial else "complete_file_roundtrip_verified") is not True
            or proof.get("status")!="passed" or proof.get("complete_file_roundtrip_verified") is not True
            or proof.get("decoded_from_persisted_file_only") is not True
            or proof.get("source_sha256")!=job["sha256"] or proof.get("decoded_sha256")!=job["sha256"]
            or proof.get("decoded_values")!=job["values"] or proof.get("container_sha256")!=row[sha]
            or proof.get("complete_file_bits")!=row[bits]):
        raise ValueError("saved initial/final bytes or persisted-only decode proof differs")
    return choices


def validate_paid_options(choices,charged,by_trial,job):
    for choice in choices:
        start,length=choice["start"],choice["length"]
        for option in choice["options"]:
            codec=option["codec"]
            if codec.startswith("alp+tail:"):
                prefix=length//1024*1024
                expected=[(start,prefix,"alp"),(start+prefix,length-prefix,codec.split(":",1)[1])]
            elif codec=="alp" and length<1024: expected=[]
            else: expected=[(start,length,codec)]
            keys=[tuple(k) for k in option.get("physical_keys",[])]
            if keys!=expected or any(k not in charged or k not in by_trial for k in keys):
                raise ValueError("codec option borrowed unpaid/different-region physical trials")
            trials=[by_trial[k] for k in keys]
            statuses=[t["status"] for t in trials]
            status="failed" if "failed" in statuses else "unsupported" if "unsupported" in statuses or not keys else "passed"
            if (status!=option["status"] or any(t.get("sha256")!=job["sha256"] for t in trials)
                    or status=="passed" and (any(t.get("roundtrip_verified") is not True for t in trials)
                        or sum(t["frame_bytes"] for t in trials)!=option["frame_bytes"])):
                raise ValueError("semantic codec option differs from its independently paid physical trials")


def collect(job,folder,rc):
    folder=Path(folder).resolve()
    state=frozen.read_json(folder/"series_manifest.json")
    if (rc not in (0,1) or state.get("version")!=VERSION or state.get("complete") is not True
            or state.get("protected_files_unchanged") is not True or state.get("job")!=job
            or state.get("active_actions")!=list(DEFAULT_ACTIONS) or state.get("predeclared_methods")!=list(METHODS)
            or state.get("detector_provenance_version")!=detection.VERSION
            or state.get("boundary_search_used") is not True
            or state.get("numeric_detection_shared_same_run_between_A_and_B") is not True
            or state.get("initial_boundaries_individually_accepted") is not False
            or any(state.get(k) is not False for k in ("selector_prediction_used","selector_training_used",
                "pool_pruning_used","original_full_btrblocks_in_pool","whole_fallback_used","best_seed_competition_used"))
            or set(state.get("detectors",{}))!=set(DETECTORS)):
        raise ValueError("worker lacks declared same-run boundary-origin protocol")
    intro.check_protected(state["protected_sha256"])
    cfg=RepairConfig(**state["repair_config"])
    cfg.validate()
    if cfg.grid!=64 or cfg.min_segment!=64: raise ValueError("repair grid/minimum changed")
    plan_path=folder/"frozen_initial_layouts.json"
    plan=frozen.read_json(plan_path)
    if (replay.sha256(plan_path)!=state["frozen_initial_layouts_sha256"] or plan.get("version")!=VERSION
            or plan.get("job")!=job or plan.get("active_actions")!=list(DEFAULT_ACTIONS)
            or plan.get("layouts_frozen_before_native_trials") is not True
            or plan.get("candidate_protocol")!=state["candidate_protocol"]
            or plan.get("detector_configuration")!=state["detector_configuration"]):
        raise ValueError("pre-price initial layout/candidate plan changed")
    initial={REFERENCE:fixed_partition(job["values"],1024)}
    for method,mode in METHOD_DETECTOR.items():
        result=state["detectors"][mode]
        if result.get("status")=="passed":
            initial[method]=detection.validate_detector_result(result,job,mode,state["detector_configuration"])
    if plan["initial"]!=json.loads(json.dumps(initial)) or set(plan["candidates"])!=set(METHODS):
        raise ValueError("seed geometry differs from actual detected points")
    if plan["candidates"][RAW] or plan["candidates"][REFERENCE]:
        raise ValueError("unrepaired A/reference must not acquire post-hoc feature nominations")
    trials=json.loads((folder/"native_trials.json").read_text(encoding="utf-8"))
    if not isinstance(trials,list): raise ValueError("physical trials must be list")
    by_trial={(t["start"],t["length"],t["codec"]):t for t in trials}
    if len(by_trial)!=len(trials): raise ValueError("duplicate physical trials")
    rows=state["results"]
    if [r["method"] for r in rows]!=list(METHODS): raise ValueError("unexpected/missing worker methods")
    reports={}
    for row in rows:
        method=row["method"]
        sub=folder/method
        if any(row[k]!=job[k] for k in ("dataset","series","values","sha256")):
            raise ValueError("worker input identity differs")
        if (row.get("seed_boundary_acceptance_individually_proven") is not False
                or row["candidate_count"]!=len(plan["candidates"][method])
                or row["candidate_sha256"]!=replay.sha256(sub/"candidates.csv")):
            raise ValueError("candidate count/hash or inherited seed claim differs")
        with (sub/"candidates.csv").open(newline="",encoding="utf-8-sig") as stream:
            table=[dict(position=int(r["position"]),score=float(r["score"]),sources=r["sources"],
                        strongest_group=r["strongest_group"],state_contrast=float(r["state_contrast"])) for r in csv.DictReader(stream)]
        if table!=plan["candidates"][method]: raise ValueError("candidate table differs from pre-price plan")
        if row["status"]!="passed":
            if row.get("complete_file_bits") is not None: raise ValueError("failed arm has usable final size")
            continue
        if method not in initial: raise ValueError("successful arm without authentic detector seed")
        seed=initial[method]
        if row["initial_segments"]!=len(seed): raise ValueError("initial segment count differs")
        seed_choices=validate_saved(sub,row,job,seed,True)
        final_table=json.loads((sub/"codec_choices.json").read_text(encoding="utf-8"))
        final=[(r["start"],r["length"]) for r in final_table]
        final_choices=validate_saved(sub,row,job,final)
        if row["boundary_segments"]!=len(final): raise ValueError("final segment count differs")
        operations=[]
        if method in REPAIRED_METHODS:
            repair=frozen.read_json(sub/"repair.json")
            validate_repair(repair,seed,final,row["initial_bits"],row["complete_file_bits"])
            operations=repair["operations"]
        elif final!=seed or row["complete_file_bits"]!=row["initial_bits"] or row["container_sha256"]!=row["initial_container_sha256"]:
            raise ValueError("A/reference layout was repaired or changed")
        if row["accepted_operations"]!=len(operations): raise ValueError("accepted operation count differs")
        if method!=REFERENCE:
            report=origin.build_origin_report(seed,final,plan["candidates"][method],operations,job["values"],
                                               METHOD_DETECTOR[method]+"_class")
            if frozen.read_json(sub/"origin_report.json")!=report: raise ValueError("origin report disagrees with replayed operations")
            reports[method]=report
        ledger=frozen.read_json(sub/"ledger.json")
        keys=ledger["charged_keys"]
        validate_paid_options(seed_choices,{tuple(k) for k in keys},by_trial,job)
        validate_paid_options(final_choices,{tuple(k) for k in keys},by_trial,job)
        charged=[by_trial.get(tuple(k)) for k in keys]
        if (ledger["active_actions"]!=list(DEFAULT_ACTIONS) or len({tuple(k) for k in keys})!=len(keys)
                or any(t is None for t in charged) or any(k[2] not in DEFAULT_ACTIONS for k in keys)
                or row["trial_count"]!=len(keys) or row["trial_value_volume"]!=sum(k[1] for k in keys)
                or row["trial_count"]>ledger["trial_limit"] or row["trial_value_volume"]>ledger["value_budget"]
                or ledger["trial_limit"]!=state["each_arm_trial_limit"] or ledger["value_budget"]!=state["each_arm_value_budget"]
                or row["pool_native_failures"]!=sum(t["status"]=="failed" for t in charged)
                or row["pool_unsupported_trials"]!=sum(t["status"]=="unsupported" for t in charged)):
            raise ValueError("arm full-pool work/coverage ledger differs")
    by={r["method"]:r for r in rows}
    if all(by[m]["status"]=="passed" for m in (RAW,REPAIRED)):
        if (by[RAW]["complete_file_bits"]!=by[REPAIRED]["initial_bits"]
                or by[RAW]["container_sha256"]!=by[REPAIRED]["initial_container_sha256"]):
            raise ValueError("A final differs from B initial; not the same seed/pool/prices")
    bundle=frozen.read_json(folder/"origin_report.json")
    recorded={job["series"]:reports}
    if bundle.get("version")!=VERSION or bundle.get("complete") is not True or bundle.get("reports")!=recorded:
        raise ValueError("per-series origin aggregate differs")
    boundaries,operations,summaries=origin_tables([job],rows,recorded)
    for name,data in (("final_boundary_origins.csv",boundaries),("operation_origins.csv",operations),("origin_summary.csv",summaries)):
        check_csv(folder/name,data)
    return rows,state["protected_sha256"],reports


def batch(args,*,executor=None):
    detection.detector_config(args)
    repair_config(args)
    args.btr_exe=args.fixed_btr_exe
    if not (Path(args.source_config).resolve().parent/"baselines/data_manifest.json").is_file():
        raise ValueError("preserved certified baselines/data_manifest.json required")
    jobs,protected,exporter,fixed_exe=intro.preflight(args)
    if any(job["values"]<args.class_history for job in jobs): raise ValueError("complete input shorter than detector warmup")
    protected.update(protected_sources())
    out=Path(args.out_dir).resolve()
    out.mkdir(parents=True,exist_ok=True)
    (out/"logs").mkdir()
    rows,children,reports=[],[],{}
    state=dict(version=VERSION,complete=False,status="running",jobs=jobs,results=rows,children=children,
        protected_sha256=protected,configuration=vars(args),active_actions=list(DEFAULT_ACTIONS),
        predeclared_methods=list(METHODS),selector_training_used=False,pool_pruning_used=False,
        detector_provenance_version=detection.VERSION)
    def save():
        mainline.stream_json(out/"manifest.json",state)
        frozen.write_csv(out/"summary.csv",rows,FIELDS)
        frozen.write_csv(out/"comparison.csv",[c for job in jobs for c in comparisons([r for r in rows if r["series"]==job["series"]],job)],COMPARISON_FIELDS)
        write_origins(out,jobs,rows,reports,state["complete"])
    save()
    for index,job in enumerate(jobs,1):
        folder,log=out/job["series"],out/"logs"/(job["series"]+".log")
        command=[sys.executable,"-u",str(Path(__file__).resolve()),"--worker","--source-config",args.source_config,
            "--data-repo",args.data_repo,"--out-dir",str(folder),"--series",job["series"],"--exporter",str(exporter),
            "--fixed-btr-exe",str(fixed_exe)]
        for key in ("class_history","class_subsequence","class_jump","progress_every","class_timeout","timeout",
                    "value_budget_multiplier","trial_limit","max_rounds","max_proposals","max_split_candidates","max_candidates","window"):
            command.extend(["--"+key.replace("_","-"),str(getattr(args,key))])
        child=dict(series=job["series"],status="running",command=command,log=str(log))
        children.append(child)
        save()
        print("===== %s/%s %s: numeric A / same-seed repaired B / XOR-change repaired C ====="%(index,len(jobs),job["series"]),flush=True)
        try:
            if executor is None:
                with log.open("w",encoding="utf-8") as stream:
                    rc=subprocess.run(command,stdout=stream,stderr=subprocess.STDOUT,timeout=args.worker_timeout).returncode
            else: rc=executor(command,log)
            result,proof,recorded=collect(job,folder,rc)
            for path,digest in proof.items():
                if path in protected and protected[path]!=digest: raise ValueError("worker changed protected input")
            protected.update(proof)
            rows.extend(result)
            reports[job["series"]]=recorded
            child.update(exit_code=rc,status="passed" if all(r["status"]=="passed" and not r["pool_native_failures"] for r in result)
                         else "completed_with_unavailable_cases")
        except (OSError,ValueError,KeyError,TypeError,ImportError,subprocess.SubprocessError) as error:
            child.update(status="failed",detail=str(error))
            rows.extend(dict(job,method=method,status="failed",detail=str(error),complete_file_bits=None,
                             complete_file_roundtrip_verified=False) for method in METHODS)
        try: intro.check_protected(protected)
        except ValueError:
            state.update(complete=False,status="failed")
            reports.clear()
            for row in rows: row.update(status="protected_input_changed",complete_file_bits=None,complete_file_roundtrip_verified=False)
            save()
            raise
        save()
    state.update(complete=True,protected_files_unchanged=True,status="passed" if all(c["status"]=="passed" for c in children)
                 else "completed_with_unavailable_cases")
    save()
    return int(state["status"]!="passed")


def parser():
    result=argparse.ArgumentParser(description=__doc__)
    for key in ("source-config","data-repo","exporter","fixed-btr-exe"): result.add_argument("--"+key,required=True)
    result.add_argument("--out-dir",required=True)
    result.add_argument("--series",nargs="+")
    result.add_argument("--worker",action="store_true")
    for key,value in (("class-history",4096),("class-subsequence",32),("class-jump",5),("progress-every",1000),
                      ("value-budget-multiplier",128),("trial-limit",8192),("max-rounds",12),("max-proposals",64),
                      ("max-split-candidates",6),("max-candidates",256),("window",256)):
        result.add_argument("--"+key,type=int,default=value)
    for key,value in (("timeout",300),("class-timeout",1800),("worker-timeout",14400)):
        result.add_argument("--"+key,type=float,default=value)
    result.set_defaults(native_manifest=None,max_numeric_boundaries=64,numeric_threshold=1.)
    return result


def main():
    cli=parser()
    args=cli.parse_args()
    try:
        detection.detector_config(args)
        repair_config(args)
        if args.progress_every<0 or any(not math.isfinite(getattr(args,k)) or getattr(args,k)<=0 for k in ("timeout","class_timeout","worker_timeout")):
            raise ValueError("finite positive timeouts and nonnegative progress-every required")
        args.btr_exe=args.fixed_btr_exe
        if args.worker:
            jobs,_,exporter,fixed_exe=intro.preflight(args)
            if len(jobs)!=1: raise ValueError("worker requires exactly one original series")
            rows=run_series(jobs[0],args.out_dir,exporter,fixed_exe,args)
            return int(any(r["status"]!="passed" or r["pool_native_failures"] for r in rows))
        return batch(args)
    except (OSError,ValueError,KeyError,TypeError,ImportError,subprocess.SubprocessError) as error:
        cli.exit(2,VERSION+": "+str(error)+"\n")


if __name__=="__main__":
    raise SystemExit(main())
