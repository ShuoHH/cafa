"""Paid native trials over explicit fixed pipelines, with a compact disk cache.

There is no original BtrBlocks action, recursive codec chooser, or output
fallback. ALP tails are explicitly enumerated outer-pool pipelines. Temporary
slices and decoded buffers are removed only inside new trial scratch trees;
the exact priced compressed bytes, hashes, statuses and logs are retained.
"""
import hashlib
from pathlib import Path
import struct
import tempfile

import btr_boundary_costs as native
import btr_mixed_file as legacy
import class_native_evaluator as original_evaluator
import frozen_btr_extension as frozen
import replay_frozen_files as replay
import flat_pool_format as fmt
from codec_pool import (RETAINED_ACTIONS, RETAINED_STREAM_ACTIONS,
                        RETAINED_PHYSICAL_IDS, check_physical_keys)

STREAM_CODECS = RETAINED_STREAM_ACTIONS
DEFAULT_ACTIONS = RETAINED_ACTIONS


def validate_actions(actions):
    actions = tuple(actions)
    if not actions or len(actions) != len(set(actions)) or any(a not in DEFAULT_ACTIONS for a in actions):
        raise ValueError("declare unique strict-eight flat-pool actions; excluded codecs are not actions")
    return actions


class FlatPoolService(native.NativeRegionService):
    def __init__(self, *args, actions=DEFAULT_ACTIONS, **kwargs):
        self.actions = validate_actions(actions)
        super().__init__(*args, **kwargs)
        self.payloads = self.folder/"payloads"
        self.logs = self.folder/"logs"
        self.payloads.mkdir()
        self.logs.mkdir()

    def _action_keys(self, region):
        validate_actions(self.actions)
        start, length = self._region(region)
        options = []
        for action in self.actions:
            if action != "alp":
                options.append((action, ((start, length, action),)))
                continue
            full = length//1024*1024
            if full == length:
                options.append(("alp", ((start, full, "alp"),)))
            elif full:
                for tail in self.actions:
                    if tail in STREAM_CODECS:
                        options.append(("alp+tail:"+tail,
                            ((start, full, "alp"), (start+full, length-full, tail))))
        return options

    def _needed(self, region):
        return list(dict.fromkeys(key for _, keys in self._action_keys(region) for key in keys))

    def _run_trial(self, key):
        # Reject an injected/resumed key BEFORE creating logs or invoking native code.
        validate_actions(self.actions)
        check_physical_keys((key,))
        start, length, codec = key
        self._region((start, length))
        if codec not in self.actions:
            raise ValueError("physical trial escaped the service's declared strict-eight subset")
        serial = len(self.trials)
        target = self.payloads/("%06d%s" % (serial, ".bfx" if codec in fmt.FIXED_ACTIONS else ".f64blk"))
        log = self.logs/("%06d.log" % serial)
        trial = dict(start=start, length=length, codec=codec, status="failed", detail="",
            candidate_path=str(target), sha256=self.source_hash, roundtrip_verified=False,
            frame_bytes=None, executable_sha256=self.tool_hashes[str(self.btr if codec in fmt.FIXED_ACTIONS else self.exporter)],
            attempted=True, intermediate_inputs_ephemeral=True, exact_compressed_payload_retained=True)
        try:
            with tempfile.TemporaryDirectory(prefix="trial_", dir=self.folder) as temporary:
                scratch = Path(temporary)
                sliced = scratch/"slice.raw"
                with self.raw.open("rb") as source:
                    source.seek(start*8)
                    data = source.read(length*8)
                if len(data) != length*8:
                    raise ValueError("source interval was truncated")
                sliced.write_bytes(data)
                digest = replay.sha256(sliced)
                trial["slice_sha256"] = digest
                if codec in fmt.FIXED_ACTIONS:
                    rc = self.execute([str(self.btr), "--encode", codec, str(sliced), str(target)], log, self.timeout)
                    trial["exit_code"] = rc
                    if rc == 3:
                        trial.update(status="unsupported", detail="fixed native pipeline not applicable; no substitution")
                        if target.exists():
                            raise ValueError("unsupported native action unexpectedly published a payload")
                    elif rc:
                        raise ValueError("fixed native encode failed, exit=%s" % rc)
                    else:
                        payload = target.read_bytes()
                        if len(payload) < 24:
                            raise ValueError("truncated fixed native file")
                        magic, version, action_id, n = struct.unpack(">8sIIQ", payload[:24])
                        if (magic, version, action_id, n) != (b"BTRFIX01", 1, fmt.FIXED_IDS[codec], length):
                            raise ValueError("fixed native action/count differs from requested pipeline")
                        decoded, decode_log = scratch/"decoded.raw", scratch/"decode.log"
                        rc = self.execute([str(self.btr), "--decode", str(target), str(decoded)], decode_log, self.timeout)
                        with log.open("a", encoding="utf-8") as stream:
                            stream.write("\nINDEPENDENT DECODE\n"+decode_log.read_text(encoding="utf-8"))
                        if rc or not decoded.is_file() or decoded.stat().st_size != 8*length or replay.sha256(decoded) != digest:
                            raise ValueError("fixed native persisted file did not restore all original bits")
                        frames = [dict(codec_id=fmt.FIXED_IDS[codec], count=length, start=0, payload=payload)]
                        trial.update(status="passed", roundtrip_verified=True,
                            frame_bytes=frozen.frame_cost(frames), physical_frames=1)
                else:
                    plan, report = scratch/"plan.csv", scratch/"report.json"
                    identity = (self.job["key"][0], self.job["key"][1], digest, "region_trial")
                    rows = [dict(zip(replay.IDENTITY, identity), start=0, length=length, codec=codec)]
                    frozen.write_csv(plan, rows, replay.PLAN_FIELDS)
                    rc = self.execute([str(self.exporter), str(sliced), str(plan), identity[3], str(target), str(report)], log, self.timeout)
                    trial["exit_code"] = rc
                    if rc:
                        raise ValueError("legacy native encode failed, exit=%s" % rc)
                    replay.validate_export(report, target, sliced, identity, rows, length)
                    decoded = legacy.decode_file(target, scratch/"decoded.raw", scratch/"decode_work",
                        self.exporter, self.btr, self.timeout, self.execute)
                    if decoded["decoded_sha256"] != digest:
                        raise ValueError("legacy persisted decoder differs from source bits")
                    _, frames = legacy.read_frames(target)
                    if any(f.get("codec_id") not in RETAINED_PHYSICAL_IDS for f in frames):
                        raise ValueError("native trial emitted a pool-external physical codec ID")
                    trial.update(status="passed", roundtrip_verified=True,
                        frame_bytes=frozen.frame_cost(frames), physical_frames=len(frames))
                if replay.sha256(sliced) != digest:
                    raise ValueError("native codec changed its original input")
                if trial["status"] == "passed":
                    trial["candidate_sha256"] = replay.sha256(target)
        except (OSError, ValueError, KeyError, TypeError, struct.error) as error:
            trial.update(status="failed", detail=str(error), frame_bytes=None, roundtrip_verified=False)
        self._check_inputs()
        self.cache[key] = trial
        self.trials.append(trial)
        self.counters["failed_trials"] += trial["status"] == "failed"

    def _frames(self, key):
        validate_actions(self.actions)
        check_physical_keys((key,))
        trial = self.cache[key]
        if (key[2] not in self.actions or
                tuple(trial.get(k) for k in ("start", "length", "codec")) != tuple(key)):
            raise ValueError("paid strict-eight trial identity changed")
        if trial["status"] != "passed":
            return None
        frozen.assert_hash(trial["candidate_path"], trial["candidate_sha256"], "paid fixed-pool payload")
        start, length, codec = key
        if codec in fmt.FIXED_ACTIONS:
            return [dict(codec_id=fmt.FIXED_IDS[codec], count=length, start=start,
                         payload=Path(trial["candidate_path"]).read_bytes())]
        info, frames = legacy.read_frames(trial["candidate_path"])
        if info["values"] != length:
            raise ValueError("paid legacy coverage changed")
        if any(f.get("codec_id") not in RETAINED_PHYSICAL_IDS for f in frames):
            raise ValueError("paid payload contains a pool-external physical codec ID")
        return [dict(f, start=start+f["start"]) for f in frames]

    def options_for(self, region):
        result = []
        for action, keys in self._action_keys(region):
            trials = [self.cache.get(key) for key in keys]
            statuses = [t["status"] if t else "pending" for t in trials]
            status = ("failed" if "failed" in statuses else "pending" if "pending" in statuses
                      else "unsupported" if "unsupported" in statuses else "passed")
            result.append(dict(codec=action, status=status, sha256=self.source_hash,
                frame_bytes=sum(t["frame_bytes"] for t in trials) if status == "passed" else None,
                roundtrip_verified=status == "passed", physical_keys=[list(k) for k in keys]))
        if "alp" in self.actions and not any(o["codec"] == "alp" or o["codec"].startswith("alp+tail:") for o in result):
            result.append(dict(codec="alp", status="unsupported", sha256=self.source_hash,
                frame_bytes=None, roundtrip_verified=False, physical_keys=[],
                detail="no aligned prefix or no predeclared streaming tail"))
        return result

    def _best(self, region):
        options = [o for o in self.options_for(region) if o["status"] == "passed"]
        if not options:
            return None
        chosen = min(options, key=lambda o: o["frame_bytes"])
        frames = [f for k in chosen["physical_keys"] for f in self._frames(tuple(k))]
        return dict(start=region[0], length=region[1], codec=chosen["codec"], frames=frames,
                    physical_keys=[tuple(k) for k in chosen["physical_keys"]])


def signature(item):
    value = hashlib.sha256()
    for f in item["frames"]:
        value.update(struct.pack(">QIIQ", f["start"], f["count"], f["codec_id"], len(f["payload"])))
        value.update(f["payload"])
    return value.hexdigest()


class FlatPoolEvaluator(original_evaluator.NativeRegionEvaluator):
    cost_source = "actual_flat_fixed_pipeline_frames"

    def evaluate(self, intervals, *, phase="trial"):
        if phase not in ("seed", "trial"):
            raise ValueError("unknown evaluator phase")
        regions = self._intervals(intervals)
        missing = [r for r in regions if r not in self.costs]
        if missing:
            if not self.can_evaluate(missing):
                raise ValueError("flat-pool request exceeds its independent budget")
            chosen = self.view.best_many(missing)
            if chosen is None:
                raise ValueError("no lossless fixed-pool action for a requested region")
            for region, item in zip(missing, chosen):
                # Validate the new IDs without widening old format validators.
                relative = [dict(f, start=f["start"]-region[0]) for f in item["frames"]]
                fmt.checked_frames(relative, region[1])
                keys = tuple(item["physical_keys"])
                if any(k not in self.view.charged_keys for k in keys):
                    raise ValueError("chosen fixed pipeline contains unpaid native work")
                bits = frozen.frame_cost(item["frames"])*8
                self.costs[region] = bits
                self.choices[region] = dict(start=region[0], length=region[1], codec=item["codec"],
                    physical_keys=keys, signature=signature(item), bits=bits)
        return {r: self.costs[r] for r in regions}

    def native_segments(self, intervals):
        selected = []
        for region in self._intervals(intervals):
            choice = self.choices[region]
            frames = [f for key in choice["physical_keys"] for f in self.service._frames(key)]
            item = dict(start=region[0], length=region[1], codec=choice["codec"], frames=frames)
            if signature(item) != choice["signature"] or frozen.frame_cost(frames)*8 != self.costs[region]:
                raise ValueError("priced fixed bytes changed before persistence")
            selected.append(item)
        return selected

    def usage_segments(self, regions):
        return [dict(start=s, length=n, codec=self.choices[(s,n)]["codec"],
            frame_bytes=self.costs[(s,n)]//8, options_complete=True,
            options=self.service.options_for((s,n))) for s,n in regions]
