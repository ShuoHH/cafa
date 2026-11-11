"""Budgeted, persisted native region trials for Btr-aware boundary search.

No predicted price authorizes a move. Each unique (source, start, length,
backend) is encoded at most once in this service, including failed attempts.
The returned frames contain those exact measured bytes, never re-encodings.
Every physical frame header is charged; an ALP prefix has an explicit cold
streaming tail. This is an expensive trial-compression diagnostic, not a
prediction-only selector or an end-to-end throughput benchmark.
"""

import math
from pathlib import Path

import btr_mixed_file as bridge
import frozen_btr_extension as frozen
import replay_frozen_files as replay
from external_file_format import inspect_container


STREAM_CODECS = tuple(name for name in replay.CODEC_IDS if name != "alp")


def _integer(value, name, minimum=0):
    if type(value) is not int or value < minimum:
        raise ValueError(name + " must be a strict integer >= " + str(minimum))
    return value


class NativeRegionService:
    """Atomic reservation of the full declared codec pool for each operation.

    Budgets measure unique attempted codec/value volume, not completed native
    work. Existing seeds are reused without encoding and their historical work
    is explicitly outside this budget. Each seed gets an independent service.
    """

    def __init__(self, job, folder, exporter, btr, timeout, value_budget,
                 trial_limit, execute=replay.execute_export):
        self.job = job
        self.n = _integer(job["n"], "values", 1)
        self.raw = Path(job["raw"]).resolve()
        self.exporter, self.btr = Path(exporter).resolve(), Path(btr).resolve()
        self.source_hash = replay.digest_string(job["key"][2], "source sha256")
        if self.raw.stat().st_size != 8 * self.n:
            raise ValueError("raw size differs from source count")
        frozen.assert_hash(self.raw, self.source_hash, "raw source")
        if (isinstance(timeout, bool) or not isinstance(timeout, (int, float))
                or not math.isfinite(timeout) or timeout <= 0):
            raise ValueError("timeout must be a finite positive number")
        self.timeout = timeout
        self.value_budget = _integer(value_budget, "value budget", 1)
        self.trial_limit = _integer(trial_limit, "trial limit", 1)
        self.folder = Path(folder).resolve()
        # Never create a cache in the data/tool directories or overwrite a run.
        for path in (self.raw, self.exporter, self.btr):
            if replay.inside(path, self.folder) or replay.inside(self.folder, path.parent):
                raise ValueError("trial folder overlaps raw input or native tools")
        self.folder.mkdir(parents=True, exist_ok=False)
        self.btr_cache = self.folder / "btr_trials"
        self.old_cache = self.folder / "legacy_trials"
        self.btr_cache.mkdir()
        self.old_cache.mkdir()
        self.execute = execute
        self.tool_hashes = {str(path): replay.sha256(path)
                            for path in (self.exporter, self.btr)}
        self.cache, self.trials = {}, []
        self.counters = dict(trial_count=0, trial_value_volume=0,
                             budget_rejections=0, cache_hits=0, failed_trials=0)

    def _check_inputs(self):
        frozen.assert_hash(self.raw, self.source_hash, "native trial raw source")
        for path, digest in self.tool_hashes.items():
            frozen.assert_hash(path, digest, "native executable")

    def _region(self, region):
        if not isinstance(region, (tuple, list)) or len(region) != 2:
            raise ValueError("region must be (start, length)")
        start = _integer(region[0], "start")
        length = _integer(region[1], "length", 1)
        if start + length > self.n or length > 102400:
            raise ValueError("region exceeds raw input / supported rowgroup")
        return start, length

    def _needed(self, region):
        start, length = region
        keys = [(start, length, codec) for codec in STREAM_CODECS]
        keys.append((start, length, "btrblocks"))
        full = length // 1024 * 1024
        if full:
            keys.append((start, full, "alp"))
            if full != length:
                keys.extend((start + full, length - full, codec) for codec in STREAM_CODECS)
        return keys

    def _old_trial(self, start, length, codec):
        folder = self.old_cache / ("{:06d}".format(len(list(self.old_cache.iterdir()))))
        folder.mkdir()
        sliced, plan = folder / "slice.double_bin", folder / "plan.csv"
        container, report, log = folder / "candidate.f64blk", folder / "report.json", folder / "encode.log"
        with self.raw.open("rb") as stream:
            stream.seek(start * 8)
            data = stream.read(length * 8)
        if len(data) != length * 8:
            raise ValueError("raw interval changed while creating slice")
        with sliced.open("xb") as stream:
            stream.write(data)
        digest = replay.sha256(sliced)
        key = (self.job["key"][0], self.job["key"][1], digest, "region_trial")
        rows = [dict(zip(replay.IDENTITY, key), start=0, length=length, codec=codec)]
        frozen.write_csv(plan, rows, replay.PLAN_FIELDS)
        plan_hash = replay.sha256(plan)
        trial = dict(start=start, length=length, codec=codec, status="unavailable", detail="",
                     candidate_path=str(container), slice_path=str(sliced), slice_sha256=digest,
                     plan_path=str(plan), plan_sha256=plan_hash, encoded_values=length,
                     executable_sha256=self.tool_hashes[str(self.exporter)])
        try:
            code = self.execute([self.exporter, sliced, plan, "region_trial", container, report],
                                log, self.timeout)
            trial["exit_code"] = code
            if code != 0:
                raise ValueError("native " + codec + " trial exited " + str(code))
            info, checksum = replay.validate_export(report, container, sliced, key, rows, length)
            _, frames = bridge.read_frames(container)
            decoded = frozen.decode_exact(container, sliced, folder / "decode", self.exporter,
                                          self.btr, self.timeout, self.execute)
            trial.update(status="passed", frames=frames, candidate_sha256=checksum,
                         frame_bytes=frozen.frame_cost(frames), physical_frames=len(frames),
                         decode_report=decoded)
        except (OSError, ValueError, KeyError, TypeError) as error:
            trial.update(status="unavailable", detail=str(error))
        frozen.assert_hash(sliced, digest, "native input slice")
        frozen.assert_hash(plan, plan_hash, "native trial plan")
        frozen.write_json(folder / "trial_manifest.json", {k: v for k, v in trial.items() if k != "frames"})
        return trial

    def _run_trial(self, key):
        start, length, codec = key
        if codec == "btrblocks":
            _, trial = frozen.encode_trial(self.job, start, length, self.btr_cache, self.btr,
                                            self.exporter, self.tool_hashes[str(self.btr)],
                                            self.timeout, self.execute)
            trial["codec"] = codec
        else:
            trial = self._old_trial(start, length, codec)
        self._check_inputs()
        self.cache[key] = trial
        self.trials.append({k: v for k, v in trial.items() if k != "frames"})
        self.counters["failed_trials"] += trial["status"] != "passed"

    def _frames(self, key):
        trial = self.cache[key]
        if trial["status"] != "passed":
            return None
        frozen.assert_hash(trial["candidate_path"], trial["candidate_sha256"], "priced candidate payload")
        # Read persisted bytes again; do not trust mutable in-memory trial frames.
        info, frames = bridge.read_frames(trial["candidate_path"])
        if info["values"] != key[1]:
            raise ValueError("priced candidate coverage changed")
        return [dict(frame, start=key[0] + frame["start"],
                     codec_id=25 if frame["codec_id"] == 4 else frame["codec_id"])
                for frame in frames]

    def _best(self, region):
        start, length = region
        options = []
        for codec in STREAM_CODECS + ("btrblocks",):
            frames = self._frames((start, length, codec))
            if frames is not None:
                options.append((codec, frames))
        full = length // 1024 * 1024
        if full:
            prefix = self._frames((start, full, "alp"))
            if prefix is not None:
                if full == length:
                    options.append(("alp", prefix))
                else:
                    tails = []
                    for codec in STREAM_CODECS:
                        frames = self._frames((start + full, length - full, codec))
                        if frames is not None:
                            tails.append((codec, frames))
                    if tails:
                        codec, tail = min(tails, key=lambda pair: frozen.frame_cost(pair[1]))
                        options.append(("alp+tail:" + codec, prefix + tail))
        if not options:
            return None
        codec, frames = min(options, key=lambda pair: frozen.frame_cost(pair[1]))
        return dict(start=start, length=length, codec=codec, frames=frames)

    def best_many(self, regions):
        """Price an operation's regions or reject its entire reservation first.

        Cached failures are not retried. If a native candidate is unavailable,
        another valid codec can still be selected; no valid codec -> None.
        """
        regions = [self._region(region) for region in regions]
        if not regions:
            raise ValueError("at least one region is required")
        needed = list(dict.fromkeys(key for region in regions for key in self._needed(region)))
        fresh = [key for key in needed if key not in self.cache]
        self.counters["cache_hits"] += len(needed) - len(fresh)
        if (self.counters["trial_count"] + len(fresh) > self.trial_limit
                or self.counters["trial_value_volume"] + sum(k[1] for k in fresh) > self.value_budget):
            self.counters["budget_rejections"] += 1
            return None
        self._check_inputs()
        # Reserve every unique attempt before the first native invocation.
        self.counters["trial_count"] += len(fresh)
        self.counters["trial_value_volume"] += sum(key[1] for key in fresh)
        for key in fresh:
            self._run_trial(key)
        return_value = [self._best(region) for region in regions]
        return None if any(value is None for value in return_value) else return_value

