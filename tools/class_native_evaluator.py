"""Native stored-byte costing for the original ClaSS boundary-repair planner.

This adapter changes neither candidate generation nor ``repair_partition``.
It measures the declared full feasible codec pool on visited regions, not a
learned/top-k selector. Each attempted physical codec key is paid through a
BudgetedRegionView, including shared-cache hits not yet paid by that view.
Region prices include all physical frame headers. The single 24-byte file
header is a constant added only when the complete file is written.
"""

from __future__ import annotations

import btr_boundary_costs as native
import btr_boundary_search as framing
import frozen_btr_extension as frozen
import replay_frozen_files as replay


class DiskBackedNativeRegionService(native.NativeRegionService):
    """Keep candidate payloads on disk rather than in every trial cache row."""

    def _run_trial(self, key):
        super()._run_trial(key)
        # NativeRegionService._frames checks and reopens the priced file. Its
        # original cached frame list therefore only duplicates payload memory.
        self.cache[key].pop("frames", None)


def _quota(value, name):
    if value is None:
        return None
    if type(value) is not int or value < 0:
        raise ValueError(name + " must be None or a strict nonnegative integer")
    return value


def _identity(segment):
    start, length, digest = framing._signature(segment)
    return start, length, digest.hex()


class NativeRegionEvaluator:
    """Expose exact native region costs to the unchanged repair interface.

    One view may be shared by fixed controls and multiple seeds. The global
    view budget is never reset. Calling set_seed_quota snapshots its charged
    counters and places an additional cap on subsequent new physical keys.
    Pure preflight does not reserve work, invoke codecs, or change counters.
    """

    cost_source = "actual_native_full_feasible_codec_pool_stored_frame_bytes"

    def __init__(self, view):
        self.view = view
        self.service = view.backend
        self.source = tuple(self.service.job["key"][:3])
        if len(self.source) != 3:
            raise ValueError("native evaluator needs dataset, series and source SHA")
        self.costs, self.choices = {}, {}
        self.quota_start = self.quota_values_start = 0
        self.quota_limit = self.quota_values_limit = None

    @property
    def remaining_edges(self):
        return max(0, self.view.trial_limit - self.view.counters["trial_count"])

    @property
    def remaining_values(self):
        return max(0, self.view.value_budget - self.view.counters["trial_value_volume"])

    def _intervals(self, intervals):
        regions = [self.service._region(region) for region in intervals]
        return sorted(set(regions))

    def _request(self, intervals):
        regions = self._intervals(intervals)
        if not regions:
            return regions, []
        # Use the ledger's exact physical-key validation and expansion, without
        # entering best_many (which would reserve counters or invoke codecs).
        return self.view._needed(regions)

    def set_seed_quota(self, maximum_new_edges=None, maximum_new_values=None):
        edges = _quota(maximum_new_edges, "seed edge quota")
        values = _quota(maximum_new_values, "seed value quota")
        self.quota_start = self.view.counters["trial_count"]
        self.quota_values_start = self.view.counters["trial_value_volume"]
        self.quota_limit, self.quota_values_limit = edges, values

    def can_evaluate(self, intervals):
        _, requested = self._request(intervals)
        fresh = [key for key in requested if key not in self.view.charged_keys]
        new_edges, new_values = len(fresh), sum(key[1] for key in fresh)
        used_edges = self.view.counters["trial_count"]
        used_values = self.view.counters["trial_value_volume"]
        if used_edges + new_edges > self.view.trial_limit:
            return False
        if used_values + new_values > self.view.value_budget:
            return False
        if (self.quota_limit is not None and
                used_edges - self.quota_start + new_edges > self.quota_limit):
            return False
        if (self.quota_values_limit is not None and
                used_values - self.quota_values_start + new_values > self.quota_values_limit):
            return False
        # Production uses one series-wide view/backend. Also honor a narrower
        # backend cap if a caller deliberately provides one: reject atomically
        # before asking the ledger to reserve a request it cannot execute.
        native_fresh = [key for key in requested if key not in self.service.cache]
        native_trials = self.service.counters["trial_count"]
        native_values = self.service.counters["trial_value_volume"]
        edge_cap = getattr(self.service, "trial_limit", None)
        value_cap = getattr(self.service, "value_budget", None)
        if edge_cap is not None and native_trials + len(native_fresh) > edge_cap:
            return False
        if value_cap is not None and native_values + sum(k[1] for k in native_fresh) > value_cap:
            return False
        return True

    def _keys(self, segment):
        start, length, codec = segment["start"], segment["length"], segment["codec"]
        if codec in native.STREAM_CODECS or codec == "btrblocks":
            return ((start, length, codec),)
        full = length // 1024 * 1024
        if codec == "alp" and full == length:
            return ((start, length, "alp"),)
        if codec.startswith("alp+tail:") and 0 < full < length:
            tail = codec.split(":", 1)[1]
            if tail in native.STREAM_CODECS:
                return ((start, full, "alp"), (start + full, length - full, tail))
        raise ValueError("native service returned an unknown/infeasible selected codec")

    def evaluate(self, intervals, *, phase="trial"):
        if phase not in ("seed", "trial"):
            raise ValueError("measurement phase must be seed or trial")
        regions = self._intervals(intervals)
        if not self.can_evaluate(regions):
            raise ValueError("native full-codec budget is insufficient for the complete request")
        if not regions:
            return {}
        selected = self.view.best_many(regions)
        if selected is None or len(selected) != len(regions):
            raise ValueError("native full-codec service could not score every requested region")
        # Validate every selected physical plan before committing semantic
        # prices. Failed native attempts remain paid in the ledger regardless.
        updates = {}
        for region, item in zip(regions, selected):
            item = framing._segment(item, *region)
            bits = 8 * frozen.frame_cost(item["frames"])
            identity = _identity(item)
            keys = self._keys(item)
            if any(key not in self.view.charged_keys for key in keys):
                raise ValueError("selected physical plan was not paid by this budget view")
            if region in self.costs and (self.costs[region] != bits or
                                         self.choices[region]["signature"] != identity):
                raise ValueError("native evaluator changed a frozen region's stored bytes")
            updates[region] = dict(start=region[0], length=region[1],
                                   codec=item["codec"], signature=identity,
                                   physical_keys=keys, phase=phase, bits=bits)
        for region, choice in updates.items():
            self.choices[region] = choice
            # No frame/payload arrays are kept in the evaluator cache.
            self.costs[region] = choice["bits"]
        return {region: self.costs[region] for region in regions}

    def native_segments(self, intervals):
        """Reopen the selected immutable native bytes; never re-encode/reselect."""
        regions = self._intervals(intervals)
        if any(region not in self.costs for region in regions):
            raise ValueError("native plan requires completely evaluated regions")
        selected = []
        for region in regions:
            choice, frames = self.choices[region], []
            for key in choice["physical_keys"]:
                part = self.service._frames(key)
                if part is None:
                    raise ValueError("selected persisted native codec became unavailable")
                frames.extend(part)
            item = framing._segment(dict(start=region[0], length=region[1],
                                         codec=choice["codec"], frames=frames), *region)
            if (8 * frozen.frame_cost(frames) != self.costs[region] or
                    _identity(item) != choice["signature"]):
                raise ValueError("selected persisted native bytes differ from the priced region")
            selected.append(item)
        return selected

    def physical_plan(self, intervals, method):
        """Return physical-frame metadata for an audit, not a re-encoding plan."""
        if not isinstance(method, str) or not method:
            raise ValueError("physical plan method must be nonempty text")
        names = {value: key for key, value in replay.CODEC_IDS.items()}
        names[25] = "btrblocks"
        return [dict(zip(("dataset", "series", "sha256"), self.source),
                     method=method, start=frame["start"], length=frame["count"],
                     codec=names[frame["codec_id"]], codec_id=frame["codec_id"],
                     payload_bytes=len(frame["payload"]),
                     frame_bytes=12 + len(frame["payload"]),
                     region_start=segment["start"], region_length=segment["length"])
                for segment in self.native_segments(intervals) for frame in segment["frames"]]
