"""ClaSS on patterns of ADJACENT binary64 changes, not absolute bit patterns.

Integration adapts ClaSPy 0.2.8's streaming kNN/control flow; the upstream
project is BSD-3-Clause. See third_party_licenses/ClaSPy-BSD-3-Clause.txt.

Delta[i] = bits(value[i]) XOR bits(value[i-1]); delta[0] = 0 exactly once.
Mask histories are uint64 arrays separate from original float64 histories.
The old raw-bit metric, numeric detector and installed packages are unchanged.
"""
import os
import time

import numpy as np
from claspy.nearest_neighbour import _argkmin
from claspy.streaming.nearest_neighbour import _roll_knns
from claspy.utils import roll_array

from class_bit_hamming import BitClaSSSegmentation, BitStreamingNeighbours, distance_row
from segment_class import SUPPORTED_CLASPY, claspy_disk_cache_policy, package_versions


ACTUAL_METRIC = "adjacent_xor_change_subsequence_hamming"
REPRESENTATION = "delta[i]=bits(v[i]) XOR bits(v[i-1]); delta[0]=0"


class XorChangeNeighbours(BitStreamingNeighbours):
    """Author online-kNN coordinates; exact change-mask distance recurrence.

    previous_word records the previous ORIGINAL global value, not the first
    currently retained value. It is never reset by history eviction or a CP.
    """
    actual_metric = ACTUAL_METRIC

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.metric_words = np.zeros(self.n_timepoints, dtype=np.uint64)
        self.previous_word = None
        self.words_seen = 0
        self.last_delta = None

    def update(self, timepoint, change_point=0):
        value = np.float64(timepoint)
        word = value.view(np.uint64)
        delta = np.uint64(0) if self.previous_word is None else word ^ self.previous_word
        self.previous_word = word
        self.words_seen += 1
        self.last_delta = delta
        # Two distinct histories: masks are NEVER viewed/cast as float inputs
        # and NEVER used as integer magnitudes in a numeric distance.
        self.metric_words = roll_array(self.metric_words, -1, delta)
        self.time_series = roll_array(self.time_series, -1, value)
        self.lbound = self.knn_insert_idx - self.knn_filled + 1 + change_point
        self.n_filled = min(self.n_filled + 1, self.n_windows)
        if self.n_filled < self.window_size + self.exclusion_radius + self.k_neighbours:
            return self
        if self.knn_filled > 0:
            self.dists = roll_array(self.dists, -1, np.full(self.k_neighbours, np.inf, dtype=np.float64))
            self.knns = roll_array(self.knns, -1)
            self.knns[self.knn_insert_idx - self.knn_filled:self.knn_insert_idx] -= 1
            self.knns[-1, :] = -1
        start = self.lbound - 1
        if not 0 <= start < self.n_windows:
            raise ValueError("change_point gives an invalid author-kNN reference range")
        raw, self.partial = distance_row(self.metric_words, self.knn_insert_idx,
            self.window_size, self.partial, start, self.first, self.brute_force)
        self.first = False
        self.last_raw_distances = raw
        dist = raw.copy()
        q, radius = self.knn_insert_idx, self.exclusion_radius
        dist[max(0, q-radius):min(q+radius, self.n_windows)] = np.max(dist)
        knns = _argkmin(dist, self.k_neighbours, self.lbound)
        self.dists, self.knns, self.lbound, self.knn_filled = _roll_knns(
            self.dists, self.knns, dist, knns, self.knn_insert_idx,
            self.knn_filled, self.n_windows, self.k_neighbours, self.lbound)
        return self

    def transform(self):
        result = super().transform()
        result.actual_metric = result.distance_name = ACTUAL_METRIC
        result.representation = REPRESENTATION
        return result


class XorChangeClaSSSegmentation(BitClaSSSegmentation):
    """Only allocate the new neighbour representation during author warmup."""
    actual_metric = ACTUAL_METRIC

    def __init__(self, **config):
        super().__init__(**config)
        self.distance = ACTUAL_METRIC

    def _warmup(self, timepoint):
        # Install before replay: a post-super hook would already have scored
        # all warmup data using the old representation. No mask is preinserted.
        self.warmup_counter += 1
        self.warmup = roll_array(self.warmup, -1)
        self.warmup[-1] = timepoint
        if self.warmup_counter != self.n_warmup:
            return self
        self.min_seg_size = 5*self.window_size
        if self.threshold == "default":
            self.threshold = 0.75 if self.validation == "score_threshold" else 1e-50
        self.knn_stream = XorChangeNeighbours(self.n_timepoints, self.window_size,
            self.k_neighbours, brute_force=self.brute_force)
        self.knn_stream_lag = (self.knn_stream.window_size+self.knn_stream.exclusion_radius
                               +self.knn_stream.k_neighbours)
        for value in self.warmup:
            self.update(value)
        return self


def make_detector(config, *, brute_force=False):
    versions = package_versions()
    if versions.get("claspy") != SUPPORTED_CLASPY:
        raise ValueError("XOR-change ClaSS adapter requires claspy=="+SUPPORTED_CLASPY)
    config = dict(config)
    for name, default in (("n_timepoints",4096),("window_size",32),("k_neighbours",3),
                          ("jump",5),("excl_radius",5)):
        value = config.get(name,default)
        if type(value) is not int or value < 1:
            raise ValueError(name+" must be a fixed positive integer")
        config[name] = value
    if config["n_timepoints"] <= 10*config["window_size"]:
        raise ValueError("history must exceed ten subsequence lengths")
    if "n_warmup" in config and (type(config["n_warmup"]) is not int or config["n_warmup"] < 1):
        raise ValueError("n_warmup must be a positive integer")
    if config["excl_radius"] <= config["k_neighbours"]:
        raise ValueError("author exclusion factor must exceed k_neighbours")
    if config.get("score","f1") != "f1":
        raise ValueError("this experiment preserves author macro-F1 scoring")
    if config.get("validation","significance_test") not in ("significance_test","score_threshold"):
        raise ValueError("unsupported author validation setting")
    return XorChangeClaSSSegmentation(brute_force=brute_force, **config)


def segment_change_events(values, config, progress_every=1000):
    """Same four-result protocol as segment_class; no double pass or flush."""
    array = np.asarray(values)
    if array.ndim != 1 or array.dtype != np.dtype(np.float64) or not len(array) or not np.all(np.isfinite(array)):
        raise ValueError("require original finite float64 values, not XOR mask magnitudes")
    policy = os.environ.get("CLASS_NUMBA_DISK_CACHE","1")
    if policy not in ("0","1"):
        raise ValueError("CLASS_NUMBA_DISK_CACHE must be 0 or 1")
    versions = package_versions()
    events, seen = [], 0
    with claspy_disk_cache_policy(policy == "1") as cache_info:
        detector = make_detector(config)
        started = time.perf_counter()
        for index,value in enumerate(array):
            detector.update(value)
            for point in detector.change_points[seen:]:
                events.append(dict(change_point=int(point),detected_after_values=index+1))
            seen = len(detector.change_points)
            if progress_every and ((index+1)%progress_every == 0 or index+1 == len(array)):
                print("XOR-change ClaSS progress: %s/%s values; raw_cps=%s; elapsed=%.1fs" %
                      (index+1,len(array),seen,time.perf_counter()-started),flush=True)
        elapsed = time.perf_counter()-started
    resolved = dict(config)
    resolved.update(distance=ACTUAL_METRIC,actual_metric=ACTUAL_METRIC,representation=REPRESENTATION,
        distance_definition="sum popcount(delta[query+r] XOR delta[reference+r]) over aligned change positions",
        distance_update="exact shared diagonal recurrence; unmasked uint64 change-history distance",
        first_delta_policy="delta[0]=0 exactly once at global sequence start",
        chronology="delta[i] belongs to original value i; CP indices remain original zero-based value indices",
        source_support="w changes normally use w+1 original values; the first delta is the declared zero sentinel",
        previous_original_word_continuous_across_history_eviction_and_change_points=True,
        history_entry_reinitialized_at_window_or_cp=False,synthetic_flush_used=False,
        original_float64_history_preserved=True,masks_cast_to_float_or_numeric_magnitude=False,
        old_raw_bit_adapter_modified=False,installed_sources_modified=False,
        author_streaming_update_reused=True,author_macro_f1_reused=True,author_validation=detector.validation,
        author_significance_reused=detector.validation == "significance_test",
        window_size=detector.window_size,n_warmup=detector.n_warmup,threshold=detector.threshold,
        log_cps=True,warmup_completed=detector.warmup_counter == detector.n_warmup,numba_disk_cache=cache_info)
    return events,resolved,elapsed,versions


def segment_change_values(values, config, progress_every=1000):
    return [row["change_point"] for row in segment_change_events(values,config,progress_every)[0]]
