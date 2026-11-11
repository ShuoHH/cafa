"""Binary64-subsequence Hamming neighbours with the author's ClaSS detector.

Integration adapts ClaSPy 0.2.8's streaming kNN/control flow; the upstream
project is BSD-3-Clause. See third_party_licenses/ClaSPy-BSD-3-Clause.txt.

Only the streaming nearest-neighbour metric is replaced. Author ClaSS macro-F1,
significance test, streaming update, lag and change-point bookkeeping are reused.
No installed file is changed. uint64 words are bit views, NEVER numeric inputs
to a Euclidean metric. New Numba kernels never write an on-disk cache.
"""
import os
import time
from types import SimpleNamespace

import numpy as np
from numba import njit
from claspy.nearest_neighbour import _argkmin
from claspy.streaming.nearest_neighbour import _roll_knns
from claspy.streaming.segmentation import StreamingClaSPSegmentation
from claspy.utils import roll_array

from segment_class import SUPPORTED_CLASPY, claspy_disk_cache_policy, package_versions


ACTUAL_METRIC = "binary64_subsequence_hamming"


@njit(cache=False)
def popcount64(word):
    """Exact portable SWAR popcount; uint64 wraparound is intentional."""
    x = np.uint64(word)
    x -= (x >> np.uint64(1)) & np.uint64(0x5555555555555555)
    x = (x & np.uint64(0x3333333333333333)) + ((x >> np.uint64(2)) & np.uint64(0x3333333333333333))
    x = (x + (x >> np.uint64(4))) & np.uint64(0x0f0f0f0f0f0f0f0f)
    return np.int64((x * np.uint64(0x0101010101010101)) >> np.uint64(56))


@njit(cache=False)
def distance_row(words, query, width, partial, start, first, brute=False):
    """Return raw distances and the next diagonal partial row.

    partial[j] stores last distance MINUS its departing first pair. Both query
    and reference subsequences advance one position per streaming update. Thus
    adding the arriving final pair produces the new exact distance in O(1).
    A newly activated leftmost reference is initialized by a width-long sum.
    Exclusion masks are deliberately absent: inf masks must not pollute state.
    """
    n_windows = len(words) - width + 1
    row = np.full(n_windows, np.inf, dtype=np.float64)
    following = np.full(n_windows, np.inf, dtype=np.float64)
    for j in range(start, n_windows):
        total = 0
        if first or brute or j == start or not np.isfinite(partial[j]):
            for offset in range(width):
                total += popcount64(words[query + offset] ^ words[j + offset])
        else:
            total = np.int64(partial[j]) + popcount64(words[query + width - 1] ^ words[j + width - 1])
        row[j] = total
        following[j] = total - popcount64(words[query] ^ words[j])
    return row, following


class BitStreamingNeighbours:
    """Same online kNN maintenance/coordinates as claspy0.2.8, new distance row.

    brute_force is a correctness-test oracle, not a production experiment arm.
    Storage is O(history*k); the fast distance update is O(history) per value.
    """
    actual_metric = ACTUAL_METRIC

    def __init__(self, n_timepoints=4096, window_size=32, k_neighbours=3, *, brute_force=False):
        self.n_timepoints = n_timepoints
        self.window_size = window_size
        self.k_neighbours = k_neighbours
        self.exclusion_radius = window_size // 2
        self.n_windows = n_timepoints - window_size + 1
        self.knn_insert_idx = self.n_windows - self.exclusion_radius - k_neighbours - 1
        if self.knn_insert_idx < 1:
            raise ValueError("history is too short for the author streaming-kNN geometry")
        self.lbound = self.n_filled = self.knn_filled = 0
        self.time_series = np.full(n_timepoints, np.nan, dtype=np.float64)
        self.dists = np.full((self.n_windows, k_neighbours), np.inf, dtype=np.float64)
        self.knns = np.full((self.n_windows, k_neighbours), -1, dtype=np.int64)
        self.partial = np.full(self.n_windows, np.inf, dtype=np.float64)
        self.first = True
        self.brute_force = bool(brute_force)
        self.last_raw_distances = None

    def update(self, timepoint, change_point=0):
        self.lbound = self.knn_insert_idx - self.knn_filled + 1 + change_point
        self.n_filled = min(self.n_filled + 1, self.n_windows)
        # Assignment/rolling copies the raw float64 values; there is no float
        # arithmetic in the metric. Unfilled words are excluded by lbound.
        self.time_series = roll_array(self.time_series, -1, timepoint)
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
        raw, self.partial = distance_row(self.time_series.view(np.uint64), self.knn_insert_idx,
            self.window_size, self.partial, start, self.first, self.brute_force)
        self.first = False
        self.last_raw_distances = raw
        dist = raw.copy()
        q, radius = self.knn_insert_idx, self.exclusion_radius
        dist[max(0, q - radius):min(q + radius, self.n_windows)] = np.max(dist)
        knns = _argkmin(dist, self.k_neighbours, self.lbound)
        self.dists, self.knns, self.lbound, self.knn_filled = _roll_knns(
            self.dists, self.knns, dist, knns, self.knn_insert_idx,
            self.knn_filled, self.n_windows, self.k_neighbours, self.lbound)
        return self

    def transform(self):
        # Identical slice/shift/clip geometry to author StreamingKSN.transform.
        # ClaSS.fit(knn=...) and its significance test consume offsets. A
        # lightweight object avoids falsely labelling Hamming as Euclidean.
        offsets = self.knns[self.lbound:self.knn_insert_idx] - self.lbound
        offsets = np.clip(offsets, 0, offsets.shape[0] - 1)
        return SimpleNamespace(time_series=self.time_series,
            temporal_constraints=np.asarray([(0, self.n_timepoints)], dtype=int),
            distances=self.dists[self.lbound:self.knn_insert_idx], offsets=offsets,
            window_size=self.window_size, k_neighbours=self.k_neighbours,
            actual_metric=ACTUAL_METRIC, distance_name=ACTUAL_METRIC)

    def update_transform(self, timepoint):
        return self.update(timepoint).transform()


class BitClaSSSegmentation(StreamingClaSPSegmentation):
    """Override only warmup allocation; inherit the complete author update()."""
    actual_metric = ACTUAL_METRIC

    def __init__(self, *, brute_force=False, **config):
        # ClaSS.__init__ stores this real metric name; fit(knn=...) does not
        # call map_distances. There is NO Euclidean placeholder or fallback.
        config = dict(config)
        config["distance"] = ACTUAL_METRIC
        config["log_cps"] = True
        super().__init__(**config)
        self.brute_force = bool(brute_force)

    def _warmup(self, timepoint):
        # A post-super() hook is too late: author _warmup immediately replays
        # the entire buffer through self.update, including scoring/testing.
        self.warmup_counter += 1
        self.warmup = roll_array(self.warmup, -1)
        self.warmup[-1] = timepoint
        if self.warmup_counter != self.n_warmup:
            return self
        self.min_seg_size = 5 * self.window_size
        if self.threshold == "default":
            self.threshold = 0.75 if self.validation == "score_threshold" else 1e-50
        self.knn_stream = BitStreamingNeighbours(self.n_timepoints, self.window_size,
            self.k_neighbours, brute_force=self.brute_force)
        self.knn_stream_lag = (self.knn_stream.window_size + self.knn_stream.exclusion_radius
                               + self.knn_stream.k_neighbours)
        for value in self.warmup:
            self.update(value)
        return self


def make_detector(config, *, brute_force=False):
    versions = package_versions()
    if versions.get("claspy") != SUPPORTED_CLASPY:
        raise ValueError("Binary ClaSS adapter requires claspy==" + SUPPORTED_CLASPY)
    config = dict(config)
    for name, default in (("n_timepoints", 4096), ("window_size", 32), ("k_neighbours", 3),
                          ("jump", 5), ("excl_radius", 5)):
        value = config.get(name, default)
        if type(value) is not int or value < 1:
            raise ValueError(name + " must be a fixed positive integer")
        config[name] = value
    if config["n_timepoints"] <= 10 * config["window_size"]:
        raise ValueError("history must exceed ten subsequence lengths")
    if "n_warmup" in config and (type(config["n_warmup"]) is not int or config["n_warmup"] < 1):
        raise ValueError("n_warmup must be a positive integer")
    if config["excl_radius"] <= config["k_neighbours"]:
        raise ValueError("author exclusion factor must exceed k_neighbours")
    if config.get("score", "f1") != "f1":
        raise ValueError("this experiment preserves author macro-F1 scoring")
    if config.get("validation", "significance_test") not in ("significance_test", "score_threshold"):
        raise ValueError("unsupported author validation setting")
    return BitClaSSSegmentation(brute_force=brute_force, **config)


def segment_bit_events(values, config, progress_every=1000):
    """Same return protocol as segment_class.segment_values, one actual pass."""
    array = np.asarray(values)
    if array.ndim != 1 or array.dtype != np.dtype(np.float64) or not len(array) or not np.all(np.isfinite(array)):
        raise ValueError("require original finite float64 values, not integer word magnitudes")
    policy = os.environ.get("CLASS_NUMBA_DISK_CACHE", "1")
    if policy not in ("0", "1"):
        raise ValueError("CLASS_NUMBA_DISK_CACHE must be 0 or 1")
    versions = package_versions()
    events, seen = [], 0
    with claspy_disk_cache_policy(policy == "1") as cache_info:
        detector = make_detector(config)
        started = time.perf_counter()
        for index, value in enumerate(array):
            detector.update(value)
            for point in detector.change_points[seen:]:
                events.append(dict(change_point=int(point), detected_after_values=index + 1))
            seen = len(detector.change_points)
            if progress_every and ((index + 1) % progress_every == 0 or index + 1 == len(array)):
                elapsed = time.perf_counter() - started
                print("Binary ClaSS progress: %s/%s values; raw_cps=%s; elapsed=%.1fs" %
                      (index + 1, len(array), seen, elapsed), flush=True)
        elapsed = time.perf_counter() - started
    resolved = dict(config)
    resolved.update(distance=ACTUAL_METRIC, actual_metric=ACTUAL_METRIC,
        distance_definition="sum popcount(bits(query[r]) XOR bits(reference[r])) over aligned subsequence positions",
        distance_update="exact diagonal recurrence; remove first aligned pair and add last aligned pair",
        distance_complexity="O(N*history) plus author kNN maintenance and ClaSS scoring",
        bit_words_used_as_numeric_magnitudes=False, euclidean_metric_fallback=False,
        installed_sources_modified=False, author_streaming_update_reused=True,
        author_macro_f1_and_significance_reused=detector.validation == "significance_test",
        author_macro_f1_reused=True, author_validation=detector.validation,
        window_size=detector.window_size, n_warmup=detector.n_warmup,
        threshold=detector.threshold, log_cps=True,
        warmup_completed=detector.warmup_counter == detector.n_warmup, numba_disk_cache=cache_info)
    # No future padding/flush; global timing and late-CP behavior remain author semantics.
    return events, resolved, elapsed, versions


def segment_bit_values(values, config, progress_every=1000):
    return [r["change_point"] for r in segment_bit_events(values, config, progress_every)[0]]
