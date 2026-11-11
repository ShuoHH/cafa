#!/usr/bin/env python3
"""Freeze raw-value ClaSS (PVLDB 2024) boundaries before codec selection.

The maintained author implementation is ``claspy.streaming.segmentation``.
Only its global ``change_points`` log is exported: ``predict()`` returns a
sliding-window-relative position, not a position in the original file.

This pilot uses an explicit 4096-point history and a 32-point subsequence.
These are pilot parameters, not a reproduction of the paper's default setup.
No codec, cost table, oracle plan, or training label is used by this program.
"""

import argparse
from contextlib import contextmanager
import csv
import hashlib
import json
import math
import os
import platform
import sys
import time

import numpy as np


ADAPTER_VERSION = "class_boundaries_v1"
SUPPORTED_CLASPY = "0.2.8"
BOUNDARY_FIELDS = ("dataset", "series", "sha256", "start", "length")
SOURCE_URL = (
    "https://github.com/ermshaua/claspy/blob/v0.2.8/"
    "claspy/streaming/segmentation.py"
)


def file_sha256(path):
    digest = hashlib.sha256()
    with open(path, "rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def read_float64(path, max_values=0):
    """Read original little-endian doubles without replacing or rounding values."""
    size = os.path.getsize(path)
    if size % 8:
        raise ValueError("input byte size is not a multiple of 8: {}".format(size))
    if size >= 8:
        with open(path, "rb") as stream:
            first = int.from_bytes(stream.read(8), "little")
        # CodecHarness detects and skips this legacy uint64 count header.
        # ALP and NumPy otherwise treat it as a data value, shifting every
        # boundary. Require one unambiguous, shared headerless representation.
        if first != 0 and first == size // 8 - 1:
            raise ValueError("input has a uint64 count header; convert it to headerless float64 first: " + path)
    if max_values < 0:
        raise ValueError("max_values must be nonnegative")
    count = max_values if max_values else -1
    values = np.fromfile(path, dtype="<f8", count=count)
    if not len(values):
        raise ValueError("input contains no float64 values")
    bad = np.flatnonzero(~np.isfinite(values))
    if len(bad):
        raise ValueError(
            "ClaSS requires finite values; {} NaN/Inf values found (first at {}). "
            "The original input has not been transformed.".format(len(bad), int(bad[0]))
        )
    return values


def normalize_boundaries(change_points, n_values, snap_grid=64, min_segment=64):
    """Turn global CPs into a contiguous partition, including the exact suffix.

    Interior CPs snap to the nearest grid position (ties to the earlier one).
    Duplicates and out-of-range positions are ignored.  A boundary too near
    either endpoint or the preceding accepted boundary is omitted.  We never
    round the series end or drop a short final suffix: it joins the last segment.
    """
    if n_values <= 0:
        raise ValueError("n_values must be positive")
    if snap_grid <= 0 or min_segment <= 0:
        raise ValueError("snap_grid and min_segment must be positive")
    candidates = set()
    for point in change_points:
        numeric = float(point)
        if not math.isfinite(numeric) or numeric != int(numeric):
            raise ValueError("ClaSS returned a noninteger change point: {!r}".format(point))
        point = int(numeric)
        if not 0 < point < n_values:
            continue
        quotient, remainder = divmod(point, snap_grid)
        point = (quotient + int(2 * remainder > snap_grid)) * snap_grid
        if 0 < point < n_values:
            candidates.add(point)

    endpoints = [0]
    for point in sorted(candidates):
        if point - endpoints[-1] >= min_segment and n_values - point >= min_segment:
            endpoints.append(point)
    endpoints.append(int(n_values))
    return [(start, end - start) for start, end in zip(endpoints, endpoints[1:])]


def package_versions():
    # The adapter's pure boundary tests can run without claspy installed.
    from importlib.metadata import PackageNotFoundError, version

    result = {}
    for name in ("claspy", "numpy", "numba", "scipy", "pandas", "scikit-learn"):
        try:
            result[name] = version(name)
        except PackageNotFoundError:
            result[name] = None
    return result


@contextmanager
def claspy_disk_cache_policy(enabled):
    """Optionally bypass ClaSPy's on-disk JIT cache, keeping native JIT active.

    Numba 0.61's cache index can retain expired dispatcher weak references;
    serializing that index then raises ``underlying object has vanished``.
    Only loaded claspy-owned dispatchers are affected in this process. Third-
    party files and cached artifacts are never deleted or rewritten here.
    This guarded integration uses Numba's per-cache enable/disable methods.
    """
    from numba import config as numba_config
    from numba.core.registry import CPUDispatcher

    info = {"enabled": bool(enabled), "jit_enabled": not bool(numba_config.DISABLE_JIT),
            "scope": "loaded claspy-owned CPU dispatchers in this process",
            "disabled_kernels": [], "third_party_source_modified": False}
    states, seen = [], set()
    if not enabled:
        if numba_config.DISABLE_JIT:
            raise ValueError("unset NUMBA_DISABLE_JIT: the disk-cache workaround keeps native JIT enabled")
        for name, module in list(sys.modules.items()):
            if not (name=="claspy" or name.startswith("claspy.")) or module is None:
                continue
            for value in list(vars(module).values()):
                if not isinstance(value, CPUDispatcher) or id(value) in seen:
                    continue
                seen.add(id(value))
                function = value.py_func
                if not function.__module__.startswith("claspy."):
                    continue
                cache = getattr(value,"_cache",None)
                if cache is None or not callable(getattr(cache,"disable",None)) or not callable(getattr(cache,"enable",None)):
                    raise ValueError("unsupported Numba per-dispatcher cache API; cannot apply safe ClaSS workaround")
                states.append((cache,getattr(cache,"_enabled",None)))
                info["disabled_kernels"].append(function.__module__+"."+function.__name__)
        if not states:
            raise ValueError("no ClaSPy JIT kernels found for disk-cache workaround")
    try:
        for cache, _ in states:
            cache.disable()
        yield info
    finally:
        for cache, was_enabled in reversed(states):
            if was_enabled is True:
                cache.enable()


def segment_values(values, config, progress_every=1000):
    """Run the real author implementation and return global detection events."""
    versions = package_versions()
    if versions.get("claspy") != SUPPORTED_CLASPY:
        raise ValueError(
            "this adapter requires claspy=={} (installed: {}). Install "
            "requirements-class.txt in a Python 3.10-3.12 virtual environment".format(
                SUPPORTED_CLASPY, versions.get("claspy") or "missing")
        )
    from claspy.streaming.segmentation import StreamingClaSPSegmentation

    policy = os.environ.get("CLASS_NUMBA_DISK_CACHE","1")
    if policy not in ("0","1"):
        raise ValueError("CLASS_NUMBA_DISK_CACHE must be 0 or 1")
    with claspy_disk_cache_policy(policy=="1") as cache_info:
        print("ClaSS Numba disk cache: {}; native JIT: {}; bypassed kernels={}".format(
            "enabled" if cache_info["enabled"] else "disabled",
            "enabled" if cache_info["jit_enabled"] else "disabled",
            len(cache_info["disabled_kernels"])),flush=True)
        detector = StreamingClaSPSegmentation(log_cps=True, **config)
        events = []
        seen = 0
        started = time.perf_counter()
        for index, value in enumerate(values):
            detector.update(float(value))
            for point in detector.change_points[seen:]:
                events.append({"change_point": int(point), "detected_after_values": index + 1})
            seen = len(detector.change_points)
            if progress_every and ((index + 1) % progress_every == 0 or index + 1 == len(values)):
                elapsed = time.perf_counter() - started
                print("ClaSS progress: {}/{} values ({:.1f}%), raw_cps={}, elapsed={:.1f}s".format(
                    index + 1, len(values), 100.0 * (index + 1) / len(values), seen, elapsed), flush=True)
        elapsed = time.perf_counter() - started
    resolved = dict(config)
    resolved.update({"window_size": detector.window_size, "threshold": detector.threshold,
                     "log_cps": True, "warmup_completed": detector.warmup_counter == detector.n_warmup,
                     "numba_disk_cache": cache_info})
    # A finite file has no future observations to confirm late CPs.  Do not pad
    # or repeat data to flush the model.  Coverage is added by normalize_boundaries.
    return events, resolved, elapsed, versions


def write_outputs(out_dir, identity, intervals, events, manifest):
    os.makedirs(out_dir, exist_ok=True)
    boundaries_path = os.path.join(out_dir, "boundaries.csv")
    with open(boundaries_path, "w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=BOUNDARY_FIELDS)
        writer.writeheader()
        for start, length in intervals:
            writer.writerow(dict(identity, start=start, length=length))
    raw_path = os.path.join(out_dir, "raw_change_points.csv")
    with open(raw_path, "w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=("change_point", "detected_after_values"))
        writer.writeheader()
        writer.writerows(events)
    with open(os.path.join(out_dir, "segmentation_manifest.json"), "w", encoding="utf-8") as stream:
        json.dump(manifest, stream, indent=2, allow_nan=False)
        stream.write("\n")
    return boundaries_path


def window_size_arg(value):
    if value in ("suss", "fft", "acf"):
        return value
    try:
        result = int(value)
    except ValueError:
        raise argparse.ArgumentTypeError("window size must be a positive integer or suss/fft/acf")
    if result < 2:
        raise argparse.ArgumentTypeError("window size must be at least 2")
    return result


def threshold_arg(value):
    if value == "default":
        return value
    try:
        result = float(value)
    except ValueError:
        raise argparse.ArgumentTypeError("threshold must be default or a number between 0 and 1")
    if not math.isfinite(result) or not 0 < result <= 1:
        raise argparse.ArgumentTypeError("threshold must be between 0 and 1")
    return result


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("input", help="headerless little-endian float64 .bin file")
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--series", required=True)
    parser.add_argument("--out-dir", required=True)
    parser.add_argument("--n-timepoints", type=int, default=4096, help="ClaSS sliding history length")
    parser.add_argument("--n-warmup", type=int, default=None, help="default: n-timepoints")
    parser.add_argument("--window-size", type=window_size_arg, default=32,
                        help="ClaSS subsequence length or suss/fft/acf (pilot default: 32)")
    parser.add_argument("--jump", type=int, default=5)
    parser.add_argument("--k-neighbours", type=int, default=3)
    parser.add_argument("--distance", choices=("znormed_euclidean_distance", "cinvariant_euclidean_distance",
                                               "euclidean_distance"), default="znormed_euclidean_distance")
    parser.add_argument("--score", choices=("f1", "accuracy"), default="f1")
    parser.add_argument("--validation", choices=("significance_test", "score_threshold"),
                        default="significance_test")
    parser.add_argument("--threshold", type=threshold_arg, default="default")
    parser.add_argument("--excl-radius", type=int, default=5)
    parser.add_argument("--snap-grid", type=int, default=64, help="nearest boundary grid; 1 disables snapping")
    parser.add_argument("--min-segment", type=int, default=64)
    parser.add_argument("--max-values", type=int, default=0, help="0 uses the complete input")
    parser.add_argument("--progress-every", type=int, default=1000, help="0 disables progress output")
    args = parser.parse_args(argv)
    try:
        warmup = args.n_warmup if args.n_warmup is not None else args.n_timepoints
        if args.n_timepoints < 2 or not 1 <= warmup <= args.n_timepoints:
            raise ValueError("require n-timepoints >= 2 and 1 <= n-warmup <= n-timepoints")
        if min(args.jump, args.k_neighbours, args.excl_radius, args.snap_grid, args.min_segment) < 1:
            raise ValueError("jump, k-neighbours, excl-radius, snap-grid, and min-segment must be positive")
        if args.progress_every < 0:
            raise ValueError("progress-every must be nonnegative")
        if isinstance(args.window_size, int) and 10 * args.window_size >= args.n_timepoints:
            raise ValueError("n-timepoints must exceed 10 * window-size to allow two ClaSS segments")
        config = {"n_timepoints": args.n_timepoints, "n_warmup": warmup,
                  "window_size": args.window_size, "k_neighbours": args.k_neighbours,
                  "distance": args.distance, "score": args.score, "jump": args.jump,
                  "validation": args.validation, "threshold": args.threshold, "excl_radius": args.excl_radius}
        values = read_float64(args.input, args.max_values)
        identity = {"dataset": args.dataset, "series": args.series, "sha256": file_sha256(args.input)}
        print("ClaSS pilot configuration: {}".format(json.dumps(config, sort_keys=True)), flush=True)
        print("ClaSS input: {} values; boundary snap={}, min-segment={}".format(
            len(values), args.snap_grid, args.min_segment), flush=True)
        events, resolved, elapsed, versions = segment_values(values, config, args.progress_every)
        intervals = normalize_boundaries([event["change_point"] for event in events], len(values),
                                         args.snap_grid, args.min_segment)
        manifest = {"adapter_version": ADAPTER_VERSION, "method": "ClaSS (PVLDB 2024)",
                    "implementation": "claspy.streaming.segmentation.StreamingClaSPSegmentation",
                    "implementation_source": SOURCE_URL, "configuration_kind": "ClaSS pilot configuration",
                    "input": os.path.abspath(args.input), "identity": identity, "dtype": "<f8",
                    "values": len(values), "input_file_values": os.path.getsize(args.input) // 8,
                    "max_values": args.max_values, "original_values_transformed": False,
                    "codec_costs_or_labels_used": False, "python": platform.python_version(),
                    "dependency_versions": versions, "requested_configuration": config,
                    "resolved_configuration": resolved, "snap_grid": args.snap_grid,
                    "snap_policy": "nearest grid point; ties to earlier point",
                    "min_segment": args.min_segment, "raw_change_points": [e["change_point"] for e in events],
                    "final_boundaries": [start for start, _ in intervals] + [len(values)],
                    "segments": len(intervals), "segmentation_seconds": elapsed,
                    "terminal_policy": "no synthetic flush; append exact series end; preserve every input value"}
        path = write_outputs(args.out_dir, identity, intervals, events, manifest)
        print("ClaSS boundaries frozen: {} ({} segments, {:.1f}s)".format(path, len(intervals), elapsed), flush=True)
        return 0
    except (ValueError, OSError, ImportError) as exc:
        print("segment_class: {}".format(exc), file=sys.stderr)
        return 2


if __name__ == "__main__":
    sys.exit(main())
