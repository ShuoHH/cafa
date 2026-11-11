"""Raw-only APPROXIMATE stored-bit priorities for boundary modifications.

This module is not an ALP/Gorilla/Chimp implementation, a learned cost model,
an exact cost oracle, or a certified pruning bound.  Its fixed models merely
rank proposed changes before the caller spends its native measurement budget.
Only actual measured stored bits may authorize a boundary modification.

The XOR model charges a raw first value, a one-bit repeat, and a significant
XOR field plus a fixed descriptor.  The RLE model adds gamma-coded run lengths.
The decimal model uses one trial decimal scale per segment and packs ranges in
1024-value vectors starting at THAT segment's start, with raw exceptions and an
explicit streaming tail.  Real codec state reuse, ALP's scale search, headers,
exception layout and integer representation differ from these approximations.

Preprocessing scans the input and each frozen decimal scale once.  Thereafter
stream queries use prefixes/run summaries, and decimal queries use per-offset
vector prefixes; queries never rescan all values of the requested region.
"""

from __future__ import annotations

import math
from numbers import Integral, Real
import time

import numpy as np


MODEL_VERSION = "approximate_compression_gain_v1"
VECTOR_SIZE = 1024
MAX_VALUES = 102400
MAX_SAFE_INTEGER = 2**53 - 1


def _integer(value, name, minimum=0):
    if isinstance(value, bool) or not isinstance(value, Integral) or value < minimum:
        raise ValueError(f"{name} must be an integer >= {minimum}")
    return int(value)


def _bit_width(values):
    """Exact integer bit widths without rounding uint64 through float64."""
    work = np.asarray(values, dtype=np.uint64).copy()
    result = np.zeros(work.shape, dtype=np.int64)
    for shift in (32, 16, 8, 4, 2, 1):
        mask = work >= np.uint64(1 << shift)
        result += mask.astype(np.int64) * shift
        work = np.where(mask, work >> np.uint64(shift), work)
    return result + (work != 0)


def _prefix(values):
    return np.concatenate((np.zeros(1, dtype=np.int64),
                           np.cumsum(values, dtype=np.int64)))


def _gamma_bits(length):
    return 2 * (int(length).bit_length() - 1) + 1


class CompressionGainProxy:
    """Fixed raw-data cost proxy; smaller estimates have higher priority.

    Legal regions start on ``grid`` and end on ``grid`` or at the input end.
    A short final region (including a final 2--63-value region) is retained.
    ``grid`` must divide ALP's 1024-value vector.  Singleton regions can be
    scored, but the caller remains responsible for its segmentation minimum.
    The input is copied and is never modified, even if preprocessing fails.
    """

    def __init__(self, values, metadata_bits=32, grid=64,
                 decimal_digits=tuple(range(13))):
        begin = time.perf_counter()
        self.grid = _integer(grid, "grid", 1)
        if VECTOR_SIZE % self.grid:
            raise ValueError("grid must divide the 1024-value vector size")
        if (isinstance(metadata_bits, bool) or not isinstance(metadata_bits, Real)
                or not math.isfinite(float(metadata_bits)) or metadata_bits < 0):
            raise ValueError("metadata_bits must be finite and nonnegative")
        self.metadata_bits = float(metadata_bits)
        digits = tuple(decimal_digits)
        if any(isinstance(d, bool) or not isinstance(d, Integral)
               or not 0 <= int(d) <= 12 for d in digits):
            raise ValueError("decimal_digits must contain integers from 0 to 12")
        if len(set(map(int, digits))) != len(digits):
            raise ValueError("decimal_digits must not contain duplicates")
        self.decimal_digits = tuple(sorted(map(int, digits)))
        array = np.asarray(values)
        if array.ndim != 1 or np.iscomplexobj(array):
            raise ValueError("values must be a one-dimensional real sequence")
        self.values = np.array(array, dtype=np.float64, copy=True)
        self.n = len(self.values)
        if not 1 <= self.n <= MAX_VALUES or not np.isfinite(self.values).all():
            raise ValueError("values must contain 1..102400 finite float64 values")
        self.values.setflags(write=False)
        self.config = {
            "model_version": MODEL_VERSION,
            "approximate": True,
            "certified_bound": False,
            "native_codec_calls": False,
            "training_used": False,
            "grid": self.grid,
            "metadata_bits": self.metadata_bits,
            "decimal_digits": list(self.decimal_digits),
            "decimal_scale_scope": "one_shared_scale_per_segment",
            "vector_size": VECTOR_SIZE,
            "vector_origin": "segment_start",
            "xor_changed_descriptor_bits": 13,
            "rle_descriptor_bits": 8,
            "decimal_vector_descriptor_bits": 32,
            "decimal_exception_bits": 80,
            "decimal_validity": "bitwise_scaled_roundtrip_and_safe_integer",
            "decimal_safe_integer_limit": MAX_SAFE_INTEGER,
            "negative_zero_is_decimal_exception": True,
            "tail_metadata_bits": self.metadata_bits,
            "query_value_rescans": False,
        }
        bits = self.values.view(np.uint64)
        xor = np.zeros(self.n, dtype=np.uint64)
        xor[1:] = bits[1:] ^ bits[:-1]
        # The low set bit gives trailing zeros; zero stays zero intentionally.
        low = xor & np.subtract(np.uint64(0), xor, dtype=np.uint64)
        significant = np.where(xor != 0, _bit_width(xor) - _bit_width(low) + 1, 0)
        changed = xor != 0
        increments = np.where(changed, significant + 13, 1).astype(np.int64)
        increments[0] = 0
        self._xor_prefix = _prefix(increments)
        self._changed_prefix = _prefix(np.where(changed, significant + 13, 0))
        self._run_starts = np.concatenate((np.array([0]), np.flatnonzero(changed)))
        self._run_ends = np.concatenate((self._run_starts[1:], np.array([self.n])))
        lengths = self._run_ends - self._run_starts
        self._run_gamma_prefix = _prefix(np.array([_gamma_bits(k) for k in lengths]))
        self._run_ids = np.cumsum(changed, dtype=np.int64)
        self._decimal_prefixes = {}
        self._decimal_exception_prefixes = {}
        self._prepare_decimal_vectors()
        self._cache = {}
        self.counters = {
            "preprocess_seconds": time.perf_counter() - begin,
            "preprocess_values": self.n,
            "preprocess_decimal_scale_values": (
                self.n * len(self.decimal_digits) if self.n >= VECTOR_SIZE else 0),
            "query_count": 0,
            "gain_query_count": 0,
            "cache_hits": 0,
            "unique_regions": 0,
            "query_value_rescans": 0,
            "native_codec_calls": 0,
            "training_rows": 0,
        }

    def _prepare_decimal_vectors(self):
        atoms_per_vector = VECTOR_SIZE // self.grid
        starts = np.arange(0, self.n, self.grid, dtype=np.int64)
        # There is no full decimal vector on shorter inputs.
        count = max(0, (self.n - VECTOR_SIZE) // self.grid + 1)
        if not count:
            return
        original_bits = self.values.view(np.uint64)
        negative_zero = (self.values == 0) & np.signbit(self.values)
        for digits in self.decimal_digits:
            factor = float(10**digits)
            with np.errstate(over="ignore", invalid="ignore", divide="ignore"):
                scaled = np.rint(self.values * factor)
                safe = np.isfinite(scaled) & (np.abs(scaled) <= MAX_SAFE_INTEGER)
                reconstructed = scaled / factor
            valid = safe & (reconstructed.view(np.uint64) == original_bits) & ~negative_zero
            integers = np.zeros(self.n, dtype=np.int64)
            integers[valid] = scaled[valid].astype(np.int64)
            exceptions = _prefix((~valid).astype(np.int64))
            self._decimal_exception_prefixes[digits] = exceptions
            min_atoms = np.minimum.reduceat(
                np.where(valid, integers, np.iinfo(np.int64).max), starts)
            max_atoms = np.maximum.reduceat(
                np.where(valid, integers, np.iinfo(np.int64).min), starts)
            # Sliding windows are views, not N x 1024 copies.  At most 1600 x
            # 16 entries for the default 64 grid; each point is not revisited
            # by any subsequent region query.
            low = np.lib.stride_tricks.sliding_window_view(min_atoms, atoms_per_vector)
            high = np.lib.stride_tricks.sliding_window_view(max_atoms, atoms_per_vector)
            lo = np.min(low[:count], axis=1)
            hi = np.max(high[:count], axis=1)
            vector_starts = np.arange(count, dtype=np.int64) * self.grid
            bad = exceptions[vector_starts + VECTOR_SIZE] - exceptions[vector_starts]
            good = VECTOR_SIZE - bad
            widths = np.zeros(count, dtype=np.int64)
            # Safe signed integers span fewer than 2**54, so subtraction fits.
            has_valid = good > 0
            widths[has_valid] = _bit_width((hi[has_valid] - lo[has_valid]).astype(np.uint64))
            # Allocate integer slots even for exceptions, then charge their
            # raw value/position separately.  This is deliberately a model,
            # not an assertion about ALP's actual exception representation.
            prices = 32 + VECTOR_SIZE * widths + 80 * bad
            self._decimal_prefixes[digits] = tuple(
                _prefix(prices[offset::atoms_per_vector])
                for offset in range(atoms_per_vector))

    def _region(self, start, length):
        start = _integer(start, "start")
        length = _integer(length, "length", 1)
        end = start + length
        if start % self.grid or end > self.n or (end != self.n and end % self.grid):
            raise ValueError("region must start on grid and end on grid or at input end")
        return start, end

    def _stream_payloads(self, start, end):
        length = end - start
        xor = 64 + int(self._xor_prefix[end] - self._xor_prefix[start + 1])
        first, last = int(self._run_ids[start]), int(self._run_ids[end - 1])
        if first == last:
            gamma = _gamma_bits(length)
        else:
            gamma = _gamma_bits(int(self._run_ends[first]) - start)
            gamma += _gamma_bits(end - int(self._run_starts[last]))
            gamma += int(self._run_gamma_prefix[last] - self._run_gamma_prefix[first + 1])
        rle = 64 + 8 + gamma + int(self._changed_prefix[end] - self._changed_prefix[start + 1])
        return {"raw_approx": float(64 * length),
                "xor_approx": float(xor), "rle_xor_approx": float(rle)}

    def _breakdown(self, start, end):
        components = {name: payload + self.metadata_bits
                      for name, payload in self._stream_payloads(start, end).items()}
        length = end - start
        full_vectors, tail_values = divmod(length, VECTOR_SIZE)
        best_decimal, best_digits = math.inf, None
        if full_vectors and self._decimal_prefixes:
            atom = start // self.grid
            stride = VECTOR_SIZE // self.grid
            offset, index = atom % stride, atom // stride
            for digits, offsets in self._decimal_prefixes.items():
                prefix = offsets[offset]
                price = int(prefix[index + full_vectors] - prefix[index])
                if price < best_decimal:
                    best_decimal, best_digits = float(price), digits
            best_decimal += self.metadata_bits
            if tail_values:
                tail_start = start + full_vectors * VECTOR_SIZE
                best_decimal += min(self._stream_payloads(tail_start, end).values())
                best_decimal += self.metadata_bits
            components["decimal_vector_approx"] = best_decimal
        winner = min(components, key=lambda name: (components[name], name))
        return {"cost_bits": float(components[winner]), "winning_proxy": winner,
                "components": components, "decimal_digits": best_digits,
                "full_vectors": full_vectors, "tail_values": tail_values,
                "approximate": True, "certified_bound": False}

    def cost(self, start, length):
        """Approximate stored bits; not an actual or guaranteed codec size."""
        start, end = self._region(start, length)
        self.counters["query_count"] += 1
        key = (start, end)
        if key in self._cache:
            self.counters["cache_hits"] += 1
        else:
            self._cache[key] = self._breakdown(start, end)
            self.counters["unique_regions"] = len(self._cache)
        return self._cache[key]["cost_bits"]

    def breakdown(self, start, length):
        """JSON-safe diagnostic copy including separate approximate channels."""
        start, end = self._region(start, length)
        self.cost(start, end - start)
        row = self._cache[(start, end)]
        return dict(row, components=dict(row["components"]))

    def gain(self, old_intervals, new_intervals):
        """Estimated NET saving, including every segment/tail descriptor.

        Inputs are ``(start, length)`` pairs.  They must partition the exact
        same contiguous parent region; changing/deleting coverage is not gain.
        The result only orders trials.  A positive value is not proof that a
        real codec will save space.
        """
        self.counters["gain_query_count"] += 1
        def checked(intervals):
            rows = [self._region(start, length) for start, length in intervals]
            if not rows or any(a[1] != b[0] for a, b in zip(rows, rows[1:])):
                raise ValueError("intervals must be a nonempty contiguous partition")
            return rows
        old, new = checked(old_intervals), checked(new_intervals)
        if (old[0][0], old[-1][1]) != (new[0][0], new[-1][1]):
            raise ValueError("old and new intervals must cover the same parent")
        return (sum(self.cost(s, e - s) for s, e in old)
                - sum(self.cost(s, e - s) for s, e in new))
