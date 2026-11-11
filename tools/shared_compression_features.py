"""Exact raw-only features shared by boundary construction and codec inference.

The bank owns one immutable binary64 snapshot, not codec prices or labels.
Value-local decimal digits and bit arrays are prepared once. Region statistics
remain region-local: window medians/scales are never substituted for a larger
or smaller segment. Existing feature definitions and old modules stay intact.
"""
from __future__ import annotations

import hashlib
import math
from pathlib import Path
import time

import numpy as np
import pandas as pd

import btr_selector_learning as learning


VERSION = "exact_shared_raw_feature_bank_v1"
KEY = learning.KEY
BASE_COLUMNS = learning.TSEC_COLUMNS + learning.LENGTH_COLUMNS
BOUNDARY_COLUMNS = (
    "ident_rate", "rle_bits", "center_bits", "exp_change_rate", "exp_mean",
    "dec_star", "dec_star_exception_share", "dec_star_int_bitwidth",
    "gap_med", "residue_dominated_share",
)
REPRESENTATION_COLUMNS = (
    "dec_star", "dec_star_exception_share", "dec_star_int_bitwidth",
    "dec_star_delta_bitwidth", "dec_scale_change_rate", "xbits_med", "gap_med",
    "residue_bits_med", "residue_dominated_share", "eligible_share",
)
COMPACT_COLUMNS = BASE_COLUMNS + REPRESENTATION_COLUMNS
COMPACT20_COLUMNS = COMPACT_COLUMNS
BOUNDARY20_COLUMNS = BASE_COLUMNS + BOUNDARY_COLUMNS
UNION_COLUMNS = tuple(dict.fromkeys(COMPACT_COLUMNS + BOUNDARY_COLUMNS))
_DECIMAL = REPRESENTATION_COLUMNS[:5]
_RELATION = REPRESENTATION_COLUMNS[5:]
_BOUNDARY_ONLY = BOUNDARY_COLUMNS[:5]
_FACTORS = tuple(10.0 ** digits for digits in range(18))
_LOG2_10 = math.log2(10.0)


def _readonly(array):
    array.setflags(write=False)
    return array


def _width(values):
    """Integer bit lengths; no logarithmic/float approximation at powers of 2."""
    work = np.asarray(values, dtype=np.uint64).copy()
    result = np.zeros(work.shape, dtype=np.int64)
    for shift in (32, 16, 8, 4, 2, 1):
        mask = work >= np.uint64(1 << shift)
        result += mask.astype(np.int64) * shift
        work = np.where(mask, work >> np.uint64(shift), work)
    return result + (work != 0)


def exact_decimal_digits(values):
    """Vectorized parity with the existing scalar search over scales 0..17.

    Multiplication, ties-to-even rounding, safe-integer limit and binary64
    reconstruction comparison are unchanged. In particular, -0 is rejected;
    rounded integers are cast back to float before division so integer zero
    has the same positive-zero behavior as Python's int(round(...)).
    """
    values = np.asarray(values, dtype=np.float64)
    if values.ndim != 1:
        raise ValueError("decimal digit input must be a one-dimensional binary64 array")
    bits = values.view(np.uint64)
    result = np.full(values.size, -1, dtype=np.int16)
    pending = np.isfinite(values) & ~((values == 0) & np.signbit(values))
    with np.errstate(over="ignore", invalid="ignore", divide="ignore", under="ignore"):
        for digits, factor in enumerate(_FACTORS):
            indices = np.flatnonzero(pending)
            if not indices.size:
                break
            scaled = values[indices] * factor
            safe = np.isfinite(scaled) & (np.abs(scaled) < 2.0 ** 53)
            indices, scaled = indices[safe], scaled[safe]
            if not indices.size:
                continue
            integer = np.rint(scaled).astype(np.int64)
            reconstructed = integer.astype(np.float64) / factor
            accepted = indices[reconstructed.view(np.uint64) == bits[indices]]
            result[accepted] = digits
            pending[accepted] = False
    return result


def _higher_quantile(values, q):
    if hasattr(np, "quantile"):
        try:
            return int(np.quantile(values, q, method="higher"))
        except TypeError:
            return int(np.quantile(values, q, interpolation="higher"))
    return int(np.percentile(values, q * 100.0, interpolation="higher"))


def _columns(columns):
    if isinstance(columns, (str, bytes)):
        raise ValueError("declare feature columns as an explicit sequence")
    result = tuple(columns)
    if not result or len(result) != len(set(result)) or any(c not in UNION_COLUMNS for c in result):
        raise ValueError("declare unique known shared feature columns")
    return result


def _gamma(length):
    return 2 * (int(length).bit_length() - 1) + 1


class FeatureBank:
    """A source/SHA-bound immutable snapshot plus exact region/column cache.

    ``row`` reads the immutable snapshot; it does not rehash a file per region.
    ``frame`` authenticates the source once per batch. Lifecycle callers using
    the boundary adapter must call ``assert_unchanged`` before/after their work.
    Only numeric feature scalars are retained per region, never full per-region
    pair arrays; DP's many unions therefore do not duplicate the source arrays.
    """

    def __init__(self, raw, identity):
        began = time.perf_counter()
        self.raw = Path(raw).resolve()
        self.identity = learning._identity(identity)
        self.source_sha256 = self.identity[2]
        tick = time.perf_counter()
        data = self.raw.read_bytes()
        if (not data or len(data) % 8 or len(data) // 8 > learning.MAX_VALUES
                or hashlib.sha256(data).hexdigest() != self.source_sha256):
            raise ValueError("shared feature raw bytes differ from the declared SHA/count contract")
        self.n = len(data) // 8
        first = int.from_bytes(data[:8], "little")
        if first and first == self.n - 1:
            raise ValueError("shared feature input contains a legacy uint64 count header")
        self.values = _readonly(np.frombuffer(data, dtype="<f8"))
        if not np.isfinite(self.values).all():
            raise ValueError("shared feature raw values must be finite binary64")
        self.bits = _readonly(self.values.view("<u8"))
        read_seconds = time.perf_counter() - tick
        tick = time.perf_counter()
        self.decimal_digits = _readonly(exact_decimal_digits(self.values))
        decimal_seconds = time.perf_counter() - tick
        tick = time.perf_counter()
        xor = np.zeros(self.n, dtype=np.uint64)
        xor[1:] = self.bits[1:] ^ self.bits[:-1]
        self.xor = _readonly(xor)
        low = xor & np.subtract(np.uint64(0), xor, dtype=np.uint64)
        self.xor_widths = _readonly(np.where(xor != 0, _width(xor) - _width(low) + 1, 0))
        fields = ((self.bits >> np.uint64(52)) & np.uint64(0x7FF)).astype(np.int64)
        self.exponent_fields = _readonly(fields)
        self.exponents = _readonly(np.where(fields == 0, -1022, fields - 1023).astype(np.int64))
        changed = xor != 0
        self._run_starts = _readonly(np.concatenate((np.array([0]), np.flatnonzero(changed))))
        self._run_ends = _readonly(np.concatenate((self._run_starts[1:], np.array([self.n]))))
        self._run_ids = _readonly(np.cumsum(changed, dtype=np.int64))
        run_gamma = np.asarray([_gamma(n) for n in self._run_ends - self._run_starts], dtype=np.int64)
        self._run_gamma_prefix = _readonly(np.concatenate((np.zeros(1, dtype=np.int64), np.cumsum(run_gamma, dtype=np.int64))))
        bit_seconds = time.perf_counter() - tick
        self._rows = {}
        self._boundary_adapter = None
        self.counters = dict(
            preprocess_seconds=time.perf_counter() - began, preprocess_values=self.n,
            exact_decimal_digit_passes=1, requested_regions=0, new_regions=0,
            cached_regions=0, partial_regions=0, requested_fields=0, computed_fields=0,
            source_verifications=1, source_verification_seconds=read_seconds,
            stage_seconds=dict(read_and_hash=read_seconds, decimal_digits=decimal_seconds,
                bit_arrays=bit_seconds, basic=0.0, decimal=0.0, relation=0.0,
                boundary=0.0, verification=0.0),
        )

    def assert_unchanged(self):
        tick = time.perf_counter()
        data = self.raw.read_bytes()
        good = len(data) == self.n * 8 and hashlib.sha256(data).hexdigest() == self.source_sha256
        elapsed = time.perf_counter() - tick
        self.counters["source_verifications"] += 1
        self.counters["source_verification_seconds"] += elapsed
        self.counters["stage_seconds"]["verification"] += elapsed
        if not good:
            raise ValueError("shared feature source changed after its authenticated snapshot")
        return True

    def _region(self, start, length):
        start = learning._integer(start, "start", minimum=0)
        length = learning._integer(length, "length")
        if start + length > self.n:
            raise ValueError("shared feature region extends beyond source")
        return start, length

    def _basic(self, start, length):
        values = self.values[start:start + length]
        if length == 1:
            return dict(tsec_mean_ts=float(values[0]), tsec_var_ts=0.0, tsec_spread_ts=0.0,
                tsec_mean_ds=math.nan, tsec_var_ds=math.nan, tsec_spread_ds=math.nan,
                tsec_count_rs_ge8=0, tsec_count_is=0,
                segment_length=length, log2_length=math.log2(length))
        delta = np.diff(values)
        run, count = 1, 0
        for same in values[1:] == values[:-1]:
            run = run + 1 if same else 1
            count += run >= 8
        return dict(tsec_mean_ts=float(np.mean(values)), tsec_var_ts=float(np.var(values)),
            tsec_spread_ts=float(np.ptp(values)), tsec_mean_ds=float(np.mean(delta)),
            tsec_var_ds=float(np.var(delta)), tsec_spread_ds=float(np.ptp(delta)),
            tsec_count_rs_ge8=count, tsec_count_is=int(np.count_nonzero(delta > 0)),
            segment_length=length, log2_length=math.log2(length))

    def _decimal(self, start, length):
        values, digits = self.values[start:start + length], self.decimal_digits[start:start + length]
        valid = digits >= 0
        row = dict(dec_star=-1, dec_star_exception_share=1.0, dec_star_int_bitwidth=64,
            dec_star_delta_bitwidth=64, dec_scale_change_rate=0.0)
        best = None
        for candidate in np.unique(digits[valid]).tolist():
            representable = valid & (digits <= candidate)
            scaled = np.rint(values[representable] * _FACTORS[candidate])
            if not np.isfinite(scaled).all() or np.any(np.abs(scaled) >= 2.0 ** 53):
                continue
            integers = scaled.astype(np.int64)
            width = int(int(integers.max() - integers.min()).bit_length())
            exceptions = int(length - integers.size)
            cost = width * int(integers.size) + exceptions * 80
            if best is None or cost < best:
                best = cost
                row.update(dec_star=int(candidate), dec_star_exception_share=exceptions / float(length),
                    dec_star_int_bitwidth=width)
        if row["dec_star"] >= 0:
            chosen = valid & (digits <= row["dec_star"])
            pair = chosen[:-1] & chosen[1:]
            if np.any(pair):
                scaled = np.zeros(length, dtype=np.int64)
                scaled[chosen] = np.rint(values[chosen] * _FACTORS[row["dec_star"]]).astype(np.int64)
                delta = scaled[1:][pair] - scaled[:-1][pair]
                maximum = int(np.max(np.abs(delta)))
                row["dec_star_delta_bitwidth"] = maximum.bit_length() + 1 if maximum else 0
            pair = valid[:-1] & valid[1:]
            if np.any(pair):
                row["dec_scale_change_rate"] = float(np.mean(digits[1:][pair] != digits[:-1][pair]))
        return row

    def _relation(self, start, length):
        row = dict(xbits_med=math.nan, gap_med=math.nan, residue_bits_med=math.nan,
                   residue_dominated_share=math.nan, eligible_share=0.0)
        if length < 2:
            return row
        values = self.values[start:start + length]
        bits = self.bits[start:start + length]
        digits = self.decimal_digits[start:start + length]
        valid_digits = digits[digits >= 0]
        if valid_digits.size < 0.9 * length:
            return row
        digit = _higher_quantile(valid_digits, 0.9)
        valid = (digits >= 0) & (digits <= digit) & np.isfinite(values)
        scaled_float = np.zeros(length, dtype=np.float64)
        scaled_float[valid] = np.rint(values[valid] * _FACTORS[digit])
        valid &= np.isfinite(scaled_float)
        valid &= np.abs(scaled_float) < 2.0 ** 53
        scaled = np.zeros(length, dtype=np.int64)
        scaled[valid] = scaled_float[valid].astype(np.int64)
        pair = valid[:-1] & valid[1:]
        pair &= values[:-1] != 0.0
        pair &= values[1:] != 0.0
        pair &= (bits[:-1] >> np.uint64(63)) == (bits[1:] >> np.uint64(63))
        # Exclude the cross-boundary XOR at global position start.
        xor = self.xor[start + 1:start + length]
        pair &= xor != 0
        count = int(np.sum(pair))
        row["eligible_share"] = count / float(length - 1)
        if count < 16:
            return row
        indices = np.flatnonzero(pair)
        xbits = self.xor_widths[start + 1:start + length][indices].astype(np.float64)
        delta = scaled[indices + 1] - scaled[indices]
        absolute = np.abs(delta).astype(np.uint64)
        dbits = np.where(absolute != 0, _width(absolute) + 1, 0).astype(np.float64)
        gap = xbits - dbits
        larger = np.where(np.abs(values[indices + 1]) >= np.abs(values[indices]), indices + 1, indices)
        residue = np.clip(52.0 - self.exponents[start:start + length][larger] - digit * _LOG2_10, 0.0, 52.0)
        row.update(xbits_med=float(np.percentile(xbits, 50)), gap_med=float(np.percentile(gap, 50)),
            residue_bits_med=float(np.percentile(residue, 50)),
            residue_dominated_share=float(np.mean(gap >= 0.5 * xbits)))
        return row

    def _boundary(self, start, length):
        end = start + length
        xor, widths = self.xor[start + 1:end], self.xor_widths[start + 1:end]
        first, last = int(self._run_ids[start]), int(self._run_ids[end - 1])
        if first == last:
            gamma = _gamma(length)
        else:
            gamma = (_gamma(int(self._run_ends[first]) - start)
                + _gamma(end - int(self._run_starts[last]))
                + int(self._run_gamma_prefix[last] - self._run_gamma_prefix[first + 1]))
        return dict(ident_rate=float(np.mean(xor == 0)) if length > 1 else math.nan,
            rle_bits=gamma / float(length),
            center_bits=float(np.mean(widths[xor != 0])) if np.any(xor != 0) else 0.0,
            exp_change_rate=float(np.mean(self.exponent_fields[start + 1:end] != self.exponent_fields[start:end - 1])) if length > 1 else math.nan,
            exp_mean=float(np.mean(self.exponents[start:end].astype(float))))

    def row(self, start, length, columns):
        columns = _columns(columns)
        key = self._region(start, length)
        counters = self.counters
        counters["requested_regions"] += 1
        counters["requested_fields"] += len(columns)
        if key not in self._rows:
            self._rows[key] = {}
            counters["new_regions"] += 1
        elif all(name in self._rows[key] for name in columns):
            counters["cached_regions"] += 1
        else:
            counters["partial_regions"] += 1
        cached = self._rows[key]
        with np.errstate(over="ignore", invalid="ignore", divide="ignore", under="ignore"):
            for names, stage, calculate in (
                    (BASE_COLUMNS, "basic", self._basic), (_DECIMAL, "decimal", self._decimal),
                    (_RELATION, "relation", self._relation), (_BOUNDARY_ONLY, "boundary", self._boundary)):
                if any(name in columns and name not in cached for name in names):
                    tick = time.perf_counter()
                    computed = calculate(*key)
                    counters["stage_seconds"][stage] += time.perf_counter() - tick
                    counters["computed_fields"] += sum(name not in cached for name in computed)
                    cached.update(computed)
        result = {name: cached[name] for name in columns}
        if any(math.isinf(float(value)) for value in result.values()):
            raise ValueError("shared raw feature computation produced an infinite value")
        return result

    def frame(self, regions, columns):
        tick = time.perf_counter()
        columns = _columns(columns)
        parsed = [self._region(*learning._region(region)) for region in regions]
        if not parsed or len(set(parsed)) != len(parsed):
            raise ValueError("shared feature frame requires unique nonempty regions")
        self.assert_unchanged()
        rows = []
        for start, length in parsed:
            row = dict(zip(KEY[:3], self.identity), start=start, length=length)
            row.update(self.row(start, length, columns))
            rows.append(row)
        frame = pd.DataFrame(rows, columns=list(KEY) + list(columns))
        frame.attrs.update(feature_wall_seconds=time.perf_counter() - tick,
            source_sha256=self.source_sha256, shared_feature_version=VERSION,
            feature_columns=list(columns), raw_only=True)
        return frame

    def boundary_cache(self):
        if self._boundary_adapter is None:
            self._boundary_adapter = _BoundaryCache(self)
        return self._boundary_adapter


class _BoundaryCache:
    """The existing build_candidates cache interface, with all ten fields."""
    def __init__(self, bank):
        self.bank, self.values, self.source = bank, bank.values, bank.identity
        self.rows, self.seconds, self.values_scanned = {}, 0.0, 0

    def row(self, start, length):
        key = self.bank._region(start, length)
        if key[1] < 2:
            raise ValueError("boundary feature windows require at least two values")
        if key not in self.rows:
            tick = time.perf_counter()
            row = dict(zip(KEY[:3], self.source), start=key[0], length=key[1],
                source_kind="class_segment", segment_length=key[1],
                segment_log2_length=math.log2(key[1]), log2_length=math.log2(key[1]))
            row.update(self.bank.row(*key, BOUNDARY_COLUMNS))
            self.rows[key] = row
            self.seconds += time.perf_counter() - tick
            self.values_scanned += key[1]
        return dict(self.rows[key])


SourceSharedFeatureBank = FeatureBank
