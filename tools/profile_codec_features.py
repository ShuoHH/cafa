#!/usr/bin/env python3
"""Collect raw-value codec features in fixed base windows.

This is deliberately separate from the oracle.  Features are computed from
the input bits only; codec execution is needed later to produce cost labels.
Rows are appended to an existing CSV and keyed by dataset/series/window,
manifest hash, and feature version, so repeated collection does not rewrite
or duplicate old rows.
"""

import argparse
import csv
import hashlib
import math
import os
import sys
from collections import Counter, deque

import numpy as np


# v2 adds ALP-oriented decimal precision features.  The version is part of
# the append key, so an old schema is never silently mixed with this one.
# v5 adds the eight numerical features defined by TSEC. Keep the new schema
# separate from earlier append-only feature tables.
FEATURE_VERSION = "codec_features_v5"
TZ_BANDS = (("tz_0_5", 0, 5), ("tz_6_19", 6, 19),
            ("tz_20_40", 20, 40), ("tz_41_63", 41, 63),
            ("tz_64", 64, 64))
LZ_BANDS = (("lz_0_7", 0, 7), ("lz_8_11", 8, 11),
            ("lz_12_15", 12, 15), ("lz_16_19", 16, 19),
            ("lz_20_23", 20, 23), ("lz_24_64", 24, 64))


def file_sha256(path):
    digest = hashlib.sha256()
    with open(path, "rb") as source:
        for block in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def bits_of_float(value):
    return int(np.asarray(value, dtype="<f8").view("<u8"))


def bits_of_array(values):
    return values.view("<u8")


def xor_shape(xor_values):
    """Return lz/tz arrays for nonzero 64-bit XOR values."""
    leading = []
    trailing = []
    for raw in xor_values.tolist():
        value = int(raw)
        if value == 0:
            continue
        trailing.append((value & -value).bit_length() - 1)
        leading.append(64 - value.bit_length())
    return np.asarray(leading, dtype=np.int64), np.asarray(trailing, dtype=np.int64)


def gamma_bits(length):
    return 2 * (int(length).bit_length() - 1) + 1


def exact_decimal_digits(value, max_digits=17):
    """Return the smallest exact decimal scale, or -1.

    The bit comparison intentionally rejects -0.0: converting it through an
    integer round trip produces +0.0, which is an ALP exception as well.
    """
    if not math.isfinite(value) or (value == 0.0 and math.copysign(1.0, value) < 0):
        return -1
    original_bits = bits_of_float(value)
    for digits in range(max_digits + 1):
        scale = 10.0 ** digits
        scaled = value * scale
        if not math.isfinite(scaled) or abs(scaled) >= 2.0 ** 53:
            continue
        integer = int(round(scaled))
        candidate = integer / scale
        if bits_of_float(candidate) == original_bits:
            return digits
    return -1


def decimal_features(values, max_digits=17):
    digits = np.asarray([exact_decimal_digits(float(v), max_digits)
                         for v in values], dtype=np.int64)
    valid = digits >= 0
    counts = Counter(int(d) for d in digits[valid].tolist())
    mode = min(counts, key=lambda d: (-counts[d], d)) if counts else -1
    exceptions = int(np.sum(digits != mode)) if mode >= 0 else int(values.size)

    integers = []
    if mode >= 0:
        scale = 10.0 ** mode
        for value, digit in zip(values.tolist(), digits.tolist()):
            if digit != mode:
                continue
            scaled = float(value) * scale
            if not math.isfinite(scaled) or abs(scaled) >= 2.0 ** 63:
                continue
            integers.append(int(round(scaled)))
    if integers:
        span = max(integers) - min(integers)
        int_bitwidth = int(span.bit_length())
    else:
        int_bitwidth = 0

    # Analytic ALP-like choice.  This is only a predictor feature; exact ALP
    # accounting still comes from the C++ ALP cost table.
    dec_star = -1
    dec_star_exception_share = 1.0
    dec_star_int_bitwidth = 64
    dec_star_delta_bitwidth = 64
    dec_scale_change_rate = 0.0
    dec_scaled_repeat_rate = 0.0
    dec_representable_share = 0.0
    best_cost = None
    for candidate in sorted(set(int(d) for d in digits[valid].tolist())):
        representable = valid & (digits <= candidate)
        if not np.any(representable):
            continue
        scale = 10.0 ** candidate
        scaled = np.rint(values[representable] * scale)
        if (not np.all(np.isfinite(scaled)) or
                np.any(np.abs(scaled) >= 2.0 ** 53)):
            continue
        integer_values = scaled.astype(np.int64)
        span = int(integer_values.max() - integer_values.min())
        bitwidth = int(span.bit_length())
        exception_count = int(values.size - integer_values.size)
        cost = bitwidth * int(integer_values.size) + exception_count * (64 + 16)
        if best_cost is None or cost < best_cost:
            best_cost = cost
            dec_star = int(candidate)
            dec_star_exception_share = exception_count / float(values.size)
            dec_star_int_bitwidth = bitwidth

    if dec_star >= 0:
        chosen = valid & (digits <= dec_star)
        dec_representable_share = float(np.mean(chosen))
        pair_mask = chosen[:-1] & chosen[1:]
        if np.any(pair_mask):
            scales = 10.0 ** dec_star
            scaled_all = np.zeros(values.size, dtype=np.int64)
            scaled_all[chosen] = np.rint(values[chosen] * scales).astype(np.int64)
            delta = scaled_all[1:][pair_mask] - scaled_all[:-1][pair_mask]
            max_abs_delta = int(np.max(np.abs(delta))) if delta.size else 0
            dec_star_delta_bitwidth = (max_abs_delta.bit_length() + 1
                                       if max_abs_delta else 0)
            dec_scaled_repeat_rate = float(np.mean(delta == 0))
        digit_pairs = valid[:-1] & valid[1:]
        if np.any(digit_pairs):
            dec_scale_change_rate = float(
                np.mean(digits[1:][digit_pairs] != digits[:-1][digit_pairs]))

    valid_digits = digits[valid]
    return {
        "dec_mode": mode,
        "dec_max": int(valid_digits.max()) if valid_digits.size else -1,
        "dec_mean": float(valid_digits.mean()) if valid_digits.size else float("nan"),
        "dec_exception_share": exceptions / float(values.size),
        "non_decimal_share": float(np.mean(~valid)),
        "int_bitwidth": int_bitwidth,
        "dec_star": dec_star,
        "dec_star_exception_share": dec_star_exception_share,
        "dec_star_int_bitwidth": dec_star_int_bitwidth,
        "dec_star_delta_bitwidth": dec_star_delta_bitwidth,
        "dec_scale_change_rate": dec_scale_change_rate,
        "dec_scaled_repeat_rate": dec_scaled_repeat_rate,
        "dec_representable_share": dec_representable_share,
    }


def one_window(values, bits, recurrent=None):
    width = values.size
    # TSEC Table 1: value/delta mean, variance and spread; Count(RS) counts
    # *positions* whose current run length is at least eight (not the number
    # of runs); Count(IS) counts strictly increasing adjacent pairs.
    if np.all(np.isfinite(values)):
        deltas = np.diff(values)
        tsec_stats = {
            "tsec_mean_ts": float(np.mean(values)),
            "tsec_var_ts": float(np.var(values)),
            "tsec_spread_ts": float(np.ptp(values)),
            "tsec_mean_ds": float(np.mean(deltas)),
            "tsec_var_ds": float(np.var(deltas)),
            "tsec_spread_ds": float(np.ptp(deltas)),
            "tsec_count_is": int(np.count_nonzero(deltas > 0)),
        }
        run_len = 1
        repeat_count = 0
        for same in (values[1:] == values[:-1]):
            run_len = run_len + 1 if same else 1
            repeat_count += run_len >= 8
        tsec_stats["tsec_count_rs_ge8"] = repeat_count
    else:
        tsec_stats = {name: float("nan") for name in (
            "tsec_mean_ts", "tsec_var_ts", "tsec_spread_ts",
            "tsec_mean_ds", "tsec_var_ds", "tsec_spread_ds",
            "tsec_count_rs_ge8", "tsec_count_is")}
    pair_bits = bits[1:] ^ bits[:-1]
    unequal = pair_bits != 0
    nonzero_xor = pair_bits[unequal]
    leading, trailing = xor_shape(nonzero_xor)

    exponent_fields = ((bits >> np.uint64(52)) & np.uint64(0x7FF)).astype(np.int64)
    unbiased_exponent = np.where(
        exponent_fields == 0, -1022, exponent_fields - 1023).astype(float)
    finite_exponent = np.isfinite(values) & (exponent_fields != 0x7FF)
    if np.any(finite_exponent):
        exp_values = unbiased_exponent[finite_exponent]
        exp_mean = float(np.mean(exp_values))
        exp_std = float(np.std(exp_values))
        exp_range = float(np.max(exp_values) - np.min(exp_values))
    else:
        exp_mean = float("nan")
        exp_std = float("nan")
        exp_range = float("nan")

    _, value_counts = np.unique(bits, return_counts=True)
    unique_share = float(np.sum(value_counts == 1) / width)

    row = {
        "ident_rate": float(np.mean(pair_bits == 0)),
        "runs_per_value": 1.0,
        "rle_bits": 0.0,
        "recur128": 0.0,
        "distinct_share": float(np.unique(bits).size / width),
        "unique_share": unique_share,
        "center_bits": float(np.mean(64 - leading - trailing)) if leading.size else 0.0,
        "rel_change": float("nan"),
        "exp_change_rate": float("nan"),
        "sign_change_rate": float(np.mean(
            ((bits[1:] ^ bits[:-1]) & np.uint64(1 << 63)) != 0)),
        "zero_share": float(np.mean(values == 0.0)),
        "sentinel_share": float(np.mean(np.isfinite(values) & (values <= -999.0))),
        "nonfinite_share": float(np.mean(~np.isfinite(values))),
        "exp_mean": exp_mean,
        "exp_std": exp_std,
        "exp_range": exp_range,
    }

    for name, low, high in TZ_BANDS + LZ_BANDS:
        row[name] = 0.0
    histogram = np.zeros(65, dtype=np.int64)
    if trailing.size:
        histogram += np.bincount(trailing, minlength=65)
        for name, low, high in TZ_BANDS:
            row[name] = float(histogram[low:high + 1].sum() / trailing.size)
        for name, low, high in LZ_BANDS:
            row[name] = float(np.sum((leading >= low) & (leading <= high)) / leading.size)
        row["center_bits"] = float(np.mean(64 - leading - trailing))
    for i in range(65):
        row[f"tz_hist_{i}"] = float(histogram[i] / trailing.size) if trailing.size else 0.0

    run_count = 0
    total_rle_bits = 0
    current = int(bits[0])
    run_length = 1
    for raw in bits[1:].tolist():
        value = int(raw)
        if value == current:
            run_length += 1
        else:
            run_count += 1
            total_rle_bits += gamma_bits(run_length)
            current = value
            run_length = 1
    run_count += 1
    total_rle_bits += gamma_bits(run_length)
    row["runs_per_value"] = run_count / float(width)
    row["rle_bits"] = total_rle_bits / float(width)

    if recurrent is None:
        recent = deque()
        recent_counts = Counter()
        recurrent_values = []
        for raw in bits.tolist():
            value = int(raw)
            recurrent_values.append(recent_counts[value] > 0)
            recent.append(value)
            recent_counts[value] += 1
            if len(recent) > 128:
                old = recent.popleft()
                recent_counts[old] -= 1
                if recent_counts[old] == 0:
                    del recent_counts[old]
        recurrent = np.asarray(recurrent_values, dtype=np.bool_)
    row["recur128"] = float(np.mean(recurrent))

    finite = np.isfinite(values)
    if width > 1:
        previous = values[:-1]
        current_values = values[1:]
        valid_pairs = finite[:-1] & finite[1:]
        delta = np.abs(current_values - previous)
        scale = np.maximum(np.abs(current_values), np.abs(previous))
        valid_pairs &= (delta != 0.0) & (scale != 0.0)
        if np.any(valid_pairs):
            row["rel_change"] = float(np.median(
                np.log2(scale[valid_pairs] / delta[valid_pairs])))
        row["exp_change_rate"] = float(
            np.mean(exponent_fields[1:] != exponent_fields[:-1]))

    row.update(decimal_features(values))
    row.update(tsec_stats)
    return row


def fieldnames():
    fields = ["dataset", "series", "start", "length", "manifest_sha256",
              "feature_version", "ident_rate", "rle_bits", "runs_per_value",
              "recur128", "distinct_share", "unique_share", "center_bits",
              "rel_change"]
    fields += [name for name, _, _ in TZ_BANDS]
    fields += [name for name, _, _ in LZ_BANDS]
    fields += [f"tz_hist_{i}" for i in range(65)]
    fields += ["dec_mode", "dec_max", "dec_mean", "dec_exception_share",
               "non_decimal_share", "int_bitwidth", "dec_star",
               "dec_star_exception_share", "dec_star_int_bitwidth",
               "dec_star_delta_bitwidth", "dec_scale_change_rate",
               "dec_scaled_repeat_rate", "dec_representable_share",
               "exp_change_rate",
                "sign_change_rate", "zero_share", "sentinel_share",
                "nonfinite_share", "exp_mean", "exp_std", "exp_range",
                "tsec_mean_ts", "tsec_var_ts", "tsec_spread_ts",
                "tsec_mean_ds", "tsec_var_ds", "tsec_spread_ds",
                "tsec_count_rs_ge8", "tsec_count_is"]
    return fields


def key(row):
    return tuple(str(row[name]) for name in
                 ("dataset", "series", "start", "length",
                  "manifest_sha256", "feature_version"))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("series", help="headerless little-endian double file")
    parser.add_argument("--window", type=int, default=256)
    parser.add_argument("--stride", type=int, default=None,
                        help="start spacing; defaults to the window length")
    parser.add_argument("--dataset", default=None)
    parser.add_argument("--series-name", default=None)
    parser.add_argument("--max-values", type=int, default=0)
    parser.add_argument("--out", required=True, help="append-only feature CSV")
    args = parser.parse_args()
    if args.window < 2:
        parser.error("--window must be at least 2")
    stride = args.window if args.stride is None else args.stride
    if stride < 1:
        parser.error("--stride must be positive")

    values = np.fromfile(args.series, dtype="<f8")
    if args.max_values:
        values = values[:args.max_values]
    if values.size < args.window:
        parser.error("series is shorter than one complete feature window")
    # Retain the full input: overlapping windows may begin after the last
    # non-overlapping window and still contain a complete vector.
    bits = bits_of_array(values)
    recent = deque()
    recent_counts = Counter()
    recurrent_flags = np.zeros(values.size, dtype=np.bool_)
    for index, raw in enumerate(bits.tolist()):
        value = int(raw)
        recurrent_flags[index] = recent_counts[value] > 0
        recent.append(value)
        recent_counts[value] += 1
        if len(recent) > 128:
            old = recent.popleft()
            recent_counts[old] -= 1
            if recent_counts[old] == 0:
                del recent_counts[old]
    manifest = file_sha256(args.series)
    dataset = args.dataset or os.path.basename(os.path.dirname(os.path.abspath(args.series)))
    series_name = args.series_name or os.path.splitext(os.path.basename(args.series))[0]
    names = fieldnames()

    existing_keys = set()
    output_exists = os.path.exists(args.out) and os.path.getsize(args.out) > 0
    if output_exists:
        with open(args.out, newline="") as source:
            reader = csv.DictReader(source)
            if reader.fieldnames != names:
                raise RuntimeError(f"feature table header mismatch: {args.out}")
            for row in reader:
                existing_keys.add(key(row))
    parent = os.path.dirname(os.path.abspath(args.out))
    os.makedirs(parent, exist_ok=True)
    added = 0
    skipped = 0
    with open(args.out, "a", newline="") as output:
        writer = csv.DictWriter(output, fieldnames=names)
        if not output_exists:
            writer.writeheader()
        for start in range(0, values.size - args.window + 1, stride):
            feature_row = one_window(values[start:start + args.window],
                                     bits[start:start + args.window],
                                     recurrent_flags[start:start + args.window])
            row = {"dataset": dataset, "series": series_name, "start": start,
                   "length": args.window, "manifest_sha256": manifest,
                   "feature_version": FEATURE_VERSION}
            row.update(feature_row)
            row_key = key(row)
            if row_key in existing_keys:
                skipped += 1
                continue
            writer.writerow(row)
            existing_keys.add(row_key)
            added += 1

    print(f"feature windows: {added} appended, {skipped} already present, "
          f"{(values.size - args.window) // stride + 1} total; output={args.out}")


if __name__ == "__main__":
    try:
        main()
    except (OSError, RuntimeError, ValueError) as error:
        print(f"profile_codec_features: {error}", file=sys.stderr)
        sys.exit(2)
