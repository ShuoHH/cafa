#!/usr/bin/env python3
"""Compute decimal/binary residue interaction features from raw float64 values.

The features are deliberately computed without running any codec or reading
codec labels.  They describe whether a small decimal change is accompanied by
large IEEE-754 XOR changes.  The output is an append-only CSV keyed by
dataset/series/window/file hash and feature version, so repeated collection is
safe.

The main quantities are:

* ``residue_bits``: an analytic estimate of mantissa bits below the decimal
  quantum, ``52 - exponent - d*log2(10)``;
* ``xbits``: the center width of the adjacent-value XOR;
* ``dbits``: the signed bit width of the scaled-decimal difference;
* ``gap = xbits - dbits``: the part of the XOR width not explained by the
  scaled decimal change;
* ``residue_realization`` and ``residue_dominated_share``: interaction
  features connecting the decimal grid to the observed XOR structure.

These are analysis features, not exact codec costs.  Exact codec accounting
must still come from the C++ label harness.
"""

import argparse
import csv
import hashlib
import math
import os
from collections import Counter

import numpy as np


FEATURE_VERSION = "residue_features_v1"
LOG2_10 = math.log2(10.0)
MAX_EXACT_DECIMAL_DIGITS = 17


def file_sha256(path):
    digest = hashlib.sha256()
    with open(path, "rb") as source:
        for block in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def bits_of_float(value):
    return int(np.asarray(value, dtype="<f8").view("<u8"))


def exact_decimal_digits(value, max_digits=MAX_EXACT_DECIMAL_DIGITS):
    """Smallest exact decimal scale for a float, or -1 if none is found."""
    if (not math.isfinite(value) or
            (value == 0.0 and math.copysign(1.0, value) < 0)):
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


def decimal_digits(values):
    return np.asarray([exact_decimal_digits(float(value)) for value in values],
                      dtype=np.int16)


def higher_quantile_integer(values, q):
    """Return a conservative integer quantile, compatible with old NumPy."""
    if values.size == 0:
        return None
    # NumPy 1.14 (still common on the server) has percentile but not
    # quantile.  ``higher`` is important here: the selected decimal scale
    # must represent at least 90% of the values, never fewer.
    if hasattr(np, "quantile"):
        try:
            result = np.quantile(values, q, method="higher")
        except TypeError:  # NumPy < 1.22
            result = np.quantile(values, q, interpolation="higher")
    else:
        result = np.percentile(values, q * 100.0, interpolation="higher")
    return int(result)


def percentile(values, q):
    if values.size == 0:
        return float("nan")
    return float(np.percentile(values, q))


def exponent_of_bits(raw_bits):
    """Return unbiased exponent; subnormals use the IEEE minimum exponent."""
    fields = (raw_bits >> np.uint64(52)) & np.uint64(0x7FF)
    return np.where(fields == 0, -1022,
                    fields.astype(np.int64) - 1023).astype(np.int64)


def xor_center_width(value):
    value = int(value)
    if value == 0:
        return 0
    trailing = (value & -value).bit_length() - 1
    leading = 64 - value.bit_length()
    return 64 - leading - trailing


def decimal_scale(values, digits):
    """Choose a conservative 90% decimal scale for one window."""
    finite_digits = digits[digits >= 0]
    if finite_digits.size < 0.9 * values.size:
        return -1
    # Invalid/non-decimal values are allowed up to the 10% exception budget,
    # but must not become a fictitious ``d=18`` scale at the quantile itself.
    return higher_quantile_integer(finite_digits, 0.9)


def empty_interaction_row():
    return {
        "decimal_scale": -1,
        "decimal_scale_valid_share": 0.0,
        "decimal_scale_exception_share": 1.0,
        "scale_instability": float("nan"),
        "eligible_share": 0.0,
        "eligible_pairs": 0,
        "residue_bits_med": float("nan"),
        "residue_bits_q90": float("nan"),
        "xbits_med": float("nan"),
        "xbits_q90": float("nan"),
        "dbits_med": float("nan"),
        "dbits_q95": float("nan"),
        "gap_med": float("nan"),
        "gap_q90": float("nan"),
        "residue_realization": float("nan"),
        "residue_dominated_share": float("nan"),
        "residue_dominated_share_0_4": float("nan"),
        "residue_dominated_share_0_6": float("nan"),
    }


def one_window(values):
    """Compute residue features for one complete window."""
    row = empty_interaction_row()
    width = int(values.size)
    if width < 2:
        return row

    bits = values.view("<u8")
    digits = decimal_digits(values)
    scale_digit = decimal_scale(values, digits)
    row["decimal_scale"] = scale_digit
    row["decimal_scale_valid_share"] = float(np.mean(
        (digits >= 0) & (digits <= scale_digit))) if scale_digit >= 0 else 0.0
    row["decimal_scale_exception_share"] = 1.0 - row["decimal_scale_valid_share"]

    finite_digit_pairs = (digits[:-1] >= 0) & (digits[1:] >= 0)
    if np.any(finite_digit_pairs):
        row["scale_instability"] = float(np.mean(
            digits[1:][finite_digit_pairs] != digits[:-1][finite_digit_pairs]))
    if scale_digit < 0:
        return row

    scale = 10.0 ** scale_digit
    representable = ((digits >= 0) & (digits <= scale_digit) &
                     np.isfinite(values))
    scaled_float = np.zeros(width, dtype=np.float64)
    scaled_float[representable] = np.rint(values[representable] * scale)
    representable &= np.isfinite(scaled_float)
    representable &= np.abs(scaled_float) < 2.0 ** 53
    scaled = np.zeros(width, dtype=np.int64)
    scaled[representable] = scaled_float[representable].astype(np.int64)

    pair_ok = representable[:-1] & representable[1:]
    pair_ok &= values[:-1] != 0.0
    pair_ok &= values[1:] != 0.0
    pair_ok &= ((bits[:-1] >> np.uint64(63)) ==
                (bits[1:] >> np.uint64(63)))
    pair_xor = bits[1:] ^ bits[:-1]
    pair_ok &= pair_xor != 0
    pair_count = int(np.sum(pair_ok))
    row["eligible_pairs"] = pair_count
    row["eligible_share"] = pair_count / float(width - 1)
    if pair_count < 16:
        return row

    indices = np.flatnonzero(pair_ok)
    xbits = np.asarray([xor_center_width(pair_xor[index]) for index in indices],
                       dtype=np.float64)
    delta = scaled[indices + 1] - scaled[indices]
    abs_delta = np.abs(delta).astype(np.uint64)
    dbits = np.asarray([int(value).bit_length() + 1 if int(value) else 0
                        for value in abs_delta], dtype=np.float64)
    gap = xbits - dbits

    exponents = exponent_of_bits(bits)
    larger = np.where(np.abs(values[indices + 1]) >=
                      np.abs(values[indices]), indices + 1, indices)
    residue = np.clip(52.0 - exponents[larger] - scale_digit * LOG2_10,
                      0.0, 52.0)

    row["residue_bits_med"] = percentile(residue, 50)
    row["residue_bits_q90"] = percentile(residue, 90)
    row["xbits_med"] = percentile(xbits, 50)
    row["xbits_q90"] = percentile(xbits, 90)
    row["dbits_med"] = percentile(dbits, 50)
    row["dbits_q95"] = percentile(dbits, 95)
    row["gap_med"] = percentile(gap, 50)
    row["gap_q90"] = percentile(gap, 90)

    residue_median = row["residue_bits_med"]
    if residue_median > 0:
        row["residue_realization"] = float(np.clip(
            np.median(gap) / residue_median, -0.2, 1.2))
    for threshold, name in ((0.4, "residue_dominated_share_0_4"),
                            (0.5, "residue_dominated_share"),
                            (0.6, "residue_dominated_share_0_6")):
        row[name] = float(np.mean(gap >= threshold * xbits))
    return row


def fieldnames():
    return [
        "dataset", "series", "start", "length", "manifest_sha256",
        "feature_version", "decimal_scale", "decimal_scale_valid_share",
        "decimal_scale_exception_share", "scale_instability",
        "eligible_share", "eligible_pairs", "residue_bits_med",
        "residue_bits_q90", "xbits_med", "xbits_q90", "dbits_med",
        "dbits_q95", "gap_med", "gap_q90", "residue_realization",
        "residue_dominated_share", "residue_dominated_share_0_4",
        "residue_dominated_share_0_6",
    ]


def row_key(row):
    return tuple(str(row[name]) for name in (
        "dataset", "series", "start", "length", "manifest_sha256",
        "feature_version"))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("series", help="headerless little-endian float64 file")
    parser.add_argument("--window", type=int, default=1024)
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

    manifest = file_sha256(args.series)
    dataset = args.dataset or os.path.basename(
        os.path.dirname(os.path.abspath(args.series)))
    series_name = args.series_name or os.path.splitext(
        os.path.basename(args.series))[0]
    names = fieldnames()

    existing_keys = set()
    output_exists = os.path.exists(args.out) and os.path.getsize(args.out) > 0
    if output_exists:
        with open(args.out, newline="") as source:
            reader = csv.DictReader(source)
            if reader.fieldnames != names:
                raise RuntimeError(f"feature table header mismatch: {args.out}")
            for old_row in reader:
                existing_keys.add(row_key(old_row))

    parent = os.path.dirname(os.path.abspath(args.out))
    os.makedirs(parent, exist_ok=True)
    added = 0
    skipped = 0
    with open(args.out, "a", newline="") as output:
        writer = csv.DictWriter(output, fieldnames=names)
        if not output_exists:
            writer.writeheader()
        for start in range(0, values.size - args.window + 1, stride):
            row = {
                "dataset": dataset,
                "series": series_name,
                "start": start,
                "length": args.window,
                "manifest_sha256": manifest,
                "feature_version": FEATURE_VERSION,
            }
            row.update(one_window(values[start:start + args.window]))
            if row_key(row) in existing_keys:
                skipped += 1
                continue
            writer.writerow(row)
            existing_keys.add(row_key(row))
            added += 1
    print(f"residue windows: {added} appended, {skipped} already present, "
          f"{added + skipped} total; output={args.out}")


if __name__ == "__main__":
    main()
