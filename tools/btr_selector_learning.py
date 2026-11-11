"""Raw-only features and held-out-series RF recommendation on frozen regions.

This module never runs a codec and never reads a test-region cost. Both feature
arms share the same RF capacity, length controls, training labels and regions.
The extra columns are exactly the existing decimal/residue feature family, not
a new feature search. Native pricing must happen AFTER callers persist all
returned predictions and, when requested, the returned fitted models.
"""

from __future__ import annotations

from collections import Counter
import hashlib
import math
from numbers import Integral
from pathlib import Path
import re
import sys
import time

import numpy as np
import pandas as pd

import profile_codec_features as ordinary
import profile_residue_features as residue
from rf_profiles import RF_REPRO_PARAMS


FEATURE_SETS = ("tsec", "tsec+decimal+residue")
TSEC_COLUMNS = (
    "tsec_mean_ts", "tsec_var_ts", "tsec_spread_ts",
    "tsec_mean_ds", "tsec_var_ds", "tsec_spread_ds",
    "tsec_count_rs_ge8", "tsec_count_is",
)
DECIMAL_COLUMNS = (
    "dec_star", "dec_star_exception_share", "non_decimal_share",
    "dec_star_int_bitwidth", "dec_star_delta_bitwidth",
    "dec_scale_change_rate", "dec_scaled_repeat_rate",
    "dec_representable_share", "dec_mode", "dec_max", "dec_mean",
    "dec_exception_share", "int_bitwidth",
)
RESIDUE_COLUMNS = (
    "decimal_scale", "decimal_scale_valid_share",
    "decimal_scale_exception_share", "scale_instability", "eligible_share",
    "residue_bits_med", "residue_bits_q90", "xbits_med", "xbits_q90",
    "dbits_med", "dbits_q95", "gap_med", "gap_q90", "residue_realization",
    "residue_dominated_share", "residue_dominated_share_0_4",
    "residue_dominated_share_0_6",
)
LENGTH_COLUMNS = ("segment_length", "log2_length")
ALL_COLUMNS = TSEC_COLUMNS + DECIMAL_COLUMNS + RESIDUE_COLUMNS + LENGTH_COLUMNS
KEY = ("dataset", "series", "sha256", "start", "length")
MAX_VALUES = 102400


def _integer(value, name, minimum=1, maximum=MAX_VALUES):
    if (isinstance(value, bool) or not isinstance(value, Integral)
            or not minimum <= value <= maximum):
        raise ValueError(f"{name} must be an integer in [{minimum},{maximum}]")
    return int(value)


def _identity(identity):
    if (not isinstance(identity, (tuple, list)) or len(identity) != 3
            or any(not isinstance(x, str) or not x.strip() for x in identity)):
        raise ValueError("identity must be (dataset, series, sha256) strings")
    dataset, series, digest = identity
    if not re.fullmatch(r"[0-9a-f]{64}", digest):
        raise ValueError("identity sha256 must be a lower-case SHA-256 hex digest")
    return dataset, series, digest


def feature_columns(name):
    """Return only predeclared raw feature columns; never infer them from CSV."""
    if name == "tsec":
        return list(TSEC_COLUMNS + LENGTH_COLUMNS)
    if name == "tsec+decimal+residue":
        return list(ALL_COLUMNS)
    raise ValueError(f"unknown feature set: {name!r}")


def training_regions(n, per_scale=8, lengths=(256, 1024, 4096)):
    """At most eight evenly spaced grid-64 starts per scale, plus the whole file.

    The sample positions depend only on the number of values, not features,
    labels or observed codec performance. Overlap among training examples is
    allowed; every example from a held-out raw source is excluded together.
    """
    n = _integer(n, "n", minimum=64)
    per_scale = _integer(per_scale, "per_scale", maximum=8)
    requested = [_integer(x, "training length", minimum=64) for x in lengths]
    if not requested or any(x % 64 for x in requested):
        raise ValueError("training lengths must be nonempty multiples of 64")
    regions = {(0, n)}
    for length in requested:
        if length > n:
            continue
        count = (n - length) // 64 + 1
        selected = min(count, per_scale)
        if selected == 1:
            indices = [(count - 1) // 2]
        else:
            indices = [i * (count - 1) // (selected - 1)
                       for i in range(selected)]
        regions.update((i * 64, length) for i in indices)
    return sorted(regions)


def _region(row):
    if isinstance(row, dict):
        start, length = row["start"], row["length"]
    else:
        if len(row) != 2:
            raise ValueError("a region must have start and length")
        start, length = row
    return (_integer(start, "start", minimum=0),
            _integer(length, "length"))


def _singleton(values):
    # ordinary.one_window computes ptp(diff(values)), which is undefined for
    # a singleton. Keep the actual decimal value and explicitly missing pair
    # statistics; do not duplicate the value and invent an observed repeat.
    row = ordinary.decimal_features(values)
    row.update(tsec_mean_ts=float(values[0]), tsec_var_ts=0.0,
               tsec_spread_ts=0.0, tsec_mean_ds=math.nan,
               tsec_var_ds=math.nan, tsec_spread_ds=math.nan,
               tsec_count_rs_ge8=0, tsec_count_is=0)
    return row


def feature_frame(raw_path, regions, identity):
    """Compute actual region features from unchanged finite little-endian f64.

    ``regions`` may contain (start,length) tuples or dictionaries. Singleton
    physical tails are supported, but no boundary is moved, merged or dropped.
    Undefined derived statistics remain NaN for the training-only imputer.
    """
    started = time.perf_counter()
    dataset, series, expected = _identity(identity)
    path = Path(raw_path).resolve()
    raw = path.read_bytes()
    if hashlib.sha256(raw).hexdigest() != expected:
        raise RuntimeError(f"raw input SHA-256 mismatch: {path}")
    if not raw or len(raw) % 8:
        raise ValueError("raw input must be nonempty headerless float64 bytes")
    n = len(raw) // 8
    if n > MAX_VALUES:
        raise ValueError(f"raw input exceeds {MAX_VALUES} values")
    first = int.from_bytes(raw[:8], "little")
    if first and first == n - 1:
        raise ValueError("input has a legacy uint64 count header")
    values = np.frombuffer(raw, dtype="<f8")
    if not np.isfinite(values).all():
        raise ValueError("raw feature input contains NaN or infinity")
    parsed = [_region(row) for row in regions]
    if not parsed:
        raise ValueError("no feature regions supplied")
    if len(set(parsed)) != len(parsed):
        raise ValueError("duplicate feature region")
    rows = []
    for start, length in parsed:
        if start + length > n:
            raise ValueError("feature region extends beyond raw input")
        window = values[start:start + length]
        with np.errstate(over="ignore", invalid="ignore", divide="ignore"):
            stats = (_singleton(window) if length == 1 else
                     ordinary.one_window(window, window.view("<u8")))
            stats.update(residue.one_window(window))
        stats.update(segment_length=length, log2_length=math.log2(length))
        row = {"dataset": dataset, "series": series, "sha256": expected,
               "start": start, "length": length}
        row.update({name: stats[name] for name in ALL_COLUMNS})
        if any(math.isinf(float(row[name])) for name in ALL_COLUMNS):
            raise ValueError("raw feature computation produced an infinite value")
        rows.append(row)
    if hashlib.sha256(path.read_bytes()).hexdigest() != expected:
        raise RuntimeError(f"raw input changed during feature computation: {path}")
    frame = pd.DataFrame(rows, columns=list(KEY) + list(ALL_COLUMNS))
    frame.attrs["feature_wall_seconds"] = time.perf_counter() - started
    frame.attrs["source_sha256"] = expected
    print(f"raw selector features {dataset}/{series}: {len(rows)} regions; "
          f"{frame.attrs['feature_wall_seconds']:.3f}s", file=sys.stderr)
    return frame


def _validate_frame(frame, *, training):
    if not isinstance(frame, pd.DataFrame) or frame.empty:
        raise ValueError("selector frame must be a nonempty DataFrame")
    required = list(KEY) + list(ALL_COLUMNS)
    if training:
        required.append("target_action")
    missing = [name for name in required if name not in frame.columns]
    if missing:
        raise ValueError("selector frame is missing: " + ",".join(missing))
    if frame.columns.duplicated().any():
        raise ValueError("selector frame has duplicate column names")
    if frame.duplicated(list(KEY)).any():
        raise ValueError("selector frame has duplicate region identities")
    source_map = {}
    hash_map = {}
    for row in frame.to_dict("records"):
        identity = _identity(tuple(row[name] for name in KEY[:3]))
        old = source_map.setdefault(identity[:2], identity[2])
        if old != identity[2]:
            raise ValueError("one source identity has multiple raw hashes")
        old_id = hash_map.setdefault(identity[2], identity[:2])
        if old_id != identity[:2]:
            raise ValueError("one raw hash has multiple source aliases")
        _integer(row["start"], "start", minimum=0)
        length = _integer(row["length"], "length")
        if row["segment_length"] != length:
            raise ValueError("segment_length differs from region length")
        if not math.isclose(float(row["log2_length"]), math.log2(length),
                            rel_tol=0, abs_tol=1e-12):
            raise ValueError("log2_length differs from region length")
    try:
        numeric = frame[list(ALL_COLUMNS)].to_numpy(dtype=np.float64)
    except (TypeError, ValueError) as error:
        raise ValueError("selector feature columns must be numeric") from error
    if np.isinf(numeric).any():
        raise ValueError("selector feature columns contain infinity")
    return sorted((dataset, series, digest)
                  for (dataset, series), digest in source_map.items())


def fit_predict(training_frame, test_frame, *, test_identity,
                feature_sets=FEATURE_SETS, top_ks=(1, 2), return_models=False):
    """Train on other raw sources and freeze feasible per-region RF rankings.

    No test labels or costs are consulted. Probability values remain the RF's
    original class probabilities after feasibility filtering (not renormalized).
    A k larger than the learned feasible class count returns the available
    classes only; ``recommended_count`` records this explicitly.
    """
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.impute import SimpleImputer
    from sklearn.pipeline import Pipeline
    from btr_selected_costs import feasible_actions

    identity = _identity(test_identity)
    training_sources = _validate_frame(training_frame, training=True)
    test_sources = _validate_frame(test_frame, training=False)
    if test_sources != [identity]:
        raise ValueError("test frame does not match the declared held-out source")
    if any(source[1] == identity[1] or source[2] == identity[2]
           for source in training_sources):
        raise ValueError("held-out series or raw SHA-256 leaks into training")
    feature_sets = tuple(feature_sets)
    top_ks = tuple(top_ks)
    if (not feature_sets or len(set(feature_sets)) != len(feature_sets)
            or not top_ks or len(set(top_ks)) != len(top_ks)):
        raise ValueError("feature sets and top-k values must be unique and nonempty")
    for name in feature_sets:
        feature_columns(name)
    for k in top_ks:
        _integer(k, "top_k", maximum=100)
    labels = training_frame["target_action"].tolist()
    for target, length in zip(labels, training_frame["length"]):
        if not isinstance(target, str) or target not in feasible_actions(int(length)):
            raise ValueError(f"training target is not a feasible action: {target!r}")
    predictions, details, models = [], [], {}
    for name in feature_sets:
        started = time.perf_counter()
        columns = feature_columns(name)
        model = Pipeline([
            ("imputer", SimpleImputer(strategy="median", keep_empty_features=True)),
            ("forest", RandomForestClassifier(**RF_REPRO_PARAMS)),
        ])
        model.fit(training_frame[columns], labels)
        training_seconds = time.perf_counter() - started
        predicting = time.perf_counter()
        probability = model.predict_proba(test_frame[columns])
        classes = list(map(str, model.named_steps["forest"].classes_))
        for index, row in enumerate(test_frame.to_dict("records")):
            feasible = set(feasible_actions(int(row["length"])))
            ranked = sorted(((action, float(probability[index, i]))
                             for i, action in enumerate(classes) if action in feasible),
                            key=lambda pair: (-pair[1], pair[0]))
            if not ranked:
                raise RuntimeError("model learned no feasible action for a test region")
            for k in top_ks:
                selected = ranked[:int(k)]
                predictions.append({
                    "dataset": identity[0], "series": identity[1], "sha256": identity[2],
                    "feature_set": name, "top_k": int(k),
                    "start": int(row["start"]), "length": int(row["length"]),
                    "ranked_actions": [action for action, _ in selected],
                    "probabilities": [value for _, value in selected],
                    "recommended_count": len(selected),
                    "all_ranked_actions": [action for action, _ in ranked],
                    "all_probabilities": [value for _, value in ranked],
                })
        prediction_seconds = time.perf_counter() - predicting
        details.append({
            "feature_set": name, "feature_columns": columns,
            "rf_parameters": dict(RF_REPRO_PARAMS),
            "imputer": {"strategy": "median", "keep_empty_features": True,
                        "fit_on_training_only": True},
            "training_rows": int(len(training_frame)),
            "training_sources": [list(source) for source in training_sources],
            "held_out_identity": list(identity),
            "training_class_counts": dict(sorted(Counter(labels).items())),
            "classes": classes,
            "training_length_counts": {str(k): int(v) for k, v in sorted(
                Counter(map(int, training_frame["length"])).items())},
            "training_wall_seconds": training_seconds,
            "prediction_wall_seconds": prediction_seconds,
            "test_costs_used": False, "boundary_search_used": False,
            "probability_normalized_after_feasibility_filter": False,
        })
        models[name] = model
        print(f"selector {name}: trained {len(training_frame)} regions from "
              f"{len(training_sources)} other sources in {training_seconds:.3f}s; "
              f"predicted {len(test_frame)} frozen regions in "
              f"{prediction_seconds:.3f}s", file=sys.stderr)
    return ((predictions, details, models) if return_models else
            (predictions, details))
