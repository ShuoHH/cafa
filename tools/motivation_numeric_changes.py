"""Ordinary numerical-change control for the introduction experiment.

This is a deterministic, untrained statistical heuristic, not ClaSS, an
implementation of an external algorithm, or a claim of optimal change-point
detection.  It inspects only numerical values: neither floating-point bits nor
codec labels, measured sizes, or compression features are used.

At each legal grid position, two adjacent windows are compared.  The score is
the standardized mean difference plus the absolute log standard-deviation
ratio.  The default threshold of 1.0 and window of 256 are declared heuristic
settings, not settings selected using the resulting compressed sizes.  A
global numerical rescaling is used only to keep arithmetic finite.  It does
not add candidates.  Greedy nonmaximum suppression separates retained changes
by at least one window (and at least the minimum segment length).
"""

from __future__ import annotations

import math
import numbers
import operator

import numpy as np


def _integer(name: str, value: object, *, minimum: int) -> int:
    if isinstance(value, (bool, np.bool_)):
        raise ValueError(f"{name} must be an integer >= {minimum}, not bool")
    try:
        result = operator.index(value)
    except TypeError as exc:
        raise ValueError(f"{name} must be an integer >= {minimum}") from exc
    if result < minimum:
        raise ValueError(f"{name} must be an integer >= {minimum}")
    return int(result)


def detect(
    values,
    *,
    window: int = 256,
    grid: int = 64,
    minimum: int = 64,
    threshold: float = 1.0,
    max_boundaries: int = 64,
) -> dict:
    """Return a complete layout and the selected numerical-change candidates.

    ``values`` must be a nonempty one-dimensional finite numerical array.
    Candidate positions are grid multiples with at least ``minimum`` values
    on either side.  Windows are clipped to the input at the ends.  A candidate
    must have a positive score at least ``threshold``; ties are resolved by
    earlier position.  Retained positions are separated by
    ``max(window, minimum)``.  There is no forced fixed-length or grid fill.

    ``intervals`` contains ``(start, length)`` pairs and preserves every value,
    including the final tail.  A short sequence that cannot be split legally
    is returned as one interval, even if its length is below ``minimum``.
    """
    window = _integer("window", window, minimum=1)
    grid = _integer("grid", grid, minimum=1)
    minimum = _integer("minimum", minimum, minimum=1)
    max_boundaries = _integer("max_boundaries", max_boundaries, minimum=0)
    if isinstance(threshold, (bool, np.bool_)) or not isinstance(threshold, numbers.Real):
        raise ValueError("threshold must be a finite nonnegative real number")
    threshold = float(threshold)
    if not math.isfinite(threshold) or threshold < 0:
        raise ValueError("threshold must be a finite nonnegative real number")

    array = np.asarray(values)
    if array.ndim != 1 or not array.size:
        raise ValueError("values must be a nonempty one-dimensional numerical array")
    if array.dtype.kind not in "biuf":
        raise ValueError("values must contain real numerical values")
    # Conversion can overflow a wider floating type; this is rejected below.
    with np.errstate(over="ignore", invalid="ignore"):
        array = array.astype(np.float64, copy=False)
    if not np.isfinite(array).all():
        raise ValueError("values must be finite and representable as float64")

    n = int(array.size)
    separation = max(window, minimum)
    first_position = ((minimum + grid - 1) // grid) * grid
    positions = range(first_position, n - minimum + 1, grid)
    config = {
        "method": "numeric_adjacent_window_mean_std_heuristic",
        "window": window,
        "grid": grid,
        "minimum": minimum,
        "threshold": threshold,
        "max_boundaries": max_boundaries,
        "nms_separation": separation,
        "edge_windows": "clipped",
        "threshold_policy": "predeclared_not_compression_tuned",
        "score": "standardized_mean_difference_plus_absolute_log_std_ratio",
        "std_floor_relative_global_std": 1e-6,
    }
    diagnostics = {
        "values": n,
        "legal_grid_positions": len(positions),
        "evaluated_positions": 0,
        "threshold_pass_candidates": 0,
        "nms_rejected_candidates": 0,
        "cap_rejected_candidates": 0,
        "selected_boundaries": 0,
    }

    def result(candidates: list[dict]) -> dict:
        boundaries = [0] + [row["position"] for row in candidates] + [n]
        return {
            "intervals": [(start, end - start) for start, end in zip(boundaries, boundaries[1:])],
            "candidates": candidates,
            "config": config,
            "diagnostics": diagnostics,
        }

    low, high = float(array.min()), float(array.max())
    if low == high or not positions or max_boundaries == 0:
        diagnostics["reason"] = (
            "constant_sequence" if low == high else
            "boundary_cap_zero" if max_boundaries == 0 else "no_legal_boundary"
        )
        return result([])

    # Half-before-add avoids overflow for finite opposite-sign extreme values.
    # Center-before-divide retains small representable changes at large offsets.
    center = low / 2.0 + high / 2.0
    normalized = array - center
    scale = float(np.max(np.abs(normalized)))
    normalized = normalized / scale
    global_std = float(np.std(normalized))
    floor = max(global_std * 1e-6, np.finfo(np.float64).eps * 32)
    passed = []
    for position in positions:
        left = normalized[max(0, position - window):position]
        right = normalized[position:min(n, position + window)]
        left_mean, right_mean = float(np.mean(left)), float(np.mean(right))
        left_std, right_std = float(np.std(left)), float(np.std(right))
        pooled_std = math.hypot(left_std, right_std) / math.sqrt(2.0)
        mean_shift = abs(left_mean - right_mean) / max(pooled_std, floor)
        std_change = abs(math.log((left_std + floor) / (right_std + floor)))
        score = mean_shift + std_change
        diagnostics["evaluated_positions"] += 1
        if score > 0.0 and score >= threshold:
            passed.append({
                "position": position,
                "score": score,
                "sources": "numeric_change",
                "strongest_group": "numeric",
                "state_contrast": score,
                "mean_shift": mean_shift,
                "std_change": std_change,
            })
    diagnostics["threshold_pass_candidates"] = len(passed)

    accepted = []
    for candidate in sorted(passed, key=lambda row: (-row["score"], row["position"])):
        if any(abs(candidate["position"] - row["position"]) < separation for row in accepted):
            diagnostics["nms_rejected_candidates"] += 1
        else:
            accepted.append(candidate)
    diagnostics["cap_rejected_candidates"] = max(0, len(accepted) - max_boundaries)
    selected = sorted(accepted[:max_boundaries], key=lambda row: row["position"])
    diagnostics["selected_boundaries"] = len(selected)
    diagnostics["reason"] = "changes_selected" if selected else "no_change_above_threshold"
    return result(selected)
