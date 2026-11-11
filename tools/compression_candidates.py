"""Raw-only boundary sources for budget-matched candidate-source ablations.

The four modes have the SAME maximum pool size, grid and legal endpoint rules.
They do not promise identical evidence or trial counts.  Class-only does not
read compression-state features; compression-only never reads ClaSS intervals;
random reads neither.  Class/compression pools expand from their own nominated
anchors, never from hidden random or fixed-grid fillers.  Without an authentic
anchor their pool is empty, and the caller must report the actual count.

Scores are dimensionless proposal priorities, NOT estimates of saved bits.
No codec execution, labels or measured costs are accepted by this module.
"""

from __future__ import annotations

import math
from numbers import Integral
import random


MODES = ("compression", "class", "union", "random")
STATE_GROUPS = {
    "repetition": (("ident_rate", 1.), ("rle_bits", 64.)),
    "xor": (("center_bits", 64.), ("exp_change_rate", 1.)),
    "magnitude": (("exp_mean", 64.),),
    "decimal": (("dec_star", 17.), ("dec_star_exception_share", 1.),
                ("dec_star_int_bitwidth", 64.)),
    "residue": (("gap_med", 64.), ("residue_dominated_share", 1.)),
}


def state_difference(left, right):
    """Normalized raw-state contrast; preserve absolute magnitude information."""
    groups = {}
    for group, entries in STATE_GROUPS.items():
        terms = []
        for name, scale in entries:
            a, b = float(left.get(name, math.nan)), float(right.get(name, math.nan))
            if math.isfinite(a) and math.isfinite(b):
                terms.append(min(abs(a-b)/scale, 2.))
            elif math.isfinite(a) != math.isfinite(b):
                terms.append(1.)
        groups[group] = sum(terms)/len(terms) if terms else 0.
    return sum(groups.values()), groups


def _positive_integer(value, name, minimum=1):
    if isinstance(value, bool) or not isinstance(value, Integral) or value < minimum:
        raise ValueError(f"{name} must be an integer >= {minimum}")
    return int(value)


def _uniform_subset(points, count):
    """Endpoint-spanning deterministic sample, including count=1 safely."""
    if len(points) <= count:
        return points
    if count == 1:
        return [points[len(points)//2]]
    # Integer arithmetic avoids rounding-dependent or duplicate selections.
    return [points[index*(len(points)-1)//(count-1)] for index in range(count)]


def build_candidates(cache, class_intervals, *, mode="union", grid=64,
                     minimum=64, window=256, maximum=256, random_seed=0):
    """Return ``position,score,sources,strongest_group,state_contrast`` rows.

    ClaSS anchors are snapped down to the common grid, then uniformly capped.
    Compression anchors are fixed-window adjacent-state contrasts, ranked by
    contrast with position as deterministic tie-break.  Union retains v1's
    anchor quotas (up to half the pool for class, at most 64 compression
    anchors); the tiny maximum=1 case now retains a real anchor rather than
    requesting a zero-element sample.

    All authentic anchors are retained before expansion whenever they fit the
    cap.  Nearby proposals first use v1's offsets -grid,+grid,-window,+window.
    If space remains, progressively farther grid rings around those SAME
    anchors fill the pool, in contrast-priority order.  These are explicit
    source-specific expansions, not new detected changes.  Random uniformly
    samples the legal grid without replacement and uses that sampled order as
    random priority (otherwise a downstream position tie-break would always
    favor the earliest sampled locations).

    The caller must use identical non-ClaSS seeds (e.g. whole and fixed1024)
    across modes to isolate candidate sources.  A ClaSS initial partition in
    compression/random would leak ClaSS information despite this isolation.
    """
    if mode not in MODES:
        raise ValueError("candidate mode must be one of "+", ".join(MODES))
    grid = _positive_integer(grid, "grid")
    minimum = _positive_integer(minimum, "minimum", 2)
    window = _positive_integer(window, "window", minimum)
    maximum = _positive_integer(maximum, "maximum")
    if isinstance(random_seed, bool) or not isinstance(random_seed, Integral):
        raise ValueError("random_seed must be an integer")
    n = len(cache.values)
    first = (minimum+grid-1)//grid*grid
    legal = range(first, n-minimum+1, grid)
    target = min(maximum, len(legal))
    if not target:
        return []
    if mode == "random":
        selected = random.Random(int(random_seed)).sample(legal, target)
        priorities = {position: (target-rank)/target for rank, position in enumerate(selected)}
        return [dict(position=position, score=priorities[position], sources="random",
                     strongest_group="", state_contrast=0.)
                for position in sorted(selected)]

    candidates = {}

    def add(position, score, origins, detail=None):
        position = int(position)//grid*grid
        if position < minimum or n-position < minimum:
            return
        if isinstance(origins, str):
            origins = (origins,)
        if position not in candidates:
            candidates[position] = dict(position=position, score=float(score),
                sources="", strongest_group="", state_contrast=0.)
        row = candidates[position]
        row["score"] = max(row["score"], float(score))
        names = set(filter(None, row["sources"].split(";"))) | set(origins)
        row["sources"] = ";".join(sorted(names))
        if detail and score >= row["state_contrast"]:
            row["strongest_group"] = max(detail, key=detail.get)
            row["state_contrast"] = float(score)

    if mode in ("class", "union"):
        points = []
        for start, _ in class_intervals:
            if isinstance(start, bool) or not isinstance(start, Integral):
                raise ValueError("ClaSS interval starts must be integers")
            position = int(start)//grid*grid
            if minimum <= position <= n-minimum:
                points.append(position)
        quota = maximum if mode == "class" else max(1, maximum//2)
        for position in _uniform_subset(sorted(set(points)), quota):
            add(position, 1., "class")

    if mode in ("compression", "union"):
        blocks = [(s, min(window, n-s)) for s in range(0, n, window)
                  if min(window, n-s) >= minimum]
        # Fetch each raw block once even when a test cache lacks memoization.
        rows = [cache.row(start, length) for start, length in blocks]
        contrasts = []
        for index, (left, right) in enumerate(zip(rows, rows[1:])):
            score, detail = state_difference(left, right)
            if score > 1e-12:
                contrasts.append((score, blocks[index+1][0], detail))
        quota = min(64, maximum if mode == "compression" else max(1, maximum//2))
        for score, position, detail in sorted(contrasts, key=lambda row: (-row[0], row[1]))[:quota]:
            add(position, score, "compression_state", detail)

    # Tiny union pools can nominate one anchor from each source.  Keep the
    # highest real priority, not the lowest position after an arbitrary slice.
    anchors = sorted(candidates.values(), key=lambda row: (-row["score"], row["position"]))
    if len(anchors) > target:
        anchors = anchors[:target]
        candidates = {row["position"]: row for row in anchors}

    # Snapshot anchors: expansion never recursively manufactures more anchors.
    anchors = [dict(row) for row in anchors]

    def expand(offset, factor):
        for anchor in anchors:
            if len(candidates) >= target:
                return
            origins = ["nearby_"+name for name in anchor["sources"].split(";")]
            add(anchor["position"]+offset, factor*anchor["score"], origins)

    for offset in (-grid, grid, -window, window):
        expand(offset, .75)
    # Increasing distance is a transparent source-based ranking.  With no
    # anchors no expansion occurs; do not conceal missing evidence by filling.
    if anchors:
        for ring in range(2, (n+grid-1)//grid+1):
            if len(candidates) >= target:
                break
            factor = .75/(1+ring)
            expand(-ring*grid, factor)
            expand(ring*grid, factor)
    return sorted(candidates.values(), key=lambda row: row["position"])


# Kept for existing pipeline imports while the runner adopts explicit modes.
compression_candidates = build_candidates
