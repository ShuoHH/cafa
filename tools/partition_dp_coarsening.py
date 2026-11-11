"""Exact dynamic-programming coarsening of an existing partition.

This module has no codec, model, filesystem, or third-party dependencies.  A
caller prices unions of *whole consecutive input regions* using its frozen
encoding policy and passes successful, actual stored-bit costs here.  Missing
edges are unavailable to the optimizer; they are never estimated or assigned a
zero cost.  The input chain must be priced, so the result cannot worsen it.

The objective must be additive: each edge cost includes all storage that depends
on that encoded region (frame header, payload, padding, per-region directory
entry, and so on).  A constant file header may be added by the caller afterward.
If the container has non-additive layout costs, they must be represented in the
DP state instead; this module does not establish guarantees for such costs.
"""

from collections.abc import Mapping, Sequence


def _integer(value, name, *, minimum=0):
    # In particular, bool must not silently become a zero/one-bit measurement.
    if type(value) is not int:
        raise TypeError("{} must be an int, not {}".format(name, type(value).__name__))
    if value < minimum:
        raise ValueError("{} must be >= {}".format(name, minimum))
    return value


def _validated_partition(partition):
    if not isinstance(partition, Sequence) or isinstance(partition, (str, bytes)):
        raise TypeError("partition must be a sequence of (start, length) pairs")
    if not partition:
        raise ValueError("partition must contain at least one region")
    regions = []
    boundaries = []
    previous_end = None
    for index, region in enumerate(partition):
        if (
            not isinstance(region, Sequence)
            or isinstance(region, (str, bytes))
            or len(region) != 2
        ):
            raise TypeError("partition region {} must be a (start, length) pair".format(index))
        start = _integer(region[0], "region {} start".format(index))
        length = _integer(region[1], "region {} length".format(index), minimum=1)
        if previous_end is not None and start != previous_end:
            raise ValueError("partition must be ordered and exactly contiguous")
        if not boundaries:
            boundaries.append(start)
        previous_end = start + length
        boundaries.append(previous_end)
        regions.append((start, length))
    return tuple(regions), tuple(boundaries)


def enumerate_union_intervals(partition):
    """Return all valid union edges once, in a safe deterministic pricing order.

    Original regions are first, then the whole covered interval, then remaining
    unions with increasing numbers of original regions and increasing starts.
    Thus a budgeted caller can price the original chain and whole-series edge
    before optional local alternatives.  The input is not mutated.
    """
    regions, boundaries = _validated_partition(partition)
    count = len(regions)
    result = []
    seen = set()

    def append(edge):
        if edge not in seen:
            seen.add(edge)
            result.append(edge)

    for region in regions:
        append(region)
    append((boundaries[0], boundaries[-1] - boundaries[0]))
    for span in range(2, count + 1):
        for left in range(count - span + 1):
            append((boundaries[left], boundaries[left + span] - boundaries[left]))
    return result


def coarsen_partition_dp(partition, measured_cost_bits, *, full_pricing_complete=False):
    """Minimize additive stored bits over successful measured union edges.

    ``measured_cost_bits`` maps ``(start, length)`` to a nonnegative built-in int.
    Every input region must have a successful cost.  Unavailable/unpriced edges
    are omitted, not supplied as None/NaN/a sentinel.  Extra or non-union edges
    are rejected.  Ties prefer fewer output regions, then lexicographically
    earlier sequences of retained boundary positions.

    ``full_pricing_complete`` is an audited assertion supplied by the caller:
    every possible union has a terminal measured-success or unavailable status.
    It does NOT imply that every union is usable.  The returned optimum claim
    is deliberately limited to the supplied successful measured-edge graph.
    Incomplete pricing still guarantees non-worsening versus the original input
    chain, and versus the whole edge when that edge has a successful cost.
    These guarantees assume the same frozen encoding policy and additive cost
    accounting for every supplied edge; this module cannot validate that policy.
    """
    regions, boundaries = _validated_partition(partition)
    if not isinstance(measured_cost_bits, Mapping):
        raise TypeError("measured_cost_bits must be a mapping")
    if type(full_pricing_complete) is not bool:
        raise TypeError("full_pricing_complete must be a bool")

    boundary_index = {position: index for index, position in enumerate(boundaries)}
    prices = {}
    outgoing = [[] for _ in regions]
    for edge, raw_bits in measured_cost_bits.items():
        if not isinstance(edge, tuple) or len(edge) != 2:
            raise TypeError("each cost key must be a (start, length) tuple")
        start = _integer(edge[0], "edge start")
        length = _integer(edge[1], "edge length", minimum=1)
        bits = _integer(raw_bits, "edge cost bits")
        stop = start + length
        if start not in boundary_index or stop not in boundary_index:
            raise ValueError("cost edge {} is not a union of input regions".format(edge))
        left, right = boundary_index[start], boundary_index[stop]
        if left >= right:
            raise ValueError("cost edges must move forward through the partition")
        prices[(start, length)] = bits
        outgoing[left].append((right, bits))

    missing_original = [region for region in regions if region not in prices]
    if missing_original:
        raise ValueError("original chain must have successful measured costs: {}".format(missing_original))

    # A state compares (bits, region_count, endpoint_indices).  This ordering
    # explicitly handles zero-cost edges and deterministic equal-cost paths.
    count = len(regions)
    best = [None] * (count + 1)
    best[0] = (0, 0, ())
    for left in range(count):
        state = best[left]
        if state is None:
            continue
        for right, bits in sorted(outgoing[left]):
            candidate = (state[0] + bits, state[1] + 1, state[2] + (right,))
            if best[right] is None or candidate < best[right]:
                best[right] = candidate

    winning = best[count]
    # The original chain is mandatory, so an unreachable terminal is a bug.
    if winning is None:
        raise RuntimeError("validated original chain unexpectedly became unreachable")
    chosen = []
    previous = 0
    for endpoint in winning[2]:
        chosen.append((boundaries[previous], boundaries[endpoint] - boundaries[previous]))
        previous = endpoint

    original_bits = sum(prices[region] for region in regions)
    whole_edge = (boundaries[0], boundaries[-1] - boundaries[0])
    whole_bits = prices.get(whole_edge)
    retained_boundaries = [boundaries[index] for index in winning[2][:-1]]
    retained_set = set(retained_boundaries)
    return {
        "partition": chosen,
        "total_bits": winning[0],
        "original_bits": original_bits,
        "whole_bits": whole_bits,
        "saved_vs_original_bits": original_bits - winning[0],
        "original_segments": count,
        "segments": len(chosen),
        "retained_boundaries": retained_boundaries,
        "removed_boundaries": [position for position in boundaries[1:-1] if position not in retained_set],
        "measured_edges": len(prices),
        "possible_edges": count * (count + 1) // 2,
        "all_edges_have_successful_costs": len(prices) == count * (count + 1) // 2,
        "full_pricing_complete": full_pricing_complete,
        "measured_graph_optimum": True,
        "coarsening_scope": "measured_successful_edges",
        "non_worse_than_original": winning[0] <= original_bits,
        "whole_edge_available": whole_bits is not None,
        "non_worse_than_whole": None if whole_bits is None else winning[0] <= whole_bits,
        "changed": tuple(chosen) != regions,
    }

