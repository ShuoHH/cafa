"""Replay accepted semantic boundary edits and report source membership.

This is a pure reporting helper, not a boundary detector or a cost evaluator.
The caller must supply semantic regions, not physical codec frames (for
example, an ALP prefix/tail seam is not a segmentation boundary).  Membership
in a detector seed or candidate source is evidence of nomination, not causal
attribution of compression gains.  Joint edit gains are never divided among
the boundaries introduced by that edit.
"""

from __future__ import annotations

from collections import Counter
from itertools import groupby
from numbers import Integral
from typing import Any, Iterable, Mapping, Sequence


CATEGORIES = ("detector_only", "compression_only", "both",
              "nearby_expansion", "repair_generated")
KINDS = {"split": (1, 2), "pair_split": (1, 3),
         "merge": (2, 1), "move": (2, 2)}
Interval = tuple[int, int]


def _integer(value: Any, name: str, minimum: int = 0) -> int:
    if isinstance(value, bool) or not isinstance(value, Integral) or value < minimum:
        raise ValueError(f"{name} must be an integer >= {minimum}")
    return int(value)


def _regions(items: Iterable[Sequence[int]], name: str,
             start: int | None = None) -> tuple[Interval, ...]:
    result = []
    expected = start
    for item in items:
        if not isinstance(item, Sequence) or len(item) != 2:
            raise ValueError(f"{name}: each semantic interval needs start and length")
        left = _integer(item[0], f"{name} start")
        length = _integer(item[1], f"{name} length", 2)
        if expected is not None and left != expected:
            raise ValueError(f"{name}: semantic intervals must be contiguous without overlap")
        result.append((left, length))
        expected = left + length
    if not result:
        raise ValueError(f"{name} must not be empty")
    return tuple(result)


def _points(intervals: Sequence[Interval]) -> set[int]:
    return {left for left, _ in intervals[1:]}


def _sources(raw: Any) -> set[str]:
    if isinstance(raw, str):
        raw = [raw]
    if not isinstance(raw, (tuple, list, set, frozenset)):
        raise ValueError("candidate sources must be a string or a sequence of strings")
    result = set()
    for value in raw:
        if not isinstance(value, str):
            raise ValueError("candidate source tags must be strings")
        result.update(tag.strip() for tag in value.split(";") if tag.strip())
    return result


def _candidate_sources(candidates, n: int, seed_points: set[int]) -> dict[int, set[str]]:
    result: dict[int, set[str]] = {}
    for item in candidates:
        if not isinstance(item, Mapping):
            raise ValueError("candidate rows must be mappings")
        position = _integer(item.get("position"), "candidate position")
        if position > n:
            raise ValueError("candidate position exceeds sequence length")
        # Endpoints are not internal boundaries and never enter any count.
        if position in (0, n):
            continue
        result.setdefault(position, set()).update(_sources(item.get("sources", [])))
    for position, tags in result.items():
        if "class" in tags and position not in seed_points:
            raise ValueError("direct class candidate is absent from the full detector seed")
    return result


def _accepted_operation(item: Mapping[str, Any], index: int) -> dict[str, Any]:
    if not isinstance(item, Mapping) or item.get("status") != "accepted":
        raise ValueError("operations must contain accepted records only")
    kind = item.get("kind")
    if kind not in KINDS:
        raise ValueError(f"unsupported boundary operation kind: {kind!r}")
    old = _regions(item.get("old_intervals", []), "old_intervals")
    new = _regions(item.get("new_intervals", []), "new_intervals")
    if (len(old), len(new)) != KINDS[kind]:
        raise ValueError(f"{kind}: unexpected old/new region counts")
    start = old[0][0]
    length = sum(length for _, length in old)
    if new[0][0] != start or sum(length for _, length in new) != length or old == new:
        raise ValueError("an accepted operation must change geometry and preserve its full span")
    if "start" in item and _integer(item["start"], "operation start") != start:
        raise ValueError("operation start disagrees with old_intervals")
    if "length" in item and _integer(item["length"], "operation length", 1) != length:
        raise ValueError("operation length disagrees with old_intervals")
    old_bits = _integer(item.get("old_bits"), "old_bits")
    new_bits = _integer(item.get("new_bits"), "new_bits")
    gain = _integer(item.get("gain_bits"), "gain_bits", 1)
    if old_bits - new_bits != gain:
        raise ValueError("accepted gain_bits must equal old_bits - new_bits and be strictly positive")
    introduced = sorted(_points(new) - _points(old))
    removed = sorted(_points(old) - _points(new))
    if "candidate_positions" in item:
        positions = [_integer(p, "operation candidate position")
                     for p in item["candidate_positions"]]
        if positions != introduced:
            raise ValueError("operation candidate_positions disagree with introduced geometry")
    proposal_id = item.get("proposal_id")
    if proposal_id is None:
        proposal_id = f"operation_{index:06d}"
    if not isinstance(proposal_id, str) or not proposal_id:
        raise ValueError("proposal_id must be a nonempty string when supplied")
    return dict(operation_index=index, proposal_id=proposal_id,
                round=_integer(item.get("round"), "operation round", 1), kind=kind,
                start=start, length=length, old_intervals=old, new_intervals=new,
                old_bits=old_bits, new_bits=new_bits, gain_bits=gain,
                removed_positions=removed, introduced_positions=introduced,
                gain_scope="joint_operation_only")


def build_origin_report(initial_intervals, final_intervals, candidates, operations,
                        values: int, detector_name: str) -> dict[str, Any]:
    """Return verified source membership and edit lineage for final boundaries.

    ``values`` is the original number of values.  ``operations`` is exactly
    ``repair_partition(...)["operations"]``, not the unfiltered proposal log.
    Accepted records in the same round must operate on disjoint regions of
    the round-start partition.  Actual costs are checked for internal record
    consistency, not remeasured here.  No zero/one-value padding is permitted.

    Detector membership is the FULL initial seed, independently of candidate
    quotas.  ``class`` tags must agree with that seed; ``compression_state``
    is a direct feature nomination.  A remaining admitted candidate is called
    nearby expansion only if it really carries a ``nearby_*`` source tag.
    """
    n = _integer(values, "values", 2)
    if not isinstance(detector_name, str) or not detector_name.strip():
        raise ValueError("detector_name must be a nonempty string")
    initial = _regions(initial_intervals, "initial_intervals", 0)
    final = _regions(final_intervals, "final_intervals", 0)
    if sum(item[1] for item in initial) != n or sum(item[1] for item in final) != n:
        raise ValueError("initial and final semantic partitions must cover exactly values")
    seed_points = _points(initial)
    source_map = _candidate_sources(candidates, n, seed_points)
    records = [_accepted_operation(item, index)
               for index, item in enumerate(operations, 1)]
    if len({item["proposal_id"] for item in records}) != len(records):
        raise ValueError("accepted proposal identifiers must be unique")
    rounds = [item["round"] for item in records]
    if rounds != sorted(rounds):
        raise ValueError("accepted operation rounds must be nondecreasing")

    history = {position: dict(removal_count=0, introduction_operation_count=0,
                              last_introduction=None, last_removal=None)
               for position in seed_points}
    partition = initial
    for _, group in groupby(records, key=lambda item: item["round"]):
        batch = list(group)
        lookup = {region[0]: index for index, region in enumerate(partition)}
        replacements = []
        for record in batch:
            first = lookup.get(record["start"])
            old = record["old_intervals"]
            if first is None or partition[first:first + len(old)] != old:
                raise ValueError("accepted operation does not match its round-start partition")
            replacements.append((first, first + len(old), record))
        replacements.sort(key=lambda item: item[0])
        if any(left[1] > right[0] for left, right in zip(replacements, replacements[1:])):
            raise ValueError("accepted operations overlap within the same round")
        updated = []
        cursor = 0
        active = _points(partition)
        for first, stop, record in replacements:
            updated.extend(partition[cursor:first])
            updated.extend(record["new_intervals"])
            cursor = stop
            for position in record["removed_positions"]:
                if position not in active:
                    raise ValueError("operation removes an inactive boundary")
                history[position]["removal_count"] += 1
                history[position]["last_removal"] = record
                active.remove(position)
            for position in record["introduced_positions"]:
                if position in active or not 0 < position < n:
                    raise ValueError("operation introduces an existing or noninternal boundary")
                event = history.setdefault(position, dict(removal_count=0,
                    introduction_operation_count=0, last_introduction=None, last_removal=None))
                event["introduction_operation_count"] += 1
                event["last_introduction"] = record
                active.add(position)
        updated.extend(partition[cursor:])
        partition = _regions(updated, "replayed partition", 0)
        if sum(item[1] for item in partition) != n or _points(partition) != active:
            raise ValueError("replayed operation changed coverage or lost lineage")
    if partition != final:
        raise ValueError("accepted operation replay does not equal final_intervals")

    rows = []
    counts = Counter({category: 0 for category in CATEGORIES})
    by_kind = Counter({kind: 0 for kind in KINDS})
    for position in sorted(_points(final)):
        tags = source_map.get(position, set())
        detected = position in seed_points
        compression = "compression_state" in tags
        if detected and compression:
            category = "both"
        elif detected:
            category = "detector_only"
        elif compression:
            category = "compression_only"
        elif position in source_map:
            if not any(tag.startswith("nearby_") for tag in tags):
                raise ValueError("non-direct candidate lacks an actual nearby_* source tag")
            category = "nearby_expansion"
        else:
            category = "repair_generated"
        event = history[position]
        last = event["last_introduction"]
        if not detected and last is None:
            raise ValueError("new final boundary has no accepted introduction evidence")
        if last is not None:
            by_kind[last["kind"]] += 1
        counts[category] += 1
        rows.append(dict(position=position, category=category, detector_name=detector_name,
            in_initial_detector_boundaries=detected,
            direct_detector_candidate="class" in tags,
            direct_compression_candidate=compression,
            in_actual_candidate_pool=position in source_map,
            candidate_sources=";".join(sorted(tags)),
            seed_lifecycle=("untouched_seed" if detected and event["removal_count"] == 0
                            else "removed_then_reintroduced_seed" if detected
                            else "introduced_after_seed"),
            ever_removed=event["removal_count"] > 0,
            removal_count=event["removal_count"],
            introduction_operation_count=event["introduction_operation_count"],
            last_introduction_kind=last["kind"] if last else "seed",
            last_introduction_operation_index=last["operation_index"] if last else None,
            last_introduction_operation_id=last["proposal_id"] if last else None,
            last_introduction_round=last["round"] if last else None,
            last_removal_operation_index=(event["last_removal"]["operation_index"]
                                          if event["last_removal"] else None)))

    final_points = _points(final)
    retained = final_points & seed_points
    reintroduced = sum(history[p]["removal_count"] > 0 for p in retained)
    summary = dict(initial_segments=len(initial), initial_detector_boundaries=len(seed_points),
        final_segments=len(final), final_boundaries=len(final_points),
        retained_exact_seed_boundaries=len(retained),
        untouched_surviving_seed_boundaries=len(retained) - reintroduced,
        reintroduced_seed_boundaries=reintroduced,
        new_final_boundaries=len(final_points - seed_points),
        removed_seed_boundaries=len(seed_points - final_points),
        accepted_operations=len(records),
        total_accepted_gain_bits=sum(item["gain_bits"] for item in records),
        **{category: counts[category] for category in CATEGORIES},
        **{f"final_last_introduced_by_{kind}": by_kind[kind] for kind in KINDS},
        category_count_sum=sum(counts.values()))
    # Explicit list conversion keeps JSON stable even for non-JSON input tuples.
    operation_rows = [dict(item, old_intervals=[list(x) for x in item["old_intervals"]],
                           new_intervals=[list(x) for x in item["new_intervals"]])
                      for item in records]
    return dict(schema_version=1, detector_name=detector_name, values=n,
        scope="final_internal_semantic_boundaries", summary=summary,
        boundary_rows=rows, operation_rows=operation_rows,
        invariants=dict(exact_geometry_replay=True, same_round_operations_disjoint=True,
            accepted_records_have_positive_consistent_gain=True,
            category_counts_equal_final_segments_minus_one=sum(counts.values()) == len(final) - 1,
            full_seed_used_without_candidate_quota=True,
            every_new_final_boundary_has_accepted_introduction=True,
            source_membership_is_causal_attribution=False,
            joint_operation_gain_allocated_to_boundaries=False,
            codec_costs_independently_remeasured_by_this_helper=False),
        notes=["Only supplied semantic intervals count; physical codec frame seams are excluded by the caller.",
               "Detector means exact membership in the full initial seed, not a retained candidate quota.",
               "Nearby means an actual admitted candidate carrying nearby_*; no geometric neighbourhood is inferred.",
               "Source membership and last introduction describe evidence and geometry, not marginal or causal compression gains.",
               "Operation costs are checked from accepted records; this helper does not execute codecs or verify stored payloads."])
