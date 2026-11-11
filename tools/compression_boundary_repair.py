"""Bounded, actual-cost local repair of a frozen semantic partition.

This module has no file, native-codec, or model dependencies.  The evaluator
uses a predeclared codec policy and returns its actual complete region cost.
The deployed pilot uses a bounded selector. An explicitly labelled diagnostic
may instead measure the full feasible pool; that is not a free codec oracle.
Optional raw-summary gain estimates rank proposals only, never accept them.

The only guarantee is monotone actual cost relative to the supplied seed under
that evaluator.  This is neither dynamic programming nor a global optimum,
and does not imply improvement over an independently chosen strong baseline.
"""

from __future__ import annotations

from dataclasses import dataclass
import math
from numbers import Integral
from typing import Any, Iterable, Mapping, Protocol, Sequence


Interval = tuple[int, int]


class RegionEvaluator(Protocol):
    def can_evaluate(self, regions: Sequence[Interval]) -> bool: ...

    def evaluate(self, regions: Sequence[Interval]) -> Mapping[Interval, int]: ...


@dataclass(frozen=True)
class RepairConfig:
    grid: int = 64
    min_segment: int = 64
    max_rounds: int = 12
    max_proposals_per_round: int = 64
    max_split_candidates: int = 6
    min_gain_bits: int = 0
    shift_radius: int = 256
    pair_split: bool = True

    def validate(self) -> None:
        for name in ("grid", "min_segment", "max_rounds",
                     "max_proposals_per_round", "max_split_candidates"):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, Integral) or value < 1:
                raise ValueError(f"{name} must be a positive integer")
        for name in ("min_gain_bits", "shift_radius"):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, Integral) or value < 0:
                raise ValueError(f"{name} must be a nonnegative integer")
        if not isinstance(self.pair_split, bool):
            raise ValueError("pair_split must be a boolean")


@dataclass(frozen=True)
class _Proposal:
    kind: str
    first: int
    stop: int
    old: tuple[Interval, ...]
    new: tuple[Interval, ...]
    candidate_positions: tuple[int, ...]
    raw_score: float

    @property
    def length(self) -> int:
        return sum(length for _, length in self.old)

    @property
    def identity(self) -> tuple:
        # A different operation name must not duplicate the same geometry.
        return self.old, self.new


def _integer(value: Any, name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, Integral):
        raise ValueError(f"{name} must be an integer")
    return int(value)


def _partition(intervals: Iterable[Sequence[int]]) -> tuple[Interval, ...]:
    result: list[Interval] = []
    expected = 0
    for item in intervals:
        if len(item) != 2:
            raise ValueError("each interval must contain start and length")
        start = _integer(item[0], "interval start")
        length = _integer(item[1], "interval length")
        if start != expected or length < 2:
            raise ValueError("intervals must cover the sequence from zero without gaps; "
                             "every semantic region needs at least two values")
        result.append((start, length))
        expected += length
    if not result:
        raise ValueError("partition must not be empty")
    return tuple(result)


def _candidates(candidates: Iterable[Mapping[str, Any]], n: int,
                grid: int) -> dict[int, dict[str, Any]]:
    result: dict[int, dict[str, Any]] = {}
    for item in candidates:
        position = _integer(item["position"], "candidate position")
        score = float(item.get("score", 0.0))
        if not math.isfinite(score):
            raise ValueError("candidate score must be finite")
        # Raw detectors may supply off-grid points.  Do not silently move one.
        if not 0 < position < n or position % grid:
            continue
        sources = item.get("sources", [])
        if isinstance(sources, str):
            sources = [sources]
        sources = tuple(sorted({str(source) for source in sources}))
        if position not in result:
            result[position] = {"score": score, "sources": sources}
        else:
            result[position]["score"] = max(result[position]["score"], score)
            result[position]["sources"] = tuple(sorted(
                set(result[position]["sources"]) | set(sources)))
    return result


def _legal(regions: tuple[Interval, ...], n: int,
           config: RepairConfig) -> bool:
    for start, length in regions:
        # The fixed selector's difference features need at least two values.
        # A final 2..min_segment-1 tail is legal, but a one-value semantic
        # region must stay attached to its predecessor instead of being padded.
        if length < 2:
            return False
        if length < config.min_segment and start + length != n:
            return False
    # Final N is intentionally not rounded to the grid.
    return all((start + length) % config.grid == 0
               for start, length in regions[:-1])


def _generate(partition: tuple[Interval, ...], candidates: dict[int, dict],
              n: int, config: RepairConfig, proposal_scorer=None) -> dict[str, list[_Proposal]]:
    groups: dict[str, list[_Proposal]] = {
        "merge": [], "move": [], "split": [], "pair_split": []}
    seen: set[tuple] = set()
    priorities = {}

    def priority(old, new, fallback=0.0):
        if proposal_scorer is None:
            return fallback
        key = (tuple(old), tuple(new))
        if key not in priorities:
            value = proposal_scorer(*key)
            if isinstance(value, bool) or not math.isfinite(float(value)):
                raise ValueError("proposal scorer must return a finite estimated gain")
            priorities[key] = float(value)
        return priorities[key]

    def add(kind: str, first: int, stop: int, new: tuple[Interval, ...],
            positions: tuple[int, ...] = (), score: float = 0.0) -> None:
        old = partition[first:stop]
        if new == old or not _legal(new, n, config):
            return
        if old[0][0] != new[0][0] or sum(x[1] for x in old) != sum(x[1] for x in new):
            raise AssertionError("proposal must preserve its complete old span")
        proposal = _Proposal(kind, first, stop, old, new, positions,
                             priority(old, new, score))
        if proposal.identity not in seen:
            groups[kind].append(proposal)
            seen.add(proposal.identity)

    for index in range(len(partition) - 1):
        start = partition[index][0]
        boundary = partition[index + 1][0]
        end = sum(partition[index + 1])
        add("merge", index, index + 2, ((start, end - start),))
        moves = {boundary - config.grid, boundary + config.grid,
                 boundary - config.shift_radius, boundary + config.shift_radius}
        moves.update(p for p in candidates
                     if abs(p - boundary) <= config.shift_radius)
        for position in sorted(moves):
            if position == boundary or not start < position < end or position % config.grid:
                continue
            if abs(position - boundary) > config.shift_radius:
                continue
            add("move", index, index + 2,
                ((start, position - start), (position, end - position)),
                (position,), candidates.get(position, {}).get("score", 0.0))

    for index, (start, length) in enumerate(partition):
        end = start + length
        choices = [p for p in candidates if start < p < end
                   and _legal(((start, p - start), (p, end - p)), n, config)]
        choices.sort(key=lambda p: (-priority(((start, length),),
                          ((start, p-start), (p, end-p)), candidates[p]["score"]), p))
        choices = choices[:config.max_split_candidates]
        for position in choices:
            add("split", index, index + 1,
                ((start, position - start), (position, end - position)),
                (position,), candidates[position]["score"])
        if config.pair_split:
            # At most k-1 adjacent pairs among the raw top-k points: not k^2.
            ordered = sorted(choices)
            for left, right in zip(ordered, ordered[1:]):
                add("pair_split", index, index + 1,
                    ((start, left - start), (left, right - left),
                     (right, end - right)), (left, right),
                    min(candidates[left]["score"], candidates[right]["score"]))

    for group in groups.values():
        group.sort(key=lambda proposal: (-proposal.raw_score, -proposal.length,
                                        proposal.old[0][0], proposal.candidate_positions))
    return groups


def _fair_order(groups: dict[str, list[_Proposal]], round_index: int) -> list[_Proposal]:
    kinds = list(groups)
    # Rotate the first class so small round budgets do not permanently starve a class.
    offset = round_index % len(kinds)
    kinds = kinds[offset:] + kinds[:offset]
    result: list[_Proposal] = []
    for index in range(max((len(x) for x in groups.values()), default=0)):
        for kind in kinds:
            if index < len(groups[kind]):
                result.append(groups[kind][index])
    return result


def _measure(evaluator: RegionEvaluator, regions: Sequence[Interval],
             costs: dict[Interval, int]) -> None:
    if not regions:
        return
    measured = evaluator.evaluate(list(regions))
    if not isinstance(measured, Mapping):
        raise ValueError("evaluator.evaluate must return an interval-to-cost mapping")
    additions: dict[Interval, int] = {}
    for region in regions:
        if region not in measured:
            raise ValueError(f"evaluator omitted requested region {region}")
        bits = _integer(measured[region], "actual selected-codec bits")
        if bits < 0:
            raise ValueError("actual selected-codec bits must not be negative")
        if region in costs and costs[region] != bits:
            raise ValueError(f"evaluator changed the deterministic cost of {region}")
        additions[region] = bits
    costs.update(additions)


def repair_partition(intervals: Iterable[Sequence[int]],
                     candidates: Iterable[Mapping[str, Any]],
                     evaluator: RegionEvaluator, *, config: RepairConfig | None = None,
                     tag: str = "", proposal_scorer=None) -> dict[str, Any]:
    """Greedily apply disjoint profitable, bounded local edits to one seed.

    ``can_evaluate`` must preflight the whole requested union against the
    evaluator's remaining budget (including already cached intervals).  No
    unmeasured or unaffordable proposal receives an invented gain.  An
    evaluator exception propagates, and the caller's seed is never mutated.
    Semantic regions require at least two values for selector features.  Only
    the final region may be shorter than ``min_segment``; its values are never
    discarded, rounded off, or padded.
    """
    config = config or RepairConfig()
    config.validate()
    if proposal_scorer is not None and not callable(proposal_scorer):
        raise ValueError("proposal_scorer must be callable or None")
    partition = _partition(intervals)
    n = sum(length for _, length in partition)
    raw = _candidates(candidates, n, config.grid)
    initial = list(partition)
    costs: dict[Interval, int] = {}
    if not evaluator.can_evaluate(list(partition)):
        raise ValueError("evaluator budget cannot score the complete initial partition")
    _measure(evaluator, list(partition), costs)
    initial_bits = sum(costs[region] for region in partition)
    total_bits = initial_bits
    operations: list[dict[str, Any]] = []
    records: list[dict[str, Any]] = []
    round_records: list[dict[str, Any]] = []
    budget_limited = False
    termination = "max_rounds"

    for round_index in range(config.max_rounds):
        proposals = _fair_order(_generate(partition, raw, n, config, proposal_scorer), round_index)
        if not proposals:
            termination = "no_proposals"
            break
        round_items: list[tuple[_Proposal, dict[str, Any]]] = []
        wanted: set[Interval] = set()
        affordable: list[tuple[_Proposal, dict[str, Any]]] = []
        for index, proposal in enumerate(proposals):
            record: dict[str, Any] = {
                "tag": tag, "round": round_index + 1,
                "proposal_id": f"{tag}:{round_index + 1}:{index + 1}",
                "kind": proposal.kind, "start": proposal.old[0][0],
                "length": proposal.length, "old_intervals": list(proposal.old),
                "new_intervals": list(proposal.new),
                "candidate_positions": list(proposal.candidate_positions),
                "raw_score": proposal.raw_score, "old_bits": None,
                "estimated_gain_bits": proposal.raw_score if proposal_scorer is not None else None,
                "new_bits": None, "gain_bits": None,
                "status": "round_cap_skipped",
            }
            records.append(record)
            round_items.append((proposal, record))
            if index >= config.max_proposals_per_round:
                continue
            union = wanted | set(proposal.new)
            if not evaluator.can_evaluate(sorted(union)):
                record["status"] = "budget_skipped"
                budget_limited = True
                continue
            record["status"] = "pending_measurement"
            wanted = union
            affordable.append((proposal, record))

        _measure(evaluator, sorted(wanted), costs)
        profitable: list[tuple[_Proposal, dict[str, Any]]] = []
        for proposal, record in affordable:
            old_bits = sum(costs[region] for region in proposal.old)
            new_bits = sum(costs[region] for region in proposal.new)
            gain = old_bits - new_bits
            record.update(old_bits=old_bits, new_bits=new_bits, gain_bits=gain,
                          status="not_profitable")
            if gain > config.min_gain_bits:
                record["status"] = "profitable_overlap_skipped"
                profitable.append((proposal, record))
        profitable.sort(key=lambda item: (-item[1]["gain_bits"],
                                         item[0].old[0][0], item[0].kind,
                                         item[0].candidate_positions))
        occupied: set[int] = set()
        chosen: list[tuple[_Proposal, dict[str, Any]]] = []
        for proposal, record in profitable:
            positions = set(range(proposal.first, proposal.stop))
            if occupied & positions:
                continue
            occupied.update(positions)
            record["status"] = "accepted"
            chosen.append((proposal, record))

        before = total_bits
        round_record = {"round": round_index + 1, "before_bits": before,
                        "generated_proposals": len(proposals),
                        "bounded_proposals": min(len(proposals), config.max_proposals_per_round),
                        "measured_proposals": len(affordable),
                        "budget_skipped": sum(r["status"] == "budget_skipped"
                                              for _, r in round_items),
                        "accepted_operations": len(chosen)}
        if not chosen:
            round_record.update(after_bits=before, gain_bits=0)
            round_records.append(round_record)
            termination = ("no_affordable_proposals" if not affordable
                           else "no_profitable_proposals")
            break

        replacements = {proposal.first: (proposal, record) for proposal, record in chosen}
        replacement: list[Interval] = []
        index = 0
        while index < len(partition):
            if index in replacements:
                proposal, _ = replacements[index]
                replacement.extend(proposal.new)
                index = proposal.stop
            else:
                replacement.append(partition[index])
                index += 1
        updated = _partition(replacement)
        if sum(length for _, length in updated) != n:
            raise AssertionError("repair dropped or duplicated the final tail")
        after = sum(costs[region] for region in updated)
        exact_gain = sum(record["gain_bits"] for _, record in chosen)
        if not after < before or before - after != exact_gain:
            raise AssertionError("accepted disjoint edits must strictly lower exact actual cost")
        partition = updated
        total_bits = after
        for _, record in sorted(chosen, key=lambda item: item[0].first):
            operations.append(dict(record))
        round_record.update(after_bits=after, gain_bits=exact_gain)
        round_records.append(round_record)

    accepted_gain = sum(item["gain_bits"] for item in operations)
    if initial_bits - total_bits != accepted_gain:
        raise AssertionError("actual-cost gains must telescope exactly")
    return {"tag": tag, "intervals": list(partition), "initial_intervals": initial,
            "values": n, "total_bits": total_bits, "initial_bits": initial_bits,
            "gain_bits": initial_bits - total_bits, "operations": operations,
            "proposal_records": records, "rounds": len(round_records),
            "round_records": round_records, "termination": termination,
            "budget_exhausted": budget_limited, "budget_limited": budget_limited,
            "telescoping_ok": True, "dp_used": False,
            "cost_source": getattr(evaluator, "cost_source", "actual_fixed_selected_codec"),
            "proposal_priority": "estimated_net_gain_bits" if proposal_scorer is not None else "raw_contrast",
            "estimates_used_for_acceptance": False,
            "guarantee": "nonincreasing_actual_cost_relative_to_this_seed_only"}
