"""Bounded, measured-cost split/merge search over persisted codec frames.

The proxy orders proposals only.  A move is accepted solely when its already
encoded frames occupy fewer bytes, including every 12-byte frame header.  The
unchanging 24-byte file header cancels between candidates.  This is a bounded
greedy local search, not dynamic programming or a global-optimality claim.

``service.best_many(regions)`` atomically reserves native trial work and returns
one segment per requested (start, length), or None when unavailable.  The
service owns trial caching and bitwise verification.  This module never encodes,
changes, or synthesizes codec payloads.
"""

from dataclasses import dataclass
import hashlib
import math
import struct


FRAME_HEADER_BYTES = 12
_FRAME_IDS = frozenset(range(16, 26))


@dataclass(frozen=True)
class SearchConfig:
    grid: int = 64
    minimum: int = 64
    candidates_per_parent: int = 2
    max_rounds: int = 16
    max_proposals_per_round: int = 16
    min_saving_bytes: int = 1


def _positive_integer(value, name):
    if type(value) is not int or value <= 0:
        raise ValueError(name + " must be a positive integer")
    return value


def _segment(segment, expected_start=None, expected_length=None):
    """Check coverage and copy metadata without copying immutable payload bytes."""
    if not isinstance(segment, dict):
        raise ValueError("each segment must be a dictionary")
    start = segment.get("start")
    length = segment.get("length")
    if type(start) is not int or start < 0:
        raise ValueError("segment start must be an unsigned integer")
    _positive_integer(length, "segment length")
    if start + length > (1 << 64) - 1:
        raise ValueError("segment coverage exceeds uint64")
    if expected_start is not None and start != expected_start:
        raise ValueError("segment does not cover the requested start")
    if expected_length is not None and length != expected_length:
        raise ValueError("segment does not cover the requested length")
    if not isinstance(segment.get("codec"), str) or not segment["codec"]:
        raise ValueError("segment codec must be a nonempty string")
    frames = segment.get("frames")
    if not isinstance(frames, (list, tuple)) or not frames:
        raise ValueError("segment must contain persisted frames")
    copied_frames, cursor = [], start
    for frame in frames:
        if not isinstance(frame, dict):
            raise ValueError("each frame must be a dictionary")
        if type(frame.get("start")) is not int or frame["start"] != cursor:
            raise ValueError("frame coverage must be contiguous and absolute")
        count = _positive_integer(frame.get("count"), "frame count")
        codec_id = frame.get("codec_id")
        if type(codec_id) is not int or codec_id not in _FRAME_IDS:
            raise ValueError("frame codec_id must be a method-6 codec ID 16..25")
        if count > (1 << 32) - 1 or (codec_id == 25 and count > 65536):
            raise ValueError("frame count exceeds its codec/format limit")
        payload = frame.get("payload")
        if type(payload) is not bytes or not payload or len(payload) > (1 << 32) - 1:
            raise ValueError("frame payload must be nonempty immutable bytes")
        copied_frames.append(dict(frame))
        cursor += count
    if cursor != start + length:
        raise ValueError("frames do not cover the segment exactly")
    return dict(segment, frames=copied_frames)


def frame_bytes(segments):
    """Actual frame bytes; excludes the single unchanged file header."""
    return sum(FRAME_HEADER_BYTES + len(frame["payload"])
               for segment in segments for frame in segment["frames"])


def _signature(segment):
    digest = hashlib.sha256()
    for frame in segment["frames"]:
        digest.update(struct.pack(">QIIQ", frame["start"], frame["count"],
                                  frame["codec_id"], len(frame["payload"])))
        digest.update(frame["payload"])
    return (segment["start"], segment["length"], digest.digest())


def _proxy_gain(proxy, old_pairs, new_pairs):
    value = proxy.gain(old_pairs, new_pairs)
    if isinstance(value, bool):
        raise ValueError("proxy gain must be a finite number")
    try:
        value = float(value)
    except (TypeError, ValueError, OverflowError) as error:
        raise ValueError("proxy gain must be a finite number") from error
    if not math.isfinite(value):
        raise ValueError("proxy gain must be a finite number")
    return value


def _proposal(operation, index, old, regions, position, proxy, signatures):
    pairs = [(segment["start"], segment["length"]) for segment in old]
    return dict(operation=operation, index=index, old=old, regions=regions,
                position=position, start=pairs[0][0],
                length=sum(pair[1] for pair in pairs),
                proxy_gain_bits=_proxy_gain(proxy, pairs, regions),
                key=(operation, position, tuple(signatures[id(item)] for item in old)))


def _priority(proposal):
    return (-proposal["proxy_gain_bits"], proposal["start"],
            proposal["length"], -1 if proposal["position"] is None else proposal["position"])


def _proposals(segments, proxy, candidates, config, visited):
    splits, merges = [], []
    # Hash each persisted incumbent only once per batch, not once per cut.
    signatures = {id(segment): _signature(segment) for segment in segments}
    for index, segment in enumerate(segments):
        start, length = segment["start"], segment["length"]
        parent_splits = []
        for position in candidates:
            if start + config.minimum <= position <= start + length - config.minimum:
                regions = [(start, position - start),
                           (position, start + length - position)]
                proposal = _proposal("split", index, [segment], regions, position, proxy, signatures)
                if proposal["key"] not in visited:
                    parent_splits.append(proposal)
        # Negative proxy gains remain eligible: proxy error cannot veto a move.
        parent_splits.sort(key=_priority)
        splits.extend(parent_splits[:config.candidates_per_parent])
        if index + 1 < len(segments):
            old = [segment, segments[index + 1]]
            proposal = _proposal("merge", index, old,
                                 [(start, length + old[1]["length"])], None, proxy, signatures)
            if proposal["key"] not in visited:
                merges.append(proposal)
    splits.sort(key=_priority)
    merges.sort(key=_priority)
    # Alternate operation types, beginning with the highest proxy priority.
    # Thus a large family of splits cannot starve every adjacent merge.
    groups = [splits, merges]
    if merges and (not splits or _priority(merges[0]) < _priority(splits[0])):
        groups.reverse()
    offsets, ordered = [0, 0], []
    turn = 0
    while len(ordered) < config.max_proposals_per_round:
        if offsets[turn] < len(groups[turn]):
            ordered.append(groups[turn][offsets[turn]])
            offsets[turn] += 1
        elif offsets[1 - turn] >= len(groups[1 - turn]):
            break
        turn = 1 - turn
    return ordered


def search(seed_segments, service, proxy, candidates, config=SearchConfig()):
    """Return a monotone measured-cost search result without mutating the seed.

    At most one move is accepted per round.  Ties retain the incumbent.  A
    refused large native trial does not prevent a later, smaller proposal from
    fitting the remaining budget.  The search is bounded and may stop before
    testing every possible cut.  An unsuccessful batch advances to fresh
    proposals instead of treating proxy ordering as evidence of no useful cut.
    """
    if not isinstance(config, SearchConfig):
        raise ValueError("config must be SearchConfig")
    for name in config.__dataclass_fields__:
        _positive_integer(getattr(config, name), name)
    segments, cursor = [], 0
    for item in seed_segments:
        segment = _segment(item, expected_start=cursor)
        segments.append(segment)
        cursor += segment["length"]
    if not segments:
        raise ValueError("seed must contain at least one segment")
    candidates = list(candidates)
    if any(type(position) is not int for position in candidates):
        raise ValueError("candidate positions must be integers")
    candidates = sorted({position for position in candidates
                         if 0 < position < cursor and position % config.grid == 0})
    initial = frame_bytes(segments)
    audit, moves, visited, measured_improvements = [], [], set(), {}
    stop_reason, rounds = "max_rounds", 0
    refused_since_move = False
    tested_since_move = False
    for round_index in range(1, config.max_rounds + 1):
        proposals = _proposals(segments, proxy, candidates, config, visited)
        if not proposals:
            stop_reason = ("budget_or_unavailable" if refused_since_move else
                           "no_improvement" if tested_since_move else "no_candidates")
            break
        rounds = round_index
        best = None
        for proposal in proposals:
            tested_since_move = True
            old_bytes = frame_bytes(proposal["old"])
            row = dict(round=round_index, operation=proposal["operation"],
                       start=proposal["start"], length=proposal["length"],
                       position=proposal["position"],
                       proxy_gain_bits=proposal["proxy_gain_bits"],
                       measured_old_bytes=old_bytes, measured_new_bytes=None,
                       saving_bytes=None, status="budget_or_unavailable", accepted=False)
            if proposal["key"] in measured_improvements:
                replacements, new_bytes = measured_improvements[proposal["key"]]
            else:
                returned = service.best_many(list(proposal["regions"]))
                if returned is None or (isinstance(returned, (list, tuple))
                                        and any(item is None for item in returned)):
                    refused_since_move = True
                    visited.add(proposal["key"])
                    audit.append(row)
                    continue
                if not isinstance(returned, (list, tuple)) or len(returned) != len(proposal["regions"]):
                    raise ValueError("service must return one segment for every requested region")
                replacements = [_segment(item, expected_start=start, expected_length=length)
                                for item, (start, length) in zip(returned, proposal["regions"])]
                new_bytes = frame_bytes(replacements)
            saving = old_bytes - new_bytes
            row.update(measured_new_bytes=new_bytes, saving_bytes=saving,
                       status="improvement_not_selected" if saving >= config.min_saving_bytes
                       else "not_smaller")
            audit.append(row)
            if saving >= config.min_saving_bytes:
                # A beneficial independent move not selected this round remains
                # eligible without another native trial or stochastic reprice.
                measured_improvements[proposal["key"]] = (replacements, new_bytes)
            else:
                visited.add(proposal["key"])
            if saving >= config.min_saving_bytes and (best is None or saving > best[0]):
                best = (saving, proposal, replacements, row)
        if best is None:
            continue
        saving, proposal, replacements, row = best
        before = frame_bytes(segments)
        index = proposal["index"]
        segments[index:index + len(proposal["old"])] = replacements
        after = frame_bytes(segments)
        if before - after != saving or saving < config.min_saving_bytes:
            raise AssertionError("accepted move must strictly reduce actual frame bytes")
        row.update(status="accepted", accepted=True)
        moves.append(dict(row, before_frame_bytes=before, after_frame_bytes=after))
        refused_since_move = False
        tested_since_move = False
    return dict(segments=segments, moves=moves, proposals=audit, stop_reason=stop_reason,
                initial_frame_bytes=initial, final_frame_bytes=frame_bytes(segments),
                rounds=rounds, proposal_count=len(audit))
