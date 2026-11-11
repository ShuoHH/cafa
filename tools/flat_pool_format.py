"""Independent persisted format for a flat pool of explicit codec pipelines.

F64PL001 is deliberately NOT an old method-6 BtrBlocks container. Legacy
16..24 payloads retain their original bytes; 32..36 contain complete BTRFIX01
files. No original BtrBlocks auto-selector payload (ID 25) is accepted.
An envelope check is not a losslessness check: verify_file decodes the saved
file in fresh native processes, without passing the original to a decoder.
"""
import argparse
import json
from pathlib import Path
import shutil
import struct
import tempfile

import btr_mixed_file as legacy
import replay_frozen_files as replay
from codec_pool import RETAINED_PHYSICAL_IDS

MAGIC = b"F64PL001"
METHOD = 7
FILE_HEADER = struct.Struct(">8sIIQ")
FRAME_HEADER = struct.Struct(">III")
FIXED_ACTIONS = ("btr_onevalue", "btr_dict_bp", "btr_rle_fixed",
                 "btr_frequency_fixed", "btr_decimal_fixed")
FIXED_IDS = dict(zip(FIXED_ACTIONS, range(32, 37)))
CODEC_IDS = dict(replay.CODEC_IDS, **FIXED_IDS)
NAMES = {value: key for key, value in CODEC_IDS.items()}


def checked_frames(frames, count):
    if type(count) is not int or not 0 < count < 2**64:
        raise ValueError("invalid complete value count")
    result, cursor = [], 0
    for frame in frames:
        codec, n, start, payload = (frame.get(k) for k in ("codec_id", "count", "start", "payload"))
        if (type(codec) is not int or codec not in RETAINED_PHYSICAL_IDS or type(n) is not int
                or not 0 < n <= 102400 or type(start) is not int or start != cursor
                or type(payload) is not bytes or not 0 < len(payload) < 2**32):
            raise ValueError("invalid flat-pool frame identity, bytes or coverage")
        if codec == CODEC_IDS["alp"] and n % 1024:
            raise ValueError("ALP frame must be an aligned complete prefix")
        if codec in FIXED_IDS.values():
            if len(payload) < 24 or struct.unpack(">8sIIQ", payload[:24]) != (b"BTRFIX01",1,codec,n):
                raise ValueError("fixed native payload identity/count differs from its outer frame")
        result.append(dict(frame))
        cursor += n
    if not result or cursor != count:
        raise ValueError("flat-pool frames do not cover the complete sequence")
    return result


def inspect_container(path):
    path = Path(path).resolve()
    size, frames, covered, payload_bytes = path.stat().st_size, 0, 0, 0
    codec_counts = {}
    with path.open("rb") as source:
        head = source.read(FILE_HEADER.size)
        if len(head) != FILE_HEADER.size:
            raise ValueError("truncated flat-pool header")
        magic, method, reserved, count = FILE_HEADER.unpack(head)
        if magic != MAGIC or method != METHOD or reserved != 0 or count <= 0:
            raise ValueError("invalid flat-pool header")
        while covered < count:
            head = source.read(FRAME_HEADER.size)
            if len(head) != FRAME_HEADER.size:
                raise ValueError("truncated flat-pool frame header")
            codec, n, length = FRAME_HEADER.unpack(head)
            if (codec not in RETAINED_PHYSICAL_IDS or not 0 < n <= min(102400, count-covered)
                    or not 0 < length <= size-source.tell()
                    or (codec == CODEC_IDS["alp"] and n % 1024)):
                raise ValueError("invalid flat-pool frame")
            if codec in FIXED_IDS.values():
                if length < 24 or struct.unpack(">8sIIQ",source.read(24)) != (b"BTRFIX01",1,codec,n):
                    raise ValueError("invalid fixed native payload header")
                source.seek(length-24, 1)
            else:
                source.seek(length, 1)
            frames += 1
            codec_counts[NAMES[codec]] = codec_counts.get(NAMES[codec], 0) + 1
            covered += n
            payload_bytes += length
        if source.read(1):
            raise ValueError("trailing flat-pool bytes")
    framing = FILE_HEADER.size + frames*FRAME_HEADER.size
    if framing+payload_bytes != size:
        raise ValueError("flat-pool size accounting mismatch")
    return dict(path=str(path), method_id=METHOD, container_version=MAGIC.decode(),
        values=count, frames=frames, complete_file_bytes=size, complete_file_bits=8*size,
        codec_storage_bytes=payload_bytes, framing_bytes=framing, envelope_valid=True,
        decoder_losslessness_not_checked_by_inspector=True,
        actual_physical_codec_counts=codec_counts, actual_frame_ids_restricted8=True)


def write_container(path, frames, count):
    frames = checked_frames(frames, count)
    with Path(path).open("xb") as target:
        target.write(FILE_HEADER.pack(MAGIC, METHOD, 0, count))
        for frame in frames:
            target.write(FRAME_HEADER.pack(frame["codec_id"], frame["count"], len(frame["payload"])))
            target.write(frame["payload"])
    return inspect_container(path)


def read_frames(path):
    path = Path(path).resolve()
    before, info = replay.sha256(path), inspect_container(path)
    result, start = [], 0
    with path.open("rb") as source:
        source.seek(FILE_HEADER.size)
        for _ in range(info["frames"]):
            codec, n, length = FRAME_HEADER.unpack(source.read(FRAME_HEADER.size))
            result.append(dict(codec_id=codec, count=n, start=start, payload=source.read(length)))
            start += n
    result = checked_frames(result, info["values"])
    if replay.sha256(path) != before:
        raise ValueError("flat-pool container changed while reading")
    return info, result


def decode_file(container, output, exporter, fixed_exe, timeout=300, execute=replay.execute_export):
    container, output, exporter, fixed_exe = map(lambda p: Path(p).resolve(),
        (container, output, exporter, fixed_exe))
    if output.exists() or output in (container, exporter, fixed_exe):
        raise ValueError("decode target must be a fresh file distinct from inputs")
    before, (info, frames) = replay.sha256(container), read_frames(container)
    decoded_groups = []
    # Only this newly-created scratch tree is removed; no user artifacts are cleaned.
    with tempfile.TemporaryDirectory(prefix="flat_decode_", dir=output.parent) as temporary:
        scratch, index = Path(temporary), 0
        while index < len(frames):
            end = index+1
            is_fixed = frames[index]["codec_id"] in FIXED_IDS.values()
            if not is_fixed:
                while end < len(frames) and frames[end]["codec_id"] not in FIXED_IDS.values():
                    end += 1
            group = scratch/("group_%04d" % len(decoded_groups))
            group.mkdir()
            decoded = group/"decoded.raw"
            count = sum(f["count"] for f in frames[index:end])
            if is_fixed:
                wrapped = group/"payload.bfx"
                wrapped.write_bytes(frames[index]["payload"])
                if execute([str(fixed_exe), "--decode", str(wrapped), str(decoded)],
                           group/"decode.log", timeout) != 0:
                    raise ValueError("fixed native standalone decode failed")
            else:
                offset = frames[index]["start"]
                part = [dict(f, start=f["start"]-offset) for f in frames[index:end]]
                wrapped = group/"payload.f64blk"
                legacy._write_container(wrapped, part, count, 3)
                legacy._decode_group(wrapped, decoded, group, exporter, fixed_exe, timeout, execute)
            if not decoded.is_file() or decoded.stat().st_size != 8*count:
                raise ValueError("native decoder output length mismatch")
            decoded_groups.append((decoded, count))
            index = end
        if replay.sha256(container) != before:
            raise ValueError("decoder changed saved input")
        with output.open("xb") as target:
            for decoded, _ in decoded_groups:
                with decoded.open("rb") as source:
                    shutil.copyfileobj(source, target)
    if output.stat().st_size != 8*info["values"] or replay.sha256(container) != before:
        raise ValueError("complete standalone decode coverage/preservation failed")
    return dict(status="decoded", decoded_from_persisted_file_only=True,
        complete_file_roundtrip_verified=False, decoded_values=info["values"],
        decoded_sha256=replay.sha256(output), container_sha256=before)


def verify_file(container, original, exporter, fixed_exe, timeout=300, execute=replay.execute_export):
    container, original = Path(container).resolve(), Path(original).resolve()
    source_hash = replay.sha256(original)
    with tempfile.TemporaryDirectory(prefix="flat_verify_", dir=container.parent) as temporary:
        result = decode_file(container, Path(temporary)/"decoded.raw", exporter, fixed_exe, timeout, execute)
        if result["decoded_sha256"] != source_hash or original.stat().st_size != result["decoded_values"]*8:
            raise ValueError("persisted complete file is not bitwise lossless")
    if replay.sha256(original) != source_hash:
        raise ValueError("original changed during verification")
    return dict(result, status="passed", source_sha256=source_hash,
        complete_file_roundtrip_verified=True, **inspect_container(container))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--decode", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--exporter", required=True)
    parser.add_argument("--fixed-btr-exe", required=True)
    args = parser.parse_args()
    print(json.dumps(decode_file(args.decode, args.output, args.exporter, args.fixed_btr_exe), indent=2))


if __name__ == "__main__":
    main()
