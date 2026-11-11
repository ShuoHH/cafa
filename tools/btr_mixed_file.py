"""Versioned persisted-file dispatcher for the frozen-plan BtrBlocks extension.

Method 6 keeps the existing method-3 payloads (codec IDs 16..24) unchanged and
adds ID 25 containing one native BtrBlocks datablock. The original native
decoders remain unchanged. This dispatcher wraps persisted payloads in their
original envelopes and invokes those decoders; it never re-encodes a payload.
Decode-only success is NOT an external bitwise round-trip verification.
"""

import argparse
import json
import math
from pathlib import Path

from external_file_format import (BTR_EXTENSION_CODEC_ID, FILE_HEADER, FRAME_HEADER,
                                  MAGIC, MIXED_CODEC_IDS, inspect_container)
from replay_frozen_files import execute_export, sha256


def _integer(value, name, maximum=None, positive=False):
    if type(value) is not int or value < (1 if positive else 0):
        raise ValueError(name + " must be a strict " + ("positive" if positive else "unsigned") + " integer")
    if maximum is not None and value > maximum:
        raise ValueError(name + " exceeds its format limit")
    return value


def read_frames(path):
    """Return checked method-3/4/6 framing and actual persisted payload bytes."""
    path = Path(path).resolve()
    before = sha256(path)
    info = inspect_container(path)
    if info["method_id"] not in (3, 4, 6):
        raise ValueError("require a method-3, method-4 or method-6 container")
    frames, start = [], 0
    with path.open("rb") as stream:
        stream.seek(FILE_HEADER.size)
        for _ in range(info["frames"]):
            header = stream.read(FRAME_HEADER.size)
            if len(header) != FRAME_HEADER.size:
                raise ValueError("container changed/truncated while reading frames")
            codec_id, count, length = FRAME_HEADER.unpack(header)
            payload = stream.read(length)
            if len(payload) != length:
                raise ValueError("container changed/truncated while reading payload")
            frames.append(dict(codec_id=codec_id, count=count, payload=payload, start=start))
            start += count
        if stream.read(1) or start != info["values"]:
            raise ValueError("container changed while reading frames")
    if before != sha256(path):
        raise ValueError("source container changed while reading frames")
    return info, frames


def _validated_frames(frames, total_values, method):
    _integer(total_values, "total_values", (1 << 64) - 1, positive=True)
    frames = list(frames)
    cursor = 0
    for frame in frames:
        if not isinstance(frame, dict):
            raise ValueError("each frame must be a dictionary")
        codec = _integer(frame.get("codec_id"), "codec_id", (1 << 32) - 1)
        count = _integer(frame.get("count"), "count", (1 << 32) - 1, positive=True)
        start = _integer(frame.get("start"), "start", (1 << 64) - 1)
        payload = frame.get("payload")
        if type(payload) is not bytes or not payload or len(payload) > (1 << 32) - 1:
            raise ValueError("frame payload must be nonempty bytes within uint32 length")
        if start != cursor:
            raise ValueError("frames must cover contiguous values starting at zero")
        if method == 3 and codec not in MIXED_CODEC_IDS:
            raise ValueError("method 3 accepts only legacy codec IDs 16..24")
        if method == 6 and codec not in MIXED_CODEC_IDS | {BTR_EXTENSION_CODEC_ID}:
            raise ValueError("unknown method-6 codec ID")
        if codec == BTR_EXTENSION_CODEC_ID and count > 65536:
            raise ValueError("BtrBlocks extension frame exceeds 65536 values")
        if method == 4 and (codec != 4 or len(frames) != 1 or count > 65536):
            raise ValueError("temporary BtrBlocks envelope must contain one bounded frame")
        cursor += count
    if not frames or cursor != total_values:
        raise ValueError("frames do not cover total_values exactly")
    return frames


def _write_container(path, frames, total_values, method):
    frames = _validated_frames(frames, total_values, method)
    block_size = total_values if method == 4 else 0
    path = Path(path).resolve()
    with path.open("xb") as stream:
        stream.write(FILE_HEADER.pack(MAGIC, method, block_size, total_values))
        for frame in frames:
            stream.write(FRAME_HEADER.pack(frame["codec_id"], frame["count"], len(frame["payload"])))
            stream.write(frame["payload"])
    return inspect_container(path)


def write_mixed(path, frames, total_values):
    """Exclusively persist method 6; payloads must already be actual encoded bytes."""
    return _write_container(path, frames, total_values, 6)


def _validate_native_report(report, info, legacy):
    if not isinstance(report, dict):
        raise ValueError("native decoder report must be an object")
    for name in ("values", "decoded_values"):
        if _integer(report.get(name), name) != info["values"]:
            raise ValueError("native decoder " + name + " differs from container coverage")
    if _integer(report.get("blocks"), "blocks") != info["frames"]:
        raise ValueError("native decoder blocks differs from container framing")
    if legacy:
        if (report.get("status") != "passed" or report.get("operation") != "standalone_decode"
                or report.get("decoded_from_persisted_file_only") is not True
                or report.get("complete_file_roundtrip_verified") is not False
                or _integer(report.get("method_id"), "method_id") != 3
                or _integer(report.get("mismatch_count"), "mismatch_count") != 0):
            raise ValueError("legacy report does not describe a successful standalone persisted-file decode")
        for name in ("complete_file_bytes", "complete_file_bits", "codec_storage_bytes", "framing_bytes"):
            if _integer(report.get(name), name) != info[name]:
                raise ValueError("legacy decoder " + name + " disagrees with persisted file")
    elif report.get("method") != "btrblocks" or report.get("mode") != "decode":
        raise ValueError("BtrBlocks report is not a standalone decode report")


def _decode_group(container, target, folder, exporter, btr_exe, timeout, execute):
    before = sha256(container)
    info = inspect_container(container)
    legacy = info["method_id"] == 3
    if info["method_id"] not in (3, 4):
        raise ValueError("native decoder requires a legacy or BtrBlocks envelope")
    report_path, log = folder / "native_report.json", folder / "decode.log"
    command = ([str(exporter), "--decode", str(container), str(target), str(report_path)] if legacy
               else [str(btr_exe), "--decode", str(container), str(target)])
    if execute(command, log, timeout) != 0:
        raise ValueError("native persisted-file decoder failed; see " + str(log))
    if sha256(container) != before:
        raise ValueError("native decoder changed its persisted input container")
    if legacy:
        report = json.loads(report_path.read_text(encoding="utf-8"))
    else:
        reports = []
        for line in log.read_text(encoding="utf-8").splitlines():
            if line.lstrip().startswith("{"):
                try:
                    candidate = json.loads(line)
                except json.JSONDecodeError:
                    continue
                if isinstance(candidate, dict) and candidate.get("method") == "btrblocks":
                    reports.append(candidate)
        if len(reports) != 1:
            raise ValueError("require exactly one BtrBlocks standalone decoder JSON report")
        report = reports[0]
    _validate_native_report(report, info, legacy)
    if not target.is_file() or target.stat().st_size != info["values"] * 8:
        raise ValueError("native decoded byte count differs from full container coverage")
    return dict(container_path=str(container), decoded_path=str(target), values=info["values"],
                method_id=info["method_id"], log_path=str(log))


def decode_file(container, output_raw, work_dir, exporter, btr_exe, timeout=300, execute=execute_export):
    """Decode from the persisted file only; external original data is not required."""
    container, output_raw, work_dir = map(lambda item: Path(item).resolve(),
                                         (container, output_raw, work_dir))
    exporter, btr_exe = Path(exporter).resolve(), Path(btr_exe).resolve()
    if (type(timeout) not in (int, float) or not math.isfinite(timeout) or timeout <= 0):
        raise ValueError("timeout must be a finite positive number")
    if output_raw in (container, exporter, btr_exe) or output_raw.exists():
        raise ValueError("decode output must be a new path distinct from inputs/executables")
    if work_dir.exists() or work_dir == output_raw or container == work_dir:
        raise ValueError("decoder work_dir must be an exclusively new directory")
    before = sha256(container)
    info, frames = read_frames(container)
    work_dir.mkdir()
    groups = []
    if info["method_id"] in (3, 4):
        folder = work_dir / "group_0000"
        folder.mkdir()
        groups.append(_decode_group(container, folder / "decoded.raw", folder, exporter, btr_exe,
                                    timeout, execute))
    else:
        index = 0
        while index < len(frames):
            end = index + 1
            legacy = frames[index]["codec_id"] != BTR_EXTENSION_CODEC_ID
            if legacy:
                while end < len(frames) and frames[end]["codec_id"] != BTR_EXTENSION_CODEC_ID:
                    end += 1
            group_frames = [dict(frame) for frame in frames[index:end]]
            offset = group_frames[0]["start"]
            for frame in group_frames:
                frame["start"] -= offset
                if not legacy:
                    frame["codec_id"] = 4
            count = sum(frame["count"] for frame in group_frames)
            folder = work_dir / ("group_%04d" % len(groups))
            folder.mkdir()
            wrapped = folder / "persisted_payloads.f64blk"
            _write_container(wrapped, group_frames, count, 3 if legacy else 4)
            groups.append(_decode_group(wrapped, folder / "decoded.raw", folder, exporter, btr_exe,
                                        timeout, execute))
            index = end
    if sha256(container) != before:
        raise ValueError("source container changed during decoding")
    if sum(group["values"] for group in groups) != info["values"]:
        raise ValueError("decoded groups do not cover the entire persisted file")
    # Output is opened only after all native decoders and protocol checks pass.
    with output_raw.open("xb") as output:
        for group in groups:
            with Path(group["decoded_path"]).open("rb") as source:
                for chunk in iter(lambda: source.read(1024 * 1024), b""):
                    output.write(chunk)
    if output_raw.stat().st_size != info["values"] * 8 or sha256(container) != before:
        raise ValueError("decoded output length/source preservation check failed")
    return dict(status="decoded", method_id=info["method_id"], container_path=str(container),
                container_sha256=before, source_container_unchanged=True, output_path=str(output_raw),
                decoded_values=info["values"], decoded_bytes=output_raw.stat().st_size,
                decoded_sha256=sha256(output_raw), decoded_from_persisted_file_only=True,
                complete_file_roundtrip_verified=False, groups=groups)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--decode", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--work-dir", required=True)
    parser.add_argument("--exporter", required=True)
    parser.add_argument("--btr-exe", required=True)
    parser.add_argument("--timeout", type=int, default=300)
    args = parser.parse_args()
    print(json.dumps(decode_file(args.decode, args.output, args.work_dir, args.exporter,
                                 args.btr_exe, args.timeout), indent=2))


if __name__ == "__main__":
    main()
