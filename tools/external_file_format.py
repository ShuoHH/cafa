"""Inspect the shared F64BLK01 envelope without decoding its codec payloads.

An envelope check is NOT a losslessness check. Actual codec round trips must
be performed separately by the Java/native decoder using the persisted file.
All integers are unsigned big-endian: file header >8sIIQ, frame header >III.
"""

from pathlib import Path
import struct


MAGIC = b"F64BLK01"
FILE_HEADER = struct.Struct(">8sIIQ")
FRAME_HEADER = struct.Struct(">III")
METHOD_IDS = {1: "elf", 2: "camel", 3: "mixed_frozen_plan", 4: "btrblocks",
              5: "neats_exact_decimal", 6: "mixed_frozen_btr_extension"}
MIXED_CODEC_IDS = set(range(16, 25))
BTR_EXTENSION_CODEC_ID = 25


def inspect_container(path):
    """Validate framing, complete coverage and EOF; never claim decoded equality."""
    path = Path(path).resolve()
    size = path.stat().st_size
    with path.open("rb") as stream:
        header = stream.read(FILE_HEADER.size)
        if len(header) != FILE_HEADER.size:
            raise ValueError("truncated F64BLK01 file header")
        magic, method, block_size, expected = FILE_HEADER.unpack(header)
        if magic != MAGIC or method not in METHOD_IDS or expected == 0:
            raise ValueError("invalid F64BLK01 magic/method/value count")
        if ((method in (1, 2) and not 1 <= block_size <= 1000)
                or (method == 4 and not 1 <= block_size <= 65536)
                or (method in (3, 5, 6) and block_size != 0)):
            raise ValueError("invalid F64BLK01 fixed/variable block size")
        values, frames, payload_bytes = 0, 0, 0
        while values < expected:
            frame = stream.read(FRAME_HEADER.size)
            if len(frame) != FRAME_HEADER.size:
                raise ValueError("truncated F64BLK01 frame header")
            codec, count, length = FRAME_HEADER.unpack(frame)
            if count == 0 or count > expected - values or length == 0:
                raise ValueError("invalid F64BLK01 frame count/payload length")
            if method in (1, 2, 4):
                if codec != method or count != min(block_size, expected - values):
                    raise ValueError("invalid fixed-codec frame identity or tail length")
            elif method == 5:
                if codec != method or frames or count != expected:
                    raise ValueError("NeaTS must contain one whole-series frame")
            elif method == 3:
                if codec not in MIXED_CODEC_IDS:
                    raise ValueError("unknown mixed-codec frame id")
            elif method == 6:
                if codec == BTR_EXTENSION_CODEC_ID:
                    if count > 65536:
                        raise ValueError("BtrBlocks extension frame exceeds 65536 values")
                elif codec not in MIXED_CODEC_IDS:
                    raise ValueError("unknown mixed-codec extension frame id")
            if length > size - stream.tell():
                raise ValueError("truncated F64BLK01 frame payload")
            stream.seek(length, 1)
            values += count
            frames += 1
            payload_bytes += length
        if stream.read(1):
            raise ValueError("F64BLK01 has trailing data")
    framing = FILE_HEADER.size + FRAME_HEADER.size * frames
    if payload_bytes + framing != size:
        raise ValueError("F64BLK01 size accounting differs from actual file")
    return dict(path=str(path), container_version=MAGIC.decode("ascii"), method_id=method,
                method=METHOD_IDS[method], block_size=block_size, values=values,
                frames=frames, blocks=frames, codec_storage_bytes=payload_bytes,
                framing_bytes=framing, complete_file_bytes=size, complete_file_bits=size * 8,
                file_header_bytes=FILE_HEADER.size, frame_header_bytes=FRAME_HEADER.size,
                envelope_valid=True, decoder_losslessness_not_checked_by_inspector=True)
