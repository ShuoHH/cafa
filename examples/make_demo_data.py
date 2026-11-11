"""Create a small, synthetic, headerless little-endian float64 series."""

from __future__ import annotations

import argparse
import math
from pathlib import Path
import struct


OUT = Path(__file__).with_name("data") / "synthetic_1024.f64le"
ADAPTIVE_OUT = Path(__file__).with_name("data") / "synthetic_8192.f64le"


def values() -> list[float]:
    result: list[float] = []
    for i in range(1024):
        if i < 256:
            value = 1.0 + 0.0001 * i
        elif i < 512:
            value = 1.0256 + 0.00003 * (i - 256) + 0.0002 * (i % 7)
        elif i < 768:
            value = 2.0 + 0.01 * ((i // 24) % 4)
        else:
            value = 2.04 + 0.000001 * (i - 768) + 0.000004 * (i % 11)
        result.append(value)
    return result


def adaptive_values() -> list[float]:
    """A larger, deterministic four-regime fixture for boundary detection."""
    result: list[float] = []
    for i in range(8192):
        if i < 2048:
            value = 1.0 + 0.0001 * i + 0.00001 * math.sin(i / 19)
        elif i < 4096:
            value = 1.2 + 0.015 * math.sin(i / 17) + 0.003 * math.sin(i / 5)
        elif i < 6144:
            value = 2.0 + 0.01 * ((i // 40) % 4)
        else:
            value = 2.04 + 0.000002 * (i - 6144) + 0.000004 * (i % 13)
        result.append(value)
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--adaptive", action="store_true",
                        help="generate the 8,192-value boundary-detection fixture")
    args = parser.parse_args()
    output, samples = ((ADAPTIVE_OUT, adaptive_values()) if args.adaptive
                       else (OUT, values()))
    output.parent.mkdir(parents=True, exist_ok=True)
    if output.exists():
        raise SystemExit(f"refusing to overwrite {output}")
    output.write_bytes(struct.pack(f"<{len(samples)}d", *samples))
    print(output)


if __name__ == "__main__":
    main()
