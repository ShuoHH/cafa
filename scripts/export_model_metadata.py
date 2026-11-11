"""Select and relocate the authenticated frozen May selector metadata.

This is a packaging step, not model training. It never deserializes joblib.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path


SOURCE_METADATA_SHA256 = "870ff0384534b2d4d8d496d6725bac4393b445629a9fc6454b95572b0955ff36"
MODEL_SHA256 = "3c2bbc7b95c79d77cdbc514d578ffbeab08efdb02fe20f3bdf51faeba87155b0"


def digest(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("source_metadata", type=Path)
    parser.add_argument("source_model", type=Path)
    parser.add_argument("output", type=Path)
    args = parser.parse_args()
    if digest(args.source_metadata) != SOURCE_METADATA_SHA256:
        raise SystemExit("historical metadata SHA-256 differs")
    if digest(args.source_model) != MODEL_SHA256:
        raise SystemExit("frozen model SHA-256 differs")
    rows = json.loads(args.source_metadata.read_text(encoding="utf-8"))
    selected = [row for row in rows if row.get("arm") == "class_balanced"
                and row.get("feature_set") == "compact20"]
    if len(selected) != 1 or selected[0].get("model_sha256") != MODEL_SHA256:
        raise SystemExit("expected one authenticated compact20 class-balanced row")
    row = dict(selected[0])
    row["model_path"] = "class_balanced_compact20.joblib"
    row["release_source_metadata_sha256"] = SOURCE_METADATA_SHA256
    row["release_metadata_change"] = "model_path relocated; model and training metadata unchanged"
    if args.output.exists():
        raise SystemExit("refusing to overwrite existing metadata")
    args.output.write_text(json.dumps(row, ensure_ascii=False, indent=2,
                                     allow_nan=False) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
