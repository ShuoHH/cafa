"""Run CACF's raw feature, candidate and frozen Top-2 selector components.

This demonstration does not encode, decode, refine or price a segment.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import sys


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "tools"))

from compression_candidates import build_candidates  # noqa: E402
from shared_compression_features import COMPACT20_COLUMNS, FeatureBank  # noqa: E402
from shared_feature_policy import FactoryPolicy  # noqa: E402


DEFAULT_INPUT = ROOT / "examples" / "data" / "synthetic_1024.f64le"
METADATA = ROOT / "models" / "class_balanced_compact20.json"
MODEL = ROOT / "models" / "class_balanced_compact20.joblib"
MAX_VALUES = 102400


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("input", nargs="?", type=Path, default=DEFAULT_INPUT,
                        help="headerless little-endian finite float64 data (max 102400 values)")
    args = parser.parse_args()
    raw = args.input.resolve(strict=True)
    count, remainder = divmod(raw.stat().st_size, 8)
    if remainder or not 1 <= count <= MAX_VALUES:
        raise SystemExit("input must contain 1..102400 headerless float64 values")
    metadata = json.loads(METADATA.read_text(encoding="utf-8"))
    if metadata["feature_columns"] != list(COMPACT20_COLUMNS):
        raise SystemExit("frozen model's 20-feature schema differs")
    if metadata["model_path"] != MODEL.name or sha256(MODEL) != metadata["model_sha256"]:
        raise SystemExit("frozen model path or SHA-256 differs; refusing to deserialize")

    # joblib is pickle-based: load only this checked, repository-supplied artifact.
    import joblib

    model = joblib.load(MODEL)
    metadata["model_path"] = str(MODEL.resolve())
    source_hash = sha256(raw)
    identity = ("CACF_external_demo", "sample_" + source_hash[:12], source_hash)
    job = dict(dataset=identity[0], series=identity[1], sha256=source_hash,
               input=str(raw), values=count)
    bank = FeatureBank(raw, identity)
    policy = FactoryPolicy(job, model, metadata, bank, top_k=2)
    regions = [(0, count)]
    if count >= 1024:
        regions.extend((start, 256) for start in (0, 256, 512, 768))
    predictions = policy.predict_many(regions)
    candidates = build_candidates(bank.boundary_cache(), [], mode="compression",
                                  grid=64, minimum=64, window=256, maximum=64)
    bank.assert_unchanged()
    print(json.dumps({
        "scope": "features, proposed boundaries and frozen Top-2 recommendations only",
        "not_measured": ["compressed size", "lossless decoding", "ClaSS initial partition",
                         "boundary refinement", "dynamic programming"],
        "input_sha256": source_hash,
        "values": count,
        "feature_count": len(COMPACT20_COLUMNS),
        "candidate_boundary_count": len(candidates),
        "candidate_positions_first10": [r["position"] for r in candidates[:10]],
        "regions": [{"start": start, "length": length, "top2": list(actions)}
                    for (start, length), actions in zip(regions, predictions)],
        "frozen_model_sha256": metadata["model_sha256"],
        "external_model_reuse": policy.predictions[0]["external_model_reuse"],
    }, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
