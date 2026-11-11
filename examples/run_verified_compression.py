"""Write and independently decode a real CACF Top-2 stored file.

This uses one whole-series region. It exercises the frozen selector and native
codec adapters, but does not claim to run adaptive boundary detection or DP.
The two native executables must be built from external upstream dependencies.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import sys


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "tools"))

import flat_pool_format as fmt  # noqa: E402
from paper_mainline_native import run_layout  # noqa: E402
from shared_compression_features import COMPACT20_COLUMNS, FeatureBank  # noqa: E402
from shared_feature_policy import FactoryPolicy  # noqa: E402


DEFAULT_INPUT = ROOT / "examples" / "data" / "synthetic_1024.f64le"
METADATA = ROOT / "models" / "class_balanced_compact20.json"
MODEL = ROOT / "models" / "class_balanced_compact20.joblib"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def checked_model(model_path=None, metadata_path=None):
    if (model_path is None) != (metadata_path is None):
        raise ValueError("model and metadata must be supplied together")
    model_path = (MODEL if model_path is None else Path(model_path)).resolve(strict=True)
    metadata_path = (METADATA if metadata_path is None else Path(metadata_path)).resolve(strict=True)
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    if (metadata.get("feature_columns") != list(COMPACT20_COLUMNS)
            or metadata.get("model_path") != model_path.name
            or sha256(model_path) != metadata.get("model_sha256")):
        raise ValueError("frozen model schema, location or SHA-256 differs")
    # Joblib is pickle-based; never deserialize an untrusted replacement.
    import joblib

    model = joblib.load(model_path)
    metadata["model_path"] = str(model_path)
    return model, metadata


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT,
                        help="headerless little-endian binary64 file (default: synthetic fixture)")
    parser.add_argument("--out", type=Path, required=True,
                        help="new output directory; existing directories are never overwritten")
    parser.add_argument("--exporter", type=Path, required=True,
                        help="independently built FrozenPlanFile executable")
    parser.add_argument("--fixed-btr-exe", type=Path, required=True,
                        help="independently built BtrBlocksFixed executable")
    parser.add_argument("--timeout", type=int, default=300,
                        help="per-native-command timeout in seconds")
    parser.add_argument("--model", type=Path, help="trusted compact20 joblib; pair with --model-metadata")
    parser.add_argument("--model-metadata", type=Path,
                        help="trusted JSON schema and expected model SHA-256")
    parser.add_argument("--dataset", help="original source dataset identity; pair with --series")
    parser.add_argument("--series", help="original source series identity; pair with --dataset")
    args = parser.parse_args()
    if (args.model is None) != (args.model_metadata is None):
        raise SystemExit("--model and --model-metadata must be supplied together")
    if (args.dataset is None) != (args.series is None) or args.dataset == "" or args.series == "":
        raise SystemExit("--dataset and --series must be nonempty and supplied together")

    raw = args.input.resolve(strict=True)
    exporter = args.exporter.resolve(strict=True)
    fixed_exe = args.fixed_btr_exe.resolve(strict=True)
    out = args.out.resolve()
    if not raw.is_file() or not exporter.is_file() or not fixed_exe.is_file():
        raise SystemExit("input and both native executables must be regular files")
    count, remainder = divmod(raw.stat().st_size, 8)
    if remainder or not 1 <= count <= 102400:
        raise SystemExit("input must contain 1..102400 headerless binary64 values")
    if args.timeout < 1:
        raise SystemExit("timeout must be positive")
    if out.exists():
        raise SystemExit("output directory already exists; choose a fresh path")

    source_hash = sha256(raw)
    identity = (args.dataset or "CACF_external_demo",
                args.series or "sample_" + source_hash[:12], source_hash)
    job = dict(dataset=identity[0], series=identity[1], sha256=source_hash,
               input=str(raw), values=count)
    model, metadata = checked_model(args.model, args.model_metadata)

    def policy_factory():
        bank = FeatureBank(raw, identity)
        return FactoryPolicy(job, model, metadata, bank, top_k=2)

    row = run_layout(job, [(0, count)], out / "compression", exporter, fixed_exe,
                     policy_factory=policy_factory, top_k=2, timeout=args.timeout)
    stored = Path(row["container_path"])
    decoded = out / "decoded.f64le"
    decode = fmt.decode_file(stored, decoded, exporter, fixed_exe,
                             timeout=args.timeout)
    if (row.get("status") != "passed"
            or row.get("complete_file_roundtrip_verified") is not True
            or row.get("complete_file_bits") != stored.stat().st_size * 8
            or decode.get("decoded_sha256") != source_hash
            or decoded.stat().st_size != raw.stat().st_size
            or sha256(raw) != source_hash):
        raise ValueError("saved-file encoding or independent bitwise decoding failed")
    report = dict(scope="whole-series frozen Top-2 native compression and independent decode",
                  adaptive_segmentation_ran=False, dp_ran=False,
                  input=str(raw), values=count, input_sha256=source_hash,
                  complete_file_bits=row["complete_file_bits"],
                  selected_codecs=[item["codec"] for item in row["selections"]],
                  stored_file=str(stored), stored_sha256=row["container_sha256"],
                  decoded_file=str(decoded), decoded_sha256=decode["decoded_sha256"],
                  bitwise_identical=True, frozen_model_sha256=metadata["model_sha256"])
    with (out / "example_report.json").open("x", encoding="utf-8") as stream:
        json.dump(report, stream, ensure_ascii=False, indent=2)
        stream.write("\n")
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
