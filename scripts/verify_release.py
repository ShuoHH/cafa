"""Check publication files without unpickling weights or invoking native codecs.

The hash inventory detects accidental edits, not an adversary replacing both
the source tree and its inventory. This is not a security/build certificate.
"""
from __future__ import annotations
import argparse
import ast
import hashlib
import json
from pathlib import Path
import re
import sys

ROOT = Path(__file__).resolve().parents[1]
SEAL = ROOT / "release_inventory.json"
EXPECTED = ("gorilla", "chimp", "chimp128", "tsxor", "rle+tsxor", "alp",
            "btr_dict_bp", "btr_decimal_fixed")
FORBIDDEN = {"Data", "benchmark", "experiment_results", "compression_experiment_results",
             ".venv", "__pycache__", "node_modules"}
MARKERS = (rb"-----BEGIN (?:RSA |EC |OPENSSH |DSA )?PRIVATE KEY-----",
           rb"gh[pousr]_[A-Za-z0-9]{30,}", rb"github_pat_[A-Za-z0-9_]{50,}",
           rb"(?:https?|ssh)://[^\s/:]+:[^\s/@]+@")

def digest(path):
    h = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()

def path_for(name):
    path = (ROOT / name).resolve(strict=True)
    if not path.is_relative_to(ROOT) or not path.is_file():
        raise ValueError("invalid repository file: " + name)
    return path

def authenticate(name, sha, size=None):
    path = path_for(name)
    if digest(path) != sha or (size is not None and path.stat().st_size != size):
        raise ValueError("SHA/size mismatch: " + name)

def load(name):
    return json.loads(path_for(name).read_text(encoding="utf-8"))

def check_tree():
    files = []
    for p in ROOT.rglob("*"):
        rel = p.relative_to(ROOT)
        if ".git" in rel.parts:
            continue
        if p.is_symlink() or any(x in FORBIDDEN or x.startswith(".venv-") for x in rel.parts):
            raise ValueError("private/generated path: " + rel.as_posix())
        if not p.is_file():
            continue
        if p.stat().st_size > 2_000_000 or p.name in {".env", "id_rsa", "id_ed25519"} or p.suffix.lower() in {
            ".pem", ".key", ".pyc", ".pyo", ".dll", ".so", ".exe", ".zip", ".gz", ".bin", ".raw"}:
            raise ValueError("unexpected private/build artifact: " + rel.as_posix())
        data = p.read_bytes()
        if p.suffix in {".py", ".json", ".md", ".txt", ".sh", ".toml", ".cmake"} or p.name.startswith("."):
            if any(re.search(pattern, data) for pattern in MARKERS):
                raise ValueError("credential-like marker (content withheld): " + rel.as_posix())
        if p.suffix == ".py":
            ast.parse(data.decode("utf-8"), filename=str(p))
        elif p.suffix == ".json":
            json.loads(data)
        files.append(p)
    sys.path.insert(0, str(ROOT / "tools"))
    import codec_pool
    if codec_pool.RETAINED_ACTIONS != EXPECTED or codec_pool.RETAINED_PHYSICAL_IDS != {16,17,18,19,23,24,33,36}:
        raise ValueError("codec pool or physical ID contract changed")
    manifest = load("models/release_manifest.json")
    artifacts = manifest["artifacts"]
    if len(artifacts) != 36 or len({x["id"] for x in artifacts}) != 36:
        raise ValueError("expected 36 distinct frozen models")
    variants = {"base10":10, "decimal15":15, "residue15":15, "compact20":20}
    anchors = {}
    for row in artifacts:
        authenticate(row["model_path"], row["model_sha256"], row["model_bytes"])
        authenticate(row["metadata_path"], row["metadata_sha256"])
        meta = load(row["metadata_path"])
        if (meta["model_sha256"] != row["model_sha256"] or meta["model_path"] != Path(row["model_path"]).name
            or meta["feature_count"] != variants[row["variant"]] or tuple(meta["retained_actions"]) != EXPECTED
            or not all(codec_pool.permitted_action(x) for x in meta["classes"])
            or row["weights_changed"] or row["refit"]):
            raise ValueError("model contract mismatch: " + row["id"])
        anchors.setdefault(row["anchor_series"], set()).add(row["variant"])
    if len(anchors) != 9 or any(x != set(variants) for x in anchors.values()):
        raise ValueError("expected nine folds with four feature variants each")
    default = manifest["compatibility_default"]
    authenticate(default["model_path"], default["model_sha256"])
    authenticate(default["metadata_path"], default["metadata_sha256"])
    provenance = manifest["provenance"]
    authenticate(provenance["runtime_manifest_path"], provenance["runtime_manifest_sha256"])
    for row in provenance["original_fold_metadata"]:
        authenticate(row["path"], row["sha256"])
    authenticate(manifest["experiment_inputs_path"], manifest["experiment_inputs_sha256"])
    inputs = load(manifest["experiment_inputs_path"])["inputs"]
    if len(inputs) != 13 or sum(x["included_in_main12"] for x in inputs) != 12:
        raise ValueError("13-evaluated/12-main scope mismatch")
    routes = {x["canonical_name"]:x for x in manifest["input_model_routing"]}
    by_id = {x["id"]:x for x in artifacts}
    for row in inputs:
        route = routes[row["canonical_name"]]
        if (route["input_sha256"] != row["sha256"] or route["values"] != row["values"]
            or by_id[row["primary_model_id"]]["model_path"] != row["primary_model_path"]
            or row["bundled_in_repository"]):
            raise ValueError("input/model route mismatch: " + row["canonical_name"])
    authenticate("examples/data/synthetic_1024.f64le", "c7d8eaf15e384f31378580ced134a3edfc953515577216edcc193e6722812f81", 8192)
    authenticate("examples/data/synthetic_8192.f64le", "c89a123d256bf4419562733ed4ae25bcc99b98a6acaa89a13421cecabd1dc8d2", 65536)
    expected_weights = {x["model_path"] for x in artifacts} | {default["model_path"]}
    if {p.relative_to(ROOT).as_posix() for p in files if p.suffix == ".joblib"} != expected_weights:
        raise ValueError("unexpected or missing weight files")
    if {p.relative_to(ROOT).as_posix() for p in files if p.suffix == ".f64le"} != {
        "examples/data/synthetic_1024.f64le", "examples/data/synthetic_8192.f64le"}:
        raise ValueError("unexpected raw input")
    return [{"path":p.relative_to(ROOT).as_posix(), "bytes":p.stat().st_size, "sha256":digest(p)}
            for p in sorted(files) if p != SEAL]

def main():
    parser = argparse.ArgumentParser(description=__doc__)
    modes = parser.add_mutually_exclusive_group()
    modes.add_argument("--check-tree", action="store_true", help="validate before sealing the final inventory")
    modes.add_argument("--seal", action="store_true", help="create a fresh inventory; refuses overwrite")
    args = parser.parse_args()
    rows = check_tree()
    if args.seal:
        with SEAL.open("x", encoding="utf-8", newline="\n") as f:
            json.dump({"version":"cacf_strict8_publication_inventory_v1", "files":rows}, f, indent=2)
            f.write("\n")
    elif not args.check_tree and load(SEAL.name)["files"] != rows:
        raise ValueError("inventory mismatch: source content, omission, or addition")
    print(f"release checks passed: {len(rows)} files; 36 frozen weights + demo alias; "
          "13 input routes; strict-eight IDs; AST and credential-marker checks")
    print("This check does not unpickle models, compile native code, or rerun paper experiments.")

if __name__ == "__main__":
    main()
