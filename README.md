# CACF

CACF is an encoding-aware framework for lossless compression of IEEE 754
binary64 time series. Its two modules are **Encoding-aware adaptive
segmentation (EAA segmentation)** and **Adaptive encoding selection (AE
selection)**. EAA segmentation proposes and refines variable-length segments
using encoding-related features and measured storage costs. AE selection
recommends two candidate encoding schemes for each segment; verified native
trial compression determines the stored representation.

This publication candidate contains the CACF method code, native integration
adapters, unchanged frozen selector weights, two synthetic examples, and compact
experiment summaries. It is **not a one-command reproduction of the paper**.
Original research-series bytes, training records, complete experiment outputs,
upstream codec source trees, and native executables are not bundled.

## Exactly eight base encoding schemes

The public CACF workflow uses the same eight base actions for measured boundary
refinement and final DP pricing. Its guards reject excluded actions rather than
pricing them and discarding them only at the end.

| Paper name | Internal action | Stored frame ID |
| --- | --- | ---: |
| Gorilla | `gorilla` | 16 |
| Chimp | `chimp` | 17 |
| Chimp128 | `chimp128` | 18 |
| TSXor | `tsxor` | 19 |
| RLE+TSXor | `rle+tsxor` | 23 |
| ALP-T | `alp` | 24 |
| Dict-BP | `btr_dict_bp` | 33 |
| Decimal-fixed | `btr_decimal_fixed` | 36 |

ALP-tail labels such as `alp+tail:chimp128` describe an ALP block plus a tail
encoded with another member of this pool; they are not additional base
encoding schemes. Frozen classifiers retain their historical class labels.
An unavailable action is not silently replaced with an out-of-pool compressor.
See [the code map](docs/CODE_MAP.md) for the enforcement points.

## Run the examples

Use Python 3.12. Run from the repository root; keep the virtual environment
outside the publication tree. On Linux:

```sh
python3.12 -m venv ../cacf-venv
../cacf-venv/bin/python -m pip install -r requirements-demo.txt
../cacf-venv/bin/python -B examples/run_demo.py
../cacf-venv/bin/python -B -S -m unittest discover -s tests -p test_codec_pool.py -v
../cacf-venv/bin/python -B scripts/verify_release.py
```

On Windows PowerShell, confirm Python 3.12 with `python --version`, create the
environment with `python -m venv ..\cacf-venv`, and replace
`../cacf-venv/bin/python` with `..\cacf-venv\Scripts\python.exe` above.

The entrypoints have different scopes:

| Entrypoint | What it does | What it does not establish |
| --- | --- | --- |
| `examples/run_demo.py` | Computes features, proposes candidate boundaries, and runs frozen Top-2 inference on 1,024 synthetic values. | No native compression, stored-size result, ClaSS detection, boundary repair, or DP. |
| `examples/run_verified_compression.py` | Runs Top-2 native trials on one whole-series region, saves `final.f64pool`, independently decodes it, and checks exact input bytes. | No adaptive segmentation or DP; not a paper benchmark. |
| `examples/run_adaptive_verified.py` | Runs fresh XOR-change ClaSS detection, feature-based candidate generation, measured boundary repair, frozen Top-2 DP, saved-file encoding, and independent decoding on 8,192 synthetic values. | Not a replay of the paper's original inputs, experiment plans, or timings. |
| `examples/run_paper_verified.py` | Authenticates selected original inputs and frozen folds, runs the primary method with full DP pricing, and compares sizes, segments, and stored hashes with preserved results. | Does not supply raw data, reproduce all comparison/selector experiments, or establish a fresh all-13 replay merely by being included. |

The native-compression entrypoints require `requirements-method.txt` and
separately built adapters. See [native setup](docs/NATIVE_SETUP.md). With those
adapters already available, run using a new output directory:

```sh
../cacf-venv/bin/python -B examples/run_verified_compression.py \
  --exporter /absolute/path/to/FrozenPlanFile \
  --fixed-btr-exe /absolute/path/to/BtrBlocksFixed \
  --out /absolute/new/path/for/whole-series-output

../cacf-venv/bin/python -B examples/run_adaptive_verified.py \
  --exporter /absolute/path/to/FrozenPlanFile \
  --fixed-btr-exe /absolute/path/to/BtrBlocksFixed \
  --out /absolute/new/path/for/adaptive-output
```

After installing `requirements-method.txt`, run the full Python suite with
`../cacf-venv/bin/python -B -m unittest discover -s tests -v`. The lightweight pool
tests above do not need the scientific or native dependencies.

Keep native builds and run outputs outside the checkout too. The release
verifier intentionally rejects virtual environments, compiled artifacts, and
Python bytecode inside a publication tree; `-B` avoids creating bytecode.
During release preparation, use `scripts/verify_release.py --check-tree` to
check an unsealed tree. The default mode additionally compares every published
file with `release_inventory.json`; edits after sealing are expected to fail
that comparison. The inventory detects accidental changes, not an adversary
replacing both files and inventory.

The adaptive runner enforces at least 2 GiB free output disk and 512 MiB
cgroup headroom where available. These are admission guards, not peak-memory
guarantees. It does not purge caches or retry automatically. A compatible
completed boundary stage can be authenticated with `--resume-boundary` and
followed by **fresh** DP pricing in another new output directory; historical
14-pool geometry or cost ledgers must not be reused for a strict-eight run.

Fresh public-Python synthetic checks are recorded in
[evidence/native_smoke_summary.json](evidence/native_smoke_summary.json), using
authenticated existing binaries. The separately rebuilt guarded native wrappers
passed self-tests, rejected excluded actions/IDs, and produced the same two
synthetic stored-file hashes; see
[evidence/native_guards_validation_v2.json](evidence/native_guards_validation_v2.json).
Both saved-file outputs independently decoded to the exact input bytes. These
existing-host tests are not a clean-host dependency build or research benchmark.

## Run the primary method on exact research inputs

With lawful access to the exact source files and native dependencies, first
validate input identities and frozen model routing without executing codecs:

```sh
../cacf-venv/bin/python -B examples/run_paper_verified.py \
  --input-dir /absolute/path/to/canonical-input-files --dry-run
```

Then use a fresh output path under an existing parent:

```sh
../cacf-venv/bin/python -B examples/run_paper_verified.py \
  --input-dir /absolute/path/to/canonical-input-files \
  --exporter /absolute/path/to/FrozenPlanFile \
  --fixed-btr-exe /absolute/path/to/BtrBlocksFixed \
  --out /absolute/new/path/for/primary-results
```

The default is all 13 evaluated series, including BIDMC-03. Repeat `--series`
to request an explicit subset, or use `--input-map` for files stored elsewhere.
This driver preserves the primary detector/boundary settings and requests the
full frozen Top-2 DP graph. The synthetic adaptive demo's default DP cap is not
the paper protocol. See [docs/PAPER_PRIMARY.md](docs/PAPER_PRIMARY.md) for exact
inputs, settings, outputs, and mismatch exit codes. Availability of this runner
does not mean that all 13 original inputs have been rerun through it during
publication preparation, nor that all paper experiments are reproduced.

## Frozen models and research inputs

[models/release_manifest.json](models/release_manifest.json) records 36 unchanged
models: nine source-specific folds, each with `base10`, `decimal15`,
`residue15`, and `compact20` feature variants. Its `input_model_routing` maps
all 13 evaluated series to the correct primary `compact20` model and metadata;
four external series reuse the historical May-held-out fold. The `ae4` section
records the four-series selector-study variants. **Eight schemes does not mean
eight model files.** No model was retrained for this release or the strict-eight
rerun.

The default `models/class_balanced_compact20.joblib` remains the May-held-out
demo model, not a universal experimental fold. Follow the manifest when
selecting a model for a research series. The native examples accept paired
`--model /path/to/model.joblib --model-metadata /path/to/model.json` options
for a 20-feature model. For an original research input, also supply its paired
`--dataset` and `--series` identities from the input manifest; a different alias
does not authenticate a held-out source. Model SHA-256 is checked before
`joblib.load`; never load untrusted joblib artifacts.

[docs/experiment_inputs.json](docs/experiment_inputs.json) records original
input hashes, value counts, binary format, and model routing. It is an identity
manifest, **not a download or preprocessing script**. Original input bytes are
withheld pending redistribution-term verification; only the two generated
synthetic fixtures are distributed here.

## Experiment reporting scope

The strict-eight primary rerun evaluated **13 series**. The manuscript's main
aggregate reports a **post-hoc 12-series subset**; BIDMC-03 remains a separately
reported negative case. Selector experiments evaluated four series; plots of
the three improved series are a **post-hoc positive subset**, with BT and the
full four-series aggregates retained separately. These subsets must not be
presented as the complete, pre-specified evaluation.

Compact summaries are in [evidence/](evidence/README.md), and the cohort and
aggregation rules are in [docs/EXPERIMENT_SCOPE.md](docs/EXPERIMENT_SCOPE.md).
The complete run archives and individual decode certificates are preserved
outside this repository; summaries are not substitutes for those artifacts.
The older synthetic pilot used a 14-action boundary stage and does **not**
validate the current all-stage strict-eight path. See
[reproduction scope](docs/REPRODUCIBILITY.md).

## Source layout

| Path | Content |
| --- | --- |
| `tools/` | CACF method modules, strict-eight pool definitions and guards. |
| `harness/` | CACF framing, RLE, and fixed-BtrBlocks adapters; external dependencies are required. |
| `models/` | Frozen model variants, metadata, input-to-fold routing, and original metadata provenance. |
| `examples/` | Two synthetic fixtures, three demo entrypoints, and an exact-input primary-method driver. |
| `evidence/` | Compact primary and selector summaries, with archive identities. |
| `docs/` | Method map, reproduction limits, experiment scope, and publication checks. |
| `third_party_licenses/` | Notices for the ClaSPy-derived integrations and BtrBlocks-derived constant table. |

## Publication and third-party status

Native codec dependencies are obtained separately. The two ClaSPy-derived
integrations and fixed BtrBlocks adapter are identified in
[THIRD_PARTY_NOTICES.md](THIRD_PARTY_NOTICES.md); a matching hash establishes
byte identity, not authorship or permission.

The authors have not yet selected a license for CACF-authored code; there is no
root `LICENSE`. Exact external native dependencies and a clean-host build still
require validation. This candidate has not been pushed publicly. See
[docs/PUBLICATION_CHECKLIST.md](docs/PUBLICATION_CHECKLIST.md) for remaining
author decisions and validation limits.
