# Reproduction scope

## Current release versus recorded experiments

This publication candidate includes the retained CACF method path, three
synthetic entrypoints, and an exact-input primary-method driver. It does not
include a complete full-paper controller, baseline/selector experiment suite,
original research input bytes, full trial archives, or upstream dependencies.
A feature-demo pass, a native round trip, and reproduction of a paper table
are different claims.

| Level | Included execution | Required outside the repository |
| --- | --- | --- |
| Feature-only | `run_demo.py`: feature extraction, boundary proposals, frozen inference on 1,024 synthetic values. | Python dependencies. |
| Whole-series stored-file example | `run_verified_compression.py`: frozen Top-2 trials, persisted file, independent decode, byte equality. | Method Python dependencies and both native adapters. |
| Adaptive synthetic example | `run_adaptive_verified.py`: fresh boundary detection and repair, Top-2 DP, persisted file, independent decode. | Method dependencies, native adapters, and sufficient resources. |
| Exact-input primary method | `run_paper_verified.py`: input/fold authentication, fresh primary execution with full DP pricing, comparison with preserved primary results. | Exact source bytes, method/native dependencies, resources; a driver is not itself evidence of a completed all-13 replay. |
| Full paper experiments | Compact result summaries and frozen variants are supplied, not a complete comparison/selector-study controller. | Exact inputs, full provenance, native environment, and the remaining experiment runners. |

The 1,024-value fixture is shorter than the detector's historical 4,096-value
warmup, so the whole-series example deliberately uses one region. The separate
8,192-value fixture can exercise adaptive detection. Both are generated data,
not excerpts from the research datasets.

The exact-input [primary driver](PAPER_PRIMARY.md) uses full-graph DP pricing,
matching the preserved primary protocol. The synthetic adaptive demo defaults
to caps of 128N processed values and 8,192 trials per native stage; its DP caps
can be insufficient for complete pricing on longer original inputs. The paper
driver requests `--paper-full-pricing` without altering the boundary caps. It
does not silently report a budget-limited DP result as fully priced.

## Fresh publication checks

[native_smoke_summary.json](../evidence/native_smoke_summary.json) records
fresh execution of the public Python paths with preserved, authenticated native
binaries. The whole-series fixture stored 31,096 bits; the adaptive fixture
stored 285,144 bits in four final segments after all 15 DP edges were priced.
Both independently decoded files were downloaded and compared byte for byte
with the included fixtures. Matching older pilot sizes does not mean old
geometry or prices were reused.

[native_guards_validation_v2.json](../evidence/native_guards_validation_v2.json)
separately records rebuilding the two guarded native wrappers on the existing
server, their retained-action round-trip and excluded-action/ID self-tests,
and matching synthetic stored-file hashes with those rebuilt binaries. This
validates the wrapper guards without claiming a clean-host build of external
dependencies or a fresh all-13 original-input replay. The exact-input driver's
all-13 dry-run validates identities/routing only; it does not compress or load
the models.

## Strict-eight contract

The eight base actions in [the README](../README.md#exactly-eight-base-encoding-schemes)
apply to measured boundary refinement **and** DP. ALP-tail classifier labels
are retained compositions within the eight-base pool. Excluded actions must
not appear in native trial ledgers or stored frames. A fresh strict-eight run
must compute its own boundary geometry and prices, not inherit them from a
14-action run.

The November 2026 strict-eight primary rerun evaluated all 13 input series with
unchanged frozen weights. Its audit checked per-series independent decoding,
actual persisted frame IDs, trial pools, stored-file hashes/sizes, and complete
pricing. The compact [primary summary](../evidence/primary13_summary.json)
records those outcomes; full certificates and trial artifacts are preserved in
external archives identified in [evidence/README.md](../evidence/README.md).
That archived experiment is distinct from executing the newly packaged
entrypoints on a fresh machine. Its resource/cache interventions also preclude
claiming untouched paper-runtime measurements.

See [EXPERIMENT_SCOPE.md](EXPERIMENT_SCOPE.md) for the 13-evaluated/12-main
cohorts and the complete versus plotted selector-study populations.

## Frozen models

[models/release_manifest.json](../models/release_manifest.json) supplies nine
primary `compact20` folds and the three accompanying feature variants per
fold: 36 distinct frozen artifacts. `input_model_routing` records the primary
model for each evaluated input; `ae4` records selector-study variants. These
are copied historical weights, not retrained strict-eight models. Four
external sources reuse the historical May fold, as explicitly recorded.

The compatibility default is
`models/class_balanced_compact20.joblib`, SHA-256
`3c2bbc7b95c79d77cdbc514d578ffbeab08efdb02fe20f3bdf51faeba87155b0`.
It is the May-held-out fold used by the synthetic demos, not the correct model
for every paper series. Metadata records feature order, training-source
identities, original hashes, and model routing, but contains no original
training values or labels. SHA checks occur before joblib deserialization;
never substitute an untrusted model artifact.

## Data and native prerequisites

[experiment_inputs.json](experiment_inputs.json) lists all 13 original input
hashes, value counts, format, original provenance paths, and selected models.
It is not an acquisition/preprocessing pipeline. Research input bytes are not
distributed until their redistribution terms are verified. Obtaining a public
source dataset alone does not establish that preprocessing reproduced these
exact bytes.

The native sources in `harness/` are CACF integration layers, not complete
upstream libraries. [NATIVE_SETUP.md](NATIVE_SETUP.md) describes the external
ALP, streaming-header, and BtrBlocks requirements. ClaSPy 0.2.8 is required for
the detector integrations. Exact native revisions and a clean-host build remain
unvalidated; Python package pins are not a native dependency lock. The fixed
Dict-BP/Decimal-fixed adapter is not the full BtrBlocks baseline.

## Historical synthetic pilots — not strict-eight validation

These records are retained to avoid relabelling old experiments as new ones:

- On 2026-11-08, a whole-series pilot used existing server executables and
  stored 31,096 bits for the 1,024-value synthetic input. Independently decoded
  bytes matched input SHA-256
  `c7d8eaf15e384f31378580ced134a3edfc953515577216edcc193e6722812f81`.
  This establishes that narrow historical codec path, not adaptive CACF.
- A later 8,192-value pilot used **14 boundary actions but eight final DP base
  actions**. A 2 GiB-cgroup resource guard stopped its first run after boundary
  repair. It changed four initial segments into five repaired segments,
  reducing its full-pool file from 259,360 to 258,536 bits. A separate guarded
  resume freshly priced 15/15 Top-2 DP edges, ended with four segments, and
  stored 285,144 bits; independent decoding matched the synthetic input.
  Different candidate policies mean these two file sizes are not a monotonic
  comparison. Its earlier boundary manifest lacked executable hashes, so
  redecoding authenticates saved bytes but not the prior executable identity.

The older adaptive pilot built `FrozenPlanFile` but reused an existing
`BtrBlocksFixed`; it was neither a clean-host build nor a single uninterrupted
run. Most importantly, it does **not** validate the corrected all-stage-eight
entrypoint. Its geometry and prices must not be reused as strict-eight evidence.

The historical 208-slot plan also did not complete as one run; follow-up
comparison, segmentation, and selector studies have separate provenance.
Neither synthetic pilot reproduces paper compression-size or timing figures.
