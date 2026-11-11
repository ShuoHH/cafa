# Native setup for the verified examples

This repository contains CACF's framing and fixed-primitive adapters, not
upstream codec source trees or prebuilt executables. A Linux build requires
CMake 3.22+, a C++23 compiler (the historical environment used GCC 13.4),
Python 3.12, and external dependencies:

- ALP source containing `include/alp.hpp` and an already built `libALP.so`;
- Gorilla, Chimp, Chimp128, TSXor, and `lib` header directories from the
  external NeaTS benchmark source tree;
- BtrBlocks source and its transitive dependencies. The fixed adapter uses a
  checked private source copy, not the unmodified full BtrBlocks baseline.

Exact external revisions and a clean-host build are **not yet fully locked and
validated**. Existing-server smoke tests do not establish that a fresh clone
can build these dependencies. No upstream files are downloaded automatically.
Review their licenses and build instructions before obtaining them.

With dependencies available, replace each placeholder with an absolute path.
Keep environments, native builds, and outputs outside the publication checkout;
the release verifier rejects generated artifacts within it:

```sh
python3.12 -m venv ../cacf-venv
../cacf-venv/bin/python -m pip install -r requirements-method.txt
export PYTHONDONTWRITEBYTECODE=1
export PYTHON_EXE="$PWD/../cacf-venv/bin/python"

export ALP_ROOT=/absolute/path/to/ALP-source
export ALP_LIBRARY=/absolute/path/to/libALP.so
export STREAMING_HEADERS_ROOT=/absolute/path/to/NeaTS-benchmark
export FRAMED_PLAN_BUILD_DIR="$PWD/../cacf-native-build/frozen-plan"
bash harness/build_framed_plan.sh
"$FRAMED_PLAN_BUILD_DIR/FrozenPlanFile" --self-test

export BTRBLOCKS_ROOT=/absolute/path/to/BtrBlocks-source
export BTRBLOCKS_FIXED_BUILD_DIR="$PWD/../cacf-native-build/btrblocks-fixed"
bash harness/build_btrblocks_fixed.sh

mkdir -p ../cacf-example-output

../cacf-venv/bin/python -B examples/run_verified_compression.py \
  --exporter "$FRAMED_PLAN_BUILD_DIR/FrozenPlanFile" \
  --fixed-btr-exe "$BTRBLOCKS_FIXED_BUILD_DIR/native/BtrBlocksFixed" \
  --out "$PWD/../cacf-example-output/verified-demo-1"

../cacf-venv/bin/python -B examples/run_adaptive_verified.py \
  --exporter "$FRAMED_PLAN_BUILD_DIR/FrozenPlanFile" \
  --fixed-btr-exe "$BTRBLOCKS_FIXED_BUILD_DIR/native/BtrBlocksFixed" \
  --out "$PWD/../cacf-example-output/adaptive-demo-1"
```

`--out` must be new. The whole-series example defaults to 1,024 finite synthetic
values; custom `--input` files must contain 1–102,400 headerless little-endian
binary64 values. The example writes a complete compressed file, independently
decoded bytes, and `example_report.json` with input/output hashes and complete
stored-file bits. It does not run adaptive detection or DP.

The adaptive example defaults to the 8,192-value fixture and runs both stages.
Keep output paths outside protected input/native parent directories, not merely
away from the exact file names; the examples above use separate sibling trees.
It requires at least 2 GiB free output disk and 512 MiB cgroup headroom when a
cgroup limit is present. These admission checks are not peak-memory guarantees.
A refusal does not trigger cache purges or automatic retries. Both measured
boundary refinement and final DP are restricted to the same eight base actions
listed in the [README](../README.md#exactly-eight-base-encoding-schemes).
The wrapper self-tests check retained-action round trips and excluded-action/ID
rejection. Rebuilding the wrappers against existing server libraries has been
checked in [native_guards_validation_v2.json](../evidence/native_guards_validation_v2.json);
this does not validate a clean-host build of all upstream dependencies.

For a different primary fold, both native entrypoints accept paired
`--model /absolute/path/to/compact20.joblib` and
`--model-metadata /absolute/path/to/compact20.json` options. They check a
20-feature schema and the model hash before loading. Select the pair from
[models/release_manifest.json](../models/release_manifest.json), not by assuming
that the default May-fold demo model applies to every series. The 10-/15-feature
ablation weights are preserved artifacts, not accepted as 20-feature models by
these example entrypoints.

For original research inputs, supply paired `--dataset` and `--series` values
from [experiment_inputs.json](experiment_inputs.json) as well. Identity checks
include the actual input SHA; a same-byte source renamed to a synthetic alias
is not treated as a genuinely external source. For exact research-input primary
runs, prefer [run_paper_verified.py](../examples/run_paper_verified.py): it
authenticates this routing automatically and selects full DP pricing. The
synthetic adaptive example retains its capped DP defaults unless explicitly
requested otherwise; see [PAPER_PRIMARY.md](PAPER_PRIMARY.md).

The full Python tests require `requirements-method.txt`:

```sh
../cacf-venv/bin/python -B -m unittest discover -s tests -v
```

## Resume without reusing old prices

If a compatible strict-eight adaptive run completed its boundary stage but
stopped before DP, use a different, fresh output directory:

```sh
../cacf-venv/bin/python -B examples/run_adaptive_verified.py \
  --exporter "$FRAMED_PLAN_BUILD_DIR/FrozenPlanFile" \
  --fixed-btr-exe "$BTRBLOCKS_FIXED_BUILD_DIR/native/BtrBlocksFixed" \
  --resume-boundary "$PWD/../cacf-example-output/adaptive-demo-1/boundary" \
  --out "$PWD/../cacf-example-output/adaptive-demo-resumed-1"
```

Resume authenticates saved input, geometry, trials, stored output, and decode
evidence before independently decoding the saved repaired file. It does not
edit the earlier result or import its prices into DP. Evidence from an older
14-action boundary run is not a valid substitute for strict-eight boundary
evidence. Historical pilots and their limitations are retained in
[REPRODUCIBILITY.md](REPRODUCIBILITY.md).

## External snapshot identities

The historical checked BtrBlocks snapshot had:

| Upstream file | SHA-256 |
| --- | --- |
| `SchemePicker.hpp` | `683fc2923b0fca6f81a41c3a6efac52831fbd2e560a992da29b92e77824c86c9` |
| `Pseudodecimal.cpp` | `708e45c8eb2da0f7d40857b3135bae6b470a5bd47f646252a6f44dd42a28bf70` |

These files are not shipped. File hashes identify the checked bytes, not a
complete upstream commit or a transitive-dependency lock. The Python version
pins in `requirements-method.txt` likewise do not lock the native environment.
