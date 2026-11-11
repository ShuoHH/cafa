# Exact-input primary-method driver

`examples/run_paper_verified.py` runs the primary CACF method on authenticated
research inputs with their frozen primary folds and complete DP pricing. It
does not download data, train models, or run the separate baseline,
segmentation-comparison, and selector/ablation experiments. Supplying a runner
is not evidence that all original inputs have been freshly replayed with it.

Package validation freshly reran BM using this driver and the rebuilt strict-eight
native wrappers: 361,776 bits, six final segments, all 45 DP edges priced, and an
identical stored-file SHA-256 to the preserved result. Independently decoded bytes
were downloaded and compared to the local original input. This validates one
research series, not all 13; see [the BM record](../evidence/paper_BM_reproduction.json).

## Inputs and a read-only dry-run

Use Python 3.12 and, for actual execution, `requirements-method.txt` plus both
native adapters described in [NATIVE_SETUP.md](NATIVE_SETUP.md). Keep raw files,
virtual environments, native builds, and outputs outside the repository.
Research data are not bundled; obtain them lawfully and reproduce the exact
hashes, counts, and headerless little-endian binary64 bytes in
[experiment_inputs.json](experiment_inputs.json).

Choose exactly one input mode:

- `--input-dir /absolute/directory`: selected files must use the exact
  `canonical_filename` values from the input manifest; no filename guessing.
- `--input-map /absolute/path/input_map.json`: a flat JSON object maps canonical
  series names to exact file paths. Relative paths resolve beside the JSON
  file, not beside the repository. Each selected series must have an entry.

For example, this minimal map supports an explicit BT-only run:

```json
{
  "BT": "/absolute/path/to/BT.bin"
}
```

The default cohort is **all 13 evaluated series**, including BIDMC-03. To select
a subset, repeat `--series` with canonical names; subsets are explicit, not
inferred from missing input files or missing map entries. The 12-series main
cohort is a reporting subset, not the execution default.

```sh
../cacf-venv/bin/python -B examples/run_paper_verified.py \
  --input-dir /absolute/path/to/canonical-input-files --dry-run

../cacf-venv/bin/python -B examples/run_paper_verified.py \
  --input-map /absolute/path/input_map.json --series BT --dry-run
```

Dry-run validates selected raw-file SHA/size/count, manifest consistency, model
routing, model SHA/size, and metadata SHA, then prints the plan. It does not
write outputs, deserialize joblib, invoke codecs, or need `--out`/native paths.
Passing all 13 identity checks is **not** a completed 13-series compression run.

## Execute a fresh primary run

Use a new output directory under an existing parent. Native executables and raw
files should not be inside that output tree; inputs and dependencies remain
protected from overwrite. Also keep outputs outside the parent directories of
protected raw files and native tools: separate sibling input, binary, and output
directories avoid the overlap guards.

```sh
../cacf-venv/bin/python -B examples/run_paper_verified.py \
  --input-dir /absolute/path/to/canonical-input-files \
  --exporter /absolute/path/to/FrozenPlanFile \
  --fixed-btr-exe /absolute/path/to/BtrBlocksFixed \
  --out /absolute/new/path/for/primary-results
```

The driver automatically selects the primary `compact20` model and metadata
from [models/release_manifest.json](../models/release_manifest.json) and passes
the exact dataset/series/SHA identity to the adaptive worker. Four external
sources reuse the preserved May-held-out fold, as recorded in the manifest.
The run does not retrain, use historical geometry or price caches, automatically
retry failures, or select a best repetition. Selected series execute serially.

## Primary protocol versus the synthetic demo

| Setting | Exact-input primary driver |
| --- | --- |
| Base actions in boundary and DP | The same eight retained actions. |
| ClaSS history / subsequence / jump | 4,096 / 32 / 5. |
| Detector / per-native-command timeout | 1,800 / 300 seconds. |
| Boundary processed-value / trial caps | 128N / 8,192, unchanged. |
| DP pricing | Full frozen Top-2 graph, automatically funded as in the preserved primary protocol. |
| Model | Authenticated source-specific frozen `compact20` route. |

The adaptive synthetic demo's default DP caps are 128N / 8,192. They do not
implicitly reproduce the full primary graph for long original inputs. The
primary driver explicitly requests `--paper-full-pricing`, which uses the
method's automatic complete-graph budget rather than the capped demo defaults;
the boundary-stage caps do not change. This mode cannot be combined with the
capped DP options. An incomplete full-DP run is a failure, not a valid substitute
for the reference result. Complete pricing can consume substantially more time,
disk, and memory; admission guards are not peak-resource guarantees.

## Outputs and discrepancy handling

The output includes `paper_run_plan.json`, per-series commands/stdout/stderr,
adaptive stage evidence and saved/decoded files, per-series `*.comparison.json`,
and `paper_run_summary.json`. Actual bytes are independently checked; primary
complete-file bits, final segment count, and compressed-file SHA are compared
with [evidence/primary13_summary.json](../evidence/primary13_summary.json).

- Exit **0**: all selected runs verified and all three reference fields match.
- Exit **1**: a worker or proof validation failed; execution stops and preserves
  the completed/failed evidence without automatic retries.
- Exit **2**: selected runs passed proof validation, but one or more reference
  sizes, final segment counts, or stored hashes differ. Differences are reported
  rather than relabelled as an exact reproduction.

Invalid arguments or input/model authentication errors also stop the command
before a successful run can be claimed; inspect the diagnostic message.
`all13_verified` is true only after all 13 runs actually finish verification,
not after a dry-run or a smaller selected subset. A valid decode alone does not
imply byte-identical reference compression or comparable timing.

The publication checks include all-input dry-run validation and synthetic native
tests. They must not be described as a completed fresh all-13 primary replay,
or as reproduction of the remaining paper experiments. See
[REPRODUCIBILITY.md](REPRODUCIBILITY.md) and the supplied evidence summaries for
the exact validation scope.
