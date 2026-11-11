# Source and artifact provenance

## Initial source selection

The Python source in `tools/` was selected from the authenticated
`paper_final13_19AEhfbt` server code bundle. Initial selection followed an
84-module local import closure; publication cleanup removed 43 modules outside
the retained method examples and native-build paths. This is a method-focused
tree, not the complete historical project or baseline experiment suite.

The server bundle audit archive had SHA-256
`1deb88f93fd83c65f6de627651b1d12c7e2dc840ab7ec4c6bcf5bedcb894a220`.
The separate original-project source audit archive had SHA-256
`3050785216e152f48c176714bb4a4619b4f9d381bbfcf0fa91ead2cb12dcab7a`.
Those audit archives are not distributed here.

Three copied historical files initially received attribution-only comments
without executable-logic changes: `tools/class_bit_hamming.py`,
`tools/class_xor_change.py`, and
`harness/external_baselines/btrblocks_fixed/fixed_btr.cpp`.
The original server SHA-256 of `fixed_btr.cpp` was
`0274c35d80cb1b758c1e313941e6c570536e6ee505402fb8482e972f49b75b95`.
This is a historical source identity, not an assertion that every current
packaging file remains byte-identical to that source.

## Strict-eight publication corrections

The earlier draft could price 14 actions during boundary refinement even
though final DP used the retained eight. The corrected release uses explicit
strict-eight definitions and guards across boundary pricing, DP, compatible
resume evidence, native trial ledgers, and actual persisted frames. It does
not reuse old 14-pool geometry or cost caches as strict-eight evidence.

The two CACF-side native wrapper dispatchers were also narrowed: excluded
encode actions and decode IDs are rejected, with original retained IDs and
encoding/serialization arithmetic preserved. These changes are distinct from
the earlier attribution-only source edits; no upstream codec implementation
was rewritten. Rebuild/self-test evidence is recorded separately from the
earlier existing-binary Python smoke tests.

Portable build wrappers accept explicit external dependency paths. The public
whole-series and adaptive entrypoints are packaging integrations around the
retained method. Their existence is not evidence that all historical paper
experiments have been replayed from this release. The additional exact-input
`run_paper_verified.py` driver authenticates input/fold identities and requests
the primary protocol's full DP graph; it is not the original full experiment
controller or evidence of a completed all-13 replay by itself.

## Models, inputs, and results

The 36 frozen models are copied unchanged from preserved authenticated
archives; no release-time training or refitting was performed. Each model's
source identity, hash, feature variant, and original metadata location are in
[models/release_manifest.json](../models/release_manifest.json). Original
metadata is retained under `models/provenance/`; extracted runtime metadata
uses local model paths. The existing default May-fold files are kept for
compatibility rather than substituted for all source-specific models.

The original six-row May-fold metadata SHA-256 was
`870ff0384534b2d4d8d496d6725bac4393b445629a9fc6454b95572b0955ff36`.
The default model SHA-256 remains
`3c2bbc7b95c79d77cdbc514d578ffbeab08efdb02fe20f3bdf51faeba87155b0`.

Research input identities are listed in
[experiment_inputs.json](experiment_inputs.json); their raw bytes are not
distributed. Both included inputs were generated with
`examples/make_demo_data.py`, not clipped from a research dataset:

| Fixture | Values | SHA-256 |
| --- | ---: | --- |
| `synthetic_1024.f64le` | 1,024 | `c7d8eaf15e384f31378580ced134a3edfc953515577216edcc193e6722812f81` |
| `synthetic_8192.f64le` | 8,192 | `c89a123d256bf4419562733ed4ae25bcc99b98a6acaa89a13421cecabd1dc8d2` |

Compact experiment evidence and archive identities are described in
[evidence/README.md](../evidence/README.md). Complete experiment archives,
decoded output bytes, and per-worker certificates remain outside the Git
candidate. Historical and corrected-run evidence must retain their distinct
scope and dates.

Provenance and SHA checks establish byte identities, not copyright ownership,
redistribution permission, or a newly validated clean-host environment. Authors
must review those separately before public release.
