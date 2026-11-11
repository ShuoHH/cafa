# Main-method code map

The retained method uses flat imports from `tools/`. The two paper modules are
Encoding-aware adaptive segmentation (EAA segmentation) and Adaptive encoding
selection (AE selection).

```text
binary64 input -> shared raw-feature bank
  |
  +-> EAA segmentation
  |     adjacent-XOR-change ClaSS -> initial boundaries
  |     encoding-feature candidates -> measured split / merge / shift
  |     (only the eight retained base actions are priced)
  |
  +-> AE selection on candidate DP regions
        frozen classifier -> Top-2 recommendations -> verified native trials
        -> storage-cost DP coarsening -> framed file -> independent decode
```

AE selection is consulted when DP prices candidate regions, not only after the
DP has chosen its final boundaries. Boundary repair can test all eight base
actions, whereas the frozen Top-2 policy narrows final DP trials. Both stages
use the same feature bank and the same strict-eight base pool. No previous
14-action boundary geometry or prices are an input to a fresh strict-eight run.

| Stage | Main CACF-side files |
| --- | --- |
| Strict pool and persisted-action checks | `codec_pool.py`, `strict8_boundary_workflow.py`, `flat_pool_backend.py`, `flat_pool_format.py`, the two native wrapper dispatchers |
| Shared raw features | `shared_compression_features.py`, `profile_codec_features.py`, `profile_residue_features.py` |
| Bit-change detector | `paper_xor_change_class.py`, `class_xor_change.py`, `class_bit_hamming.py` |
| Candidate boundaries | `compression_candidates.py`, `paper_xor_change_class.py`, `examples/run_adaptive_verified.py` |
| Measured local edits | `compression_boundary_repair.py`, `compression_gain_proxy.py`, `flat_pool_backend.py` |
| DP coarsening and native trial pricing | `paper_dp_merge_native.py`, `partition_dp_coarsening.py` |
| Frozen selector and feature policies | `shared_feature_policy.py`, `rf_profiles.py` |
| Serialization and independent decoding | `paper_mainline_native.py`, `flat_pool_format.py`, `flat_pool_backend.py`, `harness/framed/` |

The eight actions are `gorilla`, `chimp`, `chimp128`, `tsxor`, `rle+tsxor`,
`alp`, `btr_dict_bp`, and `btr_decimal_fixed`. `alp+tail:<stream-action>` is a
composition using one of the five retained stream actions, not a ninth base
scheme. Historical classifier class names are not rewritten or retrained.
Validation must examine actual stored frame IDs and native trial ledgers,
not merely the selector's reported names.
The native wrapper entrypoints also reject excluded encode actions and decode
IDs. Legacy implementation helpers or numeric constants are not an additional
reachable candidate pool in these public wrapper dispatchers.

## Entrypoint boundaries

`examples/run_demo.py` covers features, proposed encoding-feature boundaries,
and frozen inference only. It does not execute the full flow above.

`examples/run_verified_compression.py` takes a separate whole-series branch:
frozen Top-2 recommendations -> native trials -> persisted file -> independent
decode -> exact input-byte comparison. Feature proposals do not become segment
boundaries in that example.

`examples/run_adaptive_verified.py` exercises both modules on the 8,192-value
synthetic fixture. The fresh path performs boundary detection and measured
repair before DP. Its guarded resume authenticates a compatible stored boundary
result and independently decodes it before fresh DP pricing; it does not make
old mixed-pool evidence a strict-eight result.

`examples/run_paper_verified.py` validates exact original input hashes and model
routes, then invokes the adaptive path with primary-protocol settings and
`--paper-full-pricing`. Each selected series starts with fresh geometry and
prices; results are compared with preserved primary bits, final segment count,
and container SHA. The synthetic demo retains capped DP defaults; using it with
an original input does not implicitly select the full-graph paper protocol.
See [PAPER_PRIMARY.md](PAPER_PRIMARY.md).

The original baseline/segmentation-comparison/selector experiment suite is not
repackaged as a one-command full-paper runner. Availability of a primary driver
does not prove that all original inputs have freshly reproduced their results.
