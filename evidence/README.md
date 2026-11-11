# Preserved experiment summaries

The primary/selector summaries derive from the authenticated November 11
strict-eight results. Separate package-validation records describe the fresh
synthetic and single-research-series checks performed during preparation.

- `primary13_summary.json`: all 13 evaluated series, including the separately
  reported BIDMC-03 result; the main 12-series subset was selected post hoc.
- `selector_four_series_summary.json`: all four selector-study series and the
  separately identified three-positive-series display subgroup. BT is retained.
- `native_smoke_summary.json`: freshly executed public Python sources with the
  preserved Linux native binaries. The whole-series 1,024-value fixture stored
  31,096 bits; the 8,192-value adaptive fixture stored 285,144 bits in four final
  segments, with all 15 DP edges priced. Both independently decoded outputs were
  downloaded and compared byte-for-byte with the local synthetic fixtures.
  This is not a clean-host dependency build or a research-series benchmark.
- `native_guards_validation_v2.json`: current native wrappers were rebuilt with
  existing external headers/libraries. Their retained-codec round trips and
  excluded-action/ID rejection self-tests passed; both synthetic stored files
  remained byte-identical to the authenticated older binaries' outputs.
- `paper_BM_reproduction.json`: the packaged original-input driver freshly
  reproduced BM at 361,776 bits, six semantic segments, and the same stored-file
  SHA-256 as the preserved result. All 45 DP edges were priced; downloaded
  decoded bytes matched the local preserved input. Only this research series
  was rerun for package validation, not all 13 or all comparison studies.

Per-file hashes bind these summaries to the previously verified reports and
stored artifacts. The full native trial logs, compressed payloads and independent
decode certificates are preserved in the authors' private backup archives, **not
bundled here**. This repository alone does not authenticate missing payload bytes.

Preservation archive SHA-256 fingerprints:

| Archive | SHA-256 |
| --- | --- |
| Primary completed 11 | `ec8a7bd9ed4a0c4c45f4d8e3f7f261f7cc686dcd4c6f8576fc588f5c23308da0` |
| Primary final 2 | `8414813a7f4df17a764a1a05b3b9c6c82aea6dc78cd4ead0d6b3fff08a0d22fa` |
| Selector studies | `3c864e89760081ae11a3ac2d1b63154df434c48a7213fefd2916b377377ac5e2` |

Cache/resource interventions affected the primary rerun. Its elapsed times must
not be presented as the original paper's untouched hardware-timing measurements.
