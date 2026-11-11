# Publication checklist

## Included and deliberately scoped

- [x] Retain the CACF method path and native integration adapters, with explicit
  eight-base-action pool definitions for boundary and DP stages.
- [x] Preserve 36 historical frozen model artifacts (nine folds × four feature
  variants), their original metadata, hashes, and 13-input model routing.
- [x] Include two generated synthetic inputs; keep original research bytes,
  training records, virtual environments, upstream source trees, native
  executables, and complete experiment outputs out of the Git candidate.
- [x] Identify ClaSPy-derived integrations and BtrBlocks-derived constants with
  upstream notices and license texts.
- [x] Distinguish feature-only, whole-series, adaptive-synthetic, and original
  experiment scopes; do not call any one synthetic demo a paper reproduction.
- [x] Provide an exact-input primary driver with authenticated input/fold
  routing, full DP pricing, and preserved-result comparisons; do not represent
  a dry-run or representative test as an executed all-13 replay.
- [x] Execute both synthetic native paths and independently verify their saved
  decodes; separately rebuild and self-test guarded native wrappers on the
  existing host. Neither check establishes a clean-host dependency build.
- [x] Record that the historical adaptive pilot used 14 boundary actions and
  therefore does not validate the current all-stage strict-eight path.
- [x] Preserve compact evidence for all 13 evaluated primary series, including
  the separately reported BIDMC-03 negative case.
- [x] Label the 12-series main aggregate and the three-positive-series selector
  plots as post-hoc subsets; retain BT and full four-series selector summaries.

Automated source/model integrity checks and synthetic tests should be rerun
after every code or artifact change. A test pass only supports the paths it
actually exercises; it does not imply original-data reproduction or a
clean-host native build.

## Still required before an unqualified public release

- [ ] Authors review source ownership and third-party provenance, including
  native integration files. Matching hashes do not determine authorship.
- [ ] Authors choose the license for CACF-authored code and add a root
  `LICENSE`; no license has been inferred by the release preparation.
- [ ] Lock exact upstream native revisions and dependencies, and build/test
  both native adapters on a clean Linux host. Existing-server tests are not
  equivalent to a clean-host build.
- [ ] Verify research-input redistribution terms and supply lawful acquisition
  and exact preprocessing instructions. An input hash manifest alone is not
  a dataset download pipeline.
- [ ] Validate a complete original-input replay and supply the remaining
  baseline/segmentation-comparison/selector experiment runners before claiming
  one-command full-paper reproduction. A primary driver, representative tests,
  synthetic examples, and compact summaries do not provide that claim.
- [ ] Perform final secret, accidental-data, large-file, and executable scans;
  review the exact staged Git diff before the first public push.

No public push has been performed as part of this preparation. The remaining
license and dependency/data-reproduction decisions must not be represented as
completed by a packaging or synthetic-test pass.
