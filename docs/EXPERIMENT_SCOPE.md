# Experiment populations and aggregation

This document distinguishes the evaluated populations from the subsets used
in the manuscript's principal tables or plots. Compact numerical records are
in [evidence/](../evidence/README.md); original run archives retain all outcomes.

## Strict-eight primary results

The corrected primary run evaluated 13 series with the same eight base actions
throughout measured boundary refinement and DP, unchanged model weights, and
the existing parameters/budgets. Each series has separately checked decoding,
native trial-pool evidence, actual stored-frame IDs, hashes/sizes, and complete
pricing in the preserved run. No series was replaced with a better repetition.

The manuscript's main table and abstract use a **post-hoc 12-series reporting
subset**, excluding BIDMC-03 from those aggregates and describing it separately.
All 13 remain in the input/model manifests and compact evidence. The absence
of BIDMC-03 from the main aggregate must not be described as it never having
been evaluated or as a pre-specified exclusion.

| Population | CACF complete-file bits | Meaning |
| --- | ---: | --- |
| All 13 evaluated series | 14,412,464 | Full corrected primary cohort. |
| Main 12, excluding BIDMC-03 | 13,420,672 | Post-hoc manuscript main cohort. |
| Paired 11, also excluding SCADA | 10,865,312 | Used only when comparator availability requires this common cohort. |

The aggregate storage reduction is `100 × (sum(comparator bits) - sum(CACF
bits)) / sum(comparator bits)` on the **same paired series**, not the mean of
per-series percentages. Unsupported results must remain unsupported; no
unavailable cell is treated as zero or silently imputed. Each reported
comparison must state or inherit an unambiguous common cohort.

BIDMC-03 stored 991,792 bits and ended with one segment, the same size as Whole.
It was not a case with no detected boundaries: its initial and repaired
partitions contained 28 and 10 segments before final DP coarsening. Its result
was worse than the separately recorded comparison outcomes. It remains a
negative case, not evidence for a universal segmentation gain.

Original compression timing must not be replaced by a fresh-runtime claim:
the corrected runs used a resource-constrained server and documented cache
interventions. Size/decode verification and comparable runtime measurement are
different claims.

## Adaptive encoding selection study

The selector study evaluated four series on their corrected fixed geometries:
SRRL (May), BT, BM, and GL. They contain 45, 3, 6, and 6 segments respectively:
60 total. Figures that show only SRRL (May), BM, and GL show a **post-hoc
three-positive-series subset** with 57 segments, not the entire evaluation.
BT is retained separately and the full four-series aggregate is supplied.
Unchanged geometries reused authenticated existing quality evidence; BT and GL
were freshly evaluated on changed geometries. The summary's `source_kind`
records distinguish them; this is not a claim that all four were newly run.

`base10` uses basic features; `compact20` is the full selector. `decimal15`
retains basic plus decimal features, while `residue15` retains basic plus
binary-relation features. Thus a "remove decimal" result uses `residue15`,
and "remove binary-relation" uses `decimal15`. All variants use preserved
frozen weights; the corrected experiment did not retrain them.

| Population | Base hits → full hits | Base bits → full bits | Exhaustive-search bits |
| --- | --- | --- | ---: |
| Plotted three-series subset | 49/57 → 52/57 | 1,971,032 → 1,889,192 | 1,887,800 |
| BT separately | 2/3 → 2/3 | 552,448 → 599,448 | 548,296 |
| All four evaluated series | 51/60 → 54/60 | 2,523,480 → 2,488,640 | 2,436,096 |

On the plotted subset, hit rate rises from 85.96% to 91.23% (+5.26 percentage
points), stored size falls 4.15%, and full selection uses 0.074% more space than
exhaustive search. On BT, hit rate is unchanged and full selection uses 8.51%
more space than the basic selector and 9.33% more than exhaustive search.
Across all four, hit rate rises from 85% to 90%, size falls 1.38%, and full
selection uses 2.16% more space than exhaustive search. Subset effects must
not be substituted for full-population effects.

Feature ablation plots may be restricted to the same explicitly labelled
three-series subset, but the underlying four-series ablation records remain
preserved. The presence of positive subset aggregates does not establish that
every feature group improves every individual series. For example, removing
the binary-relation group improves May's hit rate and slightly reduces its
stored size in this recorded study.

The selector timings in this study are not a new clean-host hardware benchmark.
See [REPRODUCIBILITY.md](REPRODUCIBILITY.md) for execution and evidence limits.
