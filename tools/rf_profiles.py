"""Pinned random-forest settings for the TSEC-feature comparison.

TSEC reports using a random-forest classifier but does not publish its tree
count, depth or leaf settings. These values are a reproducible experimental
profile, NOT parameters attributed to TSEC. Use the same capacity and seed
for the paper-feature and augmented-feature arms.
"""

RF_REPRO_PARAMS = {
    "n_estimators": 100,
    "max_depth": None,
    "min_samples_split": 2,
    "min_samples_leaf": 1,
    "max_features": "sqrt",
    "bootstrap": True,
    "random_state": 0,
    "n_jobs": 1,
}
