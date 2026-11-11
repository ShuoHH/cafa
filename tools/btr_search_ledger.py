"""Fair virtual trial budgets over a shared, immutable native-price cache.

Sharing a backend ensures that every arm sees the same persisted BtrBlocks
sample for a physical interval. Each arm/seed still pays its own unique
codec/value work, including work whose actual payload another arm produced.
Thus running second never gives an arm a larger effective search budget.
This ledger does not create, modify, encode, or select any payload itself.
"""


def _integer(value, name, minimum=0):
    if type(value) is not int or value < minimum:
        raise ValueError(name + " must be a strict integer >= " + str(minimum))
    return value


def fixed_partition(n, size, minimum=64, grid=64):
    """Return a complete fixed-size layout, retaining every final tail value.

    Interior boundaries lie on ``grid``. A final tail can be shorter than
    ``minimum`` (even one value); it must not be silently dropped or appended
    to change the declared fixed-length reference. The native backend deals
    with ALP feasibility and explicit streaming tails separately.
    """
    n = _integer(n, "values", 1)
    size = _integer(size, "fixed size", 1)
    minimum = _integer(minimum, "minimum", 1)
    grid = _integer(grid, "grid", 1)
    if size < minimum or size % grid:
        raise ValueError("fixed size must meet minimum and be a multiple of grid")
    return [(start, min(size, n - start)) for start in range(0, n, size)]


class BudgetedRegionView:
    """Independently charge one search arm/seed against a shared backend.

    ``backend`` must expose NativeRegionService's ``_region``, ``_needed``,
    ``cache``, ``counters`` and ``best_many`` protocol. Region validation and
    full-pool physical key expansion happen before any ledger mutation.
    A rejected operation makes no backend call and reserves no keys. Failed
    native attempts and backend-level refusals remain conservatively charged;
    they cannot be retried for free through this ledger.
    """

    def __init__(self, backend, value_budget, trial_limit=8192):
        self.backend = backend
        self.value_budget = _integer(value_budget, "value budget", 1)
        self.trial_limit = _integer(trial_limit, "trial limit", 1)
        self.charged_keys = set()
        self.counters = dict(trial_count=0, trial_value_volume=0,
                             budget_rejections=0, cache_hits=0,
                             shared_cache_reuses=0,
                             actual_new_native_trials=0,
                             actual_new_native_values=0,
                             backend_unavailable_operations=0)

    def _needed(self, regions):
        # Materialize/validate the complete request before changing counters.
        regions = [self.backend._region(region) for region in regions]
        if not regions:
            raise ValueError("at least one region is required")
        needed = []
        for region in regions:
            for key in self.backend._needed(region):
                if (not isinstance(key, tuple) or len(key) != 3
                        or type(key[0]) is not int or key[0] < 0
                        or type(key[1]) is not int or key[1] < 1
                        or not isinstance(key[2], str) or not key[2]):
                    raise ValueError("backend physical keys must be (start, length, codec)")
                needed.append(key)
        if not needed:
            raise ValueError("backend declared no physical trials")
        return regions, list(dict.fromkeys(needed))

    def best_many(self, regions):
        """Reserve all new virtual work, then reuse or obtain exact prices.

        Full-pool work includes ALP's aligned prefix and every explicit cold
        streaming-tail candidate exactly as declared by the native backend.
        Keys repeated by another requested region are paid once per ledger.
        """
        regions, needed = self._needed(regions)
        fresh = [key for key in needed if key not in self.charged_keys]
        volume = sum(key[1] for key in fresh)
        if (self.counters["trial_count"] + len(fresh) > self.trial_limit
                or self.counters["trial_value_volume"] + volume > self.value_budget):
            self.counters["budget_rejections"] += 1
            return None

        # Reserve atomically before native work. Cached failed attempts count
        # too; a warm cache must not make a search arm's budget go farther.
        self.charged_keys.update(fresh)
        self.counters["trial_count"] += len(fresh)
        self.counters["trial_value_volume"] += volume
        self.counters["cache_hits"] += len(needed) - len(fresh)
        self.counters["shared_cache_reuses"] += sum(key in self.backend.cache for key in fresh)
        before_trials = self.backend.counters["trial_count"]
        before_values = self.backend.counters["trial_value_volume"]
        try:
            result = self.backend.best_many(regions)
        finally:
            self.counters["actual_new_native_trials"] += (
                self.backend.counters["trial_count"] - before_trials)
            self.counters["actual_new_native_values"] += (
                self.backend.counters["trial_value_volume"] - before_values)
        if result is None:
            self.counters["backend_unavailable_operations"] += 1
        return result
