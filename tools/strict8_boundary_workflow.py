"""Strict-eight guards for the public, raw-only shared-feature boundary stage.

These guards preserve the frozen strict-eight experiment's candidate/repair
semantics. They do not import a private paper template or reuse its prices.
"""
from collections.abc import Mapping
from pathlib import Path

from codec_pool import RETAINED_ACTIONS, check_choices, check_trial_ledger
from compression_candidates import STATE_GROUPS
from flat_pool_backend import FlatPoolService

BOUNDARY_ACTIONS = RETAINED_ACTIONS
BOUNDARY_FIELDS = tuple(dict.fromkeys(
    name for entries in STATE_GROUPS.values() for name, _ in entries))


class StrictEightBoundaryService(FlatPoolService):
    def __init__(self, *args, actions=BOUNDARY_ACTIONS, **kwargs):
        if tuple(actions) != BOUNDARY_ACTIONS:
            raise ValueError("strict-eight boundary requires the exact fixed eight-action pool")
        super().__init__(*args, actions=BOUNDARY_ACTIONS, **kwargs)

    def _check_pool(self):
        if tuple(self.actions) != BOUNDARY_ACTIONS:
            raise ValueError("strict-eight boundary action list changed")

    def _action_keys(self, region):
        self._check_pool()
        return super()._action_keys(region)

    def _run_trial(self, key):
        self._check_pool()
        return super()._run_trial(key)

    def _frames(self, key):
        self._check_pool()
        return super()._frames(key)


def assert_boundary_choices(service, choices):
    service._check_pool()
    check_trial_ledger(service.trials)
    check_choices(choices)


class _CheckedBoundaryCache:
    def __init__(self, cache, bank, job):
        if (not callable(getattr(cache, "row", None))
                or getattr(cache, "values", None) is not bank.values):
            raise ValueError("boundary cache must share the caller bank's source values")
        self._cache, self.values = cache, bank.values
        self._identity = (job["dataset"], job["series"], job["sha256"])

    @property
    def values_scanned(self):
        return self._cache.values_scanned

    def row(self, start, length):
        row = self._cache.row(start, length)
        if (not isinstance(row, Mapping)
                or any(name not in row for name in BOUNDARY_FIELDS)):
            raise ValueError("shared boundary cache omits an original state feature")
        if (tuple(row.get(name) for name in ("dataset", "series", "sha256")) != self._identity
                or row.get("start") != start or row.get("length") != length):
            raise ValueError("shared boundary feature identity/region differs")
        try:
            for name in BOUNDARY_FIELDS:
                float(row[name])
        except (TypeError, ValueError, OverflowError) as error:
            raise ValueError("shared boundary state features must be numeric") from error
        return row


def bank_cache(bank, job):
    expected = (job["dataset"], job["series"], job["sha256"])
    if (Path(getattr(bank, "raw", "")).resolve() != Path(job["input"]).resolve()
            or tuple(getattr(bank, "identity", ())) != expected
            or getattr(bank, "n", None) != job["values"]
            or getattr(bank, "source_sha256", None) != job["sha256"]
            or not callable(getattr(bank, "assert_unchanged", None))
            or not callable(getattr(bank, "boundary_cache", None))):
        raise ValueError("caller feature bank must match the complete raw source identity")
    values = getattr(bank, "values", None)
    if (values is None or len(values) != job["values"]
            or getattr(getattr(values, "flags", None), "writeable", True)):
        raise ValueError("caller feature bank must retain immutable complete source values")
    bank.assert_unchanged()
    return _CheckedBoundaryCache(bank.boundary_cache(), bank, job)
