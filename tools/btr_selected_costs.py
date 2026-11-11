"""Measure only explicitly selected, feasible codec actions on frozen regions.

This adapter deliberately does not call ``NativeRegionService.best_many``:
doing so would silently measure every backend, making a top-k selector an
oracle trial.  ALP with a non-vector tail is an explicit action whose tail
codec is part of the label, not an uncharged best-of-eight choice.
"""

import btr_boundary_costs as costs


STREAM_CODECS = costs.STREAM_CODECS


def feasible_actions(length):
    """Return the declared action order, without reading labels or raw values."""
    costs._integer(length, "length", 1)
    if length > 102400:
        raise ValueError("length exceeds supported rowgroup")
    actions = STREAM_CODECS + ("btrblocks",)
    if length >= 1024:
        if length % 1024 == 0:
            actions += ("alp",)
        else:
            actions += tuple("alp+tail:" + codec for codec in STREAM_CODECS)
    return actions


class SelectedActionService(costs.NativeRegionService):
    """Persist and decode-check requested actions, reserving budgets atomically.

    The inherited cache stores each unique physical ``(start, length,
    backend)`` attempt, including unavailable attempts.  Several ALP-tail
    actions can share their one measured prefix.  Cached payload bytes are
    reopened and hash-checked, never re-encoded to reproduce an estimated
    price.  Native failures make only their requested action unavailable.
    """

    feasible_actions = staticmethod(feasible_actions)

    def _request(self, request):
        if not isinstance(request, (tuple, list)) or len(request) != 3:
            raise ValueError("request must be (start, length, action)")
        start, length = self._region(request[:2])
        action = request[2]
        if not isinstance(action, str) or action not in feasible_actions(length):
            raise ValueError("action is not feasible for requested length: " + repr(action))
        return start, length, action

    @staticmethod
    def _action_keys(request):
        start, length, action = request
        if action.startswith("alp+tail:"):
            full = length // 1024 * 1024
            return ((start, full, "alp"),
                    (start + full, length - full, action[len("alp+tail:"):]))
        return ((start, length, action),)

    def _selected(self, request):
        frames = []
        for key in self._action_keys(request):
            component = self._frames(key)
            if component is None:
                return None
            frames.extend(component)
        start, length, action = request
        return dict(start=start, length=length, codec=action, action=action, frames=frames)

    def measure_actions(self, requests):
        """Return results in request order, or ``None`` on atomic budget refusal.

        Validate *all* requests before invoking any encoder.  A native failure
        is represented by a ``None`` entry, unlike an operation-level budget
        refusal.  Complete physical frame headers remain in returned frames
        so callers can charge them with ``frozen.frame_cost``.
        """
        requests = [self._request(request) for request in requests]
        if not requests:
            raise ValueError("at least one selected action is required")
        needed = list(dict.fromkeys(key for request in requests
                                    for key in self._action_keys(request)))
        fresh = [key for key in needed if key not in self.cache]
        self.counters["cache_hits"] += len(needed) - len(fresh)
        volume = sum(key[1] for key in fresh)
        if (self.counters["trial_count"] + len(fresh) > self.trial_limit
                or self.counters["trial_value_volume"] + volume > self.value_budget):
            self.counters["budget_rejections"] += 1
            return None
        self._check_inputs()
        self.counters["trial_count"] += len(fresh)
        self.counters["trial_value_volume"] += volume
        for key in fresh:
            self._run_trial(key)
        return [self._selected(request) for request in requests]

