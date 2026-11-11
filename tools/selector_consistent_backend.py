"""Actual native costs restricted to one frozen selector's semantic Top-2.

The service never exhausts the eight-codec pool, silently changes an ALP tail,
or recommends again at final persistence. Unavailable recommendations carry no
invented price: candidate-level callers may skip PolicyUnavailable explicitly,
whereas integrity failures remain ordinary fatal errors.
"""
from __future__ import annotations

from flat_pool_backend import FlatPoolEvaluator, FlatPoolService
from codec_pool import RETAINED_ACTIONS
_TRIAL_FIELDS = (
    "start", "length", "codec", "status", "frame_bytes", "roundtrip_verified",
    "sha256", "slice_sha256", "candidate_path", "candidate_sha256", "executable_sha256",
)


class PolicyUnavailable(ValueError):
    """No usable price for these regions under their frozen recommendations."""

    def __init__(self, regions, message="no lossless action among frozen Top-2 recommendations", *,
                 reason="all_recommended_actions_unavailable"):
        self.regions = tuple(sorted(set(tuple(region) for region in regions)))
        self.reason = reason
        super().__init__(message + ": " + repr(self.regions))


class Top2FlatPoolService(FlatPoolService):
    """Keep exact flat-pool geometry but execute only frozen recommended actions."""

    def __init__(self, *args, policy, actions=RETAINED_ACTIONS, **kwargs):
        if tuple(actions) != RETAINED_ACTIONS:
            raise ValueError("consistent selector service requires the fixed retained-eight base pool")
        if not callable(getattr(policy, "actions_for", None)):
            raise ValueError("policy must expose frozen actions_for(region)")
        self.policy = policy
        self._frozen_actions = {}
        self._certified_trials = {}
        super().__init__(*args, actions=RETAINED_ACTIONS, **kwargs)

    def _action_keys(self, region):
        region = self._region(region)
        geometry = dict(super()._action_keys(region))
        if region not in self._frozen_actions:
            recommended = self.policy.actions_for(region)
            if isinstance(recommended, (str, bytes)):
                raise ValueError("frozen Top-2 actions must be a sequence, not a string")
            try:
                recommended = tuple(recommended)
            except TypeError as error:
                raise ValueError("frozen Top-2 actions must be an iterable") from error
            if (len(recommended) > 2 or any(not isinstance(a, str) for a in recommended)
                    or len(set(recommended)) != len(recommended)
                    or any(a not in geometry for a in recommended)):
                raise ValueError("frozen Top-2 actions are duplicated, unknown, or geometrically infeasible")
            self._frozen_actions[region] = recommended
        recommended = self._frozen_actions[region]
        if not recommended:
            raise PolicyUnavailable([region], "frozen selector has no geometry-feasible learned action",
                                    reason="no_feasible_recommendation")
        # Rank order is preserved. The inherited stable minimum therefore uses
        # the policy's order, not a hidden full-pool tie breaker.
        return [(action, geometry[action]) for action in recommended]

    def _run_trial(self, key):
        super()._run_trial(key)
        self._certified_trials[key] = tuple(self.cache[key].get(k) for k in _TRIAL_FIELDS)

    def _check_trials(self, keys):
        for key in keys:
            if key not in self.cache:
                continue
            current = tuple(self.cache[key].get(k) for k in _TRIAL_FIELDS)
            if key not in self._certified_trials or self._certified_trials[key] != current:
                raise ValueError("frozen Top-2 paid native record changed or is unbound")

    def options_for(self, region):
        options = self._action_keys(region)
        self._check_trials(key for _, keys in options for key in keys)
        names = {action for action, _ in options}
        # The parent appends a short unsupported bare-ALP diagnostic. It is not
        # a recommendation and must not appear in a strict Top-2 option matrix.
        return [o for o in super().options_for(region) if o["codec"] in names]

    def _frames(self, key):
        self._check_trials([key])
        return super()._frames(key)


class ConsistentTop2Evaluator(FlatPoolEvaluator):
    """Freeze measured winner frames and expose only typed policy unavailability."""

    cost_source = "actual_frozen_selector_top2_frames"

    def __init__(self, view):
        if not isinstance(view.backend, Top2FlatPoolService):
            raise ValueError("consistent evaluator requires a Top2FlatPoolService")
        super().__init__(view)
        self.unavailable_regions = {}

    def _record_empty(self, error):
        for region in error.regions:
            self.unavailable_regions[region] = dict(reason=error.reason, actions=[], options=[])

    def can_evaluate(self, intervals):
        # This is a read-only native/budget preflight. Policy inference may
        # freeze new recommendations, but no trial or ledger charge occurs.
        regions = self._intervals(intervals)
        known = [r for r in regions if r in self.unavailable_regions]
        if known:
            raise PolicyUnavailable(known, "previously measured frozen Top-2 region is unavailable")
        try:
            return super().can_evaluate(regions)
        except PolicyUnavailable as error:
            self._record_empty(error)
            raise

    def _completed_unavailable(self, regions):
        unavailable = []
        for region in regions:
            action_keys = self.service._action_keys(region)
            keys = [key for _, physical in action_keys for key in physical]
            if not keys or any(key not in self.service.cache for key in keys):
                continue
            self.service._check_trials(keys)
            if any(self.service.cache[k].get("status") not in ("passed", "unsupported", "failed") for k in keys):
                raise ValueError("frozen Top-2 native cache contains an incomplete status")
            options = self.service.options_for(region)
            if options and not any(o["status"] == "passed" for o in options):
                self.unavailable_regions[region] = dict(
                    reason="all_recommended_actions_unavailable",
                    actions=[a for a, _ in action_keys],
                    options=[dict(action=o["codec"], status=o["status"]) for o in options],
                )
                unavailable.append(region)
        return unavailable

    def evaluate(self, intervals, *, phase="trial"):
        regions = self._intervals(intervals)
        try:
            return super().evaluate(regions, phase=phase)
        except PolicyUnavailable as error:
            if error.reason == "no_feasible_recommendation":
                self._record_empty(error)
            raise
        except ValueError as error:
            # Translate ONLY the parent's explicit no-winner result. A corrupt
            # payload/signature/price, malformed policy, or budget refusal must
            # never be disguised as an optional candidate's unavailability.
            if str(error) != "no lossless fixed-pool action for a requested region":
                raise
            unavailable = self._completed_unavailable(regions)
            if not unavailable:
                raise
            raise PolicyUnavailable(unavailable) from error

    def usage_segments(self, regions):
        result = super().usage_segments(regions)
        for row in result:
            row.update(options_scope="frozen_top2_only", full_pool_exhaustive=False,
                       recommended_actions=[o["codec"] for o in row["options"]])
        return result


Top2FlatPoolEvaluator = ConsistentTop2Evaluator
