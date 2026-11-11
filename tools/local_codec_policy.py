"""Declared candidate actions for local parent/child stored-cost comparisons.

This is an explicit, fixed candidate policy, NOT a learned codec selector.
Only the one or two actions declared before the run may be tried.  The same
declaration is used for parents, children, and any fixed-layout reference.
There is no whole-sequence prepass, all-pool trial, automatic tail choice, or
external-baseline fallback in this adapter.
"""

from __future__ import annotations

import btr_selected_costs as selected
import class_mainline_btr as mainline
import frozen_btr_extension as frozen


ALLOWED_ACTIONS = (selected.STREAM_CODECS + ("btrblocks", "alp")
                   + tuple("alp+tail:" + codec for codec in selected.STREAM_CODECS))


def validate_declared_actions(actions):
    """Return an ordered immutable declaration, rejecting implicit defaults."""
    if not isinstance(actions, (tuple, list)) or not 1 <= len(actions) <= 2:
        raise ValueError("declare exactly one or two codec actions before the run")
    if any(not isinstance(action, str) or action not in ALLOWED_ACTIONS
           for action in actions):
        raise ValueError("unknown declared codec action; ALP tails require an explicit stream codec")
    if len(set(actions)) != len(actions):
        raise ValueError("declared codec actions must be unique")
    return tuple(actions)


class DeclaredActionService(mainline.DiskCosts):
    """The existing persisted native service restricted to declared actions.

    Native feasibility depends on region length: ``alp`` requires a multiple
    of 1024, and ``alp+tail:<codec>`` requires both a full ALP prefix and a
    nonempty tail.  Infeasible declarations are never silently rewritten.
    If no declared action is feasible, the complete request fails before any
    encoder runs.  Native failures can only be replaced by another feasible
    action in the same declaration, never by an undeclared codec.
    """

    cost_policy = "declared_codec_actions"

    def __init__(self, *args, declared_actions, **kwargs):
        self._declared_actions = validate_declared_actions(declared_actions)
        super().__init__(*args, **kwargs)

    @property
    def declared_actions(self):
        return self._declared_actions

    def actions_for(self, region):
        """Return the declared feasible subset; this method never encodes."""
        _, length = self._region(region)
        feasible = set(selected.feasible_actions(length))
        return tuple(action for action in self.declared_actions if action in feasible)

    def _needed(self, region):
        start, length = self._region(region)
        actions = self.actions_for((start, length))
        if not actions:
            raise ValueError("no declared codec action is feasible for region length "
                             + str(length) + "; no substitution or fallback")
        return list(dict.fromkeys(key for action in actions
                                  for key in selected.SelectedActionService._action_keys(
                                      (start, length, action))))

    def _best(self, region):
        start, length = self._region(region)
        options = []
        for action in self.actions_for((start, length)):
            frames = []
            for key in selected.SelectedActionService._action_keys((start, length, action)):
                component = self._frames(key)
                if component is None:
                    break
                frames.extend(component)
            else:
                options.append((action, frames))
        if not options:
            return None
        # Stable declaration order resolves equal measured sizes.  No new
        # action is requested while selecting or reopening persisted frames.
        action, frames = min(options, key=lambda item: frozen.frame_cost(item[1]))
        return dict(start=start, length=length, codec=action, action=action, frames=frames)
