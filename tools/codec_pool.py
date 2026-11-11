"""One immutable public codec pool, retaining the original on-disk IDs.

An ALP prefix with a retained streaming tail is a composite of these codecs,
not a ninth codec. These constants do not import any backend or file format.
"""

RETAINED_ACTIONS = (
    "gorilla", "chimp", "chimp128", "tsxor", "rle+tsxor", "alp",
    "btr_dict_bp", "btr_decimal_fixed",
)
RETAINED_STREAM_ACTIONS = RETAINED_ACTIONS[:5]
RETAINED_PHYSICAL_IDS = frozenset((16, 17, 18, 19, 23, 24, 33, 36))


def permitted_action(action):
    if not isinstance(action, str):
        return False
    if action.startswith("alp+tail:"):
        return action.split(":", 1)[1] in RETAINED_STREAM_ACTIONS
    return action in RETAINED_ACTIONS


def check_physical_keys(keys):
    for key in keys:
        if (not isinstance(key, (tuple, list)) or len(key) != 3
                or type(key[0]) is not int or key[0] < 0
                or type(key[1]) is not int or key[1] <= 0
                or key[2] not in RETAINED_ACTIONS):
            raise ValueError("strict-eight pool-external or malformed physical key")


def check_trial_ledger(trials):
    if not isinstance(trials, list):
        raise ValueError("strict-eight trial ledger must be a list")
    for trial in trials:
        if not isinstance(trial, dict):
            raise ValueError("strict-eight trial ledger contains a malformed record")
        check_physical_keys(((trial.get("start"), trial.get("length"), trial.get("codec")),))


def check_choices(choices):
    for choice in choices:
        if not isinstance(choice, dict) or not permitted_action(choice.get("codec")):
            raise ValueError("strict-eight selected a pool-external semantic action")
        keys = choice.get("physical_keys")
        if not keys:
            raise ValueError("strict-eight selected action lacks physical frame bindings")
        check_physical_keys(keys)
        if any(not permitted_action(action) for action in choice.get("recommended_actions", ())):
            raise ValueError("strict-eight assembly options escaped the declared pool")
