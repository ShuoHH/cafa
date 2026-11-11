"""Adversarial service guards; no native executables or trial execution."""
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "tools"))
from codec_pool import RETAINED_ACTIONS, RETAINED_STREAM_ACTIONS  # noqa: E402
from flat_pool_backend import DEFAULT_ACTIONS, FlatPoolService, validate_actions  # noqa: E402
from selector_consistent_backend import RETAINED_ACTIONS as SELECTOR_ACTIONS  # noqa: E402
from strict8_boundary_workflow import StrictEightBoundaryService  # noqa: E402
import flat_pool_backend as backend  # noqa: E402


class StrictEightServiceTest(unittest.TestCase):
    def test_defaults_are_the_same_object(self):
        self.assertIs(DEFAULT_ACTIONS, RETAINED_ACTIONS)
        self.assertIs(SELECTOR_ACTIONS, RETAINED_ACTIONS)

    def test_original14_rejected_before_constructor_io(self):
        original14 = RETAINED_ACTIONS + ("rle+gorilla", "rle+chimp", "rle+chimp128",
                                        "btr_onevalue", "btr_rle_fixed", "btr_frequency_fixed")
        with self.assertRaises(ValueError):
            validate_actions(original14)
        with self.assertRaises(ValueError):
            StrictEightBoundaryService(actions=original14)

    def test_injected_extra_trial_rejected_before_any_execution(self):
        service = FlatPoolService.__new__(FlatPoolService)
        service.actions, service.n = RETAINED_ACTIONS, 4096
        with self.assertRaisesRegex(ValueError, "pool-external"):
            service._run_trial((0, 64, "rle+chimp128"))
        service.actions = ("gorilla",)
        with self.assertRaisesRegex(ValueError, "declared strict-eight subset"):
            service._run_trial((0, 64, "chimp"))

    def test_alp_composite_geometry_uses_only_retained_streams(self):
        service = FlatPoolService.__new__(FlatPoolService)
        service.actions, service.n = RETAINED_ACTIONS, 4096
        options = service._action_keys((0, 1100))
        tails = [action.split(":", 1)[1] for action, _ in options if action.startswith("alp+tail:")]
        self.assertEqual(tuple(tails), RETAINED_STREAM_ACTIONS)
        self.assertTrue(all(key[2] in RETAINED_ACTIONS for _, keys in options for key in keys))

    def test_mutated_boundary_pool_rejected(self):
        service = StrictEightBoundaryService.__new__(StrictEightBoundaryService)
        service.actions, service.n = RETAINED_ACTIONS + ("btr_onevalue",), 4096
        with self.assertRaisesRegex(ValueError, "action list changed"):
            service._action_keys((0, 1024))

    def test_paid_payload_cannot_smuggle_excluded_frame_id(self):
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "payload.f64blk"
            path.write_bytes(b"fake paid payload, independent header reader is stubbed")
            service = FlatPoolService.__new__(FlatPoolService)
            service.actions, service.n = RETAINED_ACTIONS, 4096
            key = (0, 64, "gorilla")
            service.cache = {key: dict(start=0, length=64, codec="gorilla", status="passed",
                candidate_path=str(path), candidate_sha256=backend.replay.sha256(path))}
            rogue = [dict(codec_id=20, count=64, start=0, payload=b"rogue")]
            with patch.object(backend.legacy, "read_frames", return_value=({"values": 64}, rogue)):
                with self.assertRaisesRegex(ValueError, "pool-external physical codec ID"):
                    service._frames(key)


if __name__ == "__main__":
    unittest.main()
