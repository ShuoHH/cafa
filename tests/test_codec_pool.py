"""No executable or scientific-library dependencies: public pool/format contract."""
from pathlib import Path
import struct
import sys
import tempfile
import unittest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "tools"))
from codec_pool import (RETAINED_ACTIONS, RETAINED_PHYSICAL_IDS,  # noqa: E402
                        check_choices, check_trial_ledger, permitted_action)
import flat_pool_format as fmt  # noqa: E402


class CodecPoolTest(unittest.TestCase):
    def test_pool_and_physical_ids_are_fixed(self):
        self.assertEqual(RETAINED_ACTIONS, (
            "gorilla", "chimp", "chimp128", "tsxor", "rle+tsxor", "alp",
            "btr_dict_bp", "btr_decimal_fixed"))
        self.assertEqual(RETAINED_PHYSICAL_IDS, {16, 17, 18, 19, 23, 24, 33, 36})
        self.assertEqual({fmt.CODEC_IDS[name] for name in RETAINED_ACTIONS}, RETAINED_PHYSICAL_IDS)
        self.assertEqual(fmt.FIXED_IDS["btr_dict_bp"], 33)
        self.assertEqual(fmt.FIXED_IDS["btr_decimal_fixed"], 36)

    def test_alp_tails_cannot_widen_the_pool(self):
        for name in RETAINED_ACTIONS[:5]:
            self.assertTrue(permitted_action("alp+tail:" + name))
        for name in ("rle+gorilla", "rle+chimp", "rle+chimp128", "btrblocks",
                     "alp", "btr_dict_bp", "btr_onevalue"):
            self.assertFalse(permitted_action("alp+tail:" + name))

    def test_persisted_excluded_id_rejected_even_with_valid_envelope(self):
        with tempfile.TemporaryDirectory() as temporary:
            for codec in (20, 21, 22, 25, 32, 34, 35):
                path = Path(temporary) / (str(codec) + ".f64pool")
                payload = struct.pack(">8sIIQ", b"BTRFIX01", 1, codec, 4) if codec >= 32 else b"x"
                path.write_bytes(fmt.FILE_HEADER.pack(fmt.MAGIC, fmt.METHOD, 0, 4)
                                 + fmt.FRAME_HEADER.pack(codec, 4, len(payload)) + payload)
                with self.subTest(codec=codec), self.assertRaises(ValueError):
                    fmt.inspect_container(path)
                with self.subTest(codec=codec), self.assertRaises(ValueError):
                    fmt.checked_frames([dict(codec_id=codec, count=4, start=0, payload=payload)], 4)

    def test_retained_frame_ids_are_inspected_from_bytes(self):
        with tempfile.TemporaryDirectory() as temporary:
            for name in RETAINED_ACTIONS:
                codec = fmt.CODEC_IDS[name]
                n = 1024 if name == "alp" else 4
                payload = struct.pack(">8sIIQ", b"BTRFIX01", 1, codec, n) if codec >= 32 else b"x"
                info = fmt.write_container(Path(temporary) / (name.replace("+", "_") + ".f64pool"),
                    [dict(codec_id=codec, count=n, start=0, payload=payload)], n)
                self.assertEqual(info["actual_physical_codec_counts"], {name: 1})
                self.assertTrue(info["actual_frame_ids_restricted8"])

    def test_resume_ledger_and_choices_cannot_claim_extra_codec(self):
        with self.assertRaisesRegex(ValueError, "pool-external"):
            check_trial_ledger([dict(start=0, length=4, codec="rle+chimp")])
        with self.assertRaisesRegex(ValueError, "pool-external"):
            check_choices([dict(codec="alp+tail:rle+gorilla", physical_keys=[[0, 4, "gorilla"]])])
        with self.assertRaisesRegex(ValueError, "pool-external"):
            check_choices([dict(codec="gorilla", physical_keys=[[0, 4, "btr_onevalue"]])])


if __name__ == "__main__":
    unittest.main()
