"""Shared-bank input/field guards without native code or model loading."""
import hashlib
from pathlib import Path
import struct
import sys
import tempfile
import unittest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "tools"))
from shared_compression_features import FeatureBank  # noqa: E402
from strict8_boundary_workflow import bank_cache, BOUNDARY_FIELDS  # noqa: E402


class SharedBoundaryCacheTest(unittest.TestCase):
    def test_boundary_reuses_exact_bank_values_and_complete_features(self):
        with tempfile.TemporaryDirectory() as temporary:
            raw = Path(temporary) / "series.f64le"
            data = struct.pack("<64d", *(float(i) / 8 for i in range(64)))
            raw.write_bytes(data)
            digest = hashlib.sha256(data).hexdigest()
            identity = ("demo", "series", digest)
            bank = FeatureBank(raw, identity)
            job = dict(dataset=identity[0], series=identity[1], sha256=digest, values=64, input=str(raw))
            cache = bank_cache(bank, job)
            self.assertIs(cache.values, bank.values)
            self.assertIs(bank.boundary_cache(), bank.boundary_cache())
            row = cache.row(0, 32)
            self.assertTrue(all(field in row for field in BOUNDARY_FIELDS))
            self.assertEqual(row["sha256"], digest)
            self.assertEqual(cache.values_scanned, 32)
            cache.row(0, 32)
            self.assertEqual(cache.values_scanned, 32)
            raw.write_bytes(data[:-8] + struct.pack("<d", -1.0))
            with self.assertRaisesRegex(ValueError, "source changed"):
                bank_cache(bank, job)


if __name__ == "__main__":
    unittest.main()
