"""Check metadata/SHA before any pickle-based model deserialization."""
import json
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "examples"))
from run_verified_compression import checked_model, METADATA  # noqa: E402


class CheckedModelTest(unittest.TestCase):
    def test_mismatching_sha_never_reaches_joblib(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            model = root / "test.joblib"
            model.write_bytes(b"not a trusted model")
            metadata = json.loads(METADATA.read_text(encoding="utf-8"))
            metadata["model_path"] = model.name
            metadata["model_sha256"] = "0" * 64
            meta = root / "test.json"
            meta.write_text(json.dumps(metadata), encoding="utf-8")
            with patch("joblib.load") as load:
                with self.assertRaisesRegex(ValueError, "SHA-256 differs"):
                    checked_model(model, meta)
                load.assert_not_called()

    def test_model_metadata_must_be_paired(self):
        with self.assertRaisesRegex(ValueError, "supplied together"):
            checked_model(Path("anything"), None)


if __name__ == "__main__":
    unittest.main()
