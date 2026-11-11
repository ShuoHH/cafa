"""Smoke test for the synthetic, no-native-code demonstration."""

from __future__ import annotations

import json
from pathlib import Path
import runpy
import subprocess
import struct
import sys
import unittest


ROOT = Path(__file__).resolve().parents[1]


class DemoTest(unittest.TestCase):
    def test_fixture_is_generated(self) -> None:
        generator = runpy.run_path(str(ROOT / "examples" / "make_demo_data.py"))
        expected = struct.pack("<1024d", *generator["values"]())
        actual = (ROOT / "examples" / "data" / "synthetic_1024.f64le").read_bytes()
        self.assertEqual(actual, expected)

    def test_selector_and_candidates(self) -> None:
        result = subprocess.run([sys.executable, "-B", str(ROOT / "examples" / "run_demo.py")],
                                text=True, capture_output=True, check=True)
        data = json.loads(result.stdout)
        self.assertEqual(data["values"], 1024)
        self.assertEqual(data["feature_count"], 20)
        self.assertGreater(data["candidate_boundary_count"], 0)
        self.assertEqual(len(data["regions"]), 5)
        self.assertTrue(all(len(region["top2"]) == 2 for region in data["regions"]))
        self.assertTrue(data["external_model_reuse"])


if __name__ == "__main__":
    unittest.main()
