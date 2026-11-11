"""No-native tests for the read-only adaptive boundary resume gate."""

from __future__ import annotations

import json
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "examples"))
import run_adaptive_verified as adaptive  # noqa: E402


class ResumeBoundaryTest(unittest.TestCase):
    def _fixture(self, root):
        old, new = root / "old_boundary", root / "new_output"
        old.mkdir()
        new.mkdir()
        raw, exporter, fixed = root / "raw.f64le", root / "exporter", root / "fixed"
        raw.write_bytes(bytes(range(32)))
        exporter.write_bytes(b"exporter executable test identity")
        fixed.write_bytes(b"fixed executable test identity")
        source_hash = adaptive.sha256(raw)
        job = dict(dataset="demo", series="sample", values=4,
                   sha256=source_hash, input=str(raw))
        choices = [dict(start=0, length=4, codec="gorilla", physical_keys=[[0, 4, "gorilla"]],
                        recommended_actions=list(adaptive.DEFAULT_ACTIONS))]
        manifest = dict(version=adaptive.VERSION, job=job,
                        source_boundaries_historical=False,
                        fresh_xor_change_detection=True,
                        selector_or_training_used_for_boundary=False,
                        boundary_actions=list(adaptive.DEFAULT_ACTIONS),
                        strict8_at_all_stages=True,
                        shared_feature_bank_for_boundary_and_selection=True,
                        boundary_native_failures=0, boundary_trial_count=0,
                        detector=dict(status="passed", input_sha256=source_hash, values=4),
                        exporter_sha256=adaptive.sha256(exporter),
                        fixed_exe_sha256=adaptive.sha256(fixed),
                        initial_regions=[[0, 4]], repaired_regions=[[0, 4]],
                        initial_choices=choices, repaired_choices=choices, repair={})
        for stem in ("initial", "repaired"):
            container = old / (stem + ".f64pool")
            adaptive.fmt.write_container(container, [dict(codec_id=16, count=4, start=0,
                                                         payload=stem.encode("ascii"))], 4)
            manifest[stem + "_container_path"] = str(container)
            manifest[stem + "_container_sha256"] = adaptive.sha256(container)
            manifest[stem + "_complete_file_bits"] = container.stat().st_size * 8
            manifest[stem + "_verification"] = dict(decoded_sha256=source_hash,
                complete_file_roundtrip_verified=True)
        (old / "native_trials.json").write_text("[]", encoding="utf-8")
        (old / "boundary_manifest.json").write_text(json.dumps(manifest), encoding="utf-8")
        return old, new, raw, exporter, fixed, job, manifest

    def test_verified_resume_is_read_only_and_rejects_tampering(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            old = root / "old_boundary"
            new = root / "new_output"
            old.mkdir()
            new.mkdir()
            raw = root / "raw.f64le"
            raw.write_bytes(bytes(range(32)))
            exporter = root / "exporter"
            fixed = root / "fixed"
            exporter.write_bytes(b"exporter executable test identity")
            fixed.write_bytes(b"fixed executable test identity")
            source_hash = adaptive.sha256(raw)
            job = dict(dataset="demo", series="sample", values=4,
                       sha256=source_hash, input=str(raw))
            for stem in ("initial", "repaired"):
                adaptive.fmt.write_container(old / (stem + ".f64pool"), [dict(
                    codec_id=16, count=4, start=0, payload=stem.encode("ascii"))], 4)
            manifest = dict(version=adaptive.VERSION, job=job,
                            source_boundaries_historical=False,
                            fresh_xor_change_detection=True,
                            selector_or_training_used_for_boundary=False,
                            boundary_actions=list(adaptive.DEFAULT_ACTIONS),
                            strict8_at_all_stages=True,
                            shared_feature_bank_for_boundary_and_selection=True,
                            boundary_native_failures=0, boundary_trial_count=0,
                            detector=dict(status="passed", input_sha256=source_hash, values=4),
                            exporter_sha256=adaptive.sha256(exporter),
                            fixed_exe_sha256=adaptive.sha256(fixed),
                            initial_regions=[[0, 4]], repaired_regions=[[0, 4]],
                            repair={})
            for stem in ("initial", "repaired"):
                container = old / (stem + ".f64pool")
                manifest[stem + "_container_path"] = str(container)
                manifest[stem + "_container_sha256"] = adaptive.sha256(container)
                manifest[stem + "_complete_file_bits"] = container.stat().st_size * 8
                manifest[stem + "_verification"] = dict(decoded_sha256=source_hash,
                    complete_file_roundtrip_verified=True)
            (old / "native_trials.json").write_text("[]", encoding="utf-8")
            manifest_path = old / "boundary_manifest.json"
            manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
            source_before = {p.name: adaptive.sha256(p) for p in old.iterdir() if p.is_file()}

            def decode(_container, output, _exporter, _fixed, *, timeout):
                Path(output).write_bytes(raw.read_bytes())
                return dict(decoded_sha256=source_hash)

            with patch.object(adaptive.origin, "validate_repair"), patch.object(
                    adaptive.fmt, "decode_file", side_effect=decode):
                result = adaptive.resume_boundary(job, old, new, exporter, fixed, 10)
            self.assertEqual(result["repaired_regions"], [(0, 4)])
            self.assertTrue(result["independently_decoded_again"])
            self.assertEqual(source_before,
                             {p.name: adaptive.sha256(p) for p in old.iterdir() if p.is_file()})
            (old / "repaired.f64pool").write_bytes(b"tampered")
            with self.assertRaisesRegex(ValueError, "stored bytes differ"):
                adaptive.resume_boundary(job, old, new, exporter, fixed, 10)

    def test_resume_rejects_pool_external_ledger_even_if_manifest_claims_eight(self):
        with tempfile.TemporaryDirectory() as temporary:
            old, new, _raw, exporter, fixed, job, manifest = self._fixture(Path(temporary))
            manifest["boundary_trial_count"] = 1
            (old / "boundary_manifest.json").write_text(json.dumps(manifest), encoding="utf-8")
            (old / "native_trials.json").write_text(json.dumps([
                dict(start=0, length=4, codec="rle+chimp", status="passed")]), encoding="utf-8")
            with patch.object(adaptive.origin, "validate_repair"), patch.object(
                    adaptive.fmt, "decode_file") as decode:
                with self.assertRaisesRegex(ValueError, "pool-external"):
                    adaptive.resume_boundary(job, old, new, exporter, fixed, 10)
                decode.assert_not_called()

    def test_resume_rejects_actual_excluded_frame_even_with_updated_hash(self):
        with tempfile.TemporaryDirectory() as temporary:
            old, new, _raw, exporter, fixed, job, manifest = self._fixture(Path(temporary))
            path = old / "repaired.f64pool"
            data = bytearray(path.read_bytes())
            offset = adaptive.fmt.FILE_HEADER.size
            data[offset:offset + 4] = (20).to_bytes(4, "big")
            path.write_bytes(data)
            manifest["repaired_container_sha256"] = adaptive.sha256(path)
            (old / "boundary_manifest.json").write_text(json.dumps(manifest), encoding="utf-8")
            with patch.object(adaptive.fmt, "decode_file") as decode:
                with self.assertRaisesRegex(ValueError, "invalid flat-pool frame"):
                    adaptive.resume_boundary(job, old, new, exporter, fixed, 10)
                decode.assert_not_called()

    def test_resume_rejects_legacy14_manifest(self):
        with tempfile.TemporaryDirectory() as temporary:
            old, new, _raw, exporter, fixed, job, manifest = self._fixture(Path(temporary))
            manifest["boundary_actions"].append("rle+gorilla")
            (old / "boundary_manifest.json").write_text(json.dumps(manifest), encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "strict-eight"):
                adaptive.resume_boundary(job, old, new, exporter, fixed, 10)


if __name__ == "__main__":
    unittest.main()
