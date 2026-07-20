from __future__ import annotations

import json
import sys
import unittest
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from adapromptset.matching.runtime import (
    build_feature_cache_key,
    checkpoint_record,
    configure_external_paths,
    infer_dino_architecture,
    infer_sam_architecture,
)
from adapromptset.scripts.audit_matcher_runtime import run_audit


class MatcherRuntimeTests(unittest.TestCase):
    def test_architecture_inference_matches_local_checkpoint_names(self) -> None:
        self.assertEqual(infer_dino_architecture(Path("dinov2_vitl14_pretrain.pth")), "large")
        self.assertEqual(infer_dino_architecture(Path("dinov2_vitb14_pretrain.pth")), "base")
        self.assertEqual(infer_sam_architecture(Path("sam_vit_b.pth")), "vit_b")
        self.assertEqual(infer_sam_architecture(Path("sam_vit_l_0b3195.pth")), "vit_l")
        self.assertEqual(infer_sam_architecture(Path("sam_vit_h_4b8939.pth")), "vit_h")

    def test_checkpoint_record_hashes_existing_file(self) -> None:
        tmp = PROJECT_ROOT / ".test_tmp" / "dinov2_vitl14_pretrain.pth"
        tmp.parent.mkdir(parents=True, exist_ok=True)
        tmp.write_bytes(b"checkpoint")

        record = checkpoint_record(tmp, role="matcher")

        self.assertEqual(record["role"], "matcher")
        self.assertEqual(record["exists"], True)
        self.assertEqual(record["sha256"], "47320987F9A49D5B00119B960F247A956773F57543982B8BFCB6DA5BB3AFD9EF")

    def test_feature_cache_key_excludes_query_ground_truth(self) -> None:
        key = build_feature_cache_key(
            dataset="MoNuSAC",
            episode_id="episode_a",
            support_image_sha256="support-image",
            support_mask_sha256="support-mask",
            query_image_sha256="query-image",
            matcher_checkpoint_sha256="matcher",
            matcher_name="dinov2_vitl14",
        )

        serialized = json.dumps(key, sort_keys=True)
        self.assertIn("support_mask_sha256", serialized)
        self.assertNotIn("query_mask", serialized)
        self.assertNotIn("query_gt", serialized)

    def test_runtime_audit_outputs_pending_gate_c(self) -> None:
        out_path = PROJECT_ROOT / ".test_tmp" / "matcher_runtime.json"
        result = run_audit(out_path=out_path, project_root=PROJECT_ROOT)

        payload = json.loads(out_path.read_text(encoding="utf-8"))
        self.assertEqual(result, 0)
        self.assertEqual(payload["formal_gate_c"], "PENDING_REAL_MATCHER_EVIDENCE")
        self.assertEqual(payload["mvp_matcher"]["architecture"], "large")
        self.assertEqual(payload["mvp_segmenter"]["architecture"], "vit_b")
        self.assertIn("python", payload["runtime"])
        self.assertIn("torch", payload["runtime"])

    def test_external_paths_are_reported_for_matcher_imports(self) -> None:
        paths = configure_external_paths(PROJECT_ROOT.parent)

        self.assertIn(str(PROJECT_ROOT.parent / "code"), paths)
        self.assertIn(str(PROJECT_ROOT.parent / "Matcher"), paths)


if __name__ == "__main__":
    unittest.main()
