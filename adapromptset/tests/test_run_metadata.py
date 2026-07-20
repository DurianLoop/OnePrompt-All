import json
import uuid
import unittest
from pathlib import Path

from adapromptset.reproducibility.metadata import (
    canonical_config_hash,
    collect_run_metadata,
    write_run_metadata,
)

PROJECT_ROOT = Path(__file__).resolve().parents[1]


def _workspace_tmp(prefix: str) -> Path:
    path = PROJECT_ROOT / "tmp" / f"{prefix}_{uuid.uuid4().hex}"
    path.mkdir(parents=True)
    return path


class RunMetadataTests(unittest.TestCase):
    def test_canonical_config_hash_is_order_invariant(self) -> None:
        left = {"beta": [3, 1], "alpha": {"x": 1, "y": True}}
        right = {"alpha": {"y": True, "x": 1}, "beta": [3, 1]}

        self.assertEqual(canonical_config_hash(left), canonical_config_hash(right))
        self.assertEqual(len(canonical_config_hash(left)), 64)
        self.assertNotEqual(canonical_config_hash(left), canonical_config_hash({"alpha": {"x": 2}}))

    def test_collect_run_metadata_records_inputs_and_environment(self) -> None:
        root = _workspace_tmp("run_metadata_collect")
        config = root / "config.json"
        config.write_text('{"b": 2, "a": 1}\n', encoding="utf-8")
        manifest = root / "manifest.csv"
        manifest.write_text("case_id,image_sha256\ncase-1,abc\n", encoding="utf-8")

        metadata = collect_run_metadata(
            project_root=root,
            run_name="unit_run",
            config_paths=[config],
            input_paths=[manifest],
            extra={"gate": "A"},
        )

        self.assertEqual(metadata["schema_version"], "adapromptset.run_metadata.v1")
        self.assertTrue(metadata["run_id"].startswith("unit_run_"))
        self.assertEqual(metadata["extra"], {"gate": "A"})
        self.assertEqual(metadata["configs"][0]["relative_path"], "config.json")
        self.assertEqual(metadata["inputs"][0]["relative_path"], "manifest.csv")
        self.assertEqual(len(metadata["configs"][0]["sha256"]), 64)
        self.assertEqual(len(metadata["config_hash"]), 64)
        self.assertIn("python", metadata["environment"])
        self.assertIn("git", metadata)
        self.assertIn("dirty_file_count", metadata["git"])
        self.assertNotIn("status_porcelain", metadata["git"])
        serialized = json.dumps(metadata, sort_keys=True)
        self.assertNotIn("query_mask_path", serialized)
        self.assertNotIn("ground_truth", serialized.lower())

    def test_write_run_metadata_creates_json_file(self) -> None:
        root = _workspace_tmp("run_metadata_write")
        config = root / "config.json"
        config.write_text('{"a": 1}\n', encoding="utf-8")
        out_dir = root / "reports" / "run_metadata"

        out_path = write_run_metadata(
            out_dir=out_dir,
            project_root=root,
            run_name="smoke",
            config_paths=[config],
            input_paths=[],
            extra={"status": "PASS"},
        )

        payload = json.loads(out_path.read_text(encoding="utf-8"))

        self.assertEqual(out_path.name, "run_metadata.json")
        self.assertEqual(payload["extra"]["status"], "PASS")
        self.assertEqual(payload["configs"][0]["relative_path"], "config.json")


if __name__ == "__main__":
    unittest.main()
