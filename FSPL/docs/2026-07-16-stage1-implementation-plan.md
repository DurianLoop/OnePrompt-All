# FSPL Stage 1 Implementation Plan

> **For agentic workers:** Execute inline in `D:\SAM\FSPL`; no existing research source files or result folders may be modified.

**Goal:** Build a reproducible MoNuSeg reference-guided prompt-policy foundation using the workspace's official images, masks, DINOv2-L checkpoint and SAM ViT-B checkpoint.

**Architecture:** `fspl.dataset` discovers official image-mask cases and produces leakage-free reference-query episodes. `fspl.candidates` turns matcher score maps into bounded foreground/background candidate pools. `scripts/run_stage1_smoke.py` loads frozen DINOv2-L and SAM ViT-B and saves a one-episode, fixed-rule baseline artifact; no learned policy is trained in Stage 1.

**Tech Stack:** Python 3, NumPy, Pillow, PyTorch, DINOv2, Segment Anything, pytest.

## Global Constraints

- Read data only from `D:\SAM\code\datasets\MonuSeg\official`.
- Read checkpoints only from `D:\SAM\assets\checkpoints\dinov2_vitl14_pretrain.pth` and `D:\SAM\assets\checkpoints\sam_vit_b.pth`.
- Create all source, tests, manifests and artifacts below `D:\SAM\FSPL`.
- Do not use test labels to select an action; Stage 1 only creates a deterministic fixed-rule smoke baseline.
- Use CPU-compatible code; `KMP_DUPLICATE_LIB_OK=TRUE` is set only for model invocation.

---

### Task 1: Official-case audit and deterministic episode protocol

**Files:**
- Create: `fspl/__init__.py`
- Create: `fspl/dataset.py`
- Create: `scripts/audit_monuseg.py`
- Test: `tests/test_dataset.py`

**Interfaces:**
- Produces `CaseRecord(case_id, image_path, mask_path, split)`.
- Produces `Episode(reference: CaseRecord, query: CaseRecord)`.
- `discover_cases(root: Path, split: str) -> list[CaseRecord]` requires matching image/mask stems.
- `build_smoke_episode(cases: Sequence[CaseRecord]) -> Episode` deterministically chooses the first case as reference and second as query.

- [ ] Write tests that reject a missing mask and verify the deterministic reference/query ordering.
- [ ] Run `python -m pytest tests/test_dataset.py -q`; expect failure because `fspl.dataset` does not exist.
- [ ] Implement `CaseRecord`, `Episode`, `discover_cases`, `build_smoke_episode` and CSV audit output.
- [ ] Re-run `python -m pytest tests/test_dataset.py -q`; expect all tests to pass.
- [ ] Run `python -B scripts/audit_monuseg.py --root D:\SAM\code\datasets\MonuSeg\official --out artifacts\monuseg_official_audit.csv`; expect 37 train and 14 test rows.

### Task 2: Foreground-background candidate pools

**Files:**
- Create: `fspl/candidates.py`
- Test: `tests/test_candidates.py`

**Interfaces:**
- `contrast_map(fg_scores: np.ndarray, bg_scores: np.ndarray, lambda_bg: float) -> np.ndarray` returns a z-normalized foreground-minus-background score map.
- `build_candidate_pools(fg_scores, bg_scores, patch_size, positive_limit, negative_limit) -> CandidatePools` returns bounded positive and negative `Candidate` sequences with distinct coordinates.

- [ ] Write tests that require a high-foreground/low-background patch to be positive, a high-background patch to be negative, and no coordinate to appear twice.
- [ ] Run `python -m pytest tests/test_candidates.py -q`; expect failure because `fspl.candidates` does not exist.
- [ ] Implement deterministic ranking, stable tie breaking and coordinate de-duplication.
- [ ] Re-run `python -m pytest tests/test_candidates.py -q`; expect all tests to pass.

### Task 3: Frozen-model smoke baseline

**Files:**
- Create: `fspl/models.py`
- Create: `fspl/stage1.py`
- Create: `scripts/run_stage1_smoke.py`
- Test: `tests/test_stage1.py`

**Interfaces:**
- `Stage1Config` contains absolute data/checkpoint paths, `positive_limit=5`, `negative_limit=3`, and CPU device defaults.
- `write_stage1_summary(path: Path, summary: dict) -> None` writes a JSON summary containing case IDs, candidate counts, requested points, Dice and elapsed time.
- `run_stage1(config: Stage1Config) -> dict` loads frozen DINOv2-L and SAM ViT-B, extracts query/reference features, builds fixed candidate pools, prompts SAM and returns the summary.

- [ ] Write a test that verifies `write_stage1_summary` preserves mandatory keys and JSON-serializable values without loading a model.
- [ ] Run `python -m pytest tests/test_stage1.py -q`; expect failure because `fspl.stage1` does not exist.
- [ ] Implement model loading and the single-episode inference path; use only the fixed candidate policy defined in `Stage1Config`.
- [ ] Re-run `python -m pytest tests/test_stage1.py -q`; expect all tests to pass.
- [ ] Run `python -B scripts/run_stage1_smoke.py --output artifacts\stage1_smoke.json` with `KMP_DUPLICATE_LIB_OK=TRUE` and `PYTHONPATH=D:\SAM;D:\SAM\Matcher;D:\SAM\FSPL`; expect a saved JSON artifact and one finite Dice score.

### Task 4: Verification and decision gate

**Files:**
- Read: `artifacts/monuseg_official_audit.csv`
- Read: `artifacts/stage1_smoke.json`

- [ ] Run `python -m pytest tests -q`; expect all Stage 1 tests to pass.
- [ ] Check the audit contains 37 train plus 14 test cases and no duplicate case IDs.
- [ ] Check the smoke JSON records a train-only reference, a distinct query, candidate counts no greater than 5/3, finite Dice and elapsed seconds.
- [ ] Stop before launching a full 37-case grid: select whether to (A) finish fixed-rule baselines across the official protocol, or (B) start train-only action-utility generation for adaptive positive selection.
