# Pathology Dataset Protocol Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add auditable, leakage-safe GlaS and CPM-17 manifests plus reference-query episodes for the Reference-Conditioned Prompt Policy study.

**Architecture:** Keep raw datasets under `external/research_data`, with one parser per source format. A shared protocol layer assigns policy-train, validation, test, and excluded-overlap roles before it creates episodes; supports always come from policy-train and never share a patient with a query.

**Tech Stack:** Python 3, standard-library `csv`/`json`/`pathlib`, pytest, Pillow and SciPy only for data validation/conversion where already installed.

## Global Constraints

- Preserve official source split labels in every manifest.
- Never use a source-test image, label, patient, or reference for validation choice or policy training.
- Where patient IDs exist, exclude every source-train case whose patient appears in official test.
- A reference must be policy-train and must differ from the query case; for patient-labelled data it must also differ from the query patient.
- Write split metadata, source URL, SHA-256, seed, and explicit leakage rules beside every output manifest.
- Do not report a dataset as usable until image/mask pairing and instance-label semantics are validated.

---

### Task 1: GlaS source audit and patient-disjoint assignment

**Files:**
- Create: `code/benchmarks/build_glass_strict_policy_splits.py`
- Create: `code/benchmarks/test_build_glass_strict_policy_splits.py`
- Create: `runs/research_direction/dataset_audit/glas_manifest.csv`
- Create: `runs/research_direction/prompt_policy_splits/glas_patient_disjoint/protocol_metadata.json`

**Interfaces:**
- Consumes: `Grade.csv`, source image directory, and `--source-root`.
- Produces: `build_glass_protocol(rows, val_patient_count, references_per_query, seed) -> (assignments, episodes, metadata)`.

- [ ] **Step 1: Write failing tests**

```python
def test_protocol_excludes_source_train_patients_seen_in_official_test():
    assignments, episodes, metadata = build_glass_protocol(rows, 1, 2, 7)
    assert {row['patient'] for row in assignments if row['role'] == 'train'}.isdisjoint(
        {row['patient'] for row in assignments if row['role'] == 'test'}
    )
    assert metadata['excluded_overlap_case_count'] == 2
```

- [ ] **Step 2: Run test to verify it fails**

Run: `D:\conda\python.exe -m pytest code/benchmarks/test_build_glass_strict_policy_splits.py -q -p no:cacheprovider`

Expected: import failure because the new protocol module does not exist.

- [ ] **Step 3: Implement the minimal parser and protocol builder**

```python
def build_glass_protocol(rows, val_patient_count, references_per_query, seed):
    official_test_patients = {row['patient'] for row in rows if row['source_split'] != 'train'}
    eligible = [row for row in rows if row['source_split'] == 'train' and row['patient'] not in official_test_patients]
    # assign validation by whole patient, then construct episodes from train-only supports
```

- [ ] **Step 4: Run tests and create manifest outputs**

Run: `D:\conda\python.exe -m pytest code/benchmarks/test_build_glass_strict_policy_splits.py -q -p no:cacheprovider`

Expected: all GlaS protocol tests pass; generated metadata reports source and role counts.

### Task 2: CPM-17 manifest and official-holdout protocol

**Files:**
- Create: `code/benchmarks/build_cpm17_strict_policy_splits.py`
- Create: `code/benchmarks/test_build_cpm17_strict_policy_splits.py`
- Create: `runs/research_direction/dataset_audit/cpm17_manifest.csv`

**Interfaces:**
- Consumes: `train/Images/*.png`, `train/Labels/*.mat`, `test/Images/*.png`, `test/Labels/*.mat`.
- Produces: a manifest with `case_id`, `source_split`, `image_path`, `mat_path`, `instance_mask_path`, and an official-test-preserved protocol.

- [ ] **Step 1: Write failing test for complete pairing and source-test preservation**

```python
def test_cpm17_protocol_rejects_missing_label_and_never_uses_official_test_as_support():
    with pytest.raises(ValueError, match='missing label'):
        build_cpm17_manifest(incomplete_source)
    _, episodes, _ = build_cpm17_protocol(complete_rows, val_count=4, references_per_query=3)
    assert all(row['support_source_split'] == 'train' for role in episodes.values() for row in role)
```

- [ ] **Step 2: Run test to verify it fails**

Run: `D:\conda\python.exe -m pytest code/benchmarks/test_build_cpm17_strict_policy_splits.py -q -p no:cacheprovider`

Expected: import failure because CPM-17 parser does not exist.

- [ ] **Step 3: Implement manifest audit and MAT instance conversion**

```python
def load_instance_ids(mat_path: Path) -> np.ndarray:
    payload = scipy.io.loadmat(mat_path)
    instance = next(value for key, value in payload.items() if not key.startswith('__') and value.ndim == 2)
    return np.asarray(instance, dtype=np.int32)
```

- [ ] **Step 4: Run tests and audit**

Run: `D:\conda\python.exe -m pytest code/benchmarks/test_build_cpm17_strict_policy_splits.py -q -p no:cacheprovider`

Expected: tests pass and audit fails loudly if raw download has fewer than 32/32 source pairs.

### Task 3: Cross-dataset protocol validator

**Files:**
- Create: `code/benchmarks/validate_prompt_policy_protocol.py`
- Create: `code/benchmarks/test_validate_prompt_policy_protocol.py`

**Interfaces:**
- Consumes: role assignments and episode CSVs from MoNuSeg, GlaS, and CPM-17.
- Produces: `validate_protocol(assignments, episodes) -> list[str]`, empty only for valid protocols.

- [ ] **Step 1: Write a failing leakage test**

```python
def test_validator_reports_test_support_and_patient_overlap():
    errors = validate_protocol(assignments_with_overlap, episodes_with_test_support)
    assert 'support is not policy-train' in errors
    assert 'patient occurs in more than one role' in errors
```

- [ ] **Step 2: Run failure test**

Run: `D:\conda\python.exe -m pytest code/benchmarks/test_validate_prompt_policy_protocol.py -q -p no:cacheprovider`

Expected: import failure because validator does not exist.

- [ ] **Step 3: Implement validation and emit a JSON audit**

```python
def validate_protocol(assignments, episodes):
    errors = []
    # verify one role per case, patient role disjointness when available,
    # train-only support, and support/query case inequality
    return errors
```

- [ ] **Step 4: Run full protocol test set**

Run: `D:\conda\python.exe -m pytest code/benchmarks/test_build_glass_strict_policy_splits.py code/benchmarks/test_build_cpm17_strict_policy_splits.py code/benchmarks/test_validate_prompt_policy_protocol.py -q -p no:cacheprovider`

Expected: all tests pass before any policy training starts.
