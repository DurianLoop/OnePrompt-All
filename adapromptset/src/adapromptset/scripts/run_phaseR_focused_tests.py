"""Phase R focused tests — verify configs and manifests before preregistration commit."""

import csv
import json
import sys
from pathlib import Path

try:
    import yaml
except ImportError:
    print("ERROR: yaml not available")
    sys.exit(1)

ROOT = Path("D:/SAM/adapromptset")

def main():
    print("Phase R Focused Tests")
    print("=" * 50)

    errors = []

    # 1. Check all required files exist
    print("\n1. File existence checks...")
    required = [
        "manifests/phaseR/train_resampling_seed_0.csv",
        "manifests/phaseR/train_resampling_seed_1.csv",
        "manifests/phaseR/train_resampling_seed_2.csv",
        "manifests/phaseR/train_resampling_seed_3.csv",
        "manifests/phaseR/train_resampling_seed_4.csv",
        "manifests/phaseR/train_resampling_all_seeds.csv",
        "manifests/phaseR/validation_AB_fixed.csv",
        "manifests/phaseR/manifest_summary.json",
        "configs/frozen/phaseR_prompt_configs.yaml",
        "configs/frozen/phaseR_consensus_methods.yaml",
        "configs/frozen/phaseR_beta_tau_grid.yaml",
        "configs/frozen/phaseR_statistics.yaml",
        "docs/phaseR_goal_read_receipt.md",
        "docs/phaseR_problem_statement.md",
        "docs/phaseR_data_roles.md",
        "docs/phaseR_reference_resampling_protocol.md",
        "docs/phaseR_consensus_methods.md",
        "docs/phaseR_reference_reliability_protocol.md",
        "docs/phaseR_statistics_protocol.md",
        "docs/phaseR_claim_registry.md",
        "docs/phaseR_test_lock_notice.md",
        "reports/phaseR_gate_status.md",
        "reports/phaseR_compute_ledger.csv",
        "reports/phaseR_protocol_snapshot.json",
        "reports/phaseR_input_hashes.json",
    ]
    for rel in required:
        p = ROOT / rel
        if p.exists():
            print(f"  OK: {rel} ({p.stat().st_size} bytes)")
        else:
            print(f"  MISSING: {rel}")
            errors.append(rel)

    # 2. Verify manifest structure
    print("\n2. Manifest structure...")
    train = list(csv.DictReader(open(ROOT / "manifests/phaseR/train_resampling_all_seeds.csv")))
    val = list(csv.DictReader(open(ROOT / "manifests/phaseR/validation_AB_fixed.csv")))
    print(f"  Train manifest: {len(train)} rows (expect 2400)")
    print(f"  Val manifest: {len(val)} rows (expect 246)")
    if len(train) != 2400:
        errors.append("train manifest row count")
    if len(val) != 246:
        errors.append("val manifest row count")

    # 3. Verify frozen configs
    print("\n3. Frozen configs...")
    prompts = yaml.safe_load(open(ROOT / "configs/frozen/phaseR_prompt_configs.yaml"))
    print(f"  Prompt configs: {len(prompts['configs'])}")
    for c in prompts["configs"]:
        print(f"    {c['name']}: K+={c['k_positive']} K-={c['k_negative']}")
    if len(prompts["configs"]) != 3:
        errors.append("prompt config count")

    consensus = yaml.safe_load(open(ROOT / "configs/frozen/phaseR_consensus_methods.yaml"))
    print(f"  Consensus methods: {len(consensus['methods'])}")
    if len(consensus["methods"]) != 8:
        errors.append("consensus method count")

    grid = yaml.safe_load(open(ROOT / "configs/frozen/phaseR_beta_tau_grid.yaml"))
    print(f"  Beta grid: {grid['beta_grid']}")
    print(f"  Tau grid: {grid['tau_grid']}")
    if len(grid["beta_grid"]) != 3 or len(grid["tau_grid"]) != 3:
        errors.append("beta/tau grid")

    stats = yaml.safe_load(open(ROOT / "configs/frozen/phaseR_statistics.yaml"))
    print(f"  Gate R1 conditions: {len(stats['gate_r1'])} keys")
    print(f"  Gate R2 conditions: {len(stats['gate_r2'])} keys")
    print(f"  Gate R3 conditions: {len(stats['gate_r3'])} keys")
    print(f"  Gate R4 conditions: {len(stats['gate_r4'])} keys")

    # 4. Verify hashes
    print("\n4. Input hashes...")
    hashes = json.load(open(ROOT / "reports/phaseR_input_hashes.json"))
    n_files = len(hashes["files"])
    print(f"  Hashed files: {n_files}")
    missing_hashes = [k for k, v in hashes["files"].items() if not v.get("exists")]
    if missing_hashes:
        print(f"  WARNING: missing hash files: {missing_hashes}")
        errors.extend(missing_hashes)

    # 5. Verify R0 audit
    print("\n5. R0 audit files...")
    r0_dir = ROOT / "reports/phaseR_historical_audit"
    r0_files = list(r0_dir.glob("*"))
    print(f"  R0 audit files: {len(r0_files)}")
    for f in sorted(r0_files):
        print(f"    {f.name}")
    if len(r0_files) != 8:
        errors.append(f"R0 audit file count: {len(r0_files)} != 8")

    # Summary
    print("\n" + "=" * 50)
    if errors:
        print(f"FAILED: {len(errors)} errors")
        for e in errors:
            print(f"  - {e}")
        return 1
    else:
        print("ALL FOCUSED TESTS PASSED")
        return 0


if __name__ == "__main__":
    sys.exit(main())
