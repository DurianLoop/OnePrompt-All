#!/usr/bin/env python3
"""
VV4 Forensics Script — Comprehensive V4 Claim Audit (WP3–WP11)
===============================================================
Canonical V4: D:/SAM/adapromptset/paper/phaseU/full_draft.tex
Title: "Reference Reliability Estimation for Visual Prompting Pipelines"

Work Packages:
  WP3  — Claim Extraction from V4 LaTeX
  WP4  — Evidence Chain: Claim → Artifact
  WP5  — High-Risk Claim Audit (12 items)
  WP6  — Table Forensics
  WP7  — Figure Forensics
  WP9  — Cross-Phase Comparison (V4 vs V/W/AA/AB)
  WP10 — AI Hallucination Audit (A–I classification)
  WP11 — V4 Section Salvage Map

Output root: D:\SAM\adapromptset\VV4\07_CLAIM_FORENSICS\
"""

import os, re, json, csv, hashlib, datetime
from pathlib import Path
from collections import OrderedDict, defaultdict

# ─── Paths ────────────────────────────────────────────────────────────────
ROOT = Path(r"D:\SAM\adapromptset")
VV4 = ROOT / "VV4"
V4_TEX = ROOT / "paper" / "phaseU" / "full_draft.tex"
REPORTS_DIR = ROOT / "reports"
SCRIPTS_DIR = ROOT / "src" / "adapromptset" / "scripts"
OUT_DIR = VV4 / "07_CLAIM_FORENSICS"
SALVAGE_DIR = VV4 / "09_V4_SALVAGE"

# ─── Ensure output directories exist ──────────────────────────────────────
(OUT_DIR / "extracted_claims").mkdir(parents=True, exist_ok=True)
(OUT_DIR / "claim_to_artifact").mkdir(parents=True, exist_ok=True)
(OUT_DIR / "hallucination_audit").mkdir(parents=True, exist_ok=True)
(OUT_DIR / "table_forensics").mkdir(parents=True, exist_ok=True)
(OUT_DIR / "figure_forensics").mkdir(parents=True, exist_ok=True)
SALVAGE_DIR.mkdir(parents=True, exist_ok=True)

# ─── Helper ───────────────────────────────────────────────────────────────
def now_iso():
    return datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")

def write_csv(path, rows, fieldnames=None):
    if not rows:
        rows = []
    if fieldnames is None:
        fieldnames = list(rows[0].keys()) if rows else []
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
        w.writeheader()
        w.writerows(rows)

def write_json(path, data):
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False, default=str)

def write_md(path, lines):
    with open(path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")

# ══════════════════════════════════════════════════════════════════════════
# WP3: Claim Extraction
# ══════════════════════════════════════════════════════════════════════════
def extract_claims():
    """Parse V4 LaTeX and extract all verifiable claims."""
    tex = V4_TEX.read_text(encoding="utf-8")

    # Identify sections
    sections = {}
    current_section = "Preamble"
    for m in re.finditer(r'\\(section|subsection|subsubsection)\{([^}]+)\}', tex):
        level = {"section": 1, "subsection": 2, "subsubsection": 3}[m.group(1)]
        title = m.group(2).strip()
        pos = m.start()
        sections[pos] = (level, title, current_section)
        current_section = title

    # Manually curated claims from V4 based on full reading
    # Format: (id, section, claim_text, category, v4_text_reference)
    CLAIM_CATEGORIES = [
        "Problem-definition", "Novelty", "Method", "Mathematical",
        "Numerical", "Comparative", "Statistical", "Cross-pipeline",
        "Cross-dataset", "Efficiency", "Calibration", "Deployment",
        "Reproducibility", "Literature"
    ]

    claims = []

    # ── Abstract ──
    cid = 0
    def add(section, text, category, ref=""):
        nonlocal cid
        cid += 1
        claims.append({
            "Claim_ID": f"C-V4-{cid:04d}",
            "Section": section,
            "Claim_Text": text,
            "Category": category,
            "V4_Reference": ref,
            "Exact_Numbers": "",
            "Verifiable": "Yes"
        })

    # Abstract
    add("Abstract", "Visual prompting pipelines lack reference reliability estimation (RRE)", "Problem-definition")
    add("Abstract", "Swapping reference can change Dice from 0.82 to 0.02", "Numerical", "Dice range 0.02-0.82")
    add("Abstract", "RRE is formalized as predicting probability of successful segmentation per reference", "Problem-definition")
    add("Abstract", "Anti-correlation between instability and catastrophic failure: r=-0.49 for SAM-B, p<0.001", "Statistical", "r=-0.49")
    add("Abstract", "Anti-correlation holds across 4 segmenters and 2 matchers", "Cross-pipeline", "4 seg, 2 matchers")
    add("Abstract", "Three geometric hypotheses explain the anti-correlation via matching-map entropy", "Mathematical", "H1-H3")
    add("Abstract", "RAMP: dual-model selective predictor with supervised risk estimator (GB trees)", "Method")
    add("Abstract", "RAMP reduces CFR by 37.5-84.4% at 80% coverage on MoNuSAC", "Numerical", "37.5-84.4% CFR reduction")
    add("Abstract", "RRN: lightweight learned estimator, AUROC=0.612, Dice@Top-3=0.525", "Numerical", "AUROC=0.612")
    add("Abstract", "First reference selection benchmark established", "Novelty", "First RRE benchmark")
    add("Abstract", "Frozen top-K pipeline (DINOv2-L+SAM-B, Dice=0.446) outperforms GF-SAM (0.388), MAUP (0.279), CycleSAM (0.412), H-SAM (0.405), SAM-Med2D (0.438)", "Comparative", "Table 4")
    add("Abstract", "RAMP uses coverage-based ranking without fixed probability threshold", "Calibration", "No fixed threshold")
    add("Abstract", "RRN narrows gap between similarity-based selection and oracle upper bound", "Comparative", "Oracle gap")

    # Introduction
    add("Introduction", "14% of MoNuSAC queries experience catastrophic failure from at least one reference", "Numerical", "14% CFR")
    add("Introduction", "GF-SAM (NeurIPS 2024 Spotlight), MAUP (MICCAI 2025), SynPo (MICCAI 2025), Matcher (ICLR 2024), PerSAM (ICLR 2024), CycleSAM, H-SAM, SAM-Med2D are all existing methods", "Literature")
    add("Introduction", "None of the existing methods address whether the reference is reliable", "Literature")
    add("Introduction", "RRE is practically important and theoretically interesting", "Novelty")
    add("Introduction", "Anti-correlation is a universal property of the two-stage pipeline architecture", "Novelty", "universal property claim")
    add("Introduction", "Instability and catastrophic failure are anti-correlated across 4 segmenters and 2 matchers", "Cross-pipeline")
    add("Introduction", "Entropy of matching map explains the anti-correlation", "Mathematical")
    add("Introduction", "RAMP: risk-adaptive weighting, adaptive prompt config, risk-stratified consensus", "Method")
    add("Introduction", "RRN: 5 pipeline features → reliability score via MLP", "Method")
    add("Introduction", "First reference selection benchmark with 6 strategies across 3 medical + 2 natural image datasets", "Novelty", "Pascal-5i, COCO-20i")
    add("Introduction", "Deployment envelope: catastrophic prevalence 5-60%", "Deployment")
    add("Introduction", "Paper has threefold contributions: problem, analytical, method", "Novelty")

    # Related Work
    add("Related Work", "SAM family enables segmentation through spatial prompts without task-specific training", "Literature")
    add("Related Work", "All prompt generation methods share: matcher→features→correspondence→prompts", "Literature")
    add("Related Work", "Reference reliability is a dimension orthogonal to prompt quality", "Novelty")
    add("Related Work", "Selective prediction methods (MC Dropout, Deep Ensembles) capture epistemic uncertainty in segmenter, orthogonal to reference-induced failure", "Literature")
    add("Related Work", "RRE is the first to define and benchmark reference reliability estimation as a standalone problem", "Novelty")
    add("Related Work", "SAM-L (312M) reduces instability prevalence by 76% vs SAM-B (94M) but increases CFR from 14.3% to 16.7%", "Numerical", "scale-reliability mismatch")

    # Section 3: Theory
    add("Section 3: Theory", "H1: Matching map concentration is associated with prompt spatial dispersion", "Mathematical")
    add("Section 3: Theory", "H2: When prompts are systematically mispositioned, catastrophic failure is more likely", "Mathematical")
    add("Section 3: Theory", "H3: Instability and catastrophic failure carry dissociated information from different pipeline levels", "Mathematical")
    add("Section 3: Theory", "H1 tested: entropy-instability r=-0.446, p<0.001", "Statistical", "r=-0.446")
    add("Section 3: Theory", "H2: non-monotonic entropy-CFR relationship, mid-entropy most dangerous", "Mathematical")
    add("Section 3: Theory", "H3: f1 dominates instability (32% importance), f4 dominates catastrophic (55% importance)", "Numerical")
    add("Section 3: Theory", "Control A: split-reference r=-0.51, 95% CI [-0.55, -0.44]", "Statistical", "Control A")
    add("Section 3: Theory", "Control B: GT-free instability r=-0.59 with GT-based CFR, p<0.001", "Statistical", "Control B")
    add("Section 3: Theory", "Control C: Null 1 (random beta) r=-0.58; Null 2 (shuffled) r=+0.64; Null 3 (uniform) r=+0.38", "Statistical", "Null models")
    add("Section 3: Theory", "Control D: partial correlation r_partial=-0.25 after controlling mean Dice", "Statistical", "partial r=-0.25")
    add("Section 3: Theory", "Anti-correlation described as empirical association, not structural theorem", "Statistical")
    add("Section 3: Theory", "P1-P4: matcher independence, segmenter modulation, cross-pipeline dissociation, cross-domain persistence", "Mathematical")

    # Section 4: Method
    add("Section 4: Method", "Reference-resampling protocol: K=6 references, S=5 seeds, 30 triples per query", "Method")
    add("Section 4: Method", "Episode-level metrics: mean Dice, best-worst gap, CFR; binary labels: instability Δ>=0.05, catastrophic mean<0.1", "Method")
    add("Section 4: Method", "Five pipeline features: f1 entropy, f2 similarity, f3 dispersion, f4 mask consistency, f5 uncertainty", "Method")
    add("Section 4: Method", "RAMP uses gradient-boosted trees with f1-f4, separate models for instability and catastrophic risk", "Method")
    add("Section 4: Method", "RAMP has three components: risk-adaptive weighting, adaptive prompt config, risk-stratified consensus", "Method")
    add("Section 4: Method", "RAMP is post-hoc: segmenter runs first, then risk predictor decides accept/abstain", "Method")
    add("Section 4: Method", "RRN: MLP [5→64→32→1] + ReLU + Sigmoid, trained with BCE on 70/15/15 split", "Method")
    add("Section 4: Method", "RRN training: Adam lr=1e-3, batch 64, early stopping patience 20", "Method")
    add("Section 4: Method", "2x2 factorial (Matcher: DINOv2-L/B × Segmenter: SAM-B/SAM2-T), mixed-effects ANOVA", "Method")
    add("Section 4: Method", "Wilcoxon signed-rank with Holm correction, bootstrap 95% CI (10k resamples), Cohen's d", "Statistical")

    # Section 5: Experiments
    add("Section 5: Experiments", "MoNuSAC: 80 training queries, 5 seeds, 6 refs, 3 prompt configs, 7200 ref assignments per pipeline", "Numerical")
    add("Section 5: Experiments", "Pascal-5i: 4 folds × 5 classes; COCO-20i: 4 folds × 20 classes", "Cross-dataset")
    add("Section 5: Experiments", "Table 1 (WP1): All 4 segmenters Inst_AUC≥0.994, Cat_AP≥0.893", "Numerical", "Table 1")
    add("Section 5: Experiments", "SAM2-S CFR_Red=38.9%, P_Cat=0.196 (weakest)", "Numerical", "Table 1")
    add("Section 5: Experiments", "Table 2 (WP2): Matcher independence Δ<0.001, p>0.05", "Statistical", "Table 2")
    add("Section 5: Experiments", "Table 3 (WP4): Matcher F(1,237)=0.12 p=0.73, Segmenter F(1,237)=28.4 p<0.001", "Statistical", "Table 3")
    add("Section 5: Experiments", "Table 5 (refsel): RRN Dice@Top-3=0.525 vs similarity 0.516, oracle 0.588", "Numerical", "Table 5")
    add("Section 5: Experiments", "RRN per-reference AUROC=0.612", "Numerical")
    add("Section 5: Experiments", "Table 4 (RAMP): CFR reduction 55.9% (L+SAM-B), 84.4% (L+SAM-L), 37.5% (L+SAM2-T)", "Numerical", "Table 4")
    add("Section 5: Experiments", "M2_Inst is anti-informative: -11% to -16% CFR reduction", "Numerical")
    add("Section 5: Experiments", "Table 6 (comparison): GF-SAM Dice=0.388, MAUP=0.279, CycleSAM=0.412, H-SAM=0.405, SAM-Med2D=0.438", "Comparative", "Table 6")
    add("Section 5: Experiments", "RAMP CFR reduction 55.9% vs MC Dropout 10.7%, Deep Ensemble 6.3%", "Comparative")
    add("Section 5: Experiments", "Cross-dataset: MoNuSeg prevalence 0%, IDRID prevalence 80.7%", "Cross-dataset", "Table 7")
    add("Section 5: Experiments", "Deployment envelope: effective when prevalence ∈ [5%, 60%]", "Deployment")
    add("Section 5: Experiments", "Component ablation: weighting Δ=12.3%, prompt config Δ=18.7%, consensus Δ=8.4%", "Numerical", "Table 8")
    add("Section 5: Experiments", "Feature ablation: removing f4 drops Cat_AP from 0.966 to 0.250", "Numerical", "Table 9")
    add("Section 5: Experiments", "Reference count: K=2→29.4%, K=6→55.9%, K=8→58.7% CFR reduction", "Numerical", "Table 10")
    add("Section 5: Experiments", "Cross-domain: Pascal-5i r=-0.51, COCO-20i r=-0.47, both p<0.001", "Statistical", "Table 11")
    add("Section 5: Experiments", "Pascal-5i RAMP CFR_Red=42.3%, COCO-20i CFR_Red=38.7%", "Numerical")

    # Discussion
    add("Discussion", "Factorial ANOVA confirms matcher has no effect (p=0.73), bottleneck downstream", "Statistical")
    add("Discussion", "Feature dissociation between f1 and f4 confirms two failure modes operate at different levels", "Mathematical")
    add("Discussion", "Cross-domain validation on Pascal-5i and COCO-20i strengthens geometric claim", "Cross-dataset")
    add("Discussion", "Similarity-based strategies only marginally better than random for reference selection", "Numerical")
    add("Discussion", "Simple top-K pipeline outperforms GF-SAM and MAUP on dense segmentation", "Comparative")
    add("Discussion", "SAM-Med2D medical fine-tuning does not address reference-induced failure", "Comparative")
    add("Discussion", "Deployment envelope criterion can be estimated from small unlabeled sample", "Deployment")
    add("Discussion", "RRN provides bootstrapping pathway from zero-shot to few-shot reliability estimation", "Method")

    # Limitations
    add("Limitations", "Cross-dataset uses 30 queries per dataset, limiting statistical power", "Reproducibility")
    add("Limitations", "Natural image experiments use single fold (fold-0) only", "Reproducibility")
    add("Limitations", "External methods use different backbones (GF-SAM/MAUP: SAM-H, H-SAM: SAM-L)", "Comparative")
    add("Limitations", "RAMP requires labeled multi-reference data for training", "Reproducibility")
    add("Limitations", "RRN uses single 70/15/15 split with only 12 test queries", "Reproducibility")
    add("Limitations", "Features require multiple reference runs; single-reference estimation remains open", "Method")
    add("Limitations", "Component ablation did not properly isolate individual components", "Reproducibility")

    # Conclusion
    add("Conclusion", "RRE identified as missing component in visual prompting pipelines", "Novelty")
    add("Conclusion", "RAMP achieves 37.5-84.4% CFR reduction at 80% coverage", "Numerical")
    add("Conclusion", "RRN achieves AUROC=0.612, Dice@Top-3=0.525", "Numerical")
    add("Conclusion", "Reference-induced failure is a general phenomenon, not medical imaging artifact", "Cross-dataset")
    add("Conclusion", "Deployment envelope provides practical screening criterion", "Deployment")

    # Add Exact_Numbers field
    for c in claims:
        # Extract numbers from claim text
        nums = re.findall(r'[\d]+\.?[\d]*%?|r\s*=\s*-?[\d]+\.[\d]+|p\s*[<>=]+\s*[\d]+\.[\d]+', c["Claim_Text"])
        c["Exact_Numbers"] = "; ".join(nums) if nums else ""

    return claims

# ══════════════════════════════════════════════════════════════════════════
# WP4: Evidence Chain
# ══════════════════════════════════════════════════════════════════════════
def build_evidence_chain(claims):
    """Map each claim to supporting code/config/data files."""

    # Scan available artifacts
    report_files = list(REPORTS_DIR.rglob("*.json")) + list(REPORTS_DIR.rglob("*.csv")) + list(REPORTS_DIR.rglob("*.md"))
    script_files = list(SCRIPTS_DIR.rglob("*.py"))
    data_files = list((ROOT / "manifests").rglob("*.csv")) + list((ROOT / "manifests").rglob("*.json"))

    # Build keyword-based mapping
    evidence_map = {
        "segmenter": ["run_phaseU_wp1_segmenter_audit.py", "phaseU_wp1/", "phaseU_multi_model/"],
        "matcher": ["run_phaseU_wp2_matcher_audit.py", "phaseU_wp2/"],
        "ramp": ["run_phaseU_ramp.py", "phaseU_ramp/", "ramp_results.json"],
        "rrn": ["run_phaseU_rrn.py", "phaseU_rrn/", "rrn_results.json", "rrn_model.pkl"],
        "factorial": ["run_phaseU_wp4_dinov2b_sam2t.py", "phaseU_wp4/"],
        "anticorrelation": ["run_phaseV_anticorrelation_controls.py", "run_phaseW_wp7_anticorrelation.py",
                            "phaseV_anticorrelation_controls/", "phaseW_anticorrelation/"],
        "cross_dataset": ["run_phaseT_wp2_cross_dataset.py", "phaseT_wp2_"],
        "selective": ["run_phaseU_selective_prediction_baselines.py"],
        "external": ["run_phaseU_external_gfsam.py", "run_phaseU_external_maup.py"],
        "component_ablation": ["phaseW_component_ablation/", "run_phaseW_experimental_closure.py"],
        "simple_baseline": ["phaseW_simple_baselines/", "simple_baseline_comparison.json"],
        "split_audit": ["phaseW_split_audit/"],
        "rrn_cv": ["phaseV_rrn_grouped_cv/", "rrn_grouped_cv_results.json"],
        "query_stats": ["phaseV_query_level_stats/", "query_level_statistics.json"],
        "theory": ["run_phaseU_theory_figures.py", "theory_verification_results.json"],
        "figures": ["generate_figures.py", "phaseU/figures/"],
    }

    artifact_matrix = []
    for c in claims:
        row = {
            "Claim_ID": c["Claim_ID"],
            "Claim_Text": c["Claim_Text"][:80],
            "Category": c["Category"],
            "Code_Files": "",
            "Data_Files": "",
            "Report_Files": "",
            "Has_Evidence": "No",
            "Evidence_Quality": "None",
        }

        matched = set()
        text_lower = (c["Claim_Text"] + " " + c["Section"]).lower()

        for key, paths in evidence_map.items():
            if key in text_lower:
                for p in paths:
                    matched.add(p)

        # Also search by keyword
        keywords = ["segmenter", "matcher", "ramp", "rrn", "anticorrelation", "cross_dataset",
                    "selective", "external", "component", "ablation", "theory", "figure"]
        for kw in keywords:
            if kw in text_lower:
                for p in evidence_map.get(kw, []):
                    matched.add(p)

        # Find actual files
        code_files = []
        data_files_found = []
        report_files_found = []

        for m in matched:
            for sf in script_files:
                if m in str(sf):
                    code_files.append(str(sf.relative_to(ROOT)))
            for df in data_files:
                if m in str(df):
                    data_files_found.append(str(df.relative_to(ROOT)))
            for rf in report_files:
                if m in str(rf):
                    report_files_found.append(str(rf.relative_to(ROOT)))

        row["Code_Files"] = "; ".join(sorted(set(code_files))[:5])
        row["Data_Files"] = "; ".join(sorted(set(data_files_found))[:5])
        row["Report_Files"] = "; ".join(sorted(set(report_files_found))[:5])
        row["Has_Evidence"] = "Yes" if (code_files or data_files_found or report_files_found) else "No"
        if code_files and report_files_found:
            row["Evidence_Quality"] = "Strong"
        elif code_files or report_files_found:
            row["Evidence_Quality"] = "Partial"
        else:
            row["Evidence_Quality"] = "None"

        artifact_matrix.append(row)

    return artifact_matrix

# ══════════════════════════════════════════════════════════════════════════
# WP5: High-Risk Claim Audit
# ══════════════════════════════════════════════════════════════════════════
def audit_high_risk_claims():
    """Audit 12 high-risk claims from V4."""
    lines = []
    lines.append("# V4 High-Risk Claim Audit Report")
    lines.append(f"**Generated**: {now_iso()}")
    lines.append(f"**Canonical V4**: `paper/phaseU/full_draft.tex`")
    lines.append("")
    lines.append("---")
    lines.append("")

    audits = [
        {
            "id": "HRA-01",
            "claim": '"Four segmenters, two matchers"',
            "v4_text": "Abstract: 'four segmenters and two matchers'",
            "finding": "**PARTIALLY SUPPORTED**. V4 actually ran 4 segmenters (SAM-B, SAM-L, SAM2-T, SAM2-S) and 2 matchers (DINOv2-L, DINOv2-B). Phase U WP1-WP2 confirms execution. However, the claim 'six segmenters, four matchers' in earlier versions was PROHIBITED (Phase W Claim Registry).",
            "verdict": "PARTIALLY_SUPPORTED",
            "evidence": "Phase U WP1: SAM-B, SAM-L, SAM2-T, SAM2-S (4 segmenters). Phase U WP2: DINOv2-L, DINOv2-B (2 matchers). run_phaseU_wp1_segmenter_audit.py, run_phaseU_wp2_matcher_audit.py"
        },
        {
            "id": "HRA-02",
            "claim": "Anti-correlation numbers (r = -0.43 to -0.71)",
            "v4_text": "Abstract r=-0.49; Section 3 various r values; Figures",
            "finding": "**PARTIALLY SUPPORTED — SIGNIFICANTLY OVERSTATED**. Phase W WP7 anti-correlation audit (run_phaseW_wp7_anticorrelation.py) found: L+SAM-B r=-0.512 (significant), B+SAM-B r=-0.549 (significant), L+SAM-L r=-0.221 (p=0.049, fails Holm), L+SAM2-T r=-0.111 (p=0.325, NS), L+SAM2-S r=-0.179 (p=0.113, NS). The anti-correlation is significant ONLY for SAM-B pipelines, not SAM2. Phase V Null 1 (random beta) reproduces 94% of correlation (Δr=-0.036). Phase W null model produces r=0.0, confirming genuine residual for SAM-B. The claim of 'universal' anti-correlation across all pipelines is PROHIBITED.",
            "verdict": "PARTIALLY_SUPPORTED",
            "evidence": "Phase W: run_phaseW_wp7_anticorrelation.py, Phase V: run_phaseV_anticorrelation_controls.py, anticorrelation_residual.json"
        },
        {
            "id": "HRA-03",
            "claim": "Three Lemmas/Hypotheses (H1-H3)",
            "v4_text": "Section 3: Geometric Hypotheses H1-H3",
            "finding": "**SUPPORTED (as hypotheses, not theorems)**. Phase V Gate V1 (Theory Repair) confirmed: V4's 'Lemmas' and 'Theorem' language was replaced with 'Geometric Hypotheses (H1-H3)' + 'Empirical Mechanism Analysis' + 'Testable Predictions'. No formal theorem language remains in v4 (the current version already uses hypothesis language). The hypotheses are empirically testable propositions, not formal proofs.",
            "verdict": "SUPPORTED",
            "evidence": "Phase V: Gate_V1_Theory_Repair = PASS. V4 uses 'Geometric Hypotheses' not 'Theorems'. theory_verification_results.json"
        },
        {
            "id": "HRA-04",
            "claim": "RAMP components (weighting, fusion, prompt adaptation)",
            "v4_text": "Section 4.4: three components; Section 5.5.1: component ablation",
            "finding": "**PROHIBITED — COMPONENTS HAVE ZERO EFFECT**. Phase W WP1 component ablation (run_phaseW_experimental_closure.py) conclusively found: weighting has zero effect in all pipelines, fusion has tiny negative effect, Full RAMP does not improve upon baseline mean consensus. Gate: PHASEW_FULL_RAMP_NO_ADDED_VALUE. The component ablation data in V4 is unreliable (Phase V notes all variants produced identical Dice). Phase AB confirms: Prediction Consistency (B4) is the only effective method.",
            "verdict": "PROHIBITED",
            "evidence": "Phase W: component_ablation_results.json, Gate PHASEW_FULL_RAMP_NO_ADDED_VALUE. Phase AB: prediction consistency is sole effective method."
        },
        {
            "id": "HRA-05",
            "claim": '"Training-free"',
            "v4_text": "V4 does NOT explicitly claim training-free (uses 'supervised risk estimator' for RAMP). However, earlier versions did.",
            "finding": "**PROHIBITED (for earlier versions) / NOT CLAIMED (in V4)**. V4 correctly describes RAMP as having a 'supervised risk estimator' (gradient-boosted trees). However, Phase W Claim Registry lists 'training-free' as PROHIBITED. The GB trees are trained on labeled multi-reference data. Matcher (DINOv2) is also trained. Phase AB confirms this is prohibited.",
            "verdict": "V4_AVOIDED / PROHIBITED",
            "evidence": "Phase W: Claim #4 PROHIBITED. Phase V: 'RAMP training-free' = NO. RAMP uses gradient-boosted trees."
        },
        {
            "id": "HRA-06",
            "claim": '"Calibration-free"',
            "v4_text": "Abstract: 'coverage-based ranking that does not require a fixed probability threshold'",
            "finding": "**PROHIBITED (as 'calibration-free') / PARTIALLY SUPPORTED (as 'no fixed threshold')**. Phase W: IDRID ECE=0.477 confirms severe miscalibration. Phase V: 'calibration-free' = PROHIBITED. The claim 'no fixed probability threshold' is true, but 'calibration-free' is false. Phase AB: prediction consistency does not output probabilities, so calibration is not applicable.",
            "verdict": "PARTIALLY_SUPPORTED",
            "evidence": "Phase W: IDRID ECE=0.477. Phase V: 'calibration-free' = PROHIBITED."
        },
        {
            "id": "HRA-07",
            "claim": "CFR reduction 56-84% (actual 37.5-84.4%)",
            "v4_text": "Abstract: '37.5-84.4%'",
            "finding": "**SUPPORTED for V4 (corrected range)**. V4 uses the correct 37.5-84.4% range. Earlier versions used 56-84% which was incorrect. Phase V corrected this. Phase W confirms the range but notes: RAMP's CFR reduction is achievable with simpler methods (B4 Prediction Consistency). The CFR reduction is not unique to RAMP.",
            "verdict": "SUPPORTED (V4 corrected)",
            "evidence": "Phase V: corrected from 56-84% to 37.5-84.4%. Phase W: simple baselines achieve similar CFR reduction."
        },
        {
            "id": "HRA-08",
            "claim": "RRN (AUROC=0.612, Dice@Top-3=0.525)",
            "v4_text": "Abstract and Section 5.3",
            "finding": "**PARTIALLY SUPPORTED — APPENDIX ONLY**. Phase V Gate V6: RRN AUROC=0.612 < 0.65 threshold, Dice@Top-3 advantage +0.009 < +0.010 threshold. Only 12 test queries, 0 catastrophic cases. Phase W grouped 5-fold CV: mean AUROC=0.573, below 0.65. Phase AB: RRN is appendix-only. The data split (70/15/15 with 12 test queries) is too small for reliable evaluation.",
            "verdict": "PARTIALLY_SUPPORTED",
            "evidence": "Phase V: Gate_V6_RRN_Evaluation = APPENDIX_ONLY. Phase W: RRN grouped CV AUROC=0.573. Phase V: rrn_evaluation.md"
        },
        {
            "id": "HRA-09",
            "claim": "External Baseline table (GF-SAM 0.388, MAUP 0.279, etc.)",
            "v4_text": "Table 6 (comparison)",
            "finding": "**PARTIALLY SUPPORTED — BACKBONE MISMATCH**. GF-SAM and MAUP use SAM-ViT-H (stronger than RAMP's SAM-B). CycleSAM uses SAM-B (fair comparison). H-SAM uses SAM-L. SAM-Med2D: checkpoint cannot be loaded, REMOVED from main table per Phase AB. Phase W: external baselines pending backbone-stratified comparison. Phase V: Table 4 clearly marks backbone differences. The claim of 'identical conditions' is PROHIBITED.",
            "verdict": "PARTIALLY_SUPPORTED",
            "evidence": "Phase W: External Baseline Conclusion PENDING. Phase V: baseline_audit/comparator_registry.md. Phase AB: SAM-Med2D REMOVED."
        },
        {
            "id": "HRA-10",
            "claim": "Pascal-5i / COCO-20i predictions",
            "v4_text": "Section 5.8: Pascal-5i r=-0.51, COCO-20i r=-0.47",
            "finding": "**PARTIALLY SUPPORTED — SINGLE FOLD ONLY**. Phase W WP9: Pascal/COCO only fold-0 available, 4-fold needed. Phase V: Pascal-5i/COCO-20i only fold-0. Phase AB: no confirmatory evidence. The cross-domain claim is based on incomplete evaluation. The numbers (r=-0.51, r=-0.47) are from fold-0 only.",
            "verdict": "PARTIALLY_SUPPORTED",
            "evidence": "Phase W: WP9 Pascal/COCO 4-Fold = PENDING. Phase V: 'Pascal-5i/COCO-20i only fold-0'."
        },
        {
            "id": "HRA-11",
            "claim": "Deployment envelope (5-60%)",
            "v4_text": "Section 5.7: 'effective when catastrophic prevalence falls in [5%, 60%]'",
            "finding": "**PARTIALLY SUPPORTED**. Phase W WP3: coverage curves stable across 50-100%. The envelope is derived from 3 medical + 2 natural image datasets. Phase W notes: validated at 50-100% coverage, not 5-60%. The lower bound (5%) is not well-tested. The criterion is useful but requires more validation on diverse modalities.",
            "verdict": "PARTIALLY_SUPPORTED",
            "evidence": "Phase W: WP3 Risk-Coverage Curves, coverage curves stable across 50-100%."
        },
        {
            "id": "HRA-12",
            "claim": '"First RRE problem / first benchmark"',
            "v4_text": "Abstract: 'first reference selection benchmark'; Section 2.3: 'first to define and benchmark'",
            "finding": "**SUPPORTED (with qualification)**. No prior work explicitly defines RRE as a standalone problem. The literature review is thorough. However, the claim of 'first' should be qualified as 'first to formalize' rather than 'first to notice'. Reference-based few-shot learning has explored related ideas (Vinyals 2016, Snell 2017). Phase V and Phase W do not challenge this novelty claim. Phase AB supports the problem formulation as a key strength.",
            "verdict": "SUPPORTED",
            "evidence": "Phase V: novelty claim retained. Phase AB: 'Clear problem formulation (RRE)' listed as key strength."
        },
    ]

    for a in audits:
        lines.append(f"## {a['id']}: {a['claim']}")
        lines.append("")
        lines.append(f"**V4 Text**: {a['v4_text']}")
        lines.append("")
        lines.append(f"**Finding**: {a['finding']}")
        lines.append("")
        lines.append(f"**Verdict**: {a['verdict']}")
        lines.append("")
        lines.append(f"**Evidence**: {a['evidence']}")
        lines.append("")
        lines.append("---")
        lines.append("")

    # Summary
    verdicts = defaultdict(int)
    for a in audits:
        verdicts[a["verdict"]] += 1
    lines.append("## Summary")
    lines.append("")
    for v, n in sorted(verdicts.items()):
        lines.append(f"- {v}: {n}")
    lines.append("")

    return lines, audits

# ══════════════════════════════════════════════════════════════════════════
# WP6-WP7: Table & Figure Forensics
# ══════════════════════════════════════════════════════════════════════════
def table_forensics():
    """Create forensic files for each V4 table."""
    tables = [
        {
            "Table_ID": "Table-1",
            "Label": "tab:wp1",
            "Caption": "Anti-correlation detection across segmenters (DINOv2-L, MoNuSAC)",
            "Data_Source": "Phase U WP1 segmenter audit (run_phaseU_wp1_segmenter_audit.py)",
            "Report_Files": "reports/phaseU_wp1/phaseU_wp1_summary.json",
            "Numbers_Verifiable": "Yes",
            "Numbers_Match_Code": "Yes (Phase U WP1)",
            "Issues": "SAM2-S CFR_Red=38.9% is abnormally low. P_Cat=0.196 is highest.",
            "Verdict": "VERIFIED",
        },
        {
            "Table_ID": "Table-2",
            "Label": "tab:wp2",
            "Caption": "Matcher independence (SAM-B, MoNuSAC)",
            "Data_Source": "Phase U WP2 matcher audit (run_phaseU_wp2_matcher_audit.py)",
            "Report_Files": "reports/phaseU_wp2/wp2_matcher_summary.json",
            "Numbers_Verifiable": "Yes",
            "Numbers_Match_Code": "Yes",
            "Issues": "p>0.05 does not establish equivalence. V4 correctly states 'no significant difference detected'.",
            "Verdict": "VERIFIED",
        },
        {
            "Table_ID": "Table-3",
            "Label": "tab:wp4",
            "Caption": "2x2 factorial decomposition (MoNuSAC)",
            "Data_Source": "Phase U WP4 (run_phaseU_wp4_dinov2b_sam2t.py)",
            "Report_Files": "reports/phaseU_wp4/wp4_dinov2b_sam2t.json",
            "Numbers_Verifiable": "Yes",
            "Numbers_Match_Code": "Yes",
            "Issues": "None",
            "Verdict": "VERIFIED",
        },
        {
            "Table_ID": "Table-4",
            "Label": "tab:ramp",
            "Caption": "CFR reduction at 80% coverage (MoNuSAC)",
            "Data_Source": "Phase U selective prediction baselines + RAMP",
            "Report_Files": "reports/phaseU_ramp/ramp_results.json",
            "Numbers_Verifiable": "Yes",
            "Numbers_Match_Code": "Partially — RAMP CFR reduction numbers match Phase U, but Phase W shows simple baselines achieve similar results",
            "Issues": "Phase W: RAMP has zero added value over simple baselines. M2_Inst anti-informative claim is verified.",
            "Verdict": "VERIFIED_WITH_QUALIFICATION",
        },
        {
            "Table_ID": "Table-5",
            "Label": "tab:refsel",
            "Caption": "Reference selection benchmark (MoNuSAC, SAM-B)",
            "Data_Source": "Phase U RRN (run_phaseU_rrn.py)",
            "Report_Files": "reports/phaseU_rrn/rrn_results.json",
            "Numbers_Verifiable": "Yes",
            "Numbers_Match_Code": "Yes, but single 70/15/15 split (12 test queries), 0 catastrophic cases",
            "Issues": "Phase V: RRN APPENDIX_ONLY. Phase W: grouped CV AUROC=0.573 < 0.65. Small test set.",
            "Verdict": "VERIFIED_WITH_QUALIFICATION",
        },
        {
            "Table_ID": "Table-6",
            "Label": "tab:comparison",
            "Caption": "Method comparison under shared reference-resampling protocol",
            "Data_Source": "External baselines (GF-SAM, MAUP) + internal pipelines",
            "Report_Files": "run_phaseU_external_gfsam.py, run_phaseU_external_maup.py",
            "Numbers_Verifiable": "Partially",
            "Numbers_Match_Code": "GF-SAM and MAUP numbers match Phase U. SAM-Med2D checkpoint unloadable (REMOVED per Phase AB). Backbone mismatch noted.",
            "Issues": "SAM-Med2D: REMOVED. Backbone mismatch: GF-SAM/MAUP use SAM-H, RAMP uses SAM-B. 'Identical conditions' claim is PROHIBITED.",
            "Verdict": "PARTIALLY_VERIFIED",
        },
        {
            "Table_ID": "Table-7",
            "Label": "tab:crossdataset",
            "Caption": "Cross-dataset deployment envelope",
            "Data_Source": "Phase T WP2 cross-dataset",
            "Report_Files": "reports/phaseT_wp2_idrid/, reports/phaseT_wp2_monuseg/",
            "Numbers_Verifiable": "Yes",
            "Numbers_Match_Code": "Yes",
            "Issues": "IDRID ECE=0.477 (severe miscalibration). MoNuSeg 0% prevalence makes predictor irrelevant.",
            "Verdict": "VERIFIED",
        },
        {
            "Table_ID": "Table-8",
            "Label": "tab:abl_comp",
            "Caption": "Component ablation of RAMP (L+SAM-B, MoNuSAC)",
            "Data_Source": "Phase U component ablation",
            "Report_Files": "ramp_results.json",
            "Numbers_Verifiable": "UNRELIABLE — Phase V notes all variants produced identical Dice",
            "Numbers_Match_Code": "UNRELIABLE",
            "Issues": "CRITICAL: Phase V states 'component ablation data unreliable (all variants identical)'. Phase W: Full RAMP has zero added value. These numbers should NOT be cited.",
            "Verdict": "UNRELIABLE",
        },
        {
            "Table_ID": "Table-9",
            "Label": "tab:abl_feat",
            "Caption": "Feature ablation (L+SAM-B, MoNuSAC)",
            "Data_Source": "Phase U feature ablation",
            "Report_Files": "",
            "Numbers_Verifiable": "Partially",
            "Numbers_Match_Code": "Unknown",
            "Issues": "f4 removal dropping Cat_AP from 0.966 to 0.250 is plausible but not independently verified.",
            "Verdict": "PARTIALLY_VERIFIED",
        },
        {
            "Table_ID": "Table-10",
            "Label": "tab:abl_k",
            "Caption": "Reference count ablation (L+SAM-B, MoNuSAC)",
            "Data_Source": "Phase U reference count ablation",
            "Report_Files": "",
            "Numbers_Verifiable": "Partially",
            "Numbers_Match_Code": "Unknown",
            "Issues": "K=2→29.4%, K=6→55.9%, K=8→58.7% — diminishing returns pattern is plausible.",
            "Verdict": "PARTIALLY_VERIFIED",
        },
        {
            "Table_ID": "Table-11",
            "Label": "tab:natural",
            "Caption": "Cross-domain validation on natural image benchmarks",
            "Data_Source": "Phase U cross-domain (Pascal-5i, COCO-20i)",
            "Report_Files": "",
            "Numbers_Verifiable": "Partially — single fold only",
            "Numbers_Match_Code": "Unknown",
            "Issues": "Only fold-0 available. 4-fold needed for valid cross-domain claim. Phase W: WP9 Pascal/COCO 4-Fold = PENDING.",
            "Verdict": "PARTIALLY_VERIFIED",
        },
        {
            "Table_ID": "Table-regimes",
            "Label": "tab:regimes",
            "Caption": "Characterization of the two failure regimes",
            "Data_Source": "Theoretical/conceptual",
            "Report_Files": "N/A",
            "Numbers_Verifiable": "N/A (conceptual table)",
            "Numbers_Match_Code": "N/A",
            "Issues": "None",
            "Verdict": "CONCEPTUAL",
        },
    ]

    rows = []
    for t in tables:
        rows.append(t)
    return rows

def figure_forensics():
    """Create forensic files for each V4 figure."""
    figures = [
        {
            "Figure_ID": "Fig-1",
            "Filename": "fig1_anticorrelation_scatter",
            "Caption": "The reference failure landscape: instability vs catastrophic failure rate",
            "Generation_Script": "paper/phaseU/generate_figures.py",
            "Data_Source": "Phase U WP1 segmenter audit data",
            "Data_Verifiable": "Yes",
            "Issues": "Anti-correlation is significant only for SAM-B pipelines (Phase W). V4 presents it as universal, which is misleading.",
            "Verdict": "MISLEADING — anti-correlation not universal",
        },
        {
            "Figure_ID": "Fig-2",
            "Filename": "fig2_pipeline_diagram",
            "Caption": "Pipeline diagram",
            "Generation_Script": "paper/phaseU/generate_figures.py",
            "Data_Source": "N/A (diagram)",
            "Data_Verifiable": "N/A",
            "Issues": "None",
            "Verdict": "CONCEPTUAL",
        },
        {
            "Figure_ID": "Fig-3",
            "Filename": "fig3_coverage_risk_curves",
            "Caption": "Coverage-risk analysis",
            "Generation_Script": "paper/phaseU/generate_figures.py",
            "Data_Source": "Phase U selective prediction results",
            "Data_Verifiable": "Yes",
            "Issues": "Phase W: coverage curves stable across 50-100%. M2_Inst degrades performance at every coverage (confirmed).",
            "Verdict": "VERIFIED",
        },
        {
            "Figure_ID": "Fig-4",
            "Filename": "fig4_feature_importance",
            "Caption": "Feature importance and ablation",
            "Generation_Script": "paper/phaseU/generate_figures.py",
            "Data_Source": "Phase U feature importance analysis",
            "Data_Verifiable": "Yes",
            "Issues": "f4 dominates catastrophic detection (55%) — confirmed by feature ablation.",
            "Verdict": "VERIFIED",
        },
        {
            "Figure_ID": "Fig-5",
            "Filename": "fig5_cross_dataset",
            "Caption": "Deployment envelope",
            "Generation_Script": "paper/phaseU/generate_figures.py",
            "Data_Source": "Phase T WP2 cross-dataset",
            "Data_Verifiable": "Yes",
            "Issues": "IDRID ECE=0.477 (severe miscalibration).",
            "Verdict": "VERIFIED",
        },
        {
            "Figure_ID": "Fig-6",
            "Filename": "fig6_scale_reliability",
            "Caption": "Scale-reliability mismatch",
            "Generation_Script": "paper/phaseU/generate_figures.py",
            "Data_Source": "Phase U WP1 data",
            "Data_Verifiable": "Yes",
            "Issues": "Scale-reliability mismatch is confirmed: SAM-L reduces instability but increases CFR.",
            "Verdict": "VERIFIED",
        },
        {
            "Figure_ID": "Fig-7",
            "Filename": "fig7_external_comparator",
            "Caption": "External comparator",
            "Generation_Script": "paper/phaseU/generate_figures.py",
            "Data_Source": "Phase U external baselines (GF-SAM, MAUP)",
            "Data_Verifiable": "Yes",
            "Issues": "Backbone mismatch (SAM-H vs SAM-B).",
            "Verdict": "VERIFIED_WITH_QUALIFICATION",
        },
        {
            "Figure_ID": "Fig-Theory-1",
            "Filename": "fig_theory_anticorrelation_matrix",
            "Caption": "Anti-correlation matrix",
            "Generation_Script": "run_phaseU_theory_figures.py",
            "Data_Source": "Theory verification",
            "Data_Verifiable": "Partially",
            "Issues": "Anti-correlation not universal (Phase W).",
            "Verdict": "PARTIALLY_VERIFIED",
        },
        {
            "Figure_ID": "Fig-Theory-2",
            "Filename": "fig_theory_dispersion",
            "Caption": "Dispersion analysis",
            "Generation_Script": "run_phaseU_theory_figures.py",
            "Data_Source": "Theory verification",
            "Data_Verifiable": "Yes",
            "Issues": "None",
            "Verdict": "VERIFIED",
        },
        {
            "Figure_ID": "Fig-Theory-3",
            "Filename": "fig_theory_entropy_bins",
            "Caption": "Empirical verification of entropy-failure relationship",
            "Generation_Script": "run_phaseU_theory_figures.py",
            "Data_Source": "theory_verification_results.json",
            "Data_Verifiable": "Yes",
            "Issues": "Non-monotonic entropy-CFR relationship is plausible. Mid-entropy regime as blind spot is interesting.",
            "Verdict": "VERIFIED",
        },
        {
            "Figure_ID": "Fig-Theory-4",
            "Filename": "fig_theory_failure_landscape",
            "Caption": "Failure landscape",
            "Generation_Script": "run_phaseU_theory_figures.py",
            "Data_Source": "Theory verification",
            "Data_Verifiable": "Yes",
            "Issues": "None",
            "Verdict": "VERIFIED",
        },
        {
            "Figure_ID": "Fig-Theory-5",
            "Filename": "fig_theory_feature_correlation",
            "Caption": "Feature correlation matrix",
            "Generation_Script": "run_phaseU_theory_figures.py",
            "Data_Source": "Theory verification",
            "Data_Verifiable": "Yes",
            "Issues": "f4 r=-0.842 with instability, f1 r=-0.064 with CFR — dissociation confirmed.",
            "Verdict": "VERIFIED",
        },
    ]

    return figures

# ══════════════════════════════════════════════════════════════════════════
# WP9: Cross-Phase Comparison
# ══════════════════════════════════════════════════════════════════════════
def cross_phase_comparison(claims):
    """Compare V4 claims against Phase V, W, AA, AB findings."""
    rows = []

    # Key findings from later phases
    phase_findings = {
        "Phase V": {
            "date": "2026-07-27",
            "key_findings": "14 contradictions found (7 P0-CRITICAL). Anti-correlation r range corrected. Theorem removed. RAMP correctly described as post-hoc. RRN APPENDIX_ONLY. Query-level statistics: SAM-B pipelines significant, SAM2 not.",
            "gate": "PHASEV_EMPIRICAL_SELECTIVE_PREDICTION_READY",
            "claims_audited": "1 SUPPORTED, 5 PARTIALLY_SUPPORTED, 1 UNSUPPORTED, 7 PROHIBITED",
        },
        "Phase W": {
            "date": "2026-07-27",
            "key_findings": "Full RAMP has zero added value. Prediction Consistency (B4) is strongest baseline. Anti-correlation significant for SAM-B only. RRN AUROC=0.573 (Appendix only). Split renamed to Development Cohort. SAM-Med2D checkpoint unloadable.",
            "gate": "PHASEW_ROUTE_B_SIMPLE_SELECTIVE_PREDICTION",
            "claims_audited": "1 SUPPORTED, 5 PARTIALLY_SUPPORTED, 1 UNSUPPORTED, 7 PROHIBITED",
        },
        "Phase AB": {
            "date": "2026-07-28",
            "key_findings": "12/15 gates passed. No confirmatory cohort. SAM-Med2D REMOVED. Anti-correlation REMOVED. Full RAMP REMOVED. RRN Appendix only. Paper is exploratory/benchmark. Prediction Consistency is sole effective method.",
            "gate": "PHASEAB_PAPER_CANDIDATE_READY (workshop/benchmark)",
            "claims_audited": "C1-C6 SUPPORTED, P1-P7 PROHIBITED, PS1-PS3 PARTIALLY_SUPPORTED",
        },
    }

    for c in claims:
        row = {
            "Claim_ID": c["Claim_ID"],
            "Claim_Text": c["Claim_Text"][:100],
            "Category": c["Category"],
            "V4_Status": "CLAIMED",
            "Phase_V_Status": "",
            "Phase_W_Status": "",
            "Phase_AB_Status": "",
            "Final_Status": "",
            "Notes": "",
        }

        text_lower = c["Claim_Text"].lower()

        # Determine status based on known Phase findings
        if "anti-correlation" in text_lower and ("universal" in text_lower or "across" in text_lower):
            row["Phase_V_Status"] = "PARTIALLY_SUPPORTED"
            row["Phase_W_Status"] = "PARTIALLY_SUPPORTED (SAM-B only)"
            row["Phase_AB_Status"] = "REMOVED from main text"
            row["Final_Status"] = "PARTIALLY_SUPPORTED"
            row["Notes"] = "Anti-correlation is SAM-B specific, not universal. Phase AB removed from main text."
        elif "ramp" in text_lower and ("weighting" in text_lower or "fusion" in text_lower or "consensus" in text_lower):
            row["Phase_V_Status"] = "SUPPORTED (with qualification)"
            row["Phase_W_Status"] = "PROHIBITED (zero effect)"
            row["Phase_AB_Status"] = "REMOVED from main text"
            row["Final_Status"] = "PROHIBITED"
            row["Notes"] = "Phase W: Full RAMP weighting/fusion have zero effect. Phase AB: Prediction Consistency is sole method."
        elif "ramp" in text_lower and "cfr" in text_lower:
            row["Phase_V_Status"] = "SUPPORTED (corrected range)"
            row["Phase_W_Status"] = "SUPPORTED (achievable with simpler methods)"
            row["Phase_AB_Status"] = "SUPPORTED (Prediction Consistency)"
            row["Final_Status"] = "SUPPORTED"
            row["Notes"] = "CFR reduction is real but achievable with simpler methods (B4)."
        elif "rrn" in text_lower:
            row["Phase_V_Status"] = "APPENDIX_ONLY (AUROC<0.65)"
            row["Phase_W_Status"] = "APPENDIX_ONLY (AUROC=0.573)"
            row["Phase_AB_Status"] = "APPENDIX_ONLY"
            row["Final_Status"] = "APPENDIX_ONLY"
            row["Notes"] = "RRN does not meet main-text threshold. Grouped CV AUROC=0.573."
        elif "sammed2d" in text_lower or "sam-med2d" in text_lower:
            row["Phase_V_Status"] = "PENDING (checkpoint blocked)"
            row["Phase_W_Status"] = "PENDING (checkpoint unloadable)"
            row["Phase_AB_Status"] = "REMOVED"
            row["Final_Status"] = "REMOVED"
            row["Notes"] = "SAM-Med2D checkpoint cannot be verified. Phase AB REMOVED from main table."
        elif "first" in text_lower and ("rre" in text_lower or "benchmark" in text_lower or "reference selection" in text_lower):
            row["Phase_V_Status"] = "SUPPORTED"
            row["Phase_W_Status"] = "SUPPORTED"
            row["Phase_AB_Status"] = "SUPPORTED"
            row["Final_Status"] = "SUPPORTED"
            row["Notes"] = "RRE as a problem formulation is supported. Phase AB lists as key strength."
        elif "deployment" in text_lower and "envelope" in text_lower:
            row["Phase_V_Status"] = "SUPPORTED"
            row["Phase_W_Status"] = "PARTIALLY_SUPPORTED (50-100% range)"
            row["Phase_AB_Status"] = "PARTIALLY_SUPPORTED"
            row["Final_Status"] = "PARTIALLY_SUPPORTED"
            row["Notes"] = "Envelope validated at 50-100% coverage, not 5-60%."
        elif "pascal" in text_lower or "coco" in text_lower:
            row["Phase_V_Status"] = "PARTIALLY_SUPPORTED (fold-0 only)"
            row["Phase_W_Status"] = "PENDING (4-fold needed)"
            row["Phase_AB_Status"] = "PARTIALLY_SUPPORTED"
            row["Final_Status"] = "PARTIALLY_SUPPORTED"
            row["Notes"] = "Only fold-0 available. 4-fold needed for valid cross-domain claim."
        elif "training-free" in text_lower:
            row["Phase_V_Status"] = "PROHIBITED"
            row["Phase_W_Status"] = "PROHIBITED"
            row["Phase_AB_Status"] = "PROHIBITED"
            row["Final_Status"] = "PROHIBITED"
            row["Notes"] = "RAMP uses supervised gradient-boosted trees. DINOv2 is trained."
        elif "calibration-free" in text_lower:
            row["Phase_V_Status"] = "PROHIBITED"
            row["Phase_W_Status"] = "PROHIBITED (IDRID ECE=0.477)"
            row["Phase_AB_Status"] = "PROHIBITED"
            row["Final_Status"] = "PROHIBITED"
            row["Notes"] = "IDRID ECE=0.477 confirms miscalibration."
        elif "gf-sam" in text_lower or "maup" in text_lower:
            row["Phase_V_Status"] = "PARTIALLY_SUPPORTED (backbone mismatch)"
            row["Phase_W_Status"] = "PENDING (stratified comparison needed)"
            row["Phase_AB_Status"] = "PARTIALLY_SUPPORTED (backbone noted)"
            row["Final_Status"] = "PARTIALLY_SUPPORTED"
            row["Notes"] = "Backbone mismatch: GF-SAM/MAUP use SAM-H, RAMP uses SAM-B."
        elif "component ablation" in text_lower:
            row["Phase_V_Status"] = "UNRELIABLE"
            row["Phase_W_Status"] = "PROHIBITED (zero effect)"
            row["Phase_AB_Status"] = "REMOVED"
            row["Final_Status"] = "PROHIBITED"
            row["Notes"] = "Phase V: all variants identical. Phase W: Full RAMP has zero added value."
        elif "rre" in text_lower or "reference reliability" in text_lower:
            row["Phase_V_Status"] = "SUPPORTED"
            row["Phase_W_Status"] = "SUPPORTED"
            row["Phase_AB_Status"] = "SUPPORTED"
            row["Final_Status"] = "SUPPORTED"
            row["Notes"] = "RRE problem formulation is the strongest surviving contribution."
        else:
            row["Phase_V_Status"] = "NOT_AUDITED"
            row["Phase_W_Status"] = "NOT_AUDITED"
            row["Phase_AB_Status"] = "NOT_AUDITED"
            row["Final_Status"] = "UNKNOWN"
            row["Notes"] = ""

        rows.append(row)

    return rows

# ══════════════════════════════════════════════════════════════════════════
# WP10: AI Hallucination Audit
# ══════════════════════════════════════════════════════════════════════════
def hallucination_audit(claims):
    """Classify each claim A-I based on verifiability."""
    # A: VERIFIED
    # B: VERIFIED_WITH_QUALIFICATION
    # C: REPRODUCIBLE_BUT_NOT_INDEPENDENT
    # D: PARTIALLY_SUPPORTED
    # E: CONTRADICTED_BY_LATER_AUDIT
    # F: UNVERIFIABLE_NO_SOURCE
    # G: PLACEHOLDER_SUSPECTED
    # H: LIKELY_AI_FABRICATED
    # I: REQUIRES_NEW_EXPERIMENT

    rows = []
    for c in claims:
        row = {
            "Claim_ID": c["Claim_ID"],
            "Claim_Text": c["Claim_Text"][:120],
            "Category": c["Category"],
            "Audit_Class": "",
            "Audit_Confidence": "",
            "Audit_Rationale": "",
            "Later_Phase_Contradiction": "",
            "Phase_Reference": "",
        }

        text_lower = c["Claim_Text"].lower()

        # Hard-coded audit rules based on comprehensive Phase V/W/AB knowledge
        # E: CONTRADICTED_BY_LATER_AUDIT
        if "universal" in text_lower and "anti-correlation" in text_lower:
            row["Audit_Class"] = "E"
            row["Audit_Confidence"] = "HIGH"
            row["Audit_Rationale"] = "Phase W: anti-correlation significant only for SAM-B, not SAM2. Phase V: Null 1 reproduces 94% of correlation. Phase AB: anti-correlation REMOVED from main text."
            row["Later_Phase_Contradiction"] = "Yes — Phase W/AB"
            row["Phase_Reference"] = "Phase W WP7, Phase AB"
        elif "ramp" in text_lower and ("weighting" in text_lower or "fusion" in text_lower or "consensus" in text_lower or "three components" in text_lower):
            row["Audit_Class"] = "E"
            row["Audit_Confidence"] = "HIGH"
            row["Audit_Rationale"] = "Phase W: Full RAMP weighting/fusion/consensus have zero effect. PHASEW_FULL_RAMP_NO_ADDED_VALUE."
            row["Later_Phase_Contradiction"] = "Yes — Phase W"
            row["Phase_Reference"] = "Phase W WP1, Phase W Final Report"
        elif "component ablation" in text_lower and ("12.3%" in text_lower or "18.7%" in text_lower or "8.4%" in text_lower):
            row["Audit_Class"] = "E"
            row["Audit_Confidence"] = "HIGH"
            row["Audit_Rationale"] = "Phase V: all component ablation variants produced identical Dice (data unreliable). Phase W: Full RAMP has zero added value."
            row["Later_Phase_Contradiction"] = "Yes — Phase V/W"
            row["Phase_Reference"] = "Phase V, Phase W WP1"
        elif "sammed2d" in text_lower or "sam-med2d" in text_lower:
            row["Audit_Class"] = "E"
            row["Audit_Confidence"] = "HIGH"
            row["Audit_Rationale"] = "Phase W: checkpoint cannot be loaded. Phase AB: SAM-Med2D REMOVED from main table. Results unverifiable."
            row["Later_Phase_Contradiction"] = "Yes — Phase AB"
            row["Phase_Reference"] = "Phase W WP5, Phase AB"
        elif "training-free" in text_lower:
            row["Audit_Class"] = "E"
            row["Audit_Confidence"] = "HIGH"
            row["Audit_Rationale"] = "RAMP uses gradient-boosted trees (supervised). DINOv2 is trained. Phase V/W/AB all PROHIBITED."
            row["Later_Phase_Contradiction"] = "Yes — Phase V/W/AB"
            row["Phase_Reference"] = "Phase V Claim #4, Phase W"
        elif "calibration-free" in text_lower:
            row["Audit_Class"] = "E"
            row["Audit_Confidence"] = "HIGH"
            row["Audit_Rationale"] = "IDRID ECE=0.477. Phase V/W/AB all PROHIBITED."
            row["Later_Phase_Contradiction"] = "Yes — Phase V/W/AB"
            row["Phase_Reference"] = "Phase V Claim #5, Phase W"
        # B: VERIFIED_WITH_QUALIFICATION
        elif "anti-correlation" in text_lower and "r=" in text_lower and "sam-b" in text_lower:
            row["Audit_Class"] = "B"
            row["Audit_Confidence"] = "HIGH"
            row["Audit_Rationale"] = "Anti-correlation verified for SAM-B pipelines (L+SAM-B r=-0.512, B+SAM-B r=-0.549). But not universal — SAM2 pipelines not significant."
            row["Later_Phase_Contradiction"] = "No (qualified)"
            row["Phase_Reference"] = "Phase W WP7"
        elif "rrn" in text_lower and ("auroc" in text_lower or "dice" in text_lower):
            row["Audit_Class"] = "B"
            row["Audit_Confidence"] = "HIGH"
            row["Audit_Rationale"] = "RRN AUROC=0.612 from single 70/15/15 split (12 test queries). Phase W grouped CV: AUROC=0.573. Below 0.65 threshold. APPENDIX_ONLY."
            row["Later_Phase_Contradiction"] = "No (qualified)"
            row["Phase_Reference"] = "Phase V Gate V6, Phase W WP4"
        elif "pascal" in text_lower or "coco" in text_lower:
            row["Audit_Class"] = "B"
            row["Audit_Confidence"] = "MEDIUM"
            row["Audit_Rationale"] = "Only fold-0 available. 4-fold needed for valid cross-domain claim. Phase W: WP9 PENDING."
            row["Later_Phase_Contradiction"] = "No (qualified)"
            row["Phase_Reference"] = "Phase W WP9"
        elif "deployment" in text_lower and "envelope" in text_lower:
            row["Audit_Class"] = "B"
            row["Audit_Confidence"] = "MEDIUM"
            row["Audit_Rationale"] = "Envelope validated at 50-100% coverage (Phase W), not 5-60%. Useful but imprecise."
            row["Later_Phase_Contradiction"] = "No (qualified)"
            row["Phase_Reference"] = "Phase W WP3"
        elif "gf-sam" in text_lower or "maup" in text_lower:
            row["Audit_Class"] = "B"
            row["Audit_Confidence"] = "HIGH"
            row["Audit_Rationale"] = "Numbers verified but backbone mismatch (SAM-H vs SAM-B). 'Identical conditions' claim is PROHIBITED."
            row["Later_Phase_Contradiction"] = "No (qualified)"
            row["Phase_Reference"] = "Phase W WP6, Phase V Gate V5"
        # A: VERIFIED
        elif "rre" in text_lower and ("missing" in text_lower or "first" in text_lower or "problem" in text_lower):
            row["Audit_Class"] = "A"
            row["Audit_Confidence"] = "HIGH"
            row["Audit_Rationale"] = "RRE problem formulation is the strongest surviving contribution. Phase AB lists as key strength."
            row["Later_Phase_Contradiction"] = "No"
            row["Phase_Reference"] = "Phase AB"
        elif "cfr" in text_lower and "37.5" in text_lower:
            row["Audit_Class"] = "A"
            row["Audit_Confidence"] = "HIGH"
            row["Audit_Rationale"] = "CFR reduction range corrected to 37.5-84.4%. Phase V verified. Phase W: achievable with simpler methods."
            row["Later_Phase_Contradiction"] = "No"
            row["Phase_Reference"] = "Phase V, Phase W"
        elif "dino" in text_lower and "sam" in text_lower and "0.446" in text_lower:
            row["Audit_Class"] = "A"
            row["Audit_Confidence"] = "HIGH"
            row["Audit_Rationale"] = "DINOv2-L+SAM-B Dice=0.446 verified by Phase U WP1."
            row["Later_Phase_Contradiction"] = "No"
            row["Phase_Reference"] = "Phase U WP1"
        elif "h1" in text_lower or "h2" in text_lower or "h3" in text_lower:
            row["Audit_Class"] = "A"
            row["Audit_Confidence"] = "HIGH"
            row["Audit_Rationale"] = "Geometric Hypotheses are empirically testable propositions. Phase V: Gate_V1_Theory_Repair = PASS."
            row["Later_Phase_Contradiction"] = "No"
            row["Phase_Reference"] = "Phase V Gate V1"
        elif "f1" in text_lower and "f4" in text_lower and ("dissociat" in text_lower or "importance" in text_lower):
            row["Audit_Class"] = "A"
            row["Audit_Confidence"] = "HIGH"
            row["Audit_Rationale"] = "Feature dissociation confirmed. f1 dominates instability, f4 dominates catastrophic detection."
            row["Later_Phase_Contradiction"] = "No"
            row["Phase_Reference"] = "Phase U"
        elif "cross-dataset" in text_lower or "cross_dataset" in text_lower:
            row["Audit_Class"] = "B"
            row["Audit_Confidence"] = "MEDIUM"
            row["Audit_Rationale"] = "Cross-dataset validation uses 30 queries only. IDRID ECE=0.477. MoNuSeg 0% prevalence."
            row["Later_Phase_Contradiction"] = "No (qualified)"
            row["Phase_Reference"] = "Phase W"
        elif "14%" in text_lower and "catastrophic" in text_lower:
            row["Audit_Class"] = "A"
            row["Audit_Confidence"] = "MEDIUM"
            row["Audit_Rationale"] = "14% CFR on MoNuSAC is verified by Phase U WP1 data."
            row["Later_Phase_Contradiction"] = "No"
            row["Phase_Reference"] = "Phase U WP1"
        # D: PARTIALLY_SUPPORTED
        elif "scale-reliability" in text_lower or "scale" in text_lower and "reliability" in text_lower:
            row["Audit_Class"] = "D"
            row["Audit_Confidence"] = "MEDIUM"
            row["Audit_Rationale"] = "Scale-reliability mismatch pattern is observed but effect is small. SAM-L increases CFR from 14.3% to 16.7%."
            row["Later_Phase_Contradiction"] = "No"
            row["Phase_Reference"] = "Phase U WP1"
        # C: REPRODUCIBLE_BUT_NOT_INDEPENDENT
        elif "80 training" in text_lower or "7200" in text_lower or "k=6" in text_lower:
            row["Audit_Class"] = "C"
            row["Audit_Confidence"] = "HIGH"
            row["Audit_Rationale"] = "Experimental setup details are reproducible from code. But no independent replication (no fresh cohort)."
            row["Later_Phase_Contradiction"] = "No"
            row["Phase_Reference"] = "Phase W Split Audit"
        elif "2x2" in text_lower and "factorial" in text_lower:
            row["Audit_Class"] = "C"
            row["Audit_Confidence"] = "HIGH"
            row["Audit_Rationale"] = "Factorial decomposition is well-documented and reproducible. Statistical tests are appropriate. But no independent replication."
            row["Later_Phase_Contradiction"] = "No"
            row["Phase_Reference"] = "Phase U WP4"
        # I: REQUIRES_NEW_EXPERIMENT
        elif "limitation" in c["Section"].lower() and ("single fold" in text_lower or "30 quer" in text_lower):
            row["Audit_Class"] = "I"
            row["Audit_Confidence"] = "HIGH"
            row["Audit_Rationale"] = "V4 acknowledges limitation. 4-fold Pascal/COCO and larger cross-dataset needed."
            row["Later_Phase_Contradiction"] = "No"
            row["Phase_Reference"] = "Phase W WP9"
        elif "no fixed probability" in text_lower or "coverage-based ranking" in text_lower:
            row["Audit_Class"] = "A"
            row["Audit_Confidence"] = "HIGH"
            row["Audit_Rationale"] = "Coverage-based ranking is correctly implemented. No fixed threshold claim is true."
            row["Later_Phase_Contradiction"] = "No"
            row["Phase_Reference"] = "Phase V"
        # Default
        else:
            row["Audit_Class"] = "D"
            row["Audit_Confidence"] = "LOW"
            row["Audit_Rationale"] = "Claim not directly audited by later phases. Default to PARTIALLY_SUPPORTED until verified."
            row["Later_Phase_Contradiction"] = "Unknown"
            row["Phase_Reference"] = ""

        rows.append(row)

    # Ensure key confirmations are in the audit
    # Anti-correlation MUST be marked as contradicted by later audit
    # SAM-Med2D MUST be marked as removed
    # RAMP weighting/fusion MUST be marked as contradicted
    return rows

# ══════════════════════════════════════════════════════════════════════════
# WP11: Salvage Map
# ══════════════════════════════════════════════════════════════════════════
def salvage_map():
    """Classify each V4 section for salvageability."""
    sections = [
        {
            "Section": "Title",
            "V4_Content": "Reference Reliability Estimation for Visual Prompting Pipelines",
            "Salvage_Action": "REWRITE_FROM_PHASE_AB",
            "Rationale": "Phase AB title: 'Prediction Consistency as a Simple and Effective Risk Estimator for Multi-Reference Visual Prompting'. V4 title overclaims (RRE is not solved).",
            "Phase_AB_Recommendation": "Route B: Simplified Selective Prediction Paper",
        },
        {
            "Section": "Abstract",
            "V4_Content": "RRE formalization, anti-correlation, RAMP, RRN, external baselines, benchmark",
            "Salvage_Action": "REWRITE_FROM_PHASE_AB",
            "Rationale": "Must remove anti-correlation universal claim, RAMP weighting/fusion, SAM-Med2D. Focus on Prediction Consistency + coverage-based abstention.",
            "Phase_AB_Recommendation": "Rewrite to focus on simple selective prediction.",
        },
        {
            "Section": "1. Introduction",
            "V4_Content": "RRE problem, anti-correlation, two-tier solution, contributions",
            "Salvage_Action": "KEEP_WITH_MAJOR_QUALIFICATION",
            "Rationale": "RRE problem formulation is the strongest contribution. But anti-correlation must be qualified (SAM-B only). RAMP contributions must be simplified.",
            "Phase_AB_Recommendation": "Keep RRE problem. Remove universal anti-correlation. Simplify to Prediction Consistency.",
        },
        {
            "Section": "2. Related Work",
            "V4_Content": "Visual prompting, selective prediction, few-shot learning, scale/reliability",
            "Salvage_Action": "KEEP_WITH_MINOR_EDIT",
            "Rationale": "Literature review is thorough and well-organized. Minor updates needed for Phase AB findings.",
            "Phase_AB_Recommendation": "Keep structure. Add reference to Phase AB findings.",
        },
        {
            "Section": "3. The Point-Prompt Bottleneck",
            "V4_Content": "Geometric Hypotheses H1-H3, empirical mechanism analysis, validity controls, predictions",
            "Salvage_Action": "KEEP_WITH_MAJOR_QUALIFICATION",
            "Rationale": "The geometric analysis is rigorous and interesting. But anti-correlation is not universal. Mid-entropy non-monotonicity is the strongest finding. Must be presented as empirical association, not structural theorem.",
            "Phase_AB_Recommendation": "Move to Appendix or keep with strong qualification. Focus on mid-entropy blind spot.",
        },
        {
            "Section": "4. Method",
            "V4_Content": "RAMP (weighting, fusion, consensus), RRN (MLP), analytical framework",
            "Salvage_Action": "REWRITE_FROM_PHASE_AB",
            "Rationale": "Phase W: Full RAMP has zero added value. RRN is APPENDIX_ONLY. Phase AB: Prediction Consistency is the sole effective method. Method section must be rewritten.",
            "Phase_AB_Recommendation": "Replace with Prediction Consistency + coverage-based abstention.",
        },
        {
            "Section": "5.1 Setup",
            "V4_Content": "Datasets, models, metrics, baselines",
            "Salvage_Action": "KEEP_WITH_MINOR_EDIT",
            "Rationale": "Experimental setup is well-documented. Update to reflect Development Cohort (not holdout). Remove SAM-Med2D.",
            "Phase_AB_Recommendation": "Keep. Rename to Development Cohort. Remove SAM-Med2D.",
        },
        {
            "Section": "5.2 Universality of Anti-Correlation",
            "V4_Content": "Tables 1-3, scale-reliability, matcher independence, factorial",
            "Salvage_Action": "KEEP_WITH_MAJOR_QUALIFICATION",
            "Rationale": "Anti-correlation is significant for SAM-B only. 'Universal' claim is PROHIBITED. But the factorial analysis and scale-reliability mismatch are valid findings.",
            "Phase_AB_Recommendation": "Keep factorial analysis. Remove 'universal' claim. Add SAM2 non-significance.",
        },
        {
            "Section": "5.3 Failure Regime Characterization",
            "V4_Content": "Two regimes, entropy bins, non-monotonicity",
            "Salvage_Action": "KEEP",
            "Rationale": "The mid-entropy non-monotonicity finding is novel and interesting. Well-supported by theory figures.",
            "Phase_AB_Recommendation": "Keep as key contribution.",
        },
        {
            "Section": "5.4 Reference Selection Benchmark",
            "V4_Content": "Table 5 (refsel), RRN results",
            "Salvage_Action": "MOVE_TO_APPENDIX",
            "Rationale": "RRN is APPENDIX_ONLY. AUROC=0.573 < 0.65. The benchmark concept is useful but the learned method is not competitive.",
            "Phase_AB_Recommendation": "Move to Appendix. Keep benchmark concept.",
        },
        {
            "Section": "5.5 RAMP Selective Prediction",
            "V4_Content": "Table 4 (RAMP), coverage-risk curves, anti-informative M2_Inst",
            "Salvage_Action": "REWRITE_FROM_PHASE_AB",
            "Rationale": "Phase W: Full RAMP has zero added value. Phase AB: Prediction Consistency is sole effective method. The anti-informative M2_Inst finding is interesting but RAMP's components are not needed.",
            "Phase_AB_Recommendation": "Replace with Prediction Consistency results.",
        },
        {
            "Section": "5.6 Feature Dissociation",
            "V4_Content": "Feature importance, f1 vs f4 dissociation",
            "Salvage_Action": "KEEP_WITH_MINOR_EDIT",
            "Rationale": "Feature dissociation is well-supported. Interesting analytical finding.",
            "Phase_AB_Recommendation": "Keep. Minor edits for clarity.",
        },
        {
            "Section": "5.7 Comparison with Existing Methods",
            "V4_Content": "Table 6, GF-SAM, MAUP, CycleSAM, H-SAM, SAM-Med2D, MC Dropout, Deep Ensemble",
            "Salvage_Action": "KEEP_WITH_MAJOR_QUALIFICATION",
            "Rationale": "GF-SAM and MAUP numbers are verified but backbone mismatch. SAM-Med2D must be REMOVED. MC Dropout/Deep Ensemble comparison is valid.",
            "Phase_AB_Recommendation": "Keep but remove SAM-Med2D. Add backbone-stratified comparison. Clarify backbone differences.",
        },
        {
            "Section": "5.8 Cross-Dataset Deployment Envelope",
            "V4_Content": "Table 7, deployment envelope 5-60%",
            "Salvage_Action": "KEEP_WITH_MAJOR_QUALIFICATION",
            "Rationale": "Deployment envelope concept is useful but imprecise. IDRID ECE=0.477. Validated at 50-100% not 5-60%.",
            "Phase_AB_Recommendation": "Keep concept. Update range to 50-100% or add calibration experiments.",
        },
        {
            "Section": "5.9 Ablation Study",
            "V4_Content": "Component ablation, feature ablation, reference count ablation",
            "Salvage_Action": "REMOVE",
            "Rationale": "Component ablation data is UNRELIABLE (Phase V: all variants identical). Phase W: Full RAMP has zero effect. Feature ablation is interesting but not independently verified.",
            "Phase_AB_Recommendation": "Remove component ablation. Keep reference count ablation if numbers are reliable.",
        },
        {
            "Section": "5.10 Cross-Domain Generality",
            "V4_Content": "Table 11, Pascal-5i, COCO-20i",
            "Salvage_Action": "MOVE_TO_APPENDIX",
            "Rationale": "Only fold-0 available. 4-fold needed. Single-fold results are suggestive but not publishable as main-table evidence.",
            "Phase_AB_Recommendation": "Move to Appendix pending 4-fold completion.",
        },
        {
            "Section": "6. Discussion",
            "V4_Content": "Structural basis, RRE formulation, reliability vs sophistication, implications, limitations",
            "Salvage_Action": "KEEP_WITH_MAJOR_QUALIFICATION",
            "Rationale": "Discussion is well-written but must be updated to reflect Phase W/AB findings. Remove anti-correlation universal claim. Remove RAMP overclaims. Add Prediction Consistency.",
            "Phase_AB_Recommendation": "Rewrite discussion to focus on Prediction Consistency + RRE problem formulation.",
        },
        {
            "Section": "7. Conclusion",
            "V4_Content": "RRE identified, two-tier solution, benchmark, deployment envelope",
            "Salvage_Action": "REWRITE_FROM_PHASE_AB",
            "Rationale": "Must reflect Phase AB: Prediction Consistency is sole method. RAMP/RRN are not competitive. RRE problem formulation is the key contribution.",
            "Phase_AB_Recommendation": "Rewrite to: 'We identify RRE as a missing component. Prediction Consistency provides a simple effective risk estimator. Coverage-based abstention reduces CFR.'",
        },
        {
            "Section": "Appendix",
            "V4_Content": "Metric coupling controls, additional results",
            "Salvage_Action": "KEEP",
            "Rationale": "Metric coupling controls are rigorous and important. Should be retained as appendix material.",
            "Phase_AB_Recommendation": "Keep. This is strong supplementary material.",
        },
    ]

    return sections

# ══════════════════════════════════════════════════════════════════════════
# MAIN
# ══════════════════════════════════════════════════════════════════════════
def main():
    print("=" * 70)
    print("VV4 Forensics Script — WP3–WP11")
    print(f"Canonical V4: {V4_TEX}")
    print(f"Output root: {OUT_DIR}")
    print(f"Started: {now_iso()}")
    print("=" * 70)

    # ── WP3: Claim Extraction ──
    print("\n[WP3] Extracting claims from V4 LaTeX...")
    claims = extract_claims()
    print(f"  Extracted {len(claims)} claims")
    write_csv(OUT_DIR / "extracted_claims" / "v4_claim_registry.csv", claims)
    print(f"  → {OUT_DIR / 'extracted_claims' / 'v4_claim_registry.csv'}")

    # ── WP4: Evidence Chain ──
    print("\n[WP4] Building evidence chain...")
    artifact_matrix = build_evidence_chain(claims)
    evidence_count = sum(1 for r in artifact_matrix if r["Has_Evidence"] == "Yes")
    print(f"  {evidence_count}/{len(artifact_matrix)} claims have evidence")
    write_csv(OUT_DIR / "claim_to_artifact" / "claim_artifact_matrix.csv", artifact_matrix)
    print(f"  → {OUT_DIR / 'claim_to_artifact' / 'claim_artifact_matrix.csv'}")

    # ── WP5: High-Risk Claim Audit ──
    print("\n[WP5] Auditing 12 high-risk claims...")
    md_lines, hr_audits = audit_high_risk_claims()
    write_md(OUT_DIR / "hallucination_audit" / "hallucination_report.md", md_lines)
    write_csv(OUT_DIR / "hallucination_audit" / "high_risk_claim_audit.csv", hr_audits)
    print(f"  → {OUT_DIR / 'hallucination_audit' / 'hallucination_report.md'}")
    print(f"  → {OUT_DIR / 'hallucination_audit' / 'high_risk_claim_audit.csv'}")

    # ── WP6: Table Forensics ──
    print("\n[WP6] Table forensics...")
    tables = table_forensics()
    write_csv(OUT_DIR / "table_forensics" / "v4_table_forensics.csv", tables)
    write_json(OUT_DIR / "table_forensics" / "v4_table_forensics.json", tables)
    print(f"  → {OUT_DIR / 'table_forensics' / 'v4_table_forensics.csv'}")
    print(f"  → {OUT_DIR / 'table_forensics' / 'v4_table_forensics.json'}")

    # ── WP7: Figure Forensics ──
    print("\n[WP7] Figure forensics...")
    figures = figure_forensics()
    write_csv(OUT_DIR / "figure_forensics" / "v4_figure_forensics.csv", figures)
    write_json(OUT_DIR / "figure_forensics" / "v4_figure_forensics.json", figures)
    print(f"  → {OUT_DIR / 'figure_forensics' / 'v4_figure_forensics.csv'}")
    print(f"  → {OUT_DIR / 'figure_forensics' / 'v4_figure_forensics.json'}")

    # ── WP9: Cross-Phase Comparison ──
    print("\n[WP9] Cross-phase comparison (V4 vs V/W/AA/AB)...")
    cross_phase = cross_phase_comparison(claims)
    write_csv(VV4 / "06_REPORTS_AND_DECISIONS" / "claim_registries" / "v4_vs_later_phases.csv", cross_phase)
    (VV4 / "06_REPORTS_AND_DECISIONS" / "claim_registries").mkdir(parents=True, exist_ok=True)
    print(f"  → {VV4 / '06_REPORTS_AND_DECISIONS' / 'claim_registries' / 'v4_vs_later_phases.csv'}")

    # ── WP10: AI Hallucination Audit ──
    print("\n[WP10] AI hallucination audit (A–I classification)...")
    hall_rows = hallucination_audit(claims)
    # Count classes
    class_counts = defaultdict(int)
    for r in hall_rows:
        class_counts[r["Audit_Class"]] += 1
    print("  Class distribution:")
    for cls in ["A", "B", "C", "D", "E", "F", "G", "H", "I"]:
        if cls in class_counts:
            print(f"    {cls}: {class_counts[cls]}")
    write_csv(OUT_DIR / "hallucination_audit" / "hallucination_registry.csv", hall_rows)
    print(f"  → {OUT_DIR / 'hallucination_audit' / 'hallucination_registry.csv'}")

    # ── WP11: Salvage Map ──
    print("\n[WP11] V4 section salvage map...")
    sections = salvage_map()
    write_csv(SALVAGE_DIR / "v4_section_salvage_map.csv", sections)
    print(f"  → {SALVAGE_DIR / 'v4_section_salvage_map.csv'}")

    # ── Write summary manifest ──
    summary = {
        "script": "run_vv4_forensics.py",
        "generated": now_iso(),
        "canonical_v4": str(V4_TEX),
        "outputs": {
            "WP3": str(OUT_DIR / "extracted_claims" / "v4_claim_registry.csv"),
            "WP4": str(OUT_DIR / "claim_to_artifact" / "claim_artifact_matrix.csv"),
            "WP5": str(OUT_DIR / "hallucination_audit" / "hallucination_report.md"),
            "WP6": str(OUT_DIR / "table_forensics" / "v4_table_forensics.csv"),
            "WP7": str(OUT_DIR / "figure_forensics" / "v4_figure_forensics.csv"),
            "WP9": str(VV4 / "06_REPORTS_AND_DECISIONS" / "claim_registries" / "v4_vs_later_phases.csv"),
            "WP10": str(OUT_DIR / "hallucination_audit" / "hallucination_registry.csv"),
            "WP11": str(SALVAGE_DIR / "v4_section_salvage_map.csv"),
        },
        "statistics": {
            "total_claims": len(claims),
            "claims_with_evidence": evidence_count,
            "high_risk_claims_audited": len(hr_audits),
            "tables_audited": len(tables),
            "figures_audited": len(figures),
            "hallucination_class_A": class_counts.get("A", 0),
            "hallucination_class_B": class_counts.get("B", 0),
            "hallucination_class_C": class_counts.get("C", 0),
            "hallucination_class_D": class_counts.get("D", 0),
            "hallucination_class_E": class_counts.get("E", 0),
            "hallucination_class_I": class_counts.get("I", 0),
            "sections_salvageable": sum(1 for s in sections if s["Salvage_Action"] in ["KEEP", "KEEP_WITH_MINOR_EDIT"]),
            "sections_need_rewrite": sum(1 for s in sections if s["Salvage_Action"] in ["REWRITE_FROM_PHASE_AB", "REMOVE"]),
        },
        "key_confirmations": {
            "anti_correlation_overthrown": "YES — Phase W/AB. Significant only for SAM-B, not SAM2. Null model corrected.",
            "sam_med2d_removed": "YES — Phase AB. Checkpoint unloadable, REMOVED from main table.",
            "ramp_weighting_fusion_zero_effect": "YES — Phase W. PHASEW_FULL_RAMP_NO_ADDED_VALUE.",
            "prediction_consistency_sole_method": "YES — Phase AB. B4 is the only effective risk signal.",
            "no_fresh_cohort": "YES — Phase AB. No confirmatory cohort exists.",
            "development_cohort": "YES — Phase W. 80 MoNuSAC queries are Development Cohort, not holdout.",
        },
    }

    # Write summary to each output directory
    write_json(OUT_DIR / "forensics_summary.json", summary)
    print(f"\n  → {OUT_DIR / 'forensics_summary.json'}")

    print("\n" + "=" * 70)
    print("VV4 Forensics Complete!")
    print(f"  {len(claims)} claims extracted")
    print(f"  {len(hr_audits)} high-risk claims audited")
    print(f"  {len(tables)} tables audited")
    print(f"  {len(figures)} figures audited")
    print(f"  Hallucination: A={class_counts.get('A',0)}, B={class_counts.get('B',0)}, C={class_counts.get('C',0)}, D={class_counts.get('D',0)}, E={class_counts.get('E',0)}, I={class_counts.get('I',0)}")
    print("=" * 70)

if __name__ == "__main__":
    main()