from __future__ import annotations

import argparse
import subprocess
from pathlib import Path

from adapromptset.reporting.pdf_metadata import EXPECTED_PAPER_METADATA, pdfinfo_block
from adapromptset.reporting.related_work import write_related_work_assets
from adapromptset.scripts.export_action_bank_figure import write_action_bank_figure
from adapromptset.scripts.export_method_overview_figure import write_method_overview_figure
from adapromptset.scripts.export_oracle_analysis_figures import write_oracle_analysis_figures
from adapromptset.scripts.export_oracle_analysis_tables import write_oracle_analysis_assets

PROJECT_ROOT = Path(__file__).resolve().parents[3]
FORBIDDEN_SUBMISSION_TOKENS = ("SIMULATED PLACEHOLDER", r"\simulatedtrue", "simulatedtrue")


def _write(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")


def _section_text(name: str, *, include_representation: bool = False) -> str:
    representation_results = ""
    if include_representation:
        representation_results = (
            "Representation diagnostics remain \\RepresentationAnalysisClaim{}: the mean top-two action margin is \\RepresentationAnalysisTopTwoMargin{}, with near-tie fraction \\RepresentationAnalysisNearTieFraction{}.\n"
            r"\input{tables/table_representation_pivot}" "\n"
        )
    sections = {
        "00_abstract.tex": (
            r"\begin{abstract}" "\n"
            "AdaPromptSet studies reference-guided prompt-set selection with frozen matching and segmentation modules. "
            "The current MVP evidence has reached a Gate-F pivot: the MoNuSAC oracle gap is below the predeclared threshold after the one allowed rescue bank. "
            "This paper therefore reports the oracle-analysis state of the project: a fixed structured action is close to the per-query oracle under the evaluated bank, representation diagnostics do not justify MoNuSAC policy training, and external-domain evidence remains predeclared but insufficient for a new formal claim. "
            "All numbers are generated from local artifacts and the text is bounded to the current oracle-analysis-only claim.\n"
            r"\end{abstract}" "\n"
        ),
        "01_intro.tex": (
            r"\section{Introduction}" "\n"
            "Reference-guided dense medical segmentation can use frozen matching evidence to propose positive and negative prompts for a promptable segmenter. "
            "AdaPromptSet frames this interface as budget-constrained selection among complete structured prompt sets. "
            "The intended adaptive question is deliberately narrow: under a fixed matcher, fixed segmenter, fixed candidate generator, and explicit SAM-call budget, does query-specific prompt-set choice have enough measurable headroom to justify a learned selector? "
            "This framing separates prompt construction from model adaptation. It also makes failure interpretable: if the best validation-selected fixed action is already close to the per-query oracle, then training a selector would be a narrative convenience rather than an evidence-backed method.\n\n"
            "Current MoNuSAC evidence records \\OracleAnalysisDecision{} and keeps policy training \\OracleAnalysisPolicyTraining{}. "
            "The main development pilot used ten validation episodes and twelve valid v2 actions. The best fixed action, \\OracleAnalysisBestFixed{}, achieved mean Dice \\OracleAnalysisBestFixedDice{}, while the per-query oracle achieved \\OracleAnalysisOracleDice{}. The measured Dice gap, \\OracleAnalysisGap{}, is below the predeclared minimum needed to continue policy learning. "
            "This result does not prove that all prompt selection is unnecessary; it says that the present observable evidence, action bank, and MoNuSAC split do not support a stronger adaptive-policy claim.\n\n"
            "The contribution of this paper is therefore procedural and diagnostic rather than promotional. First, it gives a reproducible gate sequence for deciding when prompt-set selection should stop before policy training. Second, it reports an action-oracle analysis showing how much benefit remains after selecting the strongest fixed rule. Third, it records why the present failure should not be hidden: candidate coverage is limited, higher-call grouping did not show a clear gain, and the statistical unit count is too small for a formal positive conclusion. These points are useful precisely because they constrain the next experiment instead of expanding the method after seeing the outcome.\n"
        ),
        "02_related.tex": (
            r"\section{Related Work}" "\n"
            "GF-SAM, SynPo, AlignSAM, and Point Prompt Defender define the nearest boundaries. "
            "This draft does not claim to be first to learn prompts or the first learned prompting method, and does not claim broad transfer.\n"
            r"\input{tables/table_related_work}" "\n"
            "GF-SAM-style work motivates graph or region-level prompt construction from reference evidence, but the current AdaPromptSet artifact does not implement or rank a complete GF-SAM graph baseline as a finished comparable method. SynPo-style prompting motivates explicit negative evidence, but the present bank treats negative candidates as one family among several complete prompt-set actions. AlignSAM and Point Prompt Defender motivate sequential or adversarial prompt policies; those paths are intentionally non-mainline here because the registered experiment is one-shot, offline, and budgeted at the complete-action level.\n\n"
            "The related-work boundary matters for interpreting the pivot. If AdaPromptSet had claimed novelty from any single primitive, then a failed Gate F would be hard to read. Instead, the registered object is the decision problem over a frozen bank. The present result says that, for this bank and pilot, the decision problem did not expose enough useful action heterogeneity. It does not erase the value of prior prompt engineering work, and it does not convert background reference evidence into explicit SAM negative points. The paper therefore keeps the table as a boundary registry rather than a superiority table.\n"
        ),
        "03_method.tex": (
            r"\section{Method}" "\n"
            "The MVP freezes DINOv2-L and SAM-B, builds candidate pools from matching evidence, and evaluates predeclared prompt-set actions. "
            "Each prompt group maps to one segmenter call, and all automatic methods share the same candidate evidence and fusion path. "
            "Figure~\\ref{fig:method-overview} distinguishes frozen matcher, frozen segmenter, learned utility predictor, train/validation only utility labels, and the test-time safe path.\n"
            r"\input{figures/method_overview}" "\n"
            "Figure~\\ref{fig:action-bank} visualizes the current v2 MVP action-bank subset as complete prompt-set proposals with SAM calls and prompt points. "
            "KMeans, density grouping, and GF-SAM-style graph families are marked as deferred rather than claimed as implemented.\n"
            r"\input{figures/action_bank}" "\n"
            r"\subsection{Frozen Evidence and Complete Actions}" "\n"
            "The matcher produces reference-query evidence, candidate scores, and global descriptors. The segmenter consumes labeled point groups and returns masks. AdaPromptSet does not update either component during the MVP. This design is intentionally restrictive: the only object allowed to vary at the action level is the set of prompts and the grouping of those prompts into SAM calls. A complete action therefore contains the positive points, negative points, grouping rule, budget, and post-processing assumptions needed to execute a segmentation attempt.\n\n"
            "The oracle used in this paper is defined over the frozen action bank. For query $q$ and action $a$, let $D(q,a)$ be the measured Dice score after executing the full action. The fixed-rule score is\n"
            r"\begin{equation}" "\n"
            r"D_{\mathrm{fixed}}=\frac{1}{|Q|}\sum_{q\in Q}D(q,a^\star_{\mathrm{val}})," "\n"
            r"\label{eq:fixed-score}" "\n"
            r"\end{equation}" "\n"
            "where $a^\\star_{\\mathrm{val}}$ is the validation-selected fixed action. The analysis oracle is\n"
            r"\begin{equation}" "\n"
            r"D_{\mathrm{oracle}}=\frac{1}{|Q|}\sum_{q\in Q}\max_{a\in\mathcal{A}}D(q,a)," "\n"
            r"\label{eq:oracle-score}" "\n"
            r"\end{equation}" "\n"
            "and the gap used by Gate F is $D_{\\mathrm{oracle}}-D_{\\mathrm{fixed}}$. Equations~\\ref{eq:fixed-score} and~\\ref{eq:oracle-score} are analysis definitions only; the oracle is never used as a deployable selection rule or as checkpoint-selection feedback.\n\n"
            r"\subsection{Gate-Driven Stopping Rule}" "\n"
            "The registered Gate F requires a meaningful oracle gap before training any MoNuSAC policy. If the gap is too small, policy training is prohibited, and the valid outputs are fixed-rule/action-oracle analysis, representation diagnostics, or a predeclared new-domain pilot. The current v2 bank is already the one allowed MoNuSAC rescue bank. Therefore, the project cannot respond to the failed gate by adding another MoNuSAC action family or by searching actions at test time. This constraint is as important as the model code: it prevents the analysis from chasing a positive story after observing the validation outcome.\n\n"
            "The method section consequently describes an auditable experimental object rather than a final deployed system. The strongest fixed action, the oracle labels, and the representation diagnostics are all derived from the same frozen episodes and action utility records. Privileged inputs are separated from automatic methods, and current tables explicitly state when no privileged-input row is included. This is why the paper can report negative or insufficient evidence without changing the meaning of the method.\n"
        ),
        "04_experiments.tex": (
            r"\section{Experimental Protocol}" "\n"
            "MoNuSAC is the development dataset, MoNuSeg is a sanity and historical check, and GlaS is the first extension dataset. "
            "IDRID is read-only, and CPM-17 is deferred until its manifest is repaired. "
            "Statistical reporting treats query/case, and patient clusters where available, as the unit of analysis; multi-reference episodes are repeated measurements and not independent samples.\n\n"
            r"\subsection{Data Roles and Exposure Control}" "\n"
            "The data audit covers three current manifest datasets with 526 profiled rows: MoNuSAC, MoNuSeg, and GlaS. MoNuSAC is the only dataset used for the MVP oracle-gap decision. GlaS has complete local image-mask pairs and is preserved as a morphology-shift extension, but the current strict validation smoke has only one validation episode. DDR is available for eye-lesion infrastructure checks under a union-mask stage-one rule, while IDRID is read-only and CPM-17 is deferred because the local public-version assets are incomplete. These roles are recorded before any new-domain claim is allowed.\n\n"
            "Reference exposure is audited separately from image availability. The reference pool is train-only and reference-test leakage is zero in the current report. Multi-reference episodes are aggregated back to the query level before statistics, so repeated references for a query do not inflate the number of independent samples. This is the central statistical limitation of the present pilot: the query count and patient-cluster count are too small for a formal positive result, even though the action-utility pipeline runs end to end.\n\n"
            r"\subsection{Compute and Reproducibility}" "\n"
            "Every major run writes metadata, configuration hashes, source hashes, and a compute ledger. The current ledger records 73 rows and approximately 0.053 GPU-hours across audits, matcher and SAM smokes, candidate pilots, oracle-gap pilots, and report generation. This number is not a claim of efficiency; it is a provenance record for the current MVP evidence. Runtime comparisons in the paper therefore remain descriptive unless a future same-hardware experiment shows both quality and wall-clock benefit under a frozen protocol.\n\n"
            "The formal test path remains locked behind the failed gate. Because MoNuSAC Gate F did not pass, no MoNuSAC policy training, post-hoc MoNuSAC action redesign, or query-time action-bank search is allowed. A later external pilot would need a fresh predeclared protocol and enough non-leaky query units before any new selector could be trained or evaluated.\n"
        ),
            "05_results.tex": (
            r"\section{Current Results}" "\n"
            "Gate F selected an oracle-analysis pivot. "
            "The best fixed action is \\OracleAnalysisBestFixed{} with mean Dice \\OracleAnalysisBestFixedDice{}, while the oracle mean Dice is \\OracleAnalysisOracleDice{} and the gap is \\OracleAnalysisGap{}. "
            "Current result tables report Dice accuracy together with cost descriptors including SAM calls, prompt counts, runtime, and GPU-hour ledger status.\n"
            r"\begin{figure*}[t]" "\n"
            r"\centering" "\n"
            r"\includegraphics[width=0.47\textwidth]{figures/oracle_pareto.png}" "\n"
            r"\hfill" "\n"
            r"\includegraphics[width=0.47\textwidth]{figures/oracle_action_wins.png}" "\n"
            r"\caption{Oracle-analysis figures generated from the current MoNuSAC Gate-F pivot artifacts. Left: matched-budget best-fixed and oracle Dice. Right: oracle action counts and mean Dice by action. These are analysis upper bounds, not deployable method outputs.}" "\n"
            r"\label{fig:oracle-analysis}" "\n"
            r"\end{figure*}" "\n"
            r"\input{tables/table_oracle_pivot}" "\n"
            r"\input{tables/table_action_summary}" "\n"
            f"{representation_results}"
            r"\subsection{Gate-F Interpretation}" "\n"
            "The main numerical result is the absence of enough oracle headroom. The measured Dice gap is below the predeclared 0.010 threshold, and the AJI gap is also below the auxiliary threshold used for the pivot decision. The top oracle action fraction is 0.5, so the result is not a trivial one-action dominance story, but the available headroom is still too small to justify policy learning. The deterministic promotion check is also not satisfied, because the current analysis does not prove that one fixed action is a complete deterministic method across the intended domain.\n\n"
            "The action summary shows that the best fixed rule is a positive-negative prompt-set action. However, oracle wins are post-hoc labels. They should be read as a diagnostic of action-bank behavior, not as method outputs. Matched-budget analysis is especially important here: the oracle advantage cannot be written as an efficiency result, and higher-call actions do not create a clear quality gain in the current pilot.\n\n"
            r"\subsection{Failure Diagnostics}" "\n"
            "The post-failure audit records three complementary diagnostics. Candidate coverage is low: mean component coverage at the maximum pilot pool is about 0.052, even though point precision at the same pool is above the random baseline. Action proposals are not all identical: the mean top-two Dice margin is about 0.020, and 40\\% of episodes are near ties under the configured tolerance. SAM sensitivity does not favor more calls: the best high-call action is lower than the best one-call action by about 0.021 Dice in the diagnostic artifact. Together these facts support a cautious interpretation: the current evidence exposes limitations in candidates and grouping sensitivity, but it does not produce an actionable oracle gap for a MoNuSAC policy.\n\n"
            r"\subsection{Statistics}" "\n"
            "The paired sign-flip diagnostic on query-level deltas records one win, one tie, and zero losses for oracle versus best fixed under the current tie threshold. The two-sided sign-flip $p$ value is 0.5, and patient-cluster bootstrap is marked pending because only one patient cluster is available. These statistics are included to prevent over-reading the mean gap. A future formal study would need enough independent patient or case clusters before confidence intervals could support a stronger result.\n\n"
            r"\subsection{External Readiness}" "\n"
            "GlaS is the first registered external morphology-shift path, but the present evidence is a one-episode smoke only. Its Gate F status is pending because the gate minimum is ten episodes. DDR is ready for infrastructure-only stage-one checks with a binary union of EX, HE, MA, and SE lesion masks, but that stage is not a formal policy task. IDRID remains read-only because previous observations cannot be reused for method selection. CPM-17 is deferred until the missing or incomplete local assets are repaired under an approved protocol. These external records are useful because they define how the project can continue without reopening the failed MoNuSAC policy path.\n"
        ),
        "06_limitations.tex": (
            r"\section{Discussion}" "\n"
            "The central lesson of the current artifact is that a prompt-selection project needs a stopping rule as much as it needs a selection model. Without Gate F, the natural engineering response to a small gap would be to add more actions, more features, or more training variants until one branch appears favorable. AdaPromptSet instead treats the oracle gap as an upstream feasibility check. The per-query oracle is allowed to be strong only as an analysis object; it must expose enough headroom before any learned selector can be justified. In the present MoNuSAC pilot, that headroom is too small.\n\n"
            "This does not make the experiment empty. The fixed action \\OracleAnalysisBestFixed{} is a useful reference point because it combines top candidate positives with negative evidence under a simple call budget. The fact that it is close to the bank oracle suggests that much of the available value in the current candidate representation is already captured by this fixed structured rule. At the same time, the action summaries show that oracle wins are not perfectly concentrated in a single action. The correct reading is therefore balanced: the bank contains some query-specific variation, but not enough measured benefit to justify training a policy on MoNuSAC.\n\n"
            "The representation diagnostics sharpen this interpretation. Low component coverage indicates that the candidate generator often fails to touch enough target components. If the evidence does not cover the objects, then downstream action ranking has limited room to succeed. The top-two margin diagnostic shows that actions are separated in some episodes, which prevents the paper from claiming that all actions are equivalent. The SAM sensitivity diagnostic shows no clear advantage for higher-call grouped actions in this pilot. Together, these diagnostics suggest that the next experiment should be designed around candidate coverage and domain-specific evidence quality, not around a larger policy head.\n\n"
            "A second lesson is that external data must be treated as a protocol decision rather than a convenient rescue. GlaS is available locally and has a clean morphology-shift role, but its present validation evidence has only one strict episode. DDR has enough paired labels for infrastructure checks, but the stage-one union task is not the formal four-intent lesion protocol. IDRID is recorded as read-only because prior observations could bias future choices. CPM-17 remains deferred because the local assets do not meet the expected complete public-version manifest. These details make the paper less dramatic, but more useful: they tell the next experiment exactly where it can and cannot draw evidence.\n\n"
            r"\subsection{What Evidence Would Change the Conclusion}" "\n"
            "The current conclusion would change only through evidence that is both predeclared and independent of the failed MoNuSAC rescue path. Three conditions are especially important. First, a future action bank must improve candidate coverage before action selection is evaluated. The present post-failure audit points to low component coverage, so another selector trained on the same weak candidate set would mostly learn from a narrow set of already observed successes and failures. A stronger continuation would therefore register the candidate generator, coverage metric, and acceptance threshold before computing any new oracle labels.\n\n"
            "Second, action heterogeneity must be large enough to matter after matched-budget accounting. The current oracle wins are split across actions, but the mean fixed-oracle Dice gap is small. A future gate should require not only a larger mean gap, but also enough query units to estimate whether the gap is stable across cases rather than driven by one or two episodes. This protects the project from promoting an action selector when the observed variation is real but too small, or when the apparent gain is a sampling artifact. The paired sign-flip result in the present artifact is a useful warning: even a favorable mean can be statistically unconvincing when the number of independent units is tiny.\n\n"
            "Third, external-domain evidence must be collected under a fresh role assignment. The next domain should not be selected because it makes the current paper easier to write; it should be selected because its manifest, reference pools, masks, and split logic are complete enough to support the gate before any method choice is tuned. GlaS is currently the most natural morphology-shift continuation because its local assets are complete, but the one-episode smoke is only a wiring check. DDR can stress image scale and lesion morphology, but the available stage-one union protocol is a different task. These distinctions keep follow-up experiments from borrowing credibility from this MoNuSAC pilot.\n\n"
            r"\subsection{Gate Extensions After a Failed Node}" "\n"
            "A practical gate sequence should also say what happens when a node fails for a specific reason. If coverage fails, the allowed work is a candidate-generator diagnostic, such as component coverage by structure size, point precision against a random baseline, or a reference-quality sensitivity table. If action heterogeneity fails, the allowed work is an action-bank analysis that reports fixed-action rankings, oracle win counts, near ties, and matched-budget deltas without training a selector. If statistical support fails, the allowed work is a larger predeclared validation run or a patient-cluster audit, not a stronger wording of the same small pilot. If an external manifest fails, the allowed work is dataset repair and split registration, not an improvised positive result.\n\n"
            "This failure-routing view is the main gate-mechanism refinement suggested by the current paper. The gates should not be only pass/fail labels. They should map each failure mode to a narrow set of legal next actions and a narrow set of prohibited shortcuts. In the current state, the legal next actions are fixed-rule analysis, representation diagnostics, limited candidate diagnostics, GlaS readiness expansion, DDR infrastructure smoke, and CPM-17 asset repair. The prohibited shortcuts remain MoNuSAC policy training, post-hoc MoNuSAC action-bank redesign, and query-time action-bank search. This makes the project easier to continue because a failed node does not leave the researcher guessing; it opens only those experiments that can plausibly repair the missing evidence.\n\n"
            "Table~\\ref{tab:gate-routing} turns this rule into an executable review checklist. The table is intentionally conservative: each failure mode has a diagnostic that can be run without changing the claim, a minimum record needed before promotion, and a shortcut that would invalidate the current interpretation. This is the main mechanism by which AdaPromptSet can use additional local datasets without letting them become ad hoc evidence.\n"
            r"\begin{table*}[t]" "\n"
            r"\centering" "\n"
            r"\small" "\n"
            r"\caption{Failure-to-experiment routing used after the MoNuSAC Gate-F pivot. Allowed continuations preserve the oracle-analysis-only claim unless a future predeclared gate passes with enough independent units.}" "\n"
            r"\label{tab:gate-routing}" "\n"
            r"\begin{tabular}{@{}p{0.18\textwidth}p{0.30\textwidth}p{0.26\textwidth}p{0.18\textwidth}@{}}" "\n"
            r"\toprule" "\n"
            r"Failed evidence node & Allowed continuation & Minimum promotion record & Prohibited shortcut \\" "\n"
            r"\midrule" "\n"
            r"Candidate coverage & Component coverage by target size, point precision against random baseline, and reference-quality sensitivity. & Predeclared coverage threshold plus frozen train/validation split before utility labels are regenerated. & Retuning prompts on MoNuSAC validation cases after seeing oracle wins. \\" "\n"
            r"Action heterogeneity & Fixed-action ranking, oracle win counts, near-tie fraction, and matched-budget deltas. & Larger oracle gap with enough query or patient units to make the gap interpretable. & Training a selector from a small or near-tie utility matrix. \\" "\n"
            r"Statistical support & Query-level paired analysis, patient-cluster audit, and bootstrap only when clusters exist. & Independent case or patient count sufficient for the declared interval or sign test. & Rewording a small pilot as a formal positive result. \\" "\n"
            r"External manifest & Dataset repair, split registration, read-only marking, or infrastructure smoke according to dataset role. & Complete image-mask pairs, non-leaky references, and a fresh domain gate. & Choosing a dataset because it improves the story after MoNuSAC fails. \\" "\n"
            r"Cost accounting & Same-hardware runtime, SAM-call counts, prompt counts, and GPU-hour ledger rows. & Quality and wall-clock evidence reported together under one frozen protocol. & Reporting speed or cost separately from the quality gate. \\" "\n"
            r"\bottomrule" "\n"
            r"\end{tabular}" "\n"
            r"\end{table*}" "\n"
            "\n"
            r"\section{Reproducibility and Audit Trail}" "\n"
            "The current repository is organized as a self-contained MVP package with source code, configs, manifests, reports, and paper assets. The handoff report records the phase status, gate outputs, fixed-oracle summary, representation summary, new-domain readiness, bootstrap status, compute ledger, run metadata, PDF hash, and reproduction commands. The paper is generated from scripts rather than edited as a standalone artifact; submission mode rejects simulated placeholders, and the compiled PDF metadata is audited after build.\n\n"
            r"The reporting chain is intentionally redundant. \texttt{reports/final\_project\_report.json} is the machine-readable summary. \texttt{reports/scientific\_claim\_audit.json} checks that reader-facing files do not contain prohibited claims and that required boundary evidence exists. \texttt{reports/paper\_section\_audit.json} checks that the abstract, related work, method, experiments, results, limitations, and conclusion remain aligned with the oracle-analysis-only state. \texttt{reports/pdf\_metadata\_audit.json} checks the visible title and PDF info fields. This makes the paper rebuildable and keeps future edits from quietly changing the claim boundary."
            "\n\n"
            "The current PDF is not a submission-ready positive-result paper. It is an internally reviewable oracle-analysis paper that documents why the adaptive-policy branch stopped. That distinction is visible in the title metadata, the abstract, the method figure, the action-bank figure, the oracle captions, and the final report. A reviewer can reproduce the path from data audit to gate decision without treating the oracle as an automatic method. This audit trail is the main deliverable at the present state.\n\n"
            r"\section{Limitations}" "\n"
            "The current action bank is an MVP subset and therefore an action bank upper bound rather than a complete prompt design search. "
            "Reference quality sensitivity remains unquantified in the current artifacts. "
            "Frozen matcher errors cannot be recovered by the current selector because no online SAM-response refinement or continuous coordinate generation is allowed. "
            "Utility generation cost remains part of the evidence: running the action bank consumes SAM calls, runtime, and GPU-hours even when the final deployable rule is fixed. "
            "GlaS has only one strict validation episode in the present smoke, and no formal frozen test has been run. "
            "No policy, transfer, efficiency, or leaderboard-level performance claim is supported by the current artifacts.\n\n"
            r"\subsection{What the Pivot Does Not Prove}" "\n"
            "The failed gate does not prove that adaptive prompting is impossible. It proves only that the current bank, candidate evidence, and MoNuSAC MVP split do not justify the next policy-training phase. The low component coverage suggests that some target structures are not well represented by the present candidates. Conversely, the lack of higher-call gain suggests that simply adding grouped prompts is not a guaranteed fix. Both observations narrow the next experimental question, but neither is a deployable conclusion.\n\n"
            "The paper also does not claim that \\OracleAnalysisBestFixed{} is a final deterministic method. Its validation performance is near the action-bank oracle in the pilot, but deterministic promotion would require stronger stability evidence, broader query units, and a fair comparison to fully implemented baselines such as SynPo-style and GF-SAM-style variants with all costs included. Since those rows are not complete formal comparators in the current artifact set, they are not used to create a stronger title or abstract claim.\n\n"
            r"\subsection{Reporting Boundaries}" "\n"
            "All oracle rows are analysis upper bounds. Privileged prompts, when present in diagnostics, are API and coordinate checks rather than automatic candidates. Multi-reference episodes are repeated measurements. The official GlaS test partitions are preserved; the one-episode smoke cannot be promoted to a gate pass. These boundaries are deliberately repetitive across the paper and reports because they are the difference between a useful negative result and an overfit narrative.\n"
        ),
        "07_conclusion.tex": (
            r"\section{Conclusion}" "\n"
            "The current reproducible state supports an oracle-analysis-only report and preserves the registered path for future predeclared extension evidence. AdaPromptSet reached its critical gate and stopped policy training on MoNuSAC because the measured oracle advantage over the best fixed action was too small. That is a valid outcome of the registered design. The next defensible work is not to add another MoNuSAC rescue bank, but to improve candidate coverage under train/validation evidence, run a sufficiently sized predeclared external pilot, or keep the current result as a bounded analysis of why this frozen action bank is near the best fixed rule in the tested setting.\n\n"
            "For internal review, the paper now has a coherent eight-page target structure: problem framing, related-work boundaries, method, protocol, current results, diagnostics, external readiness, reproducibility, limitations, and conclusion. It should be read as a checkpoint paper, not as the end of the entire research program. The key claim is modest but concrete: under the evaluated frozen DINOv2-L and SAM-B pipeline, the current MoNuSAC v2 action bank did not expose enough oracle gap to justify learned prompt-set selection, and the repository records the evidence needed to continue without violating that result.\n"
        ),
    }
    return sections[name]


def write_paper_skeleton(project_root: Path = PROJECT_ROOT, *, draft_watermark: bool = False) -> dict[str, Path]:
    paper_dir = project_root / "paper"
    section_dir = paper_dir / "sections"
    include_representation = (
        project_root / "reports" / "representation_analysis" / "representation_summary.json"
    ).exists()
    for name in [
        "00_abstract.tex",
        "01_intro.tex",
        "02_related.tex",
        "03_method.tex",
        "04_experiments.tex",
        "05_results.tex",
        "06_limitations.tex",
        "07_conclusion.tex",
    ]:
        _write(section_dir / name, _section_text(name, include_representation=include_representation))
    _write(
        paper_dir / "references.bib",
        "% References are intentionally minimal until the formal paper stage.\n",
    )
    simulated_flag = r"\simulatedtrue" if draft_watermark else r"\simulatedfalse"
    watermark_block = (
        "\n"
        r"\ifsimulated"
        "\n"
        r"\twocolumn[{"
        "\n"
        r"\centering\color{red}\bfseries DRAFT WATERMARK: simulated placeholders are forbidden in submission mode."
        "\n"
        r"\vspace{0.5em}"
        "\n"
        r"}]"
        "\n"
        r"\fi"
        "\n"
        if draft_watermark
        else ""
    )
    main_template = r"""\documentclass[10pt,twocolumn]{article}
\usepackage[margin=0.75in]{geometry}
\usepackage{amsmath,amssymb}
\usepackage{booktabs}
\usepackage{graphicx}
\usepackage{tikz}
\usetikzlibrary{arrows.meta,positioning}
\usepackage{xcolor}

__PDFINFO_BLOCK__
\title{AdaPromptSet: Cost-Aware Prompt Set Selection for\\Reference-Guided Dense Medical Segmentation}
\author{__PAPER_AUTHOR__}
\date{__PAPER_DATE__}

% Current gate status: PIVOT_ORACLE_ANALYSIS.
\newif\ifsimulated
__SIMULATED_FLAG__
\input{generated/oracle_analysis_macros}

\begin{document}
\maketitle
__WATERMARK_BLOCK__
\input{sections/00_abstract}
\input{sections/01_intro}
\input{sections/02_related}
\input{sections/03_method}
\input{sections/04_experiments}
\input{sections/05_results}
\input{sections/06_limitations}
\input{sections/07_conclusion}
\end{document}
"""
    main = (
        main_template.replace("__PDFINFO_BLOCK__", pdfinfo_block())
        .replace("__PAPER_AUTHOR__", EXPECTED_PAPER_METADATA.author)
        .replace("__PAPER_DATE__", EXPECTED_PAPER_METADATA.date)
        .replace("__SIMULATED_FLAG__", simulated_flag)
        .replace(
            "__WATERMARK_BLOCK__",
            watermark_block.rstrip(),
        )
    )
    _write(paper_dir / "main.tex", main)
    return {
        "main": paper_dir / "main.tex",
        "references": paper_dir / "references.bib",
    }


def guard_submission_placeholders(project_root: Path = PROJECT_ROOT) -> None:
    paper_dir = project_root / "paper"
    offenders: list[Path] = []
    for path in paper_dir.rglob("*"):
        if not path.is_file() or path.suffix.lower() not in {".tex", ".bib", ".md"}:
            continue
        text = path.read_text(encoding="utf-8", errors="ignore")
        if any(token in text for token in FORBIDDEN_SUBMISSION_TOKENS):
            offenders.append(path)
    if offenders:
        joined = ", ".join(str(path) for path in offenders)
        raise ValueError(f"simulated placeholders remain: {joined}")


def build_paper_draft(project_root: Path = PROJECT_ROOT) -> dict[str, Path]:
    write_oracle_analysis_assets(project_root)
    write_oracle_analysis_figures(project_root)
    write_method_overview_figure(project_root)
    write_action_bank_figure(project_root)
    write_related_work_assets(project_root)
    return write_paper_skeleton(project_root, draft_watermark=True)


def build_paper(project_root: Path = PROJECT_ROOT, *, mode: str = "draft") -> dict[str, Path]:
    write_oracle_analysis_assets(project_root)
    write_oracle_analysis_figures(project_root)
    write_method_overview_figure(project_root)
    write_action_bank_figure(project_root)
    write_related_work_assets(project_root)
    return write_paper_skeleton(project_root, draft_watermark=mode == "draft")


def compile_pdf(project_root: Path = PROJECT_ROOT) -> None:
    for _ in range(2):
        subprocess.run(
            ["pdflatex", "-interaction=nonstopmode", "-halt-on-error", "main.tex"],
            cwd=project_root / "paper",
            check=True,
        )


def main() -> None:
    parser = argparse.ArgumentParser(description="Build the AdaPromptSet claim-free paper skeleton.")
    parser.add_argument("--project-root", type=Path, default=PROJECT_ROOT)
    parser.add_argument("--mode", choices=["draft", "submission"], default="draft")
    parser.add_argument("--compile", action="store_true")
    args = parser.parse_args()
    build_paper(args.project_root, mode=args.mode)
    if args.mode == "submission":
        guard_submission_placeholders(args.project_root)
    if args.compile:
        compile_pdf(args.project_root)


if __name__ == "__main__":
    main()
