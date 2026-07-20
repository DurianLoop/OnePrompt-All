from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class PaperMetadata:
    title: str
    author: str
    date: str
    subject: str
    keywords: str


EXPECTED_PAPER_METADATA = PaperMetadata(
    title="AdaPromptSet: Cost-Aware Prompt Set Selection for Reference-Guided Dense Medical Segmentation",
    author="Internal MVP Draft",
    date="2026-07-17",
    subject="Oracle-analysis-only AdaPromptSet MVP artifact after Gate-F pivot",
    keywords="AdaPromptSet, oracle analysis, prompt set selection, medical segmentation",
)


def pdfinfo_block(metadata: PaperMetadata = EXPECTED_PAPER_METADATA) -> str:
    return "\n".join(
        [
            r"\pdfinfo{",
            f"  /Title ({metadata.title})",
            f"  /Author ({metadata.author})",
            f"  /Subject ({metadata.subject})",
            f"  /Keywords ({metadata.keywords})",
            r"}",
        ]
    )
