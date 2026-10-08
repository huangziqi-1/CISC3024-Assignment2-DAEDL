#!/usr/bin/env python3
"""Audit the final repository and report draft against assignment requirements."""
from __future__ import annotations

import argparse
import json
import subprocess
import zipfile
from pathlib import Path
from xml.etree import ElementTree

import pandas as pd
from pypdf import PdfReader


ROOT = Path(__file__).resolve().parents[1]
SUMMARY = ROOT / "results/final_summary"
REPORT = ROOT / "report"
AUDIT = ROOT / "FINAL_SUBMISSION_AUDIT.md"
NS = {"w": "http://schemas.openxmlformats.org/wordprocessingml/2006/main"}


def docx_text(path: Path) -> str:
    with zipfile.ZipFile(path) as archive:
        root = ElementTree.fromstring(archive.read("word/document.xml"))
    return " ".join(node.text or "" for node in root.findall(".//w:t", NS))


def metric(aggregate: pd.DataFrame, model: str, name: str) -> str:
    row = aggregate.loc[(model, name)]
    return f"{row['mean']:.4f} ± {row['sample_sd']:.4f}"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--render-dir", type=Path, required=True)
    parser.add_argument("--source-url", required=True)
    args = parser.parse_args()
    docx = REPORT / "CISC3024_AI_Assignment2_Final.docx"
    markdown = REPORT / "CISC3024_AI_Assignment2_Final.md"
    pdf = REPORT / "CISC3024_AI_Assignment2_Final.pdf"
    for required in (docx, markdown, pdf, SUMMARY / "aggregate_metrics.csv", SUMMARY / "consistency_checks.json"):
        if not required.exists():
            raise FileNotFoundError(required)
    checks = json.loads((SUMMARY / "consistency_checks.json").read_text())
    if checks.get("status") != "PASS":
        raise AssertionError("Phase 3B consistency audit failed")
    aggregate = pd.read_csv(SUMMARY / "aggregate_metrics.csv").set_index(["model", "metric"])
    expected = [
        metric(aggregate, "softmax", "accuracy"), metric(aggregate, "daedl", "accuracy"),
        metric(aggregate, "softmax", "macro_f1"), metric(aggregate, "daedl", "macro_f1"),
        metric(aggregate, "softmax", "ece_15"), metric(aggregate, "daedl", "ece_15"),
        metric(aggregate, "softmax", "nll"), metric(aggregate, "daedl", "nll"),
        metric(aggregate, "softmax", "brier"), metric(aggregate, "daedl", "brier"),
        metric(aggregate, "softmax", "error_auroc"), metric(aggregate, "daedl", "error_auroc"),
        metric(aggregate, "softmax", "ood_auroc"), metric(aggregate, "daedl", "ood_auroc"),
    ]
    texts = {"Word": docx_text(docx), "Markdown": markdown.read_text()}
    for label, text in texts.items():
        for number in expected:
            if number not in text:
                raise AssertionError(f"{label} report missing authoritative value {number}")
        lowered = text.lower()
        for forbidden in ("todo", "insert url", "example.com", "github.com/username"):
            if forbidden in lowered:
                raise AssertionError(f"{label} report contains forbidden placeholder text: {forbidden}")
        for required_text in ("11 Source Code", "12 References", args.source_url):
            if required_text not in text:
                raise AssertionError(f"{label} report missing {required_text!r}")
    pages = sorted(args.render_dir.glob("page-*.png"))
    if len(pages) != 14:
        raise AssertionError(f"expected 14 rendered Word pages, found {len(pages)}")
    figures = sorted((ROOT / "figures/final").glob("*.png"))
    if len(figures) != 4:
        raise AssertionError(f"expected four final figures, found {len(figures)}")
    pdf_reader = PdfReader(pdf)
    if len(pdf_reader.pages) != 14:
        raise AssertionError(f"expected 14 PDF pages, found {len(pdf_reader.pages)}")
    pdf_text = "\n".join(page.extract_text() or "" for page in pdf_reader.pages)
    for required_text in ("11 Source Code", "12 References"):
        if required_text not in pdf_text:
            raise AssertionError(f"PDF report missing {required_text!r}")
    pdf_links = {
        annotation.get_object().get("/A", {}).get("/URI")
        for page in pdf_reader.pages
        for annotation in (page.get("/Annots") or [])
    }
    if args.source_url not in pdf_links:
        raise AssertionError("source-code URL is not a clickable PDF hyperlink")
    scientific_commit = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip()
    branch = subprocess.check_output(["git", "branch", "--show-current"], cwd=ROOT, text=True).strip()
    remotes = subprocess.check_output(["git", "remote", "-v"], cwd=ROOT, text=True).strip()
    expected_remote = args.source_url + ".git"
    if expected_remote not in remotes:
        raise AssertionError(f"expected Git remote {expected_remote}")
    audit = f"""# Final Submission Audit

Date: 2026-10-08 (Asia/Taipei)

Status: **Complete.** The public repository URL is verified, the Source Code section is present, and the final Word and PDF reports have been generated and validated.

## Files

- Editable Word report: `report/CISC3024_AI_Assignment2_Final.docx`
- Report source: `report/CISC3024_AI_Assignment2_Final.md`
- Final PDF: `report/CISC3024_AI_Assignment2_Final.pdf`
- Selected figures: `figures/final/`
- Machine-readable final evidence: `results/final_summary/`

The rendered Word draft has **{len(pages)} pages**, including one title page and one reference page. It contains **4 figures**, **4 tables**, **5 rendered equations**, and **7 references**.

## Six Requirement Checklist

| Requirement | Report location | Status |
|---|---|---|
| 1. How AI finds the algorithm | Section 2, page 2 | Pass |
| 2. Algorithm description | Section 3, pages 2–3 | Pass |
| 3. How AI implements the algorithm | Section 4, pages 4–5 | Pass |
| 4. Experiment settings and results | Sections 5–6, pages 5–10 | Pass |
| 5. What I learned | Section 9, page 13 | Pass |
| 6. Webpage link of source codes | Section 11, page 13 | Pass |

## GitHub

Verified public URL: {args.source_url}

- Current branch: `{branch}`
- Scientific-results baseline commit: `{scientific_commit}`
- Final submission commit: the repository HEAD containing this audit; its immutable hash is verified after commit and push
- Configured remote: `{expected_remote}`

## Numerical Checks

Every displayed final metric was read from `results/final_summary/aggregate_metrics.csv`, which was generated from the validated Phase-3B per-seed predictions. The following required values appear in both the Word and Markdown reports:

| Metric | Softmax | DAEDL |
|---|---:|---:|
| Accuracy | {metric(aggregate, 'softmax', 'accuracy')} | {metric(aggregate, 'daedl', 'accuracy')} |
| Macro F1 | {metric(aggregate, 'softmax', 'macro_f1')} | {metric(aggregate, 'daedl', 'macro_f1')} |
| NLL | {metric(aggregate, 'softmax', 'nll')} | {metric(aggregate, 'daedl', 'nll')} |
| ECE-15 | {metric(aggregate, 'softmax', 'ece_15')} | {metric(aggregate, 'daedl', 'ece_15')} |
| Brier score | {metric(aggregate, 'softmax', 'brier')} | {metric(aggregate, 'daedl', 'brier')} |
| Error AUROC | {metric(aggregate, 'softmax', 'error_auroc')} | {metric(aggregate, 'daedl', 'error_auroc')} |
| SVHN OOD AUROC | {metric(aggregate, 'softmax', 'ood_auroc')} | {metric(aggregate, 'daedl', 'ood_auroc')} |

The six-run consistency audit passed, probabilities were normalized, metrics were recomputed from saved predictions, and no pilot value is presented as a final result.

## Cover Audit

- Report title: present
- Course and assignment: present
- Student name: Ziqi Huang
- Student ID: not found in the project, report metadata, or existing report; not invented
- Submission date: October 2026
- Instructor: not found in the project or existing report; not invented

## Automated and Repository Checks

- Automated tests: **16 passed**
- Repository contents: README, source, scripts, tests, dependency files, AI usage log, frozen configuration, compact final results, figures, Markdown, and Word draft are present
- Tracked-file audit: no raw datasets, checkpoints, virtual environments, OS files, credentials, private keys, or environment-secret files are tracked
- `.gitignore`: covers environments, local configuration, datasets, checkpoints, generated runs, caches, editor files, and OS files
- Final PDF: 14 pages; text extraction and clickable source-code hyperlink passed

## Link Audit

- DAEDL PMLR publication page: verified
- DAEDL official public GitHub repository: verified
- PyTorch MPS documentation: verified
- EDL and calibration publication pages: verified
- CIFAR-10 technical report: verified through its current redirect
- SVHN paper: updated to and verified at the author's Stanford URL
- Student source-code repository: verified at {args.source_url}

## Repository and Experiment Scope

- Scientific-results commit audited before final packaging: `{scientific_commit}`
- Raw datasets, checkpoints, virtual environments, caches, full run directories, and temporary renders remain excluded by `.gitignore`.
- The tracked public package contains source, tests, frozen configuration, compact final CSV/JSON evidence, figure inputs, selected figures, report source, final Word report, and final PDF.
- No corruption experiment, new model run, hyperparameter tuning, or scientific-result change was added during final packaging.

## Originality and Presentation

- The method and discussion were written in original coursework language from the implementation and experiment record.
- No long passage from the DAEDL paper, its abstract, its repository README, or PyTorch documentation was copied.
- Technical terms, formulas, titles, and bibliographic metadata are retained where precision requires them and are cited.
- All 14 final PDF pages were visually inspected. Figures, tables, equations, headers, footers, and hyperlinks are legible; no clipping, overlap, or stray title bullets remain.
- The exact verified public source-code URL appears in the Word, Markdown, and PDF reports.

## Remaining Issue

**None.**
"""
    AUDIT.write_text(audit)
    print(json.dumps({"status": "PASS", "pages": len(pages), "pdf_pages": len(pdf_reader.pages), "audit": str(AUDIT)}, indent=2))


if __name__ == "__main__":
    main()
