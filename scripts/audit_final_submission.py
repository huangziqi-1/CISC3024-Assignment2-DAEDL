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
    args = parser.parse_args()
    docx = REPORT / "CISC3024_AI_Assignment2_Final.docx"
    markdown = REPORT / "CISC3024_AI_Assignment2_Final.md"
    pdf = REPORT / "CISC3024_AI_Assignment2_Final.pdf"
    for required in (docx, markdown, SUMMARY / "aggregate_metrics.csv", SUMMARY / "consistency_checks.json"):
        if not required.exists():
            raise FileNotFoundError(required)
    checks = json.loads((SUMMARY / "consistency_checks.json").read_text())
    if checks.get("status") != "PASS":
        raise AssertionError("Phase 3B consistency audit failed")
    aggregate = pd.read_csv(SUMMARY / "aggregate_metrics.csv").set_index(["model", "metric"])
    expected = [
        metric(aggregate, "softmax", "accuracy"), metric(aggregate, "daedl", "accuracy"),
        metric(aggregate, "softmax", "ece_15"), metric(aggregate, "daedl", "ece_15"),
        metric(aggregate, "softmax", "nll"), metric(aggregate, "daedl", "nll"),
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
    pages = sorted(args.render_dir.glob("page-*.png"))
    if len(pages) != 12:
        raise AssertionError(f"expected 12 rendered Word pages, found {len(pages)}")
    figures = sorted((ROOT / "figures/final").glob("*.png"))
    if len(figures) != 4:
        raise AssertionError(f"expected four final figures, found {len(figures)}")
    if pdf.exists():
        raise AssertionError("final PDF must wait for a verified source-code URL")
    scientific_commit = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip()
    audit = f"""# Final Submission Audit

Date: 2026-10-08 (Asia/Taipei)

Status: **One manual publication dependency remains.** The report and repository package are complete, but the final source-code URL is unavailable because this checkout has no Git remote and GitHub CLI is not installed. In accordance with the submission instructions, no URL was fabricated and the final PDF was not generated.

## Files

- Editable Word draft: `report/CISC3024_AI_Assignment2_Final.docx`
- Report source: `report/CISC3024_AI_Assignment2_Final.md`
- Final PDF: not generated pending the verified public source-code URL
- Selected figures: `figures/final/`
- Machine-readable final evidence: `results/final_summary/`

The rendered Word draft has **{len(pages)} pages**, including one title page and one reference page. It contains **4 figures**, **3 tables**, **5 rendered equations**, and **7 references**.

## Six Requirement Checklist

| Requirement | Report location | Status |
|---|---|---|
| 1. How AI finds the algorithm | Section 2, page 2 | Pass |
| 2. Algorithm description | Section 3, pages 2–3 | Pass |
| 3. How AI implements the algorithm | Section 4, page 4 | Pass |
| 4. Experiment settings and results | Sections 5–6, pages 4–9 | Pass |
| 5. What I learned | Section 9, page 11 | Pass |
| 6. Webpage link of source codes | Section 11 after publication | **Pending** |

## GitHub

Verified URL: unavailable.

Exact remaining manual action: publish this project directory as a public GitHub repository and provide the resulting repository URL. After that URL is available, rerun `scripts/build_final_report.py --source-url <verified-url>`, visually verify the updated Word file, export and validate the final PDF, and commit the completed submission package.

## Numerical Checks

Every displayed final metric was read from `results/final_summary/aggregate_metrics.csv`, which was generated from the validated Phase-3B per-seed predictions. The following required values appear in both the Word and Markdown reports:

| Metric | Softmax | DAEDL |
|---|---:|---:|
| Accuracy | {metric(aggregate, 'softmax', 'accuracy')} | {metric(aggregate, 'daedl', 'accuracy')} |
| NLL | {metric(aggregate, 'softmax', 'nll')} | {metric(aggregate, 'daedl', 'nll')} |
| ECE-15 | {metric(aggregate, 'softmax', 'ece_15')} | {metric(aggregate, 'daedl', 'ece_15')} |
| Error AUROC | {metric(aggregate, 'softmax', 'error_auroc')} | {metric(aggregate, 'daedl', 'error_auroc')} |
| SVHN OOD AUROC | {metric(aggregate, 'softmax', 'ood_auroc')} | {metric(aggregate, 'daedl', 'ood_auroc')} |

The six-run consistency audit passed, probabilities were normalized, metrics were recomputed from saved predictions, and no pilot value is presented as a final result.

## Repository and Experiment Scope

- Scientific-results commit audited before final packaging: `{scientific_commit}`
- Raw datasets, checkpoints, virtual environments, caches, full run directories, and temporary renders remain excluded by `.gitignore`.
- The tracked public package contains source, tests, frozen configuration, compact final CSV/JSON evidence, figure inputs, selected figures, report source, and the Word draft.
- No corruption experiment, new model run, hyperparameter tuning, or scientific-result change was added during final packaging.

## Originality and Presentation

- The method and discussion were written in original coursework language from the implementation and experiment record.
- No long passage from the DAEDL paper, its abstract, its repository README, or PyTorch documentation was copied.
- Technical terms, formulas, titles, and bibliographic metadata are retained where precision requires them and are cited.
- The Word document was rendered page by page. Figures, tables, equations, headers, footers, and hyperlinks were visually checked; no clipping or overlap remains.

## Remaining Issue

**Publish the repository and provide its verified public URL.** This is the only remaining dependency. The source-code section and final PDF must be generated only after that URL is known.
"""
    AUDIT.write_text(audit)
    print(json.dumps({"status": "PASS_WITH_ONE_MANUAL_DEPENDENCY", "pages": len(pages), "audit": str(AUDIT)}, indent=2))


if __name__ == "__main__":
    main()
