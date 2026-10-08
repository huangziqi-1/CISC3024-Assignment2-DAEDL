# Final Submission Audit

Date: 2026-10-08 (Asia/Taipei)

Status: **Complete.** The public repository URL is verified, the Source Code section is present, and the final Word and PDF reports have been generated and validated.

## Files

- Editable Word report: `report/CISC3024_AI_Assignment2_Final.docx`
- Report source: `report/CISC3024_AI_Assignment2_Final.md`
- Final PDF: `report/CISC3024_AI_Assignment2_Final.pdf`
- Selected figures: `figures/final/`
- Machine-readable final evidence: `results/final_summary/`

The rendered Word draft has **14 pages**, including one title page and one reference page. It contains **4 figures**, **4 tables**, **5 rendered equations**, and **7 references**.

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

Verified public URL: https://github.com/huangziqi-1/CISC3024-Assignment2-DAEDL

- Current branch: `main`
- Scientific-results baseline commit: `ec5d03ea106fe1d9a524b3087fc8d181bcee4460`
- Final submission commit: the repository HEAD containing this audit; its immutable hash is verified after commit and push
- Configured remote: `https://github.com/huangziqi-1/CISC3024-Assignment2-DAEDL.git`

## Numerical Checks

Every displayed final metric was read from `results/final_summary/aggregate_metrics.csv`, which was generated from the validated Phase-3B per-seed predictions. The following required values appear in both the Word and Markdown reports:

| Metric | Softmax | DAEDL |
|---|---:|---:|
| Accuracy | 0.9013 ± 0.0004 | 0.8902 ± 0.0012 |
| Macro F1 | 0.9012 ± 0.0004 | 0.8899 ± 0.0011 |
| NLL | 0.3367 ± 0.0059 | 0.4328 ± 0.0017 |
| ECE-15 | 0.0463 ± 0.0031 | 0.0261 ± 0.0030 |
| Brier score | 0.1504 ± 0.0014 | 0.1718 ± 0.0005 |
| Error AUROC | 0.9104 ± 0.0037 | 0.9034 ± 0.0031 |
| SVHN OOD AUROC | 0.8860 ± 0.0140 | 0.8762 ± 0.0238 |

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
- Student source-code repository: verified at https://github.com/huangziqi-1/CISC3024-Assignment2-DAEDL

## Repository and Experiment Scope

- Scientific-results commit audited before final packaging: `ec5d03ea106fe1d9a524b3087fc8d181bcee4460`
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
