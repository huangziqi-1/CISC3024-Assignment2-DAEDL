# Final Submission Audit

Date: 2026-10-08 (Asia/Taipei)

Status: **One manual publication dependency remains.** The report and repository package are complete, but the final source-code URL is unavailable because this checkout has no Git remote and GitHub CLI is not installed. In accordance with the submission instructions, no URL was fabricated and the final PDF was not generated.

## Files

- Editable Word draft: `report/CISC3024_AI_Assignment2_Final.docx`
- Report source: `report/CISC3024_AI_Assignment2_Final.md`
- Final PDF: not generated pending the verified public source-code URL
- Selected figures: `figures/final/`
- Machine-readable final evidence: `results/final_summary/`

The rendered Word draft has **12 pages**, including one title page and one reference page. It contains **4 figures**, **3 tables**, **5 rendered equations**, and **7 references**.

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
| Accuracy | 0.9013 ± 0.0004 | 0.8902 ± 0.0012 |
| NLL | 0.3367 ± 0.0059 | 0.4328 ± 0.0017 |
| ECE-15 | 0.0463 ± 0.0031 | 0.0261 ± 0.0030 |
| Error AUROC | 0.9104 ± 0.0037 | 0.9034 ± 0.0031 |
| SVHN OOD AUROC | 0.8860 ± 0.0140 | 0.8762 ± 0.0238 |

The six-run consistency audit passed, probabilities were normalized, metrics were recomputed from saved predictions, and no pilot value is presented as a final result.

## Repository and Experiment Scope

- Scientific-results commit audited before final packaging: `9e38d2410a811535da4167b7823ad8fc4636876e`
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
