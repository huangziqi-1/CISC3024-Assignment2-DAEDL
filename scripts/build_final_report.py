#!/usr/bin/env python3
"""Generate the final report source and editable Word document from validated data."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd
from docx import Document
from docx.enum.style import WD_STYLE_TYPE
from docx.enum.table import WD_CELL_VERTICAL_ALIGNMENT, WD_TABLE_ALIGNMENT
from docx.enum.text import WD_ALIGN_PARAGRAPH
from docx.oxml import OxmlElement
from docx.oxml.ns import qn
from docx.shared import Inches, Pt, RGBColor

ROOT = Path(__file__).resolve().parents[1]
SUMMARY = ROOT / "results/final_summary"
FIGURES = ROOT / "figures/final"
REPORT = ROOT / "report"
EQUATIONS = REPORT / "equations"

def load_results() -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    aggregate = pd.read_csv(SUMMARY / "aggregate_metrics.csv").set_index(["model", "metric"])
    per_seed = pd.read_csv(SUMMARY / "per_seed_metrics.csv")
    density = pd.read_csv(SUMMARY / "density_diagnostics_by_seed.csv")
    checks = json.loads((SUMMARY / "consistency_checks.json").read_text())
    if checks.get("status") != "PASS":
        raise AssertionError("final report requires a passing Phase 3B audit")
    if set(per_seed.seed) != {3024, 3025, 3026} or set(per_seed.model) != {"softmax", "daedl"}:
        raise AssertionError("unexpected seed/model set")
    return aggregate, per_seed, density


def value(aggregate: pd.DataFrame, model: str, metric: str) -> str:
    row = aggregate.loc[(model, metric)]
    return f"{row['mean']:.4f} ± {row['sample_sd']:.4f}"


def set_cell_shading(cell, fill: str) -> None:
    properties = cell._tc.get_or_add_tcPr()
    shading = properties.find(qn("w:shd"))
    if shading is None:
        shading = OxmlElement("w:shd")
        properties.append(shading)
    shading.set(qn("w:fill"), fill)


def set_cell_border(cell, color: str = "D9D9D9", size: str = "6") -> None:
    properties = cell._tc.get_or_add_tcPr()
    borders = properties.find(qn("w:tcBorders"))
    if borders is None:
        borders = OxmlElement("w:tcBorders")
        properties.append(borders)
    for edge in ("top", "left", "bottom", "right", "insideH", "insideV"):
        tag = borders.find(qn(f"w:{edge}"))
        if tag is None:
            tag = OxmlElement(f"w:{edge}")
            borders.append(tag)
        tag.set(qn("w:val"), "single")
        tag.set(qn("w:sz"), size)
        tag.set(qn("w:color"), color)


def set_cell_margins(cell, top=90, start=100, bottom=90, end=100) -> None:
    properties = cell._tc.get_or_add_tcPr()
    margins = properties.find(qn("w:tcMar"))
    if margins is None:
        margins = OxmlElement("w:tcMar")
        properties.append(margins)
    for name, amount in (("top", top), ("start", start), ("bottom", bottom), ("end", end)):
        node = margins.find(qn(f"w:{name}"))
        if node is None:
            node = OxmlElement(f"w:{name}")
            margins.append(node)
        node.set(qn("w:w"), str(amount))
        node.set(qn("w:type"), "dxa")


def set_repeat_table_header(row) -> None:
    properties = row._tr.get_or_add_trPr()
    repeat = OxmlElement("w:tblHeader")
    repeat.set(qn("w:val"), "true")
    properties.append(repeat)


def add_table(document: Document, headers: list[str], rows: list[list[str]], widths: list[float] | None = None):
    table = document.add_table(rows=1, cols=len(headers))
    table.alignment = WD_TABLE_ALIGNMENT.CENTER
    table.autofit = False
    set_repeat_table_header(table.rows[0])
    for index, header in enumerate(headers):
        cell = table.rows[0].cells[index]
        cell.text = header
        cell.vertical_alignment = WD_CELL_VERTICAL_ALIGNMENT.CENTER
        set_cell_shading(cell, "17365D")
        set_cell_border(cell)
        set_cell_margins(cell)
        for paragraph in cell.paragraphs:
            paragraph.alignment = WD_ALIGN_PARAGRAPH.CENTER
            for run in paragraph.runs:
                run.font.bold = True
                run.font.color.rgb = RGBColor(255, 255, 255)
                run.font.size = Pt(9)
    for row_index, values in enumerate(rows):
        cells = table.add_row().cells
        for column, item in enumerate(values):
            cells[column].text = str(item)
            cells[column].vertical_alignment = WD_CELL_VERTICAL_ALIGNMENT.CENTER
            set_cell_border(cells[column])
            set_cell_margins(cells[column])
            if row_index % 2:
                set_cell_shading(cells[column], "F2F6FA")
            for paragraph in cells[column].paragraphs:
                paragraph.alignment = WD_ALIGN_PARAGRAPH.LEFT if column == 0 else WD_ALIGN_PARAGRAPH.CENTER
                for run in paragraph.runs:
                    run.font.size = Pt(9)
    if widths:
        for row in table.rows:
            for cell, width in zip(row.cells, widths):
                cell.width = Inches(width)
    return table


def add_field(run, instruction: str) -> None:
    begin = OxmlElement("w:fldChar")
    begin.set(qn("w:fldCharType"), "begin")
    text = OxmlElement("w:instrText")
    text.set(qn("xml:space"), "preserve")
    text.text = instruction
    end = OxmlElement("w:fldChar")
    end.set(qn("w:fldCharType"), "end")
    run._r.extend([begin, text, end])


def add_hyperlink(paragraph, text: str, url: str) -> None:
    relation = paragraph.part.relate_to(url, "http://schemas.openxmlformats.org/officeDocument/2006/relationships/hyperlink", is_external=True)
    hyperlink = OxmlElement("w:hyperlink")
    hyperlink.set(qn("r:id"), relation)
    run = OxmlElement("w:r")
    properties = OxmlElement("w:rPr")
    color = OxmlElement("w:color")
    color.set(qn("w:val"), "0563C1")
    underline = OxmlElement("w:u")
    underline.set(qn("w:val"), "single")
    properties.extend([color, underline])
    run.append(properties)
    text_node = OxmlElement("w:t")
    text_node.text = text
    run.append(text_node)
    hyperlink.append(run)
    paragraph._p.append(hyperlink)


def keep_with_next(paragraph) -> None:
    paragraph.paragraph_format.keep_with_next = True


def remove_paragraph_border(paragraph_or_style) -> None:
    element = paragraph_or_style._p if hasattr(paragraph_or_style, "_p") else paragraph_or_style.element
    properties = element.get_or_add_pPr()
    border = properties.find(qn("w:pBdr"))
    if border is not None:
        properties.remove(border)


def add_caption(document: Document, label: str, text: str):
    paragraph = document.add_paragraph(style="Caption")
    paragraph.alignment = WD_ALIGN_PARAGRAPH.CENTER
    run = paragraph.add_run(f"{label}. ")
    run.bold = True
    paragraph.add_run(text)
    keep_with_next(paragraph)
    return paragraph


def add_figure(document: Document, path: Path, width: float):
    paragraph = document.add_paragraph()
    paragraph.alignment = WD_ALIGN_PARAGRAPH.CENTER
    paragraph.paragraph_format.keep_together = True
    paragraph.add_run().add_picture(str(path), width=Inches(width))
    return paragraph


def add_equation(document: Document, name: str, width: float = 6.2):
    paragraph = document.add_paragraph()
    paragraph.alignment = WD_ALIGN_PARAGRAPH.CENTER
    paragraph.paragraph_format.space_before = Pt(2)
    paragraph.paragraph_format.space_after = Pt(6)
    paragraph.add_run().add_picture(str(EQUATIONS / name), width=Inches(width))


def configure_document(document: Document) -> None:
    section = document.sections[0]
    section.page_width = Inches(8.5)
    section.page_height = Inches(11)
    section.top_margin = Inches(0.72)
    section.bottom_margin = Inches(0.72)
    section.left_margin = Inches(0.8)
    section.right_margin = Inches(0.8)
    section.different_first_page_header_footer = True
    section.first_page_header.paragraphs[0].text = ""
    section.first_page_footer.paragraphs[0].text = ""

    styles = document.styles
    normal = styles["Normal"]
    normal.font.name = "Aptos"
    normal.font.size = Pt(10.5)
    normal.font.color.rgb = RGBColor(0, 0, 0)
    normal.paragraph_format.space_after = Pt(6)
    normal.paragraph_format.line_spacing = 1.08
    for style_name, size, before, after in (
        ("Title", 22, 0, 12), ("Heading 1", 15, 10, 5),
        ("Heading 2", 12.5, 8, 4), ("Heading 3", 11, 6, 3),
    ):
        style = styles[style_name]
        style.font.name = "Aptos Display" if style_name != "Normal" else "Aptos"
        style.font.size = Pt(size)
        style.font.color.rgb = RGBColor(0, 0, 0)
        style.font.bold = True
        style.paragraph_format.space_before = Pt(before)
        style.paragraph_format.space_after = Pt(after)
        style.paragraph_format.keep_with_next = True
    styles["Title"].font.bold = True
    styles["Title"].font.color.rgb = RGBColor(0, 0, 0)
    remove_paragraph_border(styles["Title"])
    styles["Caption"].font.name = "Aptos"
    styles["Caption"].font.size = Pt(9)
    styles["Caption"].font.italic = False
    styles["Caption"].font.color.rgb = RGBColor(45, 45, 45)
    styles["Caption"].paragraph_format.space_after = Pt(4)
    if "References" not in styles:
        reference_style = styles.add_style("References", WD_STYLE_TYPE.PARAGRAPH)
        reference_style.base_style = styles["Normal"]
        reference_style.font.name = "Aptos"
        reference_style.font.size = Pt(9)
        reference_style.paragraph_format.left_indent = Inches(0.22)
        reference_style.paragraph_format.first_line_indent = Inches(-0.22)
        reference_style.paragraph_format.space_after = Pt(4)

    for page_section in document.sections:
        header = page_section.header.paragraphs[0]
        header.text = "CISC3024 Pattern Recognition  |  AI Assignment #2"
        header.alignment = WD_ALIGN_PARAGRAPH.RIGHT
        for run in header.runs:
            run.font.name = "Aptos"
            run.font.size = Pt(8)
            run.font.color.rgb = RGBColor(90, 90, 90)
        footer = page_section.footer.paragraphs[0]
        footer.alignment = WD_ALIGN_PARAGRAPH.CENTER
        run = footer.add_run("Page ")
        run.font.size = Pt(8)
        add_field(footer.add_run(), "PAGE")


def paragraph(document: Document, text: str, bold_lead: str | None = None):
    p = document.add_paragraph()
    if bold_lead and text.startswith(bold_lead):
        p.add_run(bold_lead).bold = True
        p.add_run(text[len(bold_lead):])
    else:
        p.add_run(text)
    return p


def bullet(document: Document, text: str):
    p = document.add_paragraph(style="List Bullet")
    p.paragraph_format.space_after = Pt(3)
    p.add_run(text)
    return p


def build_markdown(aggregate: pd.DataFrame, density: pd.DataFrame, source_url: str | None) -> str:
    s_acc, d_acc = value(aggregate, "softmax", "accuracy"), value(aggregate, "daedl", "accuracy")
    s_f1, d_f1 = value(aggregate, "softmax", "macro_f1"), value(aggregate, "daedl", "macro_f1")
    s_nll, d_nll = value(aggregate, "softmax", "nll"), value(aggregate, "daedl", "nll")
    s_ece, d_ece = value(aggregate, "softmax", "ece_15"), value(aggregate, "daedl", "ece_15")
    s_brier, d_brier = value(aggregate, "softmax", "brier"), value(aggregate, "daedl", "brier")
    s_err, d_err = value(aggregate, "softmax", "error_auroc"), value(aggregate, "daedl", "error_auroc")
    s_errp, d_errp = value(aggregate, "softmax", "error_aupr"), value(aggregate, "daedl", "error_aupr")
    s_ood, d_ood = value(aggregate, "softmax", "ood_auroc"), value(aggregate, "daedl", "ood_auroc")
    s_oodp, d_oodp = value(aggregate, "softmax", "ood_aupr"), value(aggregate, "daedl", "ood_aupr")
    source_section = f"\n## 11 Source Code\n\n**Source code:** {source_url}\n" if source_url else ""
    return fr"""# AI Assisted Implementation and Evaluation of Density Aware Evidential Deep Learning

**CISC3024 Pattern Recognition**  
**AI Assignment #2**  
October 2026

## 1 Introduction

Image classifiers produce probability-like scores, but a high maximum score does not by itself show that the model knows when it may be wrong. This assignment used Codex to identify, implement, and evaluate a recent probability-based recognition method. The selected method was Density Aware Evidential Deep Learning (DAEDL), published at ICML 2024 [1]. The research question was: how does DAEDL compare with a conventional Softmax classifier in classification performance, calibration, prediction-error uncertainty, and out-of-distribution uncertainty in an assignment-scale CIFAR-10 experiment?

The three-seed results show a limited trade-off rather than an overall DAEDL advantage. Softmax achieved higher accuracy ({s_acc} versus {d_acc}) and better NLL and Brier score. DAEDL achieved lower ECE in every seed ({d_ece} versus {s_ece}). Both methods assigned greater entropy to mistakes, while Softmax had a small error-AUROC advantage. SVHN OOD rankings changed by seed, so the experiment did not establish a stable OOD winner.

## 2 AI Assisted Algorithm Search and Selection

Codex was asked to find a recent classification or recognition algorithm grounded in probability or Bayesian theory. It filtered candidates using publication date, relevance to computer-vision classification, access to a primary paper and implementation, computational feasibility on a laptop, and whether the method supported a clear comparison with a conventional baseline. DAEDL met these criteria because it predicts a Dirichlet distribution, models density in learned feature space, and reports classification, calibration, error detection, and OOD experiments [1].

Codex checked the official PMLR record, the paper, and the authors' repository. It did not treat the repository as a ready-made solution: the audit found CUDA assumptions, incomplete environment information, missing released checkpoints, and reproducibility gaps. A clean implementation was therefore used, with the official code serving as a method reference. Conventional evidential deep learning [2] and calibration work [3] supplied the background needed to interpret the method and metrics.

## 3 Algorithm Description

### 3.1 Softmax classification

For class logits $z_k(x)$, Softmax forms normalized class probabilities:

$$p_{{soft}}(y=k\mid x)=\frac{{e^{{z_k(x)}}}}{{\sum_j e^{{z_j(x)}}}}.$$

### 3.2 Evidential classification

Evidential classification predicts positive concentrations $\boldsymbol{{\alpha}}(x)$ for a Dirichlet distribution over class-probability vectors $\boldsymbol{{\pi}}$. Its predictive categorical probability is the Dirichlet mean, $\mathbb{{E}}[\pi_k\mid x]=\alpha_k/S$, where $S=\sum_j\alpha_j$. Strength describes concentration, but this project used normalized predictive entropy as the common uncertainty score so that Softmax and DAEDL were compared on the same quantity.

### 3.3 DAEDL training and density-aware prediction

DAEDL uses $\alpha_k=e^{{z_k}}$ during training. Normalizing these concentrations gives the same class probabilities as Softmax, while the loss adds the expected squared-error and predictive-variance terms of the Dirichlet plus a KL penalty for misleading evidence. After training, the implementation fits class-conditional full-covariance Gaussian distributions to the 128-dimensional penultimate embeddings. The class mixture defines a feature density, which is min-max normalized against training log densities and clipped to $[0,1]$.

At inference, the density score $s(x)$ scales the logits before concentrations are formed: $\alpha_k(x)=\exp(s(x)z_k(x))$. A low-density input drives $s(x)$ toward zero, so the expected class probabilities move toward a uniform distribution. Spectral normalization is applied to both compared models in this project so that the backbone treatment is matched; for DAEDL it also supports a smoother feature geometry for density estimation.

![Figure 1 DAEDL concept](../figures/final/figure1_daedl_concept.png)

## 4 AI Assisted Implementation

Codex inspected the paper and official code, created an isolated Python 3.11 environment, and implemented the data pipeline, CIFAR-adapted ResNet-18, Softmax and DAEDL losses, CPU float64 GDA, metrics, artifact logging, and plots. Automated tests covered probability normalization, DAEDL gradients, ECE, NLL, Brier score, score direction, GDA fitting, split reproducibility, and model shapes.

The pilot exposed two important engineering issues. Reloaded spectral-normalized models could produce non-finite outputs on the observed PyTorch MPS evaluation path, although CPU evaluation of the same checkpoint was finite. Final training therefore used MPS, while checkpoint evaluation, feature extraction, and DAEDL density fitting used CPU. Full-covariance matrices were rank deficient because some ReLU embedding coordinates were constant. The implementation used scale-relative diagonal loading and recorded rank, jitter, density range, and non-finite fractions instead of changing the architecture after seeing results.

Codex also enforced matched splits within each seed, minimum validation-NLL checkpoint selection, OOD-positive score orientation, probability-sum checks, checkpoint round trips, and metric recomputation from saved predictions. All 16 automated tests and the final six-run consistency audit passed.

## 5 Experimental Setup

CIFAR-10 provided 45,000 training images, 5,000 validation images, and the untouched official 10,000-image test set [4]. SVHN supplied 10,000 deterministically selected OOD test images [5]. Both models used the same spectrally normalized CIFAR ResNet-18, 128-dimensional embedding, augmentation, optimizer, schedule, and epoch budget. Seeds 3024, 3025, and 3026 controlled the stratified split and training randomness. Results are mean ± sample standard deviation across training seeds, not across individual test images.

| Setting | Value |
|---|---|
| Models | Softmax and DAEDL |
| Backbone | Spectrally normalized CIFAR ResNet-18, base width 32 |
| Embedding | 128 dimensions |
| Split | 45,000 train / 5,000 validation / 10,000 test |
| Seeds | 3024, 3025, 3026 |
| Optimizer | AdamW, LR $10^{{-3}}$, weight decay $5\times10^{{-4}}$ |
| Schedule | Cosine decay to $10^{{-5}}$ over 30 epochs |
| Batch size | 64 |
| Selection | Minimum validation NLL |
| Hardware | Apple M5 MacBook Air, 16 GB unified memory |
| Devices | MPS training; CPU evaluation and float64 GDA |
| OOD protocol | CIFAR-10 ID, SVHN OOD, normalized entropy, OOD positive |

## 6 Results

### 6.1 Classification and calibration

| Metric | Softmax | DAEDL |
|---|---:|---:|
| Accuracy | {s_acc} | {d_acc} |
| Macro F1 | {s_f1} | {d_f1} |
| NLL | {s_nll} | {d_nll} |
| ECE-15 | {s_ece} | {d_ece} |
| Brier score | {s_brier} | {d_brier} |

Softmax was about one percentage point more accurate in every seed. DAEDL's lower ECE also repeated in every seed, but NLL and Brier score favored Softmax. The calibration metrics therefore do not support one universal ranking. ECE measures binned agreement between confidence and observed accuracy, whereas NLL and Brier score evaluate the complete predictive distribution as proper scoring rules.

![Figure 2 three-seed headline metrics](../figures/final/figure2_three_seed_headline.png)

The seed-3024 reliability diagram illustrates the binned behavior behind ECE. Sparse bins with fewer than 50 test examples are suppressed for readability; the metric itself still uses all 15 equal-width bins.

![Figure 3 reliability](../figures/final/figure3_reliability_seed3024.png)

### 6.2 Error and OOD uncertainty

| Metric | Softmax | DAEDL |
|---|---:|---:|
| Error AUROC | {s_err} | {d_err} |
| Error AUPR | {s_errp} | {d_errp} |
| SVHN OOD AUROC | {s_ood} | {d_ood} |
| SVHN OOD AUPR | {s_oodp} | {d_oodp} |

Both models assigned much greater normalized entropy to incorrect predictions than to correct ones. Softmax had a small but consistent error-AUROC advantage. DAEDL's larger absolute entropy does not itself mean that it separates errors better; AUROC measures ranking quality and is therefore more suitable for that comparison.

DAEDL consistently assigned greater entropy to SVHN than to CIFAR-10. However, Softmax had higher OOD AUROC at seeds 3024 and 3026, while DAEDL was higher at seed 3025. The aggregate difference therefore does not justify a stable superiority claim for either model.

![Figure 4 entropy distributions](../figures/final/figure4_entropy_distributions_seed3024.png)

## 7 Discussion

DAEDL added a structured probabilistic representation without improving clean classification. Its lower ECE shows that one aspect of confidence-frequency agreement improved, yet its worse NLL and Brier score show that the full probability vectors were less effective under two proper scoring rules. Reporting only ECE would therefore have produced an incomplete calibration conclusion.

The uncertainty magnitudes also require care. DAEDL produced higher entropy for correct, incorrect, ID, and OOD samples. That wider numerical range made mistakes and SVHN inputs visibly uncertain, but it did not reliably improve discrimination AUROC. The practical question is not simply whether a method emits a larger uncertainty number; it is whether that number ranks errors or distribution shifts consistently.

Multiple seeds changed the interpretation of OOD performance. Seed 3024 alone suggested a clear Softmax advantage, whereas seed 3025 reversed the order. The three-seed aggregate still favors Softmax numerically, but the change in direction limits the claim to this observed average. No hypothesis test was added for three runs because descriptive consistency is more appropriate at this scale.

The density diagnostics show another trade-off. DAEDL embeddings remained rank deficient: global centered rank was 93, 99, and 101 out of 128, with 35, 29, and 27 constant dimensions. Per-class covariance ranks ranged from 76 to 95. Adaptive jitter remained around $10^{{-8}}$ to $10^{{-7}}$, and every density computation was finite. This supports using the results while reporting the reduced-rank feature space as a limitation.

The experiment is an assignment-scale evaluation rather than an exact reproduction of the ICML study. The paper used a different architecture and larger training setup. Here, the compact ResNet-18, 128-dimensional embedding, 30 epochs, and laptop hardware made matched multi-seed experiments feasible. The local results answer the coursework research question but do not contradict or reproduce the paper's broader benchmarks.

## 8 Limitations

The evaluation uses one in-distribution dataset, one OOD dataset, and three seeds. It does not cover every corruption type, architecture, or alternative evidential method. No exhaustive hyperparameter search was performed, and settings were frozen before replications to avoid tuning to seed 3024. MPS and CPU adaptations introduce an engineering difference from the authors' CUDA-oriented environment. DAEDL's embedding space was rank deficient even though covariance stabilization remained finite. The study therefore supports claims about this controlled CIFAR-10/SVHN setup only.

## 9 What I Learned

Before this assignment, I tended to read a classifier's probability as a direct statement of confidence. The experiment showed me that uncertainty needs separate evidence. A model can be accurate and still assign probabilities that do not match how often it is correct. It can also produce high entropy on difficult images without ranking every mistake or OOD sample better than another model.

I learned that calibration is not a single number. DAEDL had lower ECE in every seed, which initially looks like a clear calibration advantage. NLL and Brier score gave the opposite ranking. This result helped me understand that ECE depends on confidence bins, while proper scoring rules evaluate the full probability distribution and penalize confident errors differently. I should therefore explain which calibration property a metric measures instead of using “better calibrated” without qualification.

I also expected a method designed for uncertainty to outperform Softmax clearly on OOD detection. The seed results showed that this expectation was too simple. Softmax was better at seed 3024, DAEDL was better at seed 3025, and Softmax was better again at seed 3026. Had I stopped after the first run, I would have described the OOD comparison too confidently. Repeating the experiment did not merely make the result look more reliable; it changed the conclusion I was allowed to make.

The project also taught me that a larger uncertainty value is not automatically more useful. DAEDL produced much greater entropy on incorrect and SVHN inputs, but Softmax had slightly better error AUROC and a higher mean OOD AUROC. The separation between groups matters more than the absolute scale. This distinction is especially important when comparing models whose uncertainty scores arise from different parameterizations.

Using Codex did not remove the need for validation. Codex wrote the code and ran the experiments, but the workflow still needed tests for formulas and metrics, matched baselines, fixed checkpoint rules, checks for OOD score direction, probability normalization, and independent recomputation from saved predictions. The most useful AI behavior was not rapid code generation alone. It was the ability to inspect failures, record what changed, and preserve negative or mixed results rather than tuning them away.

Finally, I learned how theoretical ideas meet hardware and numerical constraints. MPS was suitable for neural-network training but not float64 covariance calculations. Reloaded spectral-normalized checkpoints also required a CPU evaluation policy in this environment. The embedding covariances were rank deficient, so the implementation needed adaptive diagonal loading and diagnostics. These issues made reproducibility, saved provenance, and numerical checks part of the pattern-recognition experiment rather than optional software details.

## 10 Conclusion

Codex successfully implemented DAEDL and compared it with a matched Softmax baseline across three CIFAR-10 seeds. DAEDL consistently reduced ECE, while Softmax remained stronger in accuracy, Macro F1, NLL, Brier score, and error-detection AUROC. Both methods assigned greater uncertainty to mistakes, and DAEDL assigned greater entropy to SVHN than CIFAR-10, but the OOD method ranking was not stable. The experiment therefore shows both the value and the limits of structured probabilistic uncertainty modelling: greater model complexity did not improve every predictive or uncertainty measure.
{source_section}
## References

[1] Yoon, T., and Kim, H. (2024). “Uncertainty Estimation by Density Aware Evidential Deep Learning.” *Proceedings of the 41st International Conference on Machine Learning*, PMLR 235, 57217–57243. https://proceedings.mlr.press/v235/yoon24a.html

[2] Sensoy, M., Kaplan, L., and Kandemir, M. (2018). “Evidential Deep Learning to Quantify Classification Uncertainty.” *Advances in Neural Information Processing Systems 31*. https://papers.nips.cc/paper/2018/hash/a981f2b708044d6fb4a71a1463242520-Abstract.html

[3] Guo, C., Pleiss, G., Sun, Y., and Weinberger, K. Q. (2017). “On Calibration of Modern Neural Networks.” *Proceedings of the 34th International Conference on Machine Learning*, PMLR 70, 1321–1330. https://proceedings.mlr.press/v70/guo17a.html

[4] Krizhevsky, A. (2009). *Learning Multiple Layers of Features from Tiny Images*. Technical report, University of Toronto. https://www.cs.toronto.edu/~kriz/learning-features-2009-TR.pdf

[5] Netzer, Y., Wang, T., Coates, A., Bissacco, A., Wu, B., and Ng, A. Y. (2011). “Reading Digits in Natural Images with Unsupervised Feature Learning.” *NIPS Workshop on Deep Learning and Unsupervised Feature Learning*. https://ufldl.stanford.edu/housenumbers/nips2011_housenumbers.pdf

[6] PyTorch Contributors. “MPS backend.” *PyTorch documentation*. https://docs.pytorch.org/docs/main/notes/mps.html

[7] Yoon, T., and Kim, H. “DAEDL official implementation.” GitHub. https://github.com/TaeseongYoon/DAEDL
"""


def build_docx(aggregate: pd.DataFrame, density: pd.DataFrame, source_url: str | None, output: Path) -> None:
    document = Document()
    configure_document(document)

    title = document.add_paragraph(style="Title")
    remove_paragraph_border(title)
    title.alignment = WD_ALIGN_PARAGRAPH.CENTER
    title.paragraph_format.space_before = Pt(120)
    title.add_run("AI Assisted Implementation and Evaluation of\nDensity Aware Evidential Deep Learning")
    subtitle = document.add_paragraph()
    subtitle.alignment = WD_ALIGN_PARAGRAPH.CENTER
    subtitle.paragraph_format.space_before = Pt(16)
    run = subtitle.add_run("CISC3024 Pattern Recognition\nAI Assignment #2")
    run.bold = True
    run.font.size = Pt(15)
    date = document.add_paragraph("October 2026")
    date.alignment = WD_ALIGN_PARAGRAPH.CENTER
    date.paragraph_format.space_before = Pt(22)
    note = document.add_paragraph("Prepared through an AI-assisted workflow with Codex")
    note.alignment = WD_ALIGN_PARAGRAPH.CENTER
    note.paragraph_format.space_before = Pt(130)
    note.runs[0].font.size = Pt(9.5)
    note.runs[0].font.color.rgb = RGBColor(80, 80, 80)
    document.add_page_break()

    document.add_heading("1 Introduction", level=1)
    paragraph(document, "Image classifiers produce probability-like scores, but a high maximum score does not by itself show that the model knows when it may be wrong. This assignment used Codex to identify, implement, and evaluate a recent probability-based recognition method. The selected method was Density Aware Evidential Deep Learning (DAEDL), published at ICML 2024 [1].")
    paragraph(document, "The research question was: how does DAEDL compare with a conventional Softmax classifier in classification performance, calibration, prediction-error uncertainty, and out-of-distribution uncertainty in an assignment-scale CIFAR-10 experiment?")
    paragraph(document, f"The three-seed results show a limited trade-off rather than an overall DAEDL advantage. Softmax achieved higher accuracy ({value(aggregate, 'softmax', 'accuracy')} versus {value(aggregate, 'daedl', 'accuracy')}) and better NLL and Brier score. DAEDL achieved lower ECE in every seed ({value(aggregate, 'daedl', 'ece_15')} versus {value(aggregate, 'softmax', 'ece_15')}). Both methods assigned greater entropy to mistakes, while Softmax had a small error-AUROC advantage. SVHN OOD rankings changed by seed, so the experiment did not establish a stable OOD winner.")

    document.add_heading("2 AI Assisted Algorithm Search and Selection", level=1)
    paragraph(document, "Codex was asked to find a recent classification or recognition algorithm grounded in probability or Bayesian theory. It filtered candidates using publication date, relevance to computer-vision classification, access to a primary paper and implementation, computational feasibility on a laptop, and whether the method supported a clear comparison with a conventional baseline.")
    paragraph(document, "DAEDL met these criteria because it predicts a Dirichlet distribution, models density in learned feature space, and reports classification, calibration, error detection, and OOD experiments [1]. It was recent, directly relevant to image recognition, and small enough to adapt into a controlled CIFAR-10 study.")
    paragraph(document, "Codex checked the official PMLR record, the paper, and the authors' repository. It did not treat the repository as a ready-made solution: the audit found CUDA assumptions, incomplete environment information, missing released checkpoints, and reproducibility gaps. A clean implementation was therefore used, with the official code serving as a method reference [7]. Conventional evidential deep learning [2] and calibration work [3] supplied the background needed to interpret the method and metrics.")

    document.add_heading("3 Algorithm Description", level=1)
    document.add_heading("3.1 Softmax classification", level=2)
    paragraph(document, "A neural classifier maps an input image x to class logits. Softmax exponentiates and normalizes those logits to obtain one probability vector. The largest entry determines the predicted class. The operation guarantees non-negative probabilities summing to one, but it does not model a distribution over possible probability vectors.")
    add_equation(document, "equation1_probabilities.png", 6.1)
    paragraph(document, "The second expression above is the evidential counterpart. A Dirichlet random variable π represents uncertainty over categorical probability vectors, and its mean gives the predictive class probability. The total concentration S controls how tightly the Dirichlet mass is concentrated.")

    document.add_heading("3.2 Evidential training", level=2)
    paragraph(document, "Conventional EDL converts non-negative evidence into Dirichlet concentrations and trains a deterministic neural network to predict evidence [2]. DAEDL changes the training parameterization to an exponential of each logit. Once normalized by total strength, these concentrations produce exactly the Softmax probabilities during training. The loss combines expected squared classification error, predictive variance, and a KL regularizer that suppresses misleading evidence.")
    add_equation(document, "equation2_training.png", 6.65)
    paragraph(document, "The local implementation averages the complete per-example objective. This scales every component by the same batch-size factor, so the relative weight λ=0.05 is unchanged. Logits are clamped only at ±30 when exponentiated as an overflow guard.")

    document.add_heading("3.3 Feature density", level=2)
    paragraph(document, "After neural-network training, DAEDL extracts the 128-dimensional penultimate feature h(x) for every training image. A full-covariance Gaussian is fitted to each class, and the empirical class proportions form a mixture density. Training log-density extrema define a clipped score s(x) between zero and one.")
    add_equation(document, "equation3_density.png", 6.45)
    paragraph(document, "A test embedding near the fitted training features receives a larger density score. A distant embedding receives a smaller score. Because full covariance and log-density calculations are sensitive to numerical precision, the implementation performs this stage on CPU in float64 and adds scale-relative diagonal jitter when Cholesky factorization requires it.")

    document.add_heading("3.4 Density aware prediction", level=2)
    paragraph(document, "At inference, DAEDL multiplies each logit by the density score before constructing the Dirichlet concentrations. This is equivalent to an input-dependent temperature. As the score approaches zero, all concentrations approach one and the expected probabilities approach the uniform distribution.")
    add_equation(document, "equation4_inference.png", 6.25)
    paragraph(document, "This project used normalized predictive entropy as the primary common uncertainty score. The normalization places entropy in [0,1] for both methods, so the same error-detection and OOD metric code can be used.")
    add_equation(document, "equation5_entropy.png", 3.2)

    add_caption(document, "Figure 1", "DAEDL inference pipeline used in this project. The diagram is original and was generated from the implemented method.")
    add_figure(document, FIGURES / "figure1_daedl_concept.png", 6.7)

    document.add_heading("4 AI Assisted Implementation", level=1)
    paragraph(document, "Codex inspected the paper and official code, created an isolated Python 3.11 environment, and implemented the complete project. The code includes fixed stratified splits, indexed datasets, train-only augmentation, a CIFAR-adapted ResNet-18, Softmax cross-entropy, the DAEDL objective, CPU GDA, evaluation metrics, artifact logging, and plotting. Automated tests cover probability normalization, DAEDL gradients, calibration metrics, score direction, GDA fitting, split reproducibility, and model output shapes.")
    paragraph(document, "The pilot exposed a PyTorch MPS issue: reloaded spectral-normalized models could produce non-finite outputs during the observed post-training evaluation path, although CPU evaluation of the same checkpoint was finite. Final training therefore used MPS, while checkpoint evaluation and feature extraction used CPU. PyTorch documents that MPS cannot store float64 tensors, which independently requires the GDA stage to run on CPU [6].")
    paragraph(document, "Full-covariance matrices were rank deficient because some ReLU embedding coordinates were constant. The implementation used scale-relative diagonal loading and recorded global rank, per-class rank, jitter, training density range, and non-finite fraction. It did not change the architecture after seeing seed-3024 results. A training-only sensitivity check on that seed showed that removing globally constant dimensions did not materially change the primary outputs.")
    paragraph(document, "For experimental validity, Codex applied identical splits to both models within each seed, selected checkpoints by minimum validation NLL, saved both selected and final checkpoints, kept the test set out of model selection, and defined OOD as the positive class. Every final probability row, checkpoint label, metric, and density diagnostic was checked. The six-run audit and all 16 automated tests passed.")

    document.add_heading("5 Experimental Setup", level=1)
    paragraph(document, "CIFAR-10 provided 45,000 training images, 5,000 validation images, and the untouched official 10,000-image test set [4]. SVHN supplied 10,000 deterministically selected OOD test images [5]. Both models used the same backbone, spectral normalization, augmentation, optimizer, schedule, epoch budget, and checkpoint rule. Seeds 3024, 3025, and 3026 controlled the stratified split and training randomness. The replicate unit is a training seed and model run; results are mean ± sample SD across three seeds.")
    add_caption(document, "Table 1", "Frozen experimental configuration.")
    add_table(document, ["Setting", "Value"], [
        ["Models", "Softmax and DAEDL"],
        ["Backbone", "Spectrally normalized CIFAR ResNet-18, base width 32"],
        ["Embedding", "128 dimensions"],
        ["Split", "45,000 train / 5,000 validation / 10,000 test"],
        ["Seeds", "3024, 3025, 3026"],
        ["Optimizer", "AdamW; initial LR 1e-3; weight decay 5e-4"],
        ["Schedule", "Cosine decay to 1e-5 over 30 epochs"],
        ["Batch size", "64"],
        ["Selection", "Minimum validation NLL"],
        ["Hardware", "Apple M5 MacBook Air; 16 GB unified memory"],
        ["Devices", "MPS training; CPU evaluation and float64 GDA"],
        ["OOD protocol", "CIFAR-10 ID; SVHN OOD; normalized entropy; OOD positive"],
    ], [1.55, 5.0])
    paragraph(document, "Classification uses accuracy and Macro F1. Calibration uses negative log-likelihood (NLL), 15-bin expected calibration error (ECE), and multiclass Brier score. Error detection and OOD detection use normalized predictive entropy with AUROC and average precision (AUPR). Lower values are preferable for NLL, ECE, and Brier; higher values are preferable for accuracy, F1, AUROC, and AUPR.")

    document.add_page_break()
    document.add_heading("6 Results", level=1)
    document.add_heading("6.1 Classification and calibration", level=2)
    add_caption(document, "Table 2", "Clean CIFAR-10 classification and calibration. Values are mean ± sample SD across three training seeds.")
    add_table(document, ["Metric", "Softmax", "DAEDL", "Preferred direction"], [
        ["Accuracy", value(aggregate, "softmax", "accuracy"), value(aggregate, "daedl", "accuracy"), "Higher"],
        ["Macro F1", value(aggregate, "softmax", "macro_f1"), value(aggregate, "daedl", "macro_f1"), "Higher"],
        ["NLL", value(aggregate, "softmax", "nll"), value(aggregate, "daedl", "nll"), "Lower"],
        ["ECE-15", value(aggregate, "softmax", "ece_15"), value(aggregate, "daedl", "ece_15"), "Lower"],
        ["Brier score", value(aggregate, "softmax", "brier"), value(aggregate, "daedl", "brier"), "Lower"],
    ], [1.65, 1.65, 1.65, 1.55])
    paragraph(document, "Softmax was consistently about one percentage point more accurate. Its accuracy exceeded DAEDL by 0.0099, 0.0130, and 0.0105 at seeds 3024, 3025, and 3026. Macro F1 followed the same pattern. Both methods nevertheless reached close to 89–90% accuracy in the compact 30-epoch setup.")
    paragraph(document, "DAEDL had lower ECE in every seed, but NLL and Brier score favored Softmax in every seed. The calibration metrics therefore do not all support the same ranking. ECE summarizes binned agreement between confidence and observed accuracy. NLL and Brier score are proper scoring rules that evaluate the complete predictive distribution and respond differently to probability sharpness and errors.")
    add_caption(document, "Figure 2", "Three-seed headline comparison. Points show individual seeds; black bars show mean ± sample SD.")
    add_figure(document, FIGURES / "figure2_three_seed_headline.png", 6.6)

    document.add_heading("6.2 Reliability", level=2)
    paragraph(document, "The seed-3024 reliability diagram illustrates why a single calibration summary needs context. DAEDL tracks the ideal diagonal more closely in several high-confidence bins, consistent with its lower ECE. The plot is illustrative rather than aggregate: bins with fewer than 50 test examples are suppressed for readability, but ECE is still calculated from all 15 equal-width bins.")
    add_caption(document, "Figure 3", "CIFAR-10 reliability at the predefined representative seed 3024.")
    add_figure(document, FIGURES / "figure3_reliability_seed3024.png", 4.7)

    document.add_heading("6.3 Prediction error uncertainty", level=2)
    paragraph(document, f"Both methods assigned substantially greater normalized entropy to incorrect predictions. Across seeds, Softmax's correct and incorrect mean entropies were {value(aggregate, 'softmax', 'error_correct_mean_entropy')} and {value(aggregate, 'softmax', 'error_incorrect_mean_entropy')}; DAEDL's were {value(aggregate, 'daedl', 'error_correct_mean_entropy')} and {value(aggregate, 'daedl', 'error_incorrect_mean_entropy')}. The distributions therefore contain useful information about mistakes for both methods.")
    paragraph(document, "DAEDL's larger absolute entropy does not automatically mean better error detection. AUROC evaluates how well uncertainty ranks incorrect above correct predictions. Softmax had a small consistent AUROC advantage at every seed, while AUPR changed paired direction. The most defensible conclusion is that both methods detected errors well and Softmax ranked them slightly better under AUROC.")

    document.add_heading("6.4 SVHN out of distribution uncertainty", level=2)
    add_caption(document, "Table 3", "Error and SVHN OOD detection using normalized predictive entropy. Values are mean ± sample SD across three seeds.")
    add_table(document, ["Metric", "Softmax", "DAEDL"], [
        ["Error AUROC", value(aggregate, "softmax", "error_auroc"), value(aggregate, "daedl", "error_auroc")],
        ["Error AUPR", value(aggregate, "softmax", "error_aupr"), value(aggregate, "daedl", "error_aupr")],
        ["SVHN OOD AUROC", value(aggregate, "softmax", "ood_auroc"), value(aggregate, "daedl", "ood_auroc")],
        ["SVHN OOD AUPR", value(aggregate, "softmax", "ood_aupr"), value(aggregate, "daedl", "ood_aupr")],
    ], [2.5, 1.9, 1.9])
    paragraph(document, f"DAEDL consistently assigned greater entropy to SVHN than to CIFAR-10: its ID and OOD mean entropies were {value(aggregate, 'daedl', 'ood_id_mean_entropy')} and {value(aggregate, 'daedl', 'ood_mean_entropy')}. Softmax showed the same direction at a lower absolute scale. However, method ranking was seed dependent. Softmax had higher OOD AUROC at seeds 3024 and 3026, while DAEDL was higher at seed 3025. The aggregate values therefore do not establish a stable OOD winner.")
    paragraph(document, "DAEDL's secondary density-only uncertainty also lacked a consistent advantage over its entropy score. Density AUROC was 0.8641, 0.8420, and 0.8498 at seeds 3024–3026. It exceeded DAEDL entropy only at seed 3024. This negative result is retained because the evaluation protocol was frozen before the replications.")
    add_caption(document, "Figure 4", "Normalized predictive-entropy distributions at seed 3024. Numeric conclusions use all three seeds.")
    add_figure(document, FIGURES / "figure4_entropy_distributions_seed3024.png", 6.35)

    document.add_heading("7 Discussion", level=1)
    document.add_heading("7.1 Accuracy and uncertainty", level=2)
    paragraph(document, "DAEDL added a structured probabilistic representation without improving clean classification in this setup. This is not an implementation failure: every run completed, checkpoint validation passed, and the classification differences repeated across seeds. The result shows that a dedicated uncertainty mechanism does not guarantee a better classifier.")
    document.add_heading("7.2 Calibration is multi dimensional", level=2)
    paragraph(document, "DAEDL's ECE advantage and Softmax's NLL/Brier advantages coexist because the metrics measure different properties. Reporting only ECE would make DAEDL appear generally better calibrated, while reporting only NLL would hide its more favorable binned confidence-frequency agreement. The accurate statement is narrower: DAEDL improved ECE, while Softmax remained stronger under the two proper scoring rules.")
    document.add_heading("7.3 Magnitude and usefulness", level=2)
    paragraph(document, "DAEDL produced higher entropy for correct, incorrect, ID, and OOD samples. That wider numerical range made mistakes and SVHN inputs visibly uncertain, but it did not reliably improve discrimination AUROC. The practical question is whether a score ranks risky inputs consistently, not simply whether its numeric value is larger.")
    document.add_heading("7.4 Seed dependence", level=2)
    paragraph(document, "Multiple seeds materially changed the OOD interpretation. Seed 3024 alone suggested a clear Softmax advantage; seed 3025 reversed the ranking. The three-seed aggregate favors Softmax numerically, but the change in direction prevents a stable superiority claim. No significance test was added because three runs support descriptive consistency, not a persuasive estimate of a sampling distribution.")
    document.add_heading("7.5 Assignment scale adaptation", level=2)
    paragraph(document, "The original DAEDL work used a different architecture and larger training setup [1]. This project used a compact ResNet-18, 128-dimensional embeddings, 30 epochs, and laptop hardware so that a matched three-seed comparison was feasible. It is an implementation and evaluation of the method's central ideas, not an exact reproduction of the paper's headline results.")

    document.add_heading("8 Limitations", level=1)
    paragraph(document, "The evaluation uses one in-distribution dataset, one OOD dataset, and three seeds. It does not cover every corruption, architecture, or evidential baseline. No exhaustive hyperparameter search was performed, and settings were frozen before replications to avoid tuning to seed 3024. MPS and CPU adaptations differ from the authors' CUDA-oriented environment. The DAEDL feature space was rank deficient even though covariance stabilization remained finite.")
    paragraph(document, f"Global centered embedding rank was {int(density.iloc[0].global_embedding_rank)}, {int(density.iloc[1].global_embedding_rank)}, and {int(density.iloc[2].global_embedding_rank)} out of 128, while constant dimensions were {int(density.iloc[0].constant_dimensions)}, {int(density.iloc[1].constant_dimensions)}, and {int(density.iloc[2].constant_dimensions)}. Per-class covariance ranks spanned {int(density.class_rank_min.min())}–{int(density.class_rank_max.max())}. Jitter remained around 1e-8 to 1e-7 and every non-finite fraction was zero, so the density results are usable but the reduced-rank representation should remain visible as a limitation.")
    paragraph(document, "No corruption experiment was added after the three-seed study. This was a deliberate stopping decision: the core assignment questions were already answered, and adding another analysis after reviewing outcomes would have expanded the scope without resolving the seed-sensitive OOD ranking.")

    document.add_page_break()
    document.add_heading("9 What I Learned", level=1)
    paragraph(document, "Before this assignment, I tended to read a classifier's probability as a direct statement of confidence. The experiment showed me that uncertainty needs separate evidence. A model can be accurate and still assign probabilities that do not match how often it is correct.")
    paragraph(document, "I learned that calibration is not a single number. DAEDL had lower ECE in every seed, which initially looks like a clear calibration advantage. NLL and Brier score gave the opposite ranking. This result helped me understand that ECE depends on confidence bins, while proper scoring rules evaluate the full probability distribution and penalize confident errors differently. I should therefore explain which calibration property a metric measures instead of using ‘better calibrated’ without qualification.")
    paragraph(document, "I also expected a method designed for uncertainty to outperform Softmax clearly on OOD detection. The seed results showed that this expectation was too simple. Softmax was better at seed 3024, DAEDL was better at seed 3025, and Softmax was better again at seed 3026. Had I stopped after the first run, I would have described the OOD comparison too confidently. Repeating the experiment did not merely make the result look more reliable; it changed the conclusion I was allowed to make.")
    paragraph(document, "The project also taught me that a larger uncertainty value is not automatically more useful. DAEDL produced much greater entropy on incorrect and SVHN inputs, but Softmax had slightly better error AUROC and a higher mean OOD AUROC. The separation between groups matters more than the absolute scale. This distinction is especially important when comparing models whose uncertainty scores arise from different parameterizations.")
    paragraph(document, "Using Codex did not remove the need for validation. Codex wrote the code and ran the experiments, but the workflow still needed tests for formulas and metrics, matched baselines, fixed checkpoint rules, checks for OOD score direction, probability normalization, and independent recomputation from saved predictions. The most useful AI behavior was not rapid code generation alone. It was the ability to inspect failures, record what changed, and preserve negative or mixed results rather than tuning them away.")
    paragraph(document, "Finally, I learned how theory meets hardware and numerical constraints. MPS could train the networks but not run float64 covariance calculations, and reloaded spectral-normalized checkpoints required CPU evaluation in this environment. Rank-deficient embedding covariances required adaptive diagonal loading and diagnostics. Reproducibility and numerical checks were therefore part of the experiment, not optional software details.")

    document.add_heading("10 Conclusion", level=1)
    paragraph(document, "Codex successfully implemented DAEDL and compared it with a matched Softmax baseline across three CIFAR-10 seeds. DAEDL consistently reduced ECE, while Softmax remained stronger in accuracy, Macro F1, NLL, Brier score, and error-detection AUROC. Both methods assigned greater uncertainty to mistakes, and DAEDL assigned greater entropy to SVHN than CIFAR-10, but the OOD method ranking was not stable. The exercise demonstrated both the potential and limits of probabilistic uncertainty methods and showed why AI-generated implementation still requires careful experimental validation.")

    if source_url:
        document.add_heading("11 Source Code", level=1)
        p = document.add_paragraph()
        p.add_run("Source code: ").bold = True
        add_hyperlink(p, source_url, source_url)

    document.add_page_break()
    document.add_heading("References", level=1)
    references = [
        ("[1] Yoon, T., and Kim, H. (2024). “Uncertainty Estimation by Density Aware Evidential Deep Learning.” Proceedings of the 41st International Conference on Machine Learning, PMLR 235, 57217–57243. ", "https://proceedings.mlr.press/v235/yoon24a.html"),
        ("[2] Sensoy, M., Kaplan, L., and Kandemir, M. (2018). “Evidential Deep Learning to Quantify Classification Uncertainty.” Advances in Neural Information Processing Systems 31. ", "https://papers.nips.cc/paper/2018/hash/a981f2b708044d6fb4a71a1463242520-Abstract.html"),
        ("[3] Guo, C., Pleiss, G., Sun, Y., and Weinberger, K. Q. (2017). “On Calibration of Modern Neural Networks.” Proceedings of the 34th International Conference on Machine Learning, PMLR 70, 1321–1330. ", "https://proceedings.mlr.press/v70/guo17a.html"),
        ("[4] Krizhevsky, A. (2009). Learning Multiple Layers of Features from Tiny Images. Technical report, University of Toronto. ", "https://www.cs.toronto.edu/~kriz/learning-features-2009-TR.pdf"),
        ("[5] Netzer, Y., Wang, T., Coates, A., Bissacco, A., Wu, B., and Ng, A. Y. (2011). “Reading Digits in Natural Images with Unsupervised Feature Learning.” NIPS Workshop on Deep Learning and Unsupervised Feature Learning. ", "https://ufldl.stanford.edu/housenumbers/nips2011_housenumbers.pdf"),
        ("[6] PyTorch Contributors. “MPS backend.” PyTorch documentation. ", "https://docs.pytorch.org/docs/main/notes/mps.html"),
        ("[7] Yoon, T., and Kim, H. “DAEDL official implementation.” GitHub. ", "https://github.com/TaeseongYoon/DAEDL"),
    ]
    for text, url in references:
        p = document.add_paragraph(style="References")
        p.add_run(text)
        add_hyperlink(p, url, url)

    properties = document.core_properties
    properties.title = "AI Assisted Implementation and Evaluation of Density Aware Evidential Deep Learning"
    properties.subject = "CISC3024 Pattern Recognition AI Assignment #2"
    properties.author = "Prepared with Codex"
    properties.keywords = "DAEDL, CIFAR-10, uncertainty, calibration, evidential deep learning"
    output.parent.mkdir(parents=True, exist_ok=True)
    document.save(output)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source-url", help="Verified public source-code URL; omit until publication")
    parser.add_argument("--docx", type=Path, default=REPORT / "CISC3024_AI_Assignment2_Final.docx")
    parser.add_argument("--markdown", type=Path, default=REPORT / "CISC3024_AI_Assignment2_Final.md")
    args = parser.parse_args()
    aggregate, _, density = load_results()
    args.markdown.write_text(build_markdown(aggregate, density, args.source_url))
    build_docx(aggregate, density, args.source_url, args.docx)
    print(json.dumps({"docx": str(args.docx), "markdown": str(args.markdown), "source_url": args.source_url}, indent=2))


if __name__ == "__main__":
    main()
