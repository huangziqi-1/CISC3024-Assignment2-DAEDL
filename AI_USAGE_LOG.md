# AI Usage Log

This log records the assignment work performed by Codex. The student did not manually write the project code.

## 2026-10-07 - Assignment interpretation and algorithm verification

**Student request:** Independently verify whether Density Aware Evidential Deep Learning (DAEDL) is suitable for CISC3024 AI Assignment #2; inspect primary sources and the local machine; design a rigorous but manageable CIFAR-10 experiment; create a reproducible project structure and AI-use record; do not start expensive training.

**Codex work:**

- Read the one-page official assignment PDF and confirmed the six required report elements: AI search, algorithm description, AI implementation, experiment settings/results, learning reflection, and a webpage link to source code.
- Located and inspected the official PMLR publication page, 27-page ICML paper, OpenReview entry, arXiv record, and authors' GitHub repository.
- Verified that DAEDL was published in the Proceedings of the 41st ICML (PMLR 235, July 2024).
- Checked the mathematical method: Dirichlet-distributed class probabilities, expected probabilities, concentration/strength, EDL loss, class-conditional Gaussian feature density, normalized density scaling, and the paper's improper-Dirichlet-prior interpretation.
- Inspected the authors' code rather than relying on the README. Identified absent environment specifications and pretrained weights, CUDA defaults, reproducibility gaps, a fresh CIFAR-10 download problem, a checkpoint path problem, an output-saving error for Brier score, and an MPS-incompatible float64 density-estimation path.
- Inspected the computer: Apple M5 MacBook Air, 10 CPU cores, 16 GB unified memory, macOS 26.4.1, about 481 GiB free disk, Python 3.13.9 in the active Anaconda base environment, PyTorch 2.11.0, no CUDA, and currently unavailable MPS despite an MPS-enabled build.
- Chose an isolated Python 3.11 environment as the starting point. The existing base environment will not be modified.
- Recommended a staged CIFAR-10 comparison of Softmax, conventional EDL, and DAEDL using identical splits and a common compact backbone; density fitting will run on CPU while neural-network training may use MPS after a fresh-environment smoke test.
- Designed evaluation for classification, calibration, incorrect-prediction uncertainty, controlled corruptions, OOD detection, and qualitative examples, with every artifact generated from saved predictions.
- Created the initial repository structure, source register, report outline, environment specifications, and `.gitignore`. No dataset, checkpoint, or long training job was started.

**Decision:** Proceed with DAEDL. It is a strong fit for the assignment, but reimplement the core method cleanly instead of treating the official repository as a turnkey package.

## 2026-10-07 - Phase 2 implementation and pilot validation

**Student request:** Implement a reproducible CIFAR-10 Softmax-versus-DAEDL pipeline, validate it in an isolated Python 3.11 environment, run only a 1-2 epoch pilot, and stop before final training.

**Codex work:**

- Created `.venv` with Python 3.11.5 and installed native arm64 PyTorch 2.14.1, torchvision 0.29.1, and the minimal analysis/test dependencies. The Anaconda base environment was not changed.
- Verified CPU operations. MPS is hidden inside the Codex sandbox but is available and passed a tensor-operation test through the Mac host shell.
- Implemented fixed stratified 45,000/5,000 CIFAR-10 split logic and reusable pilot subsets; deterministic seeds; indexed examples; clean evaluation transforms; and train-only crop/flip augmentation.
- Implemented a compact CIFAR ResNet-18 with a CIFAR stem, 128-dimensional embedding, exposed logits/features, and method-specific spectral normalization for DAEDL.
- Implemented Softmax cross-entropy and the paper's DAEDL exponential concentration, expected-MSE, predictive-variance, and KL objective. Added an exponential-input clamp at +/-30 solely as an overflow guard.
- Implemented CPU float64 class-conditional full-covariance GDA, paper-compatible training-density min/max normalization, density-scaled DAEDL prediction, and adaptive scale-relative diagonal loading for rank-deficient pilot covariances.
- Implemented accuracy, macro F1, NLL, 15-bin equal-width ECE, multiclass Brier score, predictive entropy, error-detection AUROC/AUPR, OOD-positive AUROC/AUPR, reliability plotting, and per-example CSV output.
- Added 14 tests covering DAEDL probabilities/loss gradients, ECE/Brier/NLL, empty bins, entropy, OOD direction, GDA shapes and rank deficiency, fixed split reproducibility, and model output shapes. All passed.
- Encountered and fixed: standalone package import failure; missing Python CA bundle for dataset download by scoping `SSL_CERT_FILE=/etc/ssl/cert.pem`; rank-deficient pilot covariance; and a PyTorch 2.14.1 MPS issue where spectral-normalized DAEDL evaluation produced NaN embeddings. CPU evaluation of the identical checkpoint was finite, so DAEDL training remains on MPS while post-training feature extraction, GDA, and pilot inference explicitly use CPU.
- Ran deterministic two-epoch pilots with 5,000 train, 1,000 validation, and 1,000 test examples. Softmax training took 9.38 s; DAEDL training took 10.20 s, GDA 12.92 s, and evaluation 2.26 s. Both checkpoints reloaded exactly and complete artifacts were saved under unique `pilot_...` directories.
- Audited final artifacts: identical split indices across models, zero train/validation overlap, all numeric prediction fields finite, probability-row maximum errors below `1.9e-7`, positive DAEDL concentrations, finite strengths, and density scores in `[0, 1]`.

**Decision:** Phase 2 passed. Pilot values are diagnostic only. Do not start Phase 3 without approval.

