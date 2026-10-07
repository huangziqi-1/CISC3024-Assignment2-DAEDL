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

