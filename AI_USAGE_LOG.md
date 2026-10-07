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

## 2026-10-07 - Phase 3A implementation and pre-training audit

**Student request:** Run one trustworthy full-scale seed-3024 comparison of Softmax and DAEDL, including shared spectral normalization, validation-NLL checkpoint selection, complete clean/error/calibration/SVHN evaluation, density diagnostics, and analysis-quality artifacts. Do not start other seeds or corruptions.

**Codex work before training:**

- Audited the Phase-2 architecture. The pilot Softmax model had no spectral normalization while DAEDL did, which would confound the final comparison. Phase 3A enables the identical configuration for both models: spectral normalization on the CIFAR stem, every residual-block convolution, every learned shortcut projection, the 128-D embedding layer, and the classifier. Batch-normalization layers and identity shortcuts have no eligible weight matrix and are unchanged.
- Compared the local DAEDL loss directly with the authors' official `train.py`. The official implementation sums the expected squared-error component, predictive-variance component, and KL component across the batch, then adds them. The local implementation constructs the same complete per-example objective and averages it across the batch. Therefore every component, including the KL regularizer, is divided by the same batch size; relative weighting is unchanged. The mean is retained as an optimization-scale adaptation, and a regression test verifies `mean_loss * batch_size == official_sum_form` on unclamped finite logits.
- Changed the common checkpoint criterion from method-specific validation objective to minimum validation NLL. For DAEDL before density fitting, `exp(logits) / sum(exp(logits))` is exactly `softmax(logits)`, so this metric is defined without fitting density on validation data. Both best-validation-NLL and final-epoch checkpoints are saved; the test set is not used for selection.
- Set 30 epochs, AdamW, batch size 64, initial learning rate `1e-3`, weight decay `5e-4`, cosine decay to `1e-5`, seed/split seed 3024, and full 45,000/5,000/10,000 CIFAR-10 partitions.
- Retained MPS training and made post-training CPU evaluation explicit for both spectral-normalized models. This shared policy avoids the PyTorch 2.14.1 MPS non-finite-output issue observed after reloading the DAEDL pilot checkpoint. DAEDL clean feature extraction and full-covariance GDA remain CPU float64.
- Added full-run density diagnostics, deterministic equal-size SVHN evaluation, deterministic representative selection, figure-input files, analysis figures, dual checkpoints, progress logs, and consistency-test hooks. No Phase-3A training result existed at the time of this entry.

**Training and evaluation outcome:**

- Trained Softmax first and DAEDL second for all 30 epochs on MPS. No NaN/Inf, exploding gradient, interruption, or unexpected training-device fallback occurred. Softmax training took 1,389.84 seconds and DAEDL training took 1,488.74 seconds.
- Selected Softmax epoch 26 (validation NLL 0.321901) and DAEDL epoch 27 (validation NLL 0.409656). Final-epoch checkpoints were also retained separately. The test set was never consulted for selection.
- On 10,000 clean CIFAR-10 test images, Softmax achieved 0.9011 accuracy, 0.9011 macro F1, 0.3410 NLL, 0.0438 ECE, and 0.1514 Brier. DAEDL achieved 0.8912 accuracy, 0.8908 macro F1, 0.4319 NLL, 0.0261 ECE, and 0.1715 Brier.
- For error detection with normalized predictive entropy, Softmax achieved AUROC/AUPR 0.9071/0.4749 and DAEDL 0.9016/0.4866. Both methods assigned substantially higher entropy to incorrect than correct predictions.
- On equal-size CIFAR-10 versus SVHN evaluation with OOD positive, Softmax entropy achieved AUROC/AUPR 0.8994/0.8556 and DAEDL entropy 0.8514/0.7561. DAEDL density uncertainty achieved 0.8641/0.7772. DAEDL's OOD uncertainty was higher than its ID uncertainty, but it did not outperform Softmax.
- Full DAEDL embeddings had 35 constant dimensions, improved from 54 in the pilot, and global centered rank 93. Per-class covariance ranks were 77-80; adaptive jitter was approximately `5.43e-8` to `1.16e-7`; all embeddings and density values were finite. Because the remaining degeneracy was substantial, Codex performed a training-only active-dimension sensitivity refit without changing primary results. Density-score correlation was 0.9987, clean accuracy was unchanged, and entropy OOD AUROC changed by only 0.00033, showing that primary metrics were not materially caused by the globally constant dimensions.
- The first sensitivity script attempt completed computation but failed while serializing indices because PyTorch 2.14.1 lacks `torch.flatnonzero`. Replaced it with the supported equivalent `torch.nonzero(...).flatten()` and reran successfully. No primary artifact was overwritten.
- The first combined-figure attempt selected the macOS GUI Matplotlib backend in the sandbox and aborted after the numeric files were written. Forced the noninteractive Agg backend and regenerated all combined figures into a fresh directory. Partial analysis directories were not used.
- The final automated consistency audit passed: identical split/test/SVHN indices, no split overlap, correct best-NLL checkpoint selection, finite outputs, normalized probabilities, metric recomputation agreement, positive DAEDL concentrations, correct OOD/error orientation, and zero non-finite density values. All 16 automated tests passed after implementation.

**Decision:** Retain Phase 3A as the first final core result. The evidence supports running seeds 3025 and 3026 next with the configuration frozen, before adding corruptions. Do not tune away the single-seed result or start Phase 3B without approval.

## 2026-10-07 - Phase 3B frozen replication setup

**Student request:** Replicate the accepted Phase-3A Softmax and DAEDL experiments at seeds 3025 and 3026, aggregate all three seeds, and stop before corruptions or any tuning.

**Codex pre-run actions:**

- Preserved the seed-3024 FINAL_CORE artifacts without rerunning or modifying them.
- Parameterized the existing final-run driver only to permit `FINAL_REPLICATION` markers for seeds 3025 and 3026. Scientific settings remain frozen: identical spectrally normalized compact CIFAR ResNet-18, 128-D embedding, batch size 64, AdamW, `1e-3` initial learning rate, `5e-4` weight decay, cosine decay to `1e-5`, 30 epochs, validation-NLL checkpoint selection, DAEDL objective, CPU float64 full-covariance GDA, adaptive covariance loading, and clean/error/SVHN metric definitions.
- Required the training seed and split seed to match. Within each seed, Softmax and DAEDL therefore use identical train/validation/test and deterministic SVHN indices; across seeds, the stratified training/validation split and training randomness vary according to the established reproducibility protocol.
- Added an option to suppress redundant per-seed plots. Complete configs, environments, splits, progress logs, histories, both checkpoints, predictions, metrics, runtime, and DAEDL density diagnostics remain mandatory. Aggregate figures will be generated only after all four replications complete.
- No hyperparameter or implementation change was made in response to the seed-3024 outcome.
