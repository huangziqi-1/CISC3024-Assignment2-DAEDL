# AI Assisted Implementation and Evaluation of Density Aware Evidential Deep Learning

**CISC3024 Pattern Recognition**  
**AI Assignment #2**  
October 2026

## 1 Introduction

Image classifiers produce probability-like scores, but a high maximum score does not by itself show that the model knows when it may be wrong. This assignment used Codex to identify, implement, and evaluate a recent probability-based recognition method. The selected method was Density Aware Evidential Deep Learning (DAEDL), published at ICML 2024 [1]. The research question was: how does DAEDL compare with a conventional Softmax classifier in classification performance, calibration, prediction-error uncertainty, and out-of-distribution uncertainty in an assignment-scale CIFAR-10 experiment?

The three-seed results show a limited trade-off rather than an overall DAEDL advantage. Softmax achieved higher accuracy (0.9013 ± 0.0004 versus 0.8902 ± 0.0012) and better NLL and Brier score. DAEDL achieved lower ECE in every seed (0.0261 ± 0.0030 versus 0.0463 ± 0.0031). Both methods assigned greater entropy to mistakes, while Softmax had a small error-AUROC advantage. SVHN OOD rankings changed by seed, so the experiment did not establish a stable OOD winner.

## 2 AI Assisted Algorithm Search and Selection

Codex was asked to find a recent classification or recognition algorithm grounded in probability or Bayesian theory. It filtered candidates using publication date, relevance to computer-vision classification, access to a primary paper and implementation, computational feasibility on a laptop, and whether the method supported a clear comparison with a conventional baseline. DAEDL met these criteria because it predicts a Dirichlet distribution, models density in learned feature space, and reports classification, calibration, error detection, and OOD experiments [1].

Codex checked the official PMLR record, the paper, and the authors' repository. It did not treat the repository as a ready-made solution: the audit found CUDA assumptions, incomplete environment information, missing released checkpoints, and reproducibility gaps. A clean implementation was therefore used, with the official code serving as a method reference. Conventional evidential deep learning [2] and calibration work [3] supplied the background needed to interpret the method and metrics.

## 3 Algorithm Description

### 3.1 Softmax classification

For class logits $z_k(x)$, Softmax forms normalized class probabilities:

$$p_{soft}(y=k\mid x)=\frac{e^{z_k(x)}}{\sum_j e^{z_j(x)}}.$$

### 3.2 Evidential classification

Evidential classification predicts positive concentrations $\boldsymbol{\alpha}(x)$ for a Dirichlet distribution over class-probability vectors $\boldsymbol{\pi}$. Its predictive categorical probability is the Dirichlet mean, $\mathbb{E}[\pi_k\mid x]=\alpha_k/S$, where $S=\sum_j\alpha_j$. Strength describes concentration, but this project used normalized predictive entropy as the common uncertainty score so that Softmax and DAEDL were compared on the same quantity.

### 3.3 DAEDL training and density-aware prediction

DAEDL uses $\alpha_k=e^{z_k}$ during training. Normalizing these concentrations gives the same class probabilities as Softmax, while the loss adds the expected squared-error and predictive-variance terms of the Dirichlet plus a KL penalty for misleading evidence. After training, the implementation fits class-conditional full-covariance Gaussian distributions to the 128-dimensional penultimate embeddings. The class mixture defines a feature density, which is min-max normalized against training log densities and clipped to $[0,1]$.

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
| Optimizer | AdamW, LR $10^{-3}$, weight decay $5\times10^{-4}$ |
| Schedule | Cosine decay to $10^{-5}$ over 30 epochs |
| Batch size | 64 |
| Selection | Minimum validation NLL |
| Hardware | Apple M5 MacBook Air, 16 GB unified memory |
| Devices | MPS training; CPU evaluation and float64 GDA |
| OOD protocol | CIFAR-10 ID, SVHN OOD, normalized entropy, OOD positive |

## 6 Results

### 6.1 Classification and calibration

| Metric | Softmax | DAEDL |
|---|---:|---:|
| Accuracy | 0.9013 ± 0.0004 | 0.8902 ± 0.0012 |
| Macro F1 | 0.9012 ± 0.0004 | 0.8899 ± 0.0011 |
| NLL | 0.3367 ± 0.0059 | 0.4328 ± 0.0017 |
| ECE-15 | 0.0463 ± 0.0031 | 0.0261 ± 0.0030 |
| Brier score | 0.1504 ± 0.0014 | 0.1718 ± 0.0005 |

Softmax was about one percentage point more accurate in every seed. DAEDL's lower ECE also repeated in every seed, but NLL and Brier score favored Softmax. The calibration metrics therefore do not support one universal ranking. ECE measures binned agreement between confidence and observed accuracy, whereas NLL and Brier score evaluate the complete predictive distribution as proper scoring rules.

![Figure 2 three-seed headline metrics](../figures/final/figure2_three_seed_headline.png)

The seed-3024 reliability diagram illustrates the binned behavior behind ECE. Sparse bins with fewer than 50 test examples are suppressed for readability; the metric itself still uses all 15 equal-width bins.

![Figure 3 reliability](../figures/final/figure3_reliability_seed3024.png)

### 6.2 Error and OOD uncertainty

| Metric | Softmax | DAEDL |
|---|---:|---:|
| Error AUROC | 0.9104 ± 0.0037 | 0.9034 ± 0.0031 |
| Error AUPR | 0.4926 ± 0.0202 | 0.4958 ± 0.0081 |
| SVHN OOD AUROC | 0.8860 ± 0.0140 | 0.8762 ± 0.0238 |
| SVHN OOD AUPR | 0.8296 ± 0.0234 | 0.7964 ± 0.0377 |

Both models assigned much greater normalized entropy to incorrect predictions than to correct ones. Softmax had a small but consistent error-AUROC advantage. DAEDL's larger absolute entropy does not itself mean that it separates errors better; AUROC measures ranking quality and is therefore more suitable for that comparison.

DAEDL consistently assigned greater entropy to SVHN than to CIFAR-10. However, Softmax had higher OOD AUROC at seeds 3024 and 3026, while DAEDL was higher at seed 3025. The aggregate difference therefore does not justify a stable superiority claim for either model.

![Figure 4 entropy distributions](../figures/final/figure4_entropy_distributions_seed3024.png)

## 7 Discussion

DAEDL added a structured probabilistic representation without improving clean classification. Its lower ECE shows that one aspect of confidence-frequency agreement improved, yet its worse NLL and Brier score show that the full probability vectors were less effective under two proper scoring rules. Reporting only ECE would therefore have produced an incomplete calibration conclusion.

The uncertainty magnitudes also require care. DAEDL produced higher entropy for correct, incorrect, ID, and OOD samples. That wider numerical range made mistakes and SVHN inputs visibly uncertain, but it did not reliably improve discrimination AUROC. The practical question is not simply whether a method emits a larger uncertainty number; it is whether that number ranks errors or distribution shifts consistently.

Multiple seeds changed the interpretation of OOD performance. Seed 3024 alone suggested a clear Softmax advantage, whereas seed 3025 reversed the order. The three-seed aggregate still favors Softmax numerically, but the change in direction limits the claim to this observed average. No hypothesis test was added for three runs because descriptive consistency is more appropriate at this scale.

The density diagnostics show another trade-off. DAEDL embeddings remained rank deficient: global centered rank was 93, 99, and 101 out of 128, with 35, 29, and 27 constant dimensions. Per-class covariance ranks ranged from 76 to 95. Adaptive jitter remained around $10^{-8}$ to $10^{-7}$, and every density computation was finite. This supports using the results while reporting the reduced-rank feature space as a limitation.

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

## References

[1] Yoon, T., and Kim, H. (2024). “Uncertainty Estimation by Density Aware Evidential Deep Learning.” *Proceedings of the 41st International Conference on Machine Learning*, PMLR 235, 57217–57243. https://proceedings.mlr.press/v235/yoon24a.html

[2] Sensoy, M., Kaplan, L., and Kandemir, M. (2018). “Evidential Deep Learning to Quantify Classification Uncertainty.” *Advances in Neural Information Processing Systems 31*. https://papers.nips.cc/paper/2018/hash/a981f2b708044d6fb4a71a1463242520-Abstract.html

[3] Guo, C., Pleiss, G., Sun, Y., and Weinberger, K. Q. (2017). “On Calibration of Modern Neural Networks.” *Proceedings of the 34th International Conference on Machine Learning*, PMLR 70, 1321–1330. https://proceedings.mlr.press/v70/guo17a.html

[4] Krizhevsky, A. (2009). *Learning Multiple Layers of Features from Tiny Images*. Technical report, University of Toronto. https://www.cs.toronto.edu/~kriz/learning-features-2009-TR.pdf

[5] Netzer, Y., Wang, T., Coates, A., Bissacco, A., Wu, B., and Ng, A. Y. (2011). “Reading Digits in Natural Images with Unsupervised Feature Learning.” *NIPS Workshop on Deep Learning and Unsupervised Feature Learning*. https://ufldl.stanford.edu/housenumbers/nips2011_housenumbers.pdf

[6] PyTorch Contributors. “MPS backend.” *PyTorch documentation*. https://docs.pytorch.org/docs/main/notes/mps.html

[7] Yoon, T., and Kim, H. “DAEDL official implementation.” GitHub. https://github.com/TaeseongYoon/DAEDL
