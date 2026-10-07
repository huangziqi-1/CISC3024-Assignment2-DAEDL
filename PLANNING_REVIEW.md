# DAEDL Suitability and Experiment Planning Review

Date: 2026-10-07 (Asia/Taipei)

Status: planning review; no final training has been run.

## 1. Assignment interpretation

The submission must demonstrate more than a successful classifier. It needs a traceable AI-assisted process showing that AI:

1. found and justified a recent probability- or Bayes-based recognition method;
2. explained the method accurately at Pattern Recognition course level;
3. implemented it without student-written code;
4. designed and ran meaningful experiments with honest, reproducible results;
5. helped the student form a specific learning reflection; and
6. published the source code at a webpage URL, normally a public GitHub repository.

A strong submission should therefore connect method, implementation, experiments, saved evidence, and discussion. Paper results may provide context but must never be presented as local results.

## 2. DAEDL verification

### A. Recency

The full title is **Uncertainty Estimation by Density Aware Evidential Deep Learning**, by Taeseong Yoon and Heeyoung Kim. It appeared in the Proceedings of the 41st International Conference on Machine Learning, PMLR volume 235, pages 57217-57243, held 21-27 July 2024.

As of October 2026 it is about two years old. For an undergraduate assignment asking for a recent algorithm, this is reasonably and defensibly recent. The report should say "published at ICML 2024," not imply that it was published in 2026.

### B. Probability and Bayesian foundation

DAEDL qualifies on substantive grounds, not merely because a neural network emits normalized scores.

For a `K`-class problem, evidential classification places a Dirichlet distribution over the unknown class-probability vector:

`pi | x ~ Dir(alpha(x))`.

The predictive categorical probability is the Dirichlet mean:

`p(y=k | x) = E[pi_k | x] = alpha_k(x) / S(x)`, where `S(x) = sum_j alpha_j(x)`.

Conventional EDL parameterizes `alpha_k = e_k + 1`, where nonnegative `e_k` is class evidence. Its subjective-logic uncertainty mass is commonly written `u = K / S`; larger strength means a more concentrated Dirichlet distribution. The EDL training objective used by DAEDL combines expected squared classification error under the Dirichlet with a KL penalty toward a uniform Dirichlet for misleading evidence.

DAEDL deliberately changes the parameterization. During training it uses:

`alpha_k(x) = exp(z_k(x))`,

not `e_k + 1`. This makes the expected class probabilities equal to ordinary softmax probabilities during training while retaining a full Dirichlet distribution and its concentration.

After training, DAEDL fits Gaussian discriminant analysis in learned feature space. For feature `h=f(x)`, each class has an empirical weight, mean, and covariance. The marginal feature density is:

`p(h) = sum_k omega_k N(h | mu_k, Sigma_k)`.

The log density is min-max normalized against training densities and clipped to obtain `s(x)` in `[0,1]`. At prediction time:

`alpha_k(x) = exp(z_k(x) s(x))`.

Equivalently, the predictive mean is a softmax with an input-dependent temperature `T(x)=1/s(x)`. Far from training features, `s(x)` approaches zero, all concentration parameters approach one, and the predictive distribution approaches uniform. The paper also proves an interpretation as an input-dependent posterior of a Dirichlet-Categorical model with improper prior `Dir(0)`.

These components are fundamental:

- Dirichlet distribution over categorical probabilities;
- evidential expected-risk and KL training objective;
- predictive probabilities derived by integrating over the Dirichlet;
- uncertainty represented by predictive entropy and Dirichlet concentration/strength;
- probability density estimation in feature space; and
- a stated Bayesian Dirichlet-Categorical interpretation.

Important qualification: DAEDL is not a fully Bayesian neural network. Its network weights are point estimates. Its probabilistic/Bayesian content is in the second-order Dirichlet prediction, evidential objective, density model, and posterior interpretation.

### C. Pattern-recognition relevance

DAEDL is directly a supervised image-recognition method. The paper evaluates image classification on MNIST and CIFAR-10, uses VGG-16 for CIFAR-10, and tests error detection, calibration, OOD recognition, and corruption shifts. CIFAR-10 object recognition is a standard computer-vision pattern-recognition task.

### D. Reproducibility

Official code exists under the MIT license and directly supports MNIST, CIFAR-10, and CIFAR-100. For CIFAR-10 it also loads SVHN and CIFAR-100 as OOD sets. The paper used 47,500 CIFAR-10 training images, 2,500 validation images, 10,000 test images, a VGG-16 backbone, batch size 64, Adam, initial learning rate `1e-3`, `LambdaLR` decay `0.95^epoch`, regularization `5e-2`, dropout 0.5, early stopping, and at most 100 epochs. Paper results average five runs.

The official repository does **not** contain a requirements file, environment file, released model checkpoints, or a complete reproduction command. The `--pretrained` option refers to absent files, and its CIFAR-10 path begins at filesystem root. Training from scratch is therefore required unless we independently create checkpoints.

The repository is useful as a reference implementation but needs adaptation. Notable issues found by inspection:

- default device is `cuda:0`; there is no automatic CPU/MPS selection;
- CIFAR-10 training data is opened with `download=False` before any download call, so a clean first run fails;
- `val_seed` is declared but unused, and global random seeds are not set;
- GDA embeddings and full covariance matrices use float64 on the chosen device; MPS cannot store float64 tensors;
- the Brier score is calculated incorrectly in the return path and is not written to `results.json`;
- an AUROC variable intended for epistemic confidence reuses the aleatoric score;
- saved example results lack seed, environment, commit, runtime, and model settings; and
- the checked-in sample result (77.42% accuracy) is not the paper's 91.11% and must not be treated as a validated reproduction.

Conclusion: reproducibility is feasible through a clean reimplementation with tests and explicit metadata, not by running the official entry point unchanged.

## 3. Official resources

Exact URLs are recorded in `references/SOURCES.md`. The essential primary resources are:

- PMLR publication page and BibTeX;
- official PMLR paper PDF;
- OpenReview paper entry;
- arXiv record; and
- authors' official GitHub repository and MIT license.

## 4. Local environment assessment

| Item | Observed value | Assessment |
|---|---|---|
| Computer | MacBook Air, model Mac17,4 | Thin, passively cooled laptop; long training may throttle |
| Processor | Apple M5, 10 CPU cores (4 performance + 6 efficiency) | Strong CPU and Apple GPU, no NVIDIA CUDA |
| Architecture | arm64 / Apple Silicon | Native arm64 packages required |
| Memory | 16 GB unified memory | Adequate for CIFAR-10 and a compact model; avoid large batches/models |
| OS | macOS 26.4.1, build 25E253 | Meets PyTorch's documented MPS OS requirement |
| Free disk | about 481 GiB on the data volume | More than adequate for datasets and results |
| Active Python | Anaconda Python 3.13.9 | Too new and already crowded for a conservative scientific environment |
| Other Pythons | Homebrew 3.13.0; framework 3.11.5; system 3.9.6 | Python 3.11 is the preferred project target |
| PyTorch | 2.11.0 in Anaconda base | Installed, but current MPS runtime test fails |
| MPS | build support `True`, runtime availability `False` | Must be retested in a fresh environment |
| CUDA | unavailable | Expected on Apple Silicon |
| Git | 2.39.2 (Apple Git-143) | Adequate |

The current PyTorch error says MPS requires macOS 14+ even though the Mac runs macOS 26.4.1. That points to an environment/build version-detection problem, not insufficient hardware. A fresh Python 3.11 virtual environment using the current stable arm64 PyTorch wheel is recommended. The base environment must remain untouched.

Even after MPS works, DAEDL density fitting should run on CPU because PyTorch's MPS backend does not support float64. Training and neural-network inference can use MPS; embeddings should be transferred to CPU, and GDA should be fitted in float64 with regularized covariance. If the fresh MPS test still fails, Phase A can use CPU and final training should move to a CUDA service only if laptop runtime is unacceptable.

Why isolation is necessary:

- prevents assignment packages from changing the user's base environment;
- targets the widely supported Python 3.11 line rather than Python 3.13;
- makes the MPS diagnosis independent of the current Anaconda build;
- permits a tested dependency lock and exact `pip freeze`; and
- makes the public repository reproducible.

The environment files are intentionally provisional. After Phase A passes, exact installed versions and platform metadata should be frozen to `results/environment/` and the top-level dependency files updated.

## 5. Recommended experimental design

### Core research question

On a fair CIFAR-10 classification setup, does density-aware evidential prediction retain useful classification accuracy while improving calibration and assigning greater uncertainty to errors, corrupted inputs, and OOD images?

### Models

Use three models if the Phase B timing is acceptable:

1. **Softmax:** common image backbone, linear logits, cross-entropy loss.
2. **EDL:** identical backbone, conventional `alpha=e+1` parameterization, expected-MSE plus annealed KL regularization.
3. **DAEDL:** identical backbone topology with spectral normalization, exponential concentration parameterization, the paper's evidential objective, CPU-fitted GDA, and density-scaled prediction.

Use a CIFAR-adapted compact ResNet-18 with a 128-dimensional penultimate embedding rather than the paper's full VGG-16. This is an assignment-scale adaptation, not an exact paper reproduction. The smaller embedding makes full-covariance GDA stable and inexpensive. Use the same channel topology, preprocessing, train/validation split, batch size, optimizer family, early-stopping rule, and seeds. Because spectral normalization is part of DAEDL, report it as a deliberate method difference. A secondary ablation may apply spectral normalization to all models if pilot time permits, but it is not required.

If three-model training is too slow, retain Softmax and DAEDL. Do not drop Softmax: it provides the clearest conventional baseline. EDL is the first optional component because DAEDL-vs-EDL is scientifically informative but adds another full training run.

### Data and splits

- ID dataset: CIFAR-10.
- Fixed stratified split from the 50,000 training images: 45,000 train / 5,000 validation. Keep the official 10,000 test images untouched.
- Primary OOD set: SVHN test split, resized/normalized through the same input pipeline.
- Optional second OOD set: CIFAR-100 test split if download and evaluation time remain modest.
- Corruptions generated deterministically from CIFAR-10 test images before normalization: Gaussian noise, Gaussian blur, and brightness, each at three documented severity levels plus severity zero.

Do not use test or OOD data for early stopping, density normalization, or hyperparameter selection.

### Optimization and seeds

- Phase A: 5,000 stratified training images, 1,000 validation images, 1-2 epochs, one seed.
- Phase B: full split, one seed, maximum 20-30 epochs with early stopping; record wall-clock time and peak memory.
- Phase C: choose the final epoch ceiling from Phase B. Target three fixed seeds only if the projected total runtime is reasonable. Report mean and standard deviation when multiple seeds are actually completed; otherwise report the single-seed limitation honestly.
- Candidate starting settings: batch size 64, AdamW or Adam, initial learning rate `1e-3`, validation-loss early stopping with patience 7, and a documented learning-rate scheduler. Apply identical optimization settings unless a baseline becomes clearly unstable; record any exception.

### Experiment A - Classification

Report accuracy, macro precision, macro recall, and macro F1 on CIFAR-10 test data. Include a confusion matrix only if it reveals a useful class pattern. These metrics verify that uncertainty modelling has not made the classifier unusable.

### Experiment B - Calibration

From saved test probabilities, calculate 15-bin ECE, multiclass NLL, and multiclass Brier score. Generate a reliability diagram with equal-width bins and display bin counts or suppress very sparse bins. Accuracy alone cannot establish probability quality.

### Experiment C - Correct versus incorrect predictions

Compare maximum predicted probability and normalized predictive entropy for correct and incorrect CIFAR-10 predictions. For evidential models, also compare Dirichlet strength (or its inverse) as a separate epistemic proxy. Report group means/medians and AUROC for detecting an error, with "incorrect" defined as the positive class. A box/violin plot should show whether uncertainty is systematically higher on mistakes.

### Experiment D - Controlled corruption

For each corruption and severity, report accuracy, ECE, mean confidence, and mean normalized predictive entropy. Plot severity trends with clean data as level zero. This directly tests the density-awareness intuition without downloading the full CIFAR-10-C archive.

### Experiment E - OOD detection

Use clean CIFAR-10 test images as ID and SVHN test images as OOD. Use normalized predictive entropy as a common score for all models; also report inverse Dirichlet strength for EDL/DAEDL as a model-specific score. Compute AUROC and average precision with OOD consistently labelled positive. Report average ID/OOD confidence and uncertainty. Add CIFAR-100 only after the primary analysis works.

### Experiment F - Qualitative examples

Generate a deterministic grid containing a confident correct case, uncertain correct case, high-confidence mistake, corrupted sample, and OOD sample. Show ground truth where meaningful, prediction, maximum probability, predictive entropy, and evidential strength where applicable. Select examples by explicit ranking rules saved in code, not manual cherry-picking.

### Artifact traceability

Each run should save:

- resolved YAML configuration;
- split indices and random seed;
- Git commit and dirty-state flag;
- OS, Python, package, and device metadata;
- epoch-level training/validation CSV;
- best checkpoint and checkpoint hash locally (not in Git);
- per-example predictions in compressed NPZ or Parquet;
- summary metrics in JSON/CSV; and
- runtime and stopping reason.

Tables and figures must read these artifacts. Nothing should contain manually typed experimental values.

## 6. Proposed project structure

```text
CISC3024_AI_Assignment2/
├── README.md
├── AI_USAGE_LOG.md
├── PLANNING_REVIEW.md
├── requirements.txt
├── environment.yml
├── .gitignore
├── configs/                 # versioned experiment settings
├── data/README.md           # download policy; data itself ignored
├── src/                     # models, losses, density, metrics, data, reproducibility
├── scripts/                 # train, evaluate, aggregate, and figure entry points
├── tests/                   # equations, metrics, determinism, and smoke tests
├── results/                 # generated machine-readable outputs; ignored
├── figures/                 # generated figures; ignored until selected for report
├── checkpoints/             # local model files; ignored
├── report/                  # outline, later report source and selected assets
└── references/              # source register and later BibTeX
```

Compared with the tentative structure, `configs/`, `tests/`, and a documented but ignored `data/` directory are added. Tests are important because small mistakes in OOD score direction, Dirichlet strength, ECE binning, or density normalization can reverse conclusions.

## 7. Proposed report structure

1. **Introduction and assignment objective** - define the recognition and uncertainty problem.
2. **AI-assisted algorithm search and selection** - selection criteria, primary-source search, alternatives considered, and why DAEDL was chosen.
3. **Probabilistic background and DAEDL** - softmax baseline, Dirichlet EDL, DAEDL parameterization, feature density, prediction rule, and uncertainty interpretation.
4. **AI-assisted implementation** - repository audit, clean implementation architecture, reproducibility controls, and adaptations for Apple Silicon.
5. **Experimental methodology** - datasets/splits, models, fairness, training, metrics, corruption/OOD protocols, and compute environment.
6. **Results** - only automatically generated local tables and figures.
7. **Discussion and limitations** - interpret accuracy/calibration/uncertainty tradeoffs, failure cases, statistical/run limitations, and differences from the paper.
8. **What I learned** - a genuine first-person reflection written after experiments, covering probability calibration, Dirichlet evidence, density shift, reproducibility, and AI oversight.
9. **AI usage, verification, and limitations** - summarize the chronological log, distinguish AI actions from student review, and explain how claims/results were checked.
10. **Conclusion** - answer the research question without overclaiming.
11. **Source-code webpage** - public GitHub URL and tested reproduction command.
12. **References** - primary paper, foundational EDL and calibration sources, datasets, and software.

This structure maps directly to all six assignment requirements while reading like an undergraduate technical report rather than imitating a research article.

## 8. Proposed figures and tables

Keep the main report to six high-value items:

1. **DAEDL pipeline diagram** - logits, feature-space GDA density, normalized density, scaled logits, Dirichlet parameters, probabilities/uncertainty. Supports the algorithm section and distinguishes training from inference.
2. **Main results table** - accuracy/macro F1, ECE/NLL/Brier, error-detection AUROC, and OOD AUROC/AUPR for each model. Generated by the aggregation script.
3. **Reliability diagram** - one panel per model with the same bins and axes. Generated by Experiment B.
4. **Uncertainty distributions** - compact paired panels for correct versus incorrect and ID versus OOD. Generated by Experiments C and E.
5. **Corruption trend figure** - severity versus accuracy and predictive entropy, with corruption type encoded by line style or small multiples. Generated by Experiment D.
6. **Representative prediction grid** - rule-selected clean/error/corrupted/OOD examples with probabilities and uncertainty. Generated by Experiment F.

A training-curve figure belongs in an appendix or repository unless it exposes instability. A confusion matrix is optional and should replace, not add to, another figure if it provides a substantive class-level insight.

## 9. Necessary mathematics

Include only equations that carry an explanatory burden:

1. **Dirichlet model `pi~Dir(alpha)`** - teaches that the model predicts a distribution over class-probability vectors, not a single probability vector.
2. **EDL parameterization `alpha_k=e_k+1`** - explains conventional evidence and establishes exactly what DAEDL changes.
3. **Strength and predictive mean `S=sum alpha`, `E[pi_k]=alpha_k/S`** - connects evidence/concentration to ordinary class prediction.
4. **One uncertainty equation** - normalized predictive entropy for a common cross-model comparison; optionally `K/S` or inverse strength as an evidential proxy, with limitations stated.
5. **Feature GDA density** - shows why proximity to learned training features affects uncertainty.
6. **Density normalization `s(x)`** - explains clipping and the meaning of zero versus one density score.
7. **DAEDL rule `alpha=exp(s(x)z(x))` and `T=1/s(x)`** - the central equation tying density, evidence, and adaptive temperature together.
8. **Training objective at component level** - expected squared error plus KL regularization; full expansion is optional and may be placed in an appendix.

Do not present `alpha=e+1` as DAEDL's own final parameterization. Do not overload the main text with proofs, covariance estimators, or every decomposition of Dirichlet uncertainty.

## 10. Risks and mitigations

| Risk | Consequence | Mitigation |
|---|---|---|
| Current MPS environment is broken | Training falls back to slow CPU | Fresh isolated Python 3.11 environment; smoke-test MPS before any run |
| MPS has no float64 | Official GDA path fails | Move saved embeddings to CPU and fit regularized GDA in float64 |
| Full VGG-16/100-epoch replication is expensive | Excessive runtime and thermal throttling | Compact CIFAR ResNet and staged timing; state clearly that this is an assignment-scale reproduction |
| Official code lacks weights/environment | Cannot reproduce by one command | Clean implementation, tested dependencies, own checkpoints, provenance metadata |
| Full covariance can be singular | Cholesky failure or unstable density | 128-D embedding, per-class sample checks, shrinkage/jitter, CPU numerical tests |
| Density min-max scaling is sensitive to extremes | Unstable uncertainty | Match paper for primary result, log min/max, and optionally test robust quantiles as a labelled sensitivity analysis |
| EDL/DAEDL exponential outputs can overflow | NaN loss or invalid probabilities | stable computations, log-domain checks, gradient clipping only if justified, non-finite assertions |
| Baseline unfairness | Misleading comparison | identical split/topology/budget and explicit documentation of method-specific spectral normalization/losses |
| OOD score orientation mistakes | AUROC/AUPR silently reversed | define OOD as positive in one tested metrics function and store raw scores |
| Too many metrics/plots | Diluted report | prioritize one main table and five explanatory figures |
| Single-seed final result | Weak reliability | target three seeds after pilot; otherwise disclose limitation and avoid variance claims |
| Turnitin similarity | Generic or copied prose | write from actual logs/results in original wording; cite rather than paraphrase closely |

## 11. Final recommendation

**Proceed with DAEDL.** It satisfies recency, has a clear probability/Bayesian foundation, is directly relevant to computer-vision pattern recognition, has official MIT-licensed source code, and supports manageable CIFAR-10/OOD experiments.

There is no disqualifying theoretical, licensing, or hardware problem. The official code does have enough engineering and reproducibility defects that it should be treated as a reference, not copied wholesale. The strongest feasible submission is a clean assignment-scale implementation with Softmax, EDL, and DAEDL; CPU GDA; staged MPS testing; fixed splits/seeds; saved per-example predictions; and a focused calibration/corruption/OOD analysis.

Do not begin final training until Phase A confirms the fresh environment, mathematical unit tests, dataset pipeline, and result logging.

