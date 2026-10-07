# CISC3024 Pattern Recognition — AI Assignment #2

## Density Aware Evidential Deep Learning

This repository contains an AI-assisted implementation and evaluation of Density Aware Evidential Deep Learning (DAEDL), a probability-based image-classification method published at ICML 2024. Codex identified the method, inspected the paper and official implementation, wrote the project code, ran the experiments, validated the saved artifacts, and prepared the documentation. The student did not write project code, in accordance with the assignment instructions.

The local experiment compares DAEDL with a conventional Softmax classifier on CIFAR-10. Both models use the same spectrally normalized CIFAR-adapted ResNet-18 backbone and were trained at seeds 3024, 3025, and 3026. Evaluation covers clean classification, calibration, uncertainty on prediction errors, and out-of-distribution detection with SVHN.

## Method

Evidential classification predicts the concentration parameters of a Dirichlet distribution over class-probability vectors. The expected class probability is obtained by normalizing those concentrations. DAEDL uses an exponential concentration parameterization during training, then fits class-conditional Gaussian densities to the network's penultimate features. At inference time, a normalized feature-density score scales the logits before the Dirichlet concentrations are formed. Lower-density inputs therefore produce flatter class probabilities and greater predictive uncertainty.

The implementation uses spectral normalization to constrain the feature extractor, a 128-dimensional embedding, and CPU float64 Gaussian discriminant analysis (GDA). The latter is required because the PyTorch MPS backend does not support float64 tensors.

## Experiments and key results

The final values are means and sample standard deviations across three independent training/split seeds.

| Metric | Softmax | DAEDL |
|---|---:|---:|
| Accuracy | 0.9013 ± 0.0004 | 0.8902 ± 0.0012 |
| Macro F1 | 0.9012 ± 0.0004 | 0.8899 ± 0.0011 |
| NLL | 0.3367 ± 0.0059 | 0.4328 ± 0.0017 |
| ECE-15 | 0.0463 ± 0.0031 | 0.0261 ± 0.0030 |
| Error-detection AUROC | 0.9104 ± 0.0037 | 0.9034 ± 0.0031 |
| SVHN OOD AUROC | 0.8860 ± 0.0140 | 0.8762 ± 0.0238 |

Softmax consistently achieved higher accuracy and Macro F1 and lower NLL and Brier score. DAEDL consistently achieved lower ECE. Both methods assigned greater entropy to incorrect predictions, while Softmax had a small consistent advantage in error-detection AUROC. SVHN OOD rankings changed across seeds, so the experiment does not establish a stable OOD winner. Full results and limitations are documented in [PHASE3B_RESULTS.md](PHASE3B_RESULTS.md).

## Reproduction

Python 3.11 is required. The validated environment used Python 3.11.5, PyTorch 2.14.1, torchvision 0.29.1, Apple MPS for training, and CPU for checkpoint evaluation and DAEDL density fitting.

```bash
python3.11 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
python -m pip install --no-build-isolation -e .
MPLCONFIGDIR=.cache/matplotlib python -m pytest
```

One complete frozen run can be reproduced with:

```bash
python scripts/run_final_core.py \
  --model softmax \
  --device mps \
  --phase FINAL_REPLICATION \
  --seed 3025 \
  --split-seed 3025 \
  --run-dir results/final_replication_softmax_seed3025
```

Replace `softmax` with `daedl` for the evidential model. Seed 3024 uses phase `FINAL_CORE`; seeds 3025 and 3026 use `FINAL_REPLICATION`. Use `--device cpu` on systems without MPS or CUDA. The data loaders obtain CIFAR-10 and SVHN through torchvision. Generated datasets, checkpoints, and full run directories remain local and are excluded from Git.

To validate and aggregate six completed runs:

```bash
MPLCONFIGDIR=.cache/matplotlib python scripts/finalize_phase3b.py
python scripts/prepare_final_submission.py
```

The final configuration is recorded in [configs/final_experiment.json](configs/final_experiment.json). Compact public result files are in [results/final_summary](results/final_summary).

## Repository structure

| Path | Purpose |
|---|---|
| `src/daedl_project/` | Models, DAEDL loss, GDA, data loading, metrics, training, and evaluation |
| `scripts/` | Environment checks, pilot/final runs, validation, aggregation, and submission packaging |
| `tests/` | Mathematical, metric, GDA, model, and reproducibility tests |
| `configs/` | Frozen final experiment configuration |
| `results/final_summary/` | Tracked machine-readable final results and figure inputs |
| `figures/final/` | Selected report figures generated from saved data |
| `references/` | Primary-source register and checked bibliography |
| `report/` | Report source and generated submission documents |
| `AI_USAGE_LOG.md` | Chronological record of Codex's work and validation decisions |

## Source paper

Taeseong Yoon and Heeyoung Kim. “Uncertainty Estimation by Density Aware Evidential Deep Learning.” *Proceedings of the 41st International Conference on Machine Learning*, PMLR 235:57217–57243, 2024. [Official PMLR page](https://proceedings.mlr.press/v235/yoon24a.html) · [Authors' official implementation](https://github.com/TaeseongYoon/DAEDL)

This repository is a clean assignment-scale implementation, not a claim of exact reproduction of the paper's architecture or headline results.
