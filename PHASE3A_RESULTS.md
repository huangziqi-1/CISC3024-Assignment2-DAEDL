# Phase 3A Results: Seed-3024 Final Core Experiment

Phase 3A completed the first full-data comparison of Softmax and Density Aware Evidential Deep Learning (DAEDL). These are genuine local results from the fixed 45,000/5,000/10,000 CIFAR-10 split, not paper values or pilot values. No seed 3025/3026, corruption experiment, CIFAR-100 OOD experiment, EDL baseline, report generation, or publication was run.

## Fairness and fidelity decisions

Both methods used the identical compact CIFAR ResNet-18: base width 32, 128-D embedding, CIFAR 3x3/stride-1 stem, and spectral normalization on every convolution, learned shortcut projection, the embedding linear layer, and the classifier. Batch-normalization layers and identity shortcuts have no eligible weight matrix. This corrects the pilot mismatch in which only DAEDL used spectral normalization.

Both models used seed 3024, batch size 64, AdamW, initial learning rate `1e-3`, weight decay `5e-4`, 30 epochs, and cosine decay to `1e-5`. Every epoch completed. The best checkpoint was selected by minimum validation NLL, and the untouched test set was not used for model selection.

The authors' official DAEDL implementation sums every data and KL contribution across the batch. The local implementation averages the complete per-example objective, dividing the expected-error, predictive-variance, and KL components by the same batch size. This changes only the total optimization scale, not the KL/data weighting. A regression test confirms equivalence after multiplying the local mean by batch size.

MPS was used for training. Both selected spectral-normalized checkpoints were evaluated on CPU because PyTorch 2.14.1 produced non-finite outputs when a reloaded spectral-normalized pilot checkpoint was evaluated on MPS. DAEDL clean feature extraction and full-covariance GDA used CPU float64.

## Training

| Method | Epochs | Selected epoch | Training runtime | Total pipeline runtime | Final train accuracy | Best validation accuracy | Selected validation NLL |
|---|---:|---:|---:|---:|---:|---:|---:|
| Softmax | 30 | 26 | 1,389.84 s (23.16 min) | 1,552.66 s (25.88 min) | 96.49% | 90.56% | 0.321901 |
| DAEDL | 30 | 27 | 1,488.74 s (24.81 min) | 1,863.90 s (31.07 min) | 93.74% | 89.30% | 0.409656 |

## Clean CIFAR-10 classification and calibration

| Method | Accuracy | Macro precision | Macro recall | Macro F1 | NLL | ECE-15 | Brier |
|---|---:|---:|---:|---:|---:|---:|---:|
| Softmax | 0.9011 | 0.9013 | 0.9011 | 0.9011 | 0.3410 | 0.0438 | 0.1514 |
| DAEDL | 0.8912 | 0.8911 | 0.8912 | 0.8908 | 0.4319 | 0.0261 | 0.1715 |

Softmax was 0.99 percentage points more accurate and had better NLL and Brier score. DAEDL had lower 15-bin ECE. Thus this run does not support a blanket claim that either method was better calibrated: ECE favored DAEDL, while the proper scoring rules NLL and Brier favored Softmax.

## Error uncertainty

Normalized predictive entropy was the common uncertainty score; prediction error was positive.

| Method | Correct mean / median | Incorrect mean / median | Error AUROC | Error AUPR |
|---|---:|---:|---:|---:|
| Softmax | 0.0439 / 0.0010 | 0.2696 / 0.2818 | 0.9071 | 0.4749 |
| DAEDL | 0.1520 / 0.0327 | 0.6458 / 0.7313 | 0.9016 | 0.4866 |

Both models assigned substantially higher entropy to mistakes. Softmax had slightly higher error AUROC, while DAEDL had slightly higher AUPR. The methods have different error base rates (989 versus 1,088 errors), so their AUPR values require care in direct interpretation.

## DAEDL density diagnostics

- Full clean training embeddings: 45,000, exactly 4,500 per class.
- Embedding dimension: 128; global centered rank: 93.
- Exactly constant dimensions: 35; near-constant dimensions at the documented relative threshold: 35. This improved from 54 constant dimensions in the two-epoch pilot but remains substantial.
- Per-class covariance ranks: 77 to 80.
- Positive-spectrum condition numbers: approximately `3.21e6` to `2.55e10`; tiny negative minimum eigenvalues were at floating-point roundoff scale (`-5.06e-17` or closer to zero).
- Adaptive diagonal jitter: `5.43e-8` to `1.16e-7`.
- Training log-density range: `-1650.53` to `594.66`.
- Non-finite embedding fraction: zero.

The rank deficiency is consistent with dead coordinates after the explicit ReLU embedding and additional within-class degeneracy. It was not handled by repeatedly increasing jitter. A diagnostic refit removed only dimensions that were constant using training data alone. The resulting density scores had Pearson correlation 0.9987 with the primary scores and mean absolute change 0.0034 across ID and OOD samples. Clean accuracy stayed 0.8912; entropy OOD AUROC changed from 0.85142 to 0.85109. Therefore the primary result is not materially driven by the 35 globally dead coordinates, although covariance rank remains a limitation to disclose.

## SVHN OOD detection

The ID and OOD sets each contain 10,000 deterministically selected images. OOD is positive and higher normalized predictive entropy means more OOD-like.

| Method | CIFAR-10 mean / median | SVHN mean / median | OOD AUROC | OOD AUPR |
|---|---:|---:|---:|---:|
| Softmax | 0.0662 / 0.0019 | 0.3109 / 0.3143 | 0.8994 | 0.8556 |
| DAEDL | 0.2058 / 0.0466 | 0.6140 / 0.6685 | 0.8514 | 0.7561 |

DAEDL's secondary density uncertainty (`1 - density_score`) achieved AUROC 0.8641 and AUPR 0.7772, with ID mean/median 0.0317/0.0213 and SVHN mean/median 0.0648/0.0537. Although DAEDL uncertainty was numerically higher on SVHN than CIFAR-10, its primary separation was weaker than Softmax. This run therefore does not support claiming that DAEDL outperformed Softmax on SVHN OOD detection.

## Integrity checks

The automated audit passed. The methods used identical train, validation, test, and SVHN indices; train/validation overlap was zero; both selected checkpoints were the actual validation-NLL minima; all numeric prediction outputs were finite; probability maximum row-sum errors were below `2.21e-7`; saved clean/error/OOD metrics matched recomputation from CSV files; DAEDL concentrations were positive; density fitting had zero non-finite values; and score direction treated OOD/error as positive.

## Interpretation and recommendation

The result is internally consistent and suitable for retention as the seed-3024 core experiment. On this architecture and training budget, Softmax produced better classification accuracy, NLL, Brier score, error AUROC, and entropy-based SVHN OOD detection. DAEDL produced lower ECE and slightly higher error AUPR, but its primary OOD result was weaker. These are single-seed findings and should not be generalized without replication.

Phase 3B should add seeds 3025 and 3026 with the configuration frozen before adding the corruption suite. No post-hoc training adjustment is justified by an implementation failure: the runs were stable, the split/checkpoint audit passed, and the active-dimension sensitivity result was nearly unchanged. If the multi-seed result confirms the same pattern, proceed to the planned corruption experiments and report the negative OOD comparison honestly. Do not add EDL or CIFAR-100 until the approved core plan is complete.

## Artifact locations

- Softmax FINAL_CORE run: `results/final_core_softmax_seed3024_phase3a/`
- DAEDL FINAL_CORE run: `results/final_core_daedl_seed3024_phase3a/`
- Combined validated analysis: `results/phase3a_seed3024_final_analysis_v2/`
- Machine-readable consistency checks: `results/phase3a_seed3024_final_analysis_v2/consistency_checks.json`
- Machine-readable comparison table: `results/phase3a_seed3024_final_analysis_v2/comparison_metrics.csv`

Generated datasets, checkpoints, predictions, and figures remain ignored by Git; source code and documentation are version controlled.
