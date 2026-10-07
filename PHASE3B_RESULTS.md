# Phase 3B: Frozen three-seed replication

## Protocol

Seeds 3024, 3025, and 3026 are independent training/split replicates. The Phase-3A seed-3024 artifacts were reused unchanged. Seeds 3025 and 3026 used the frozen Phase-3A architecture, optimization, validation-NLL checkpoint rule, DAEDL objective, density estimator, and evaluation definitions. Values below are generated from saved per-example predictions; aggregate variability is the sample standard deviation across the three seeds (`ddof=1`). OOD is SVHN and the positive class is OOD.

## Per-seed clean classification and calibration

| seed | model | accuracy | macro_precision | macro_recall | macro_f1 |
| --- | --- | --- | --- | --- | --- |
| 3024 | softmax | 0.9011 | 0.9013 | 0.9011 | 0.9011 |
| 3024 | daedl | 0.8912 | 0.8911 | 0.8912 | 0.8908 |
| 3025 | softmax | 0.9018 | 0.9018 | 0.9018 | 0.9016 |
| 3025 | daedl | 0.8888 | 0.8886 | 0.8888 | 0.8887 |
| 3026 | softmax | 0.9011 | 0.9009 | 0.9011 | 0.9009 |
| 3026 | daedl | 0.8906 | 0.8903 | 0.8906 | 0.8904 |

## Per-seed calibration

| seed | model | nll | ece_15 | brier |
| --- | --- | --- | --- | --- |
| 3024 | softmax | 0.3410 | 0.0438 | 0.1514 |
| 3024 | daedl | 0.4319 | 0.0261 | 0.1715 |
| 3025 | softmax | 0.3299 | 0.0452 | 0.1488 |
| 3025 | daedl | 0.4317 | 0.0232 | 0.1715 |
| 3026 | softmax | 0.3391 | 0.0498 | 0.1511 |
| 3026 | daedl | 0.4347 | 0.0292 | 0.1724 |

## Per-seed error uncertainty

| seed | model | error_correct_mean_entropy | error_correct_median_entropy | error_incorrect_mean_entropy | error_incorrect_median_entropy | error_auroc | error_aupr |
| --- | --- | --- | --- | --- | --- | --- | --- |
| 3024 | softmax | 0.0439 | 0.0010 | 0.2696 | 0.2818 | 0.9071 | 0.4749 |
| 3024 | daedl | 0.1520 | 0.0327 | 0.6458 | 0.7313 | 0.9016 | 0.4866 |
| 3025 | softmax | 0.0412 | 0.0009 | 0.2736 | 0.2742 | 0.9143 | 0.5146 |
| 3025 | daedl | 0.1467 | 0.0302 | 0.6560 | 0.7522 | 0.9070 | 0.5018 |
| 3026 | softmax | 0.0384 | 0.0007 | 0.2542 | 0.2592 | 0.9098 | 0.4884 |
| 3026 | daedl | 0.1517 | 0.0290 | 0.6528 | 0.7524 | 0.9017 | 0.4990 |

## Per-seed OOD uncertainty

| seed | model | ood_id_mean_entropy | ood_id_median_entropy | ood_mean_entropy | ood_median_entropy | ood_auroc | ood_aupr |
| --- | --- | --- | --- | --- | --- | --- | --- |
| 3024 | softmax | 0.0662 | 0.0019 | 0.3109 | 0.3143 | 0.8994 | 0.8556 |
| 3024 | daedl | 0.2058 | 0.0466 | 0.6140 | 0.6685 | 0.8514 | 0.7561 |
| 3025 | softmax | 0.0641 | 0.0018 | 0.2556 | 0.2566 | 0.8716 | 0.8101 |
| 3025 | daedl | 0.2034 | 0.0426 | 0.7444 | 0.8195 | 0.8989 | 0.8308 |
| 3026 | softmax | 0.0597 | 0.0015 | 0.2572 | 0.2636 | 0.8871 | 0.8231 |
| 3026 | daedl | 0.2065 | 0.0437 | 0.6935 | 0.7731 | 0.8782 | 0.8023 |

## Three-seed aggregate

| metric | softmax | daedl |
| --- | --- | --- |
| accuracy | 0.9013 ± 0.0004 | 0.8902 ± 0.0012 |
| brier | 0.1504 ± 0.0014 | 0.1718 ± 0.0005 |
| ece_15 | 0.0463 ± 0.0031 | 0.0261 ± 0.0030 |
| error_aupr | 0.4926 ± 0.0202 | 0.4958 ± 0.0081 |
| error_auroc | 0.9104 ± 0.0037 | 0.9034 ± 0.0031 |
| error_correct_mean_entropy | 0.0412 ± 0.0028 | 0.1501 ± 0.0030 |
| error_correct_median_entropy | 0.0009 ± 0.0001 | 0.0306 ± 0.0018 |
| error_incorrect_mean_entropy | 0.2658 ± 0.0102 | 0.6515 ± 0.0052 |
| error_incorrect_median_entropy | 0.2717 ± 0.0115 | 0.7453 ± 0.0121 |
| macro_f1 | 0.9012 ± 0.0004 | 0.8899 ± 0.0011 |
| nll | 0.3367 ± 0.0059 | 0.4328 ± 0.0017 |
| ood_aupr | 0.8296 ± 0.0234 | 0.7964 ± 0.0377 |
| ood_auroc | 0.8860 ± 0.0140 | 0.8762 ± 0.0238 |
| ood_id_mean_entropy | 0.0633 ± 0.0033 | 0.2052 ± 0.0016 |
| ood_id_median_entropy | 0.0017 ± 0.0002 | 0.0443 ± 0.0021 |
| ood_mean_entropy | 0.2746 ± 0.0314 | 0.6840 ± 0.0657 |
| ood_median_entropy | 0.2782 ± 0.0315 | 0.7537 ± 0.0773 |

## Paired differences (DAEDL minus Softmax)

Positive values favor DAEDL only for accuracy, macro F1, error-detection AUROC, and OOD AUROC/AUPR. Negative values favor DAEDL for NLL, ECE, and Brier.

| metric | 3024 | 3025 | 3026 |
| --- | --- | --- | --- |
| accuracy | -0.0099 | -0.0130 | -0.0105 |
| brier | 0.0201 | 0.0227 | 0.0213 |
| ece_15 | -0.0178 | -0.0220 | -0.0207 |
| error_auroc | -0.0055 | -0.0073 | -0.0081 |
| macro_f1 | -0.0103 | -0.0130 | -0.0105 |
| nll | 0.0909 | 0.1018 | 0.0955 |
| ood_aupr | -0.0995 | 0.0207 | -0.0208 |
| ood_auroc | -0.0480 | 0.0273 | -0.0089 |

## DAEDL density diagnostics

| seed | global_embedding_rank | constant_dimensions | near_constant_dimensions | class_rank_min | class_rank_max | jitter_min | jitter_max | training_log_density_min | training_log_density_max | nonfinite_fraction | density_ood_auroc | density_ood_aupr |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 3024 | 93 | 35 | 35 | 77 | 80 | 5.43e-08 | 1.16e-07 | -1650.5328 | 594.6596 | 0.0000 | 0.8641 | 0.7772 |
| 3025 | 99 | 29 | 29 | 83 | 95 | 4.63e-08 | 1.02e-07 | -1667.3767 | 569.1288 | 0.0000 | 0.8420 | 0.7322 |
| 3026 | 101 | 27 | 27 | 76 | 87 | 5.08e-08 | 1.05e-07 | -1417.2859 | 592.3713 | 0.0000 | 0.8498 | 0.7416 |

## Consistency

- Clean accuracy and macro F1 consistently favor Softmax across seeds. DAEDL's ECE is consistently lower, while Softmax consistently has lower NLL and Brier score.
- Both methods consistently assign higher mean and median entropy to incorrect than correct predictions. Error-detection AUROC favors Softmax in all three seeds, whereas error-detection AUPR has mixed paired signs.
- DAEDL consistently assigns greater mean and median entropy to SVHN than CIFAR-10.

## Non-replicated findings

- Softmax does not outperform DAEDL on SVHN entropy AUROC in every seed: DAEDL is higher at seed 3025, while Softmax is higher at seeds 3024 and 3026. The report therefore does not claim a stable method-level OOD advantage from this benchmark.
- Error-detection AUPR and SVHN OOD AUPR also change paired direction across seeds.

## DAEDL density stability

- Rank deficiency persists but varies: global rank rises from 93 to 99 and 101, while constant dimensions fall from 35 to 29 and 27. Per-class ranks span 76–95 across the three seeds.
- Density fitting is numerically robust: all non-finite fractions are zero and required adaptive jitter remains of order `1e-8` to `1e-7`.
- The secondary density score improves over DAEDL entropy only at seed 3024; it is worse at seeds 3025 and 3026, so the improvement is not stable.

## Interpretation

The results support a narrow trade-off: DAEDL consistently improves ECE but not clean predictive quality under NLL/Brier, and it does not provide a stable SVHN OOD advantage. No validity failure was found: all six runs use the frozen scientific configuration, within-seed comparisons share data indices, ID/OOD labels and score direction are correct, checkpoint selection uses validation NLL only, and all recomputed metrics match saved values. With only three seeds, these are descriptive consistency findings rather than significance claims.

## Figures and machine-readable sources

- `results/phase3b_three_seed_analysis/three_seed_headline_metrics.png`
- `results/phase3b_three_seed_analysis/seed3024_entropy_distributions.png`
- `results/phase3b_three_seed_analysis/per_seed_metrics.csv`
- `results/phase3b_three_seed_analysis/aggregate_metrics.csv`
- `results/phase3b_three_seed_analysis/paired_differences.csv`
- `results/phase3b_three_seed_analysis/density_diagnostics_by_seed.csv`
- `results/phase3b_three_seed_analysis/consistency_checks.json`

## Recommendation

**Add small corruption experiment.** The three-seed clean/calibration pattern is stable, but SVHN OOD rankings vary by seed. A compact, pre-specified corruption evaluation would test uncertainty under a complementary covariate shift without tuning or rerunning these models.
