# Phase 2 Pilot Summary

All numbers on this page are **PILOT ONLY - NOT FINAL RESULTS**.

## Validated environment

- Python 3.11.5 in project-local `.venv`
- PyTorch 2.14.1; torchvision 0.29.1
- Apple arm64 / macOS 26.4.1
- MPS built, available, and basic tensor test passed through the host shell
- CUDA unavailable
- MPS training; CPU float64 GDA
- DAEDL post-training feature extraction/evaluation on CPU because PyTorch 2.14.1 MPS returned NaN embeddings for the spectral-normalized checkpoint in evaluation mode

## Pilot protocol

- Seed and split seed: 3024
- Fixed stratified subsets shared by both models: 5,000 train, 1,000 validation, 1,000 test
- Two epochs; batch size 64; AdamW; initial learning rate `1e-3`; weight decay `5e-4`
- Compact CIFAR ResNet-18; base width 32; 128-D embedding
- DAEDL regularization coefficient `0.05`

## Pilot outcomes

| Quantity | Softmax | DAEDL |
|---|---:|---:|
| Final train loss | 1.6823 | 0.9160 (different objective) |
| Final train accuracy | 36.72% | 34.64% |
| Final validation loss | 1.7330 | 0.9190 (different objective) |
| Final validation accuracy | 37.30% | 35.40% |
| Final validation macro F1 | 0.3271 | 0.3196 |
| Pilot test accuracy | 34.20% | 33.10% |
| Pilot test macro F1 | 0.2950 | 0.3010 |
| Training runtime | 9.38 s | 10.20 s |
| Density-fit runtime | n/a | 12.92 s |
| Evaluation runtime | 0.16 s | 2.26 s |
| Total measured pipeline | 9.65 s | 25.56 s |

The losses are not directly comparable because Softmax uses cross-entropy while DAEDL uses expected squared error, predictive variance, and KL regularization.

Final artifact directories:

- `results/pilot_softmax_seed3024_20261007T210651+0800_db4dad/`
- `results/pilot_daedl_seed3024_20261007T210705+0800_2a87e3/`

Each contains configuration, environment, split indices, history, checkpoint, predictions, metrics, logs, dependency freeze, and a pilot-labelled reliability diagram. DAEDL additionally contains its GDA state and density summary.

## Density result

GDA fitted successfully for all ten classes on 128-D CPU float64 embeddings. Two-epoch embeddings had 54 globally constant ReLU dimensions and class covariance ranks of approximately 62-69, so diagonal loading was necessary. Adaptive jitter was scaled to mean covariance variance; selected values were approximately `7.8e-8` to `1.68e-7`. This is documented as a numerical assignment adaptation.

## Validation result

- 14/14 unit tests passed.
- Model splits were byte-identical across the final Softmax and DAEDL runs.
- No train/validation index overlap.
- All saved numeric predictions finite.
- Maximum probability-sum error below `1.9e-7`.
- DAEDL concentrations positive; strengths finite; density scores within `[0, 1]`.
- Checkpoint save/load digest matched for both models.
- OOD direction synthetic test gives perfect AUROC/AUPR when OOD-positive samples receive higher scores.

## Phase 3 recommendation

- Models: Softmax and DAEDL; defer conventional EDL unless final runtime is comfortably below budget.
- Full split: 45,000 train / 5,000 validation / 10,000 test.
- Batch size: 64.
- Optimizer: AdamW, learning rate `1e-3`, weight decay `5e-4`.
- Epoch ceiling: 30.
- Scheduler: cosine decay across 30 epochs, with a nonzero minimum learning rate of `1e-5`.
- Early stopping: validation loss, patience 7, minimum improvement `1e-4`, restore best checkpoint; do not stop before epoch 12.
- Seeds: begin with one full-run seed; if stable and time permits, run seeds 3024, 3025, and 3026.
- Experiments: clean classification/calibration, correct-vs-incorrect uncertainty, three controlled corruptions at three levels, SVHN OOD, and deterministic qualitative examples.
- Device policy: MPS training; retest MPS evaluation after any PyTorch change, otherwise keep documented CPU post-training DAEDL inference/GDA.

Projected training burden from the subset pilot is roughly 15-25 minutes per 30-epoch model after allowing for full-data scaling and MacBook Air thermal throttling. DAEDL density fitting on all 45,000 embeddings will add CPU time and should be timed separately. This estimate is provisional, not a measured final runtime.

