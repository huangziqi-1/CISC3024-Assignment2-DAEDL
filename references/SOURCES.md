# Primary Source Register

Accessed 2026-10-07 unless otherwise stated.

## Assignment

- CISC3024 Pattern Recognition, AI Assignment #2, local course PDF: `../../AIassign2.pdf` relative to this file. The sheet requires a recent probability/Bayes-based computer-vision classification or recognition algorithm, AI-performed search/programming/report writing, experimental settings/results, a learning reflection, and a source-code webpage.

## DAEDL primary sources

- Official PMLR publication page: https://proceedings.mlr.press/v235/yoon24a.html
- Official PMLR PDF: https://raw.githubusercontent.com/mlresearch/v235/main/assets/yoon24a/yoon24a.pdf
- OpenReview: https://openreview.net/forum?id=JtkruFHcRK
- arXiv record: https://arxiv.org/abs/2409.08754
- Authors' official source repository: https://github.com/TaeseongYoon/DAEDL
- Repository license (MIT): https://github.com/TaeseongYoon/DAEDL/blob/master/LICENSE

## Local inspection notes

- Paper metadata: 27 pages; ICML 2024; PMLR 235:57217-57243.
- Paper experiments: MNIST and CIFAR-10 for ID classification; FMNIST/KMNIST and SVHN/CIFAR-100 for OOD; MNIST-C/CIFAR-10-C for shifts; five-run mean and standard deviation.
- Paper CIFAR-10 implementation: VGG-16, dropout 0.5, Adam, batch 64, learning rate `1e-3`, LambdaLR `0.95^epoch`, regularization `5e-2`, early stopping, up to 100 epochs.
- Repository inspected at commit `ad1fa9ef5736e64f7d74416c2b848fee2335ebd2` (commit date 2026-03-20). The code has no dependency manifest or released checkpoints.
- Phase-2 implementation consulted paper Sections 2, 3.2, and 3.3 for the EDL objective, exponential DAEDL concentration parameterization, class-conditional GDA, density normalization, and density-scaled prediction; Appendix B for the training/density/prediction algorithms; and Appendix C.2 for the authors' CIFAR-10 settings.

## Environment documentation

- PyTorch MPS backend: https://docs.pytorch.org/docs/main/notes/mps.html
- PyTorch macOS installation: https://docs.pytorch.org/get-started/locally/

The PyTorch MPS documentation states that MPS provides GPU acceleration on macOS and does not support float64 tensors. That constraint motivates CPU-side GDA density fitting.
