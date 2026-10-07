#!/usr/bin/env python3
"""Render report equations to high-resolution transparent PNG files."""
from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt


EQUATIONS = {
    "equation1_probabilities.png": r"$p_{\mathrm{soft}}(y=k\mid x)=\frac{e^{z_k(x)}}{\sum_j e^{z_j(x)}},\qquad \mathbb{E}[\pi_k\mid x]=\frac{\alpha_k(x)}{S(x)},\quad S(x)=\sum_j\alpha_j(x)$",
    "equation2_training.png": r"$\alpha_k(x)=e^{z_k(x)},\qquad \mathcal{L}=\frac{1}{N}\sum_i\left[\sum_k(y_{ik}-p_{ik})^2+\sum_k\frac{\alpha_{ik}(S_i-\alpha_{ik})}{S_i^2(S_i+1)}+\lambda\,\mathrm{KL}\!\left(\mathrm{Dir}(\tilde{\alpha}_i)\,\|\,\mathrm{Dir}(\mathbf{1})\right)\right]$",
    "equation3_density.png": r"$p(h)=\sum_{c=1}^{K}\omega_c\,\mathcal{N}(h\mid\mu_c,\Sigma_c),\qquad s(x)=\mathrm{clip}_{[0,1]}\!\left(\frac{\log p(h(x))-\ell_{\min}}{\ell_{\max}-\ell_{\min}}\right)$",
    "equation4_inference.png": r"$\alpha_k(x)=\exp\!\left(s(x)z_k(x)\right),\qquad p(y=k\mid x)=\frac{\alpha_k(x)}{\sum_j\alpha_j(x)},\qquad s(x)\rightarrow 0\;\Rightarrow\;p(y=k\mid x)\rightarrow\frac{1}{K}$",
    "equation5_entropy.png": r"$H_{\mathrm{norm}}(p)=-\frac{1}{\log K}\sum_{k=1}^{K}p_k\log p_k$",
}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("output_dir", type=Path)
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    for name, expression in EQUATIONS.items():
        width = 13 if name == "equation2_training.png" else 10
        figure = plt.figure(figsize=(width, 0.7), facecolor="none")
        figure.text(0.5, 0.5, expression, ha="center", va="center", fontsize=16, color="black")
        figure.savefig(args.output_dir / name, dpi=260, transparent=True, bbox_inches="tight", pad_inches=0.06)
        plt.close(figure)


if __name__ == "__main__":
    main()
