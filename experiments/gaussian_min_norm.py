"""Gaussian minimum-norm regression and the shared noise-amplification term.

For y = X beta_star + epsilon and beta_hat = X^+ y, the predictor AGOP is
G = beta_hat beta_hat^T, hence F = ||G||_F = ||beta_hat||^2.  This experiment
records clean population risk and F over a complete sample-size sweep.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def fit_minimum_norm(x: np.ndarray, y: np.ndarray, rcond: float) -> tuple[np.ndarray, np.ndarray]:
    """Return the SVD minimum-norm solution and retained singular values."""
    u, singular_values, vt = np.linalg.svd(x, full_matrices=False)
    inverse = np.where(singular_values > singular_values[0] * rcond, 1.0 / singular_values, 0.0)
    return vt.T @ (inverse * (u.T @ y)), singular_values


def run(args: argparse.Namespace) -> pd.DataFrame:
    output = Path(args.output_dir)
    output.mkdir(parents=True, exist_ok=True)
    sample_sizes = [int(value) for value in args.sample_sizes.split(",")]
    noise_levels = [float(value) for value in args.noise_levels.split(",")]
    beta_star = np.zeros(args.dimension)
    beta_star[0] = 1.0
    rows: list[dict[str, float | int]] = []

    for repeat in range(args.repetitions):
        generator = np.random.default_rng(args.seed + repeat)
        x_max = generator.normal(size=(max(sample_sizes), args.dimension))
        epsilon_max = generator.normal(size=max(sample_sizes))
        for n in sample_sizes:
            x = x_max[:n]
            u, singular_values, vt = np.linalg.svd(x, full_matrices=False)
            inverse = np.where(singular_values > singular_values[0] * args.rcond, 1.0 / singular_values, 0.0)
            projector_signal = vt.T @ (inverse * (u.T @ (x @ beta_star)))
            unit_noise = vt.T @ (inverse * (u.T @ epsilon_max[:n]))
            for sigma in noise_levels:
                beta_hat = projector_signal + sigma * unit_noise
                noise_component = sigma * unit_noise
                agop_norm = float(beta_hat @ beta_hat)
                clean_risk = float(np.sum((beta_hat - beta_star) ** 2))
                noise_amplification = float(sigma**2 * np.sum(inverse**2))
                identity_residual = clean_risk - (agop_norm + 1.0 - 2.0 * beta_hat[0])
                rows.append({
                    "repeat": repeat,
                    "n": n,
                    "p": args.dimension,
                    "sigma": sigma,
                    "rcond": args.rcond,
                    "effective_rank": int(np.sum(inverse > 0)),
                    "smallest_singular_value": float(singular_values[-1]),
                    "F": agop_norm,
                    "clean_risk": clean_risk,
                    "signal_energy": float(projector_signal @ projector_signal),
                    "noise_energy": float(noise_component @ noise_component),
                    "conditional_noise_amplification": noise_amplification,
                    "alignment": float(beta_hat[0]),
                    "identity_residual": float(identity_residual),
                })
    frame = pd.DataFrame(rows)
    frame.to_csv(output / "gaussian_min_norm_runs.csv", index=False)
    make_plot(frame, output, args.dimension)
    max_error = float(frame.identity_residual.abs().max())
    (output / "run_summary.txt").write_text(
        f"repetitions={args.repetitions}\np={args.dimension}\nmax_identity_residual={max_error:.3e}\n",
        encoding="utf-8",
    )
    return frame


def make_plot(frame: pd.DataFrame, output: Path, dimension: int) -> None:
    """Plot medians and 10--90% intervals for the noisy and noiseless controls."""
    figure, axes = plt.subplots(1, 2, figsize=(8.0, 3.0), sharey=False)
    for axis, sigma in zip(axes, (0.0, 0.3)):
        subset = frame[np.isclose(frame.sigma, sigma)]
        for column, color, label, marker in [
            ("clean_risk", "#276b9c", "Clean risk", "o"),
            ("F", "#cf6b25", r"$F=\|G\|_F$", "s"),
        ]:
            grouped = subset.groupby("n")[column]
            median = grouped.median()
            interval = grouped.quantile([0.1, 0.9]).unstack()
            axis.plot(median.index / dimension, median, color=color, marker=marker, ms=3, lw=1.2, label=label)
            axis.fill_between(interval.index / dimension, interval[0.1], interval[0.9], color=color, alpha=0.14)
        axis.axvline(1.0, color="0.55", linestyle=":", lw=1)
        axis.set(xscale="log", yscale="log", xlabel=r"Samples / dimension $n/p$", title=fr"$\sigma={sigma:g}$")
        axis.grid(alpha=0.2)
    axes[0].set_ylabel("Risk and AGOP norm")
    axes[1].legend(frameon=False, fontsize=8)
    figure.tight_layout()
    figure.savefig(output / "gaussian_min_norm.pdf")
    figure.savefig(output / "gaussian_min_norm.png", dpi=240)
    plt.close(figure)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", default="outputs/gaussian")
    parser.add_argument("--dimension", type=int, default=128)
    parser.add_argument("--sample-sizes", default="32,64,96,115,128,141,160,192,256,512")
    parser.add_argument("--noise-levels", default="0,0.1,0.3")
    parser.add_argument("--repetitions", type=int, default=128)
    parser.add_argument("--rcond", type=float, default=1e-12)
    parser.add_argument("--seed", type=int, default=3001)
    args = parser.parse_args()
    run(args)


if __name__ == "__main__":
    main()
