"""Full sparse-reconstruction double-descent sweep with input-AGOP norm."""

from __future__ import annotations

import argparse
import random
from dataclasses import dataclass
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from torch import nn


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def sparse_examples(count: int, dimension: int, active_probability: float, seed: int, device: torch.device) -> torch.Tensor:
    generator = torch.Generator(device=device).manual_seed(seed)
    mask = torch.rand((count, dimension), generator=generator, device=device) < active_probability
    values = torch.rand((count, dimension), generator=generator, device=device)
    x = values * mask
    return x / x.norm(dim=1, keepdim=True).clamp_min(1e-12)


class TiedReLUAutoencoder(nn.Module):
    def __init__(self, dimension: int, bottleneck: int) -> None:
        super().__init__()
        self.weight = nn.Parameter(torch.empty(bottleneck, dimension))
        nn.init.xavier_normal_(self.weight)
        self.bias = nn.Parameter(torch.zeros(dimension))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return torch.relu((x @ self.weight.T) @ self.weight + self.bias)


@torch.no_grad()
def input_agop_norm(model: TiedReLUAutoencoder, x: torch.Tensor) -> float:
    """Compute ||E[J J^T]||_F exactly from activation frequencies."""
    w = model.weight.detach()
    active = torch.zeros(x.shape[1], device=x.device)
    for batch in x.split(2048):
        logits = (batch @ w.T) @ w + model.bias
        active += (logits > 0).float().sum(0)
    probabilities = active / len(x)
    gram = w.T @ w
    agop = (gram * probabilities.unsqueeze(0)) @ gram
    return float(torch.linalg.matrix_norm(agop, ord="fro").cpu())


@dataclass(frozen=True)
class Protocol:
    dimension: int = 1000
    bottleneck: int = 2
    active_probability: float = 0.01
    test_examples: int = 5000
    steps: int = 3000
    batch_size: int = 2048
    lr: float = 5e-3
    weight_decay: float = 1e-2


def train_one(x_train: torch.Tensor, x_test: torch.Tensor, model_seed: int, protocol: Protocol) -> tuple[float, float, float]:
    seed_everything(model_seed)
    model = TiedReLUAutoencoder(protocol.dimension, protocol.bottleneck).to(x_train.device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=protocol.lr, weight_decay=protocol.weight_decay)
    warmup = max(1, protocol.steps // 4)
    warmup_schedule = torch.optim.lr_scheduler.LinearLR(optimizer, start_factor=0.1, total_iters=warmup)
    cosine_schedule = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=protocol.steps - warmup)
    for step in range(protocol.steps):
        indices = torch.randint(len(x_train), (min(protocol.batch_size, len(x_train)),), device=x_train.device)
        x = x_train[indices]
        optimizer.zero_grad(set_to_none=True)
        loss = (model(x).sub(x).square().sum(dim=1)).mean()
        loss.backward()
        optimizer.step()
        (warmup_schedule if step < warmup else cosine_schedule).step()
    with torch.no_grad():
        train_loss = float((model(x_train).sub(x_train).square().sum(dim=1)).mean().cpu())
        test_loss = float((model(x_test).sub(x_test).square().sum(dim=1)).mean().cpu())
    return train_loss, test_loss, input_agop_norm(model, x_test)


def run(args: argparse.Namespace) -> pd.DataFrame:
    protocol = Protocol(steps=args.steps)
    device = torch.device(args.device if args.device != "auto" else ("cuda" if torch.cuda.is_available() else "cpu"))
    sizes = [3, 5, 8, 10, 15, 30, 50, 100, 200, 500] + [int(round(v)) for v in np.logspace(3, np.log10(20000), 10)] + [30000, 40000]
    output = Path(args.output_dir)
    output.mkdir(parents=True, exist_ok=True)
    rows: list[dict[str, float | int]] = []
    for n in sizes:
        x_train = sparse_examples(n, protocol.dimension, protocol.active_probability, 12345 + n, device)
        x_test = sparse_examples(protocol.test_examples, protocol.dimension, protocol.active_probability, 54321 + n, device)
        for seed in range(args.seeds):
            train_loss, test_loss, norm = train_one(x_train, x_test, seed, protocol)
            rows.append({"data_size": n, "model_seed": seed, "train_loss": train_loss, "test_loss": test_loss, "agop_frobenius_norm": norm})
        print(f"completed n={n}", flush=True)
    frame = pd.DataFrame(rows)
    frame.to_csv(output / "sparse_reconstruction_runs.csv", index=False)
    summary = frame.groupby("data_size", as_index=False).agg(["mean", "sem"])
    summary.columns = ["_".join(filter(None, column)).rstrip("_") for column in summary.columns.to_flat_index()]
    summary.to_csv(output / "sparse_reconstruction_summary.csv", index=False)
    make_plot(summary, output)
    return frame


def make_plot(summary: pd.DataFrame, output: Path) -> None:
    figure, loss_axis = plt.subplots(figsize=(5.8, 3.3))
    norm_axis = loss_axis.twinx()
    loss_axis.errorbar(summary.data_size, summary.test_loss_mean, yerr=summary.test_loss_sem, color="#276b9c", marker="o", ms=3, lw=1.2, label="Reconstruction loss")
    norm_axis.errorbar(summary.data_size, summary.agop_frobenius_norm_mean, yerr=summary.agop_frobenius_norm_sem, color="#cf6b25", marker="s", ms=3, lw=1.2, linestyle="--", label=r"$\|G\|_F$")
    loss_axis.set(xscale="log", xlabel="Training examples $n$", ylabel="Test reconstruction loss")
    norm_axis.set(yscale="log", ylabel=r"$F=\|G\|_F$")
    loss_axis.grid(alpha=0.2)
    loss_axis.legend(loc="upper left", frameon=False)
    norm_axis.legend(loc="upper right", frameon=False)
    figure.tight_layout()
    figure.savefig(output / "sparse_double_descent.pdf")
    figure.savefig(output / "sparse_double_descent.png", dpi=240)
    plt.close(figure)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", default="outputs/sparse")
    parser.add_argument("--device", default="auto")
    parser.add_argument("--steps", type=int, default=3000)
    parser.add_argument("--seeds", type=int, default=5)
    run(parser.parse_args())


if __name__ == "__main__":
    main()
