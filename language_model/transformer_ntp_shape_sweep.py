#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
transformer_ntp_shape_sweep.py
==============================

Goal
----
Chinchilla-style shape sweep for decoder-only Transformers on byte-level
next-token prediction.

For multiple fixed parameter budgets N, we sweep the depth-to-width ratio
(aspect ratio α = depth / d_model) and record:
  • test cross-entropy (nats / byte)  — the primary performance metric
  • AOFE_ratio = off-diagonal AGOP energy / total AGOP energy  — the coupling metric

The key prediction (AOFE hypothesis):
  1. At fixed N, lower-loss shapes have lower AOFE and higher AOFE_ratio.
  2. Each N has an intermediate optimal aspect ratio α* rather than a monotonic
     preference for either depth or width.
  3. The interaction-efficient α* range is stable enough to inform larger runs.

Task: Byte-level next-token prediction (NTP)
--------------------------------------------
• Tokenisation: raw bytes, vocab_size = 256
• Dataset: a locally supplied pre-tokenized byte corpus
• Training budget: D = data_ratio × N bytes  (default data_ratio = 60,
  roughly equivalent to Chinchilla's D = 20N BPE tokens at ~3 bytes/token)
• Loss: per-token cross-entropy in nats
• Eval: sequential non-overlapping windows on held-out test split

Input-space AGOP (matrix-free)
------------------------------
  The differentiable input is the per-position one-hot/simplex token vector
  q ∈ R^{T×V}, not the post-embedding activation.  With e = qE + p, the code
  maps gradients and tangents through E by the exact chain rule and estimates

      G = E_data[J_q J_q^T] ∈ R^{(T V)×(T V)}.

  G is 65,536×65,536 for T=V=256, so Hutchinson estimators recover ||G||_F²
  and ||diag(G)||² without projecting or materializing G.  The previous
  embedding-to-output projected metric remains available as ``--agop_mode
  legacy`` only for paired auditing.

Usage
-----
  # ~20M params, depths 6..14 only, low-VRAM AGOP, one-GPU workers (see run_20M_parallel_6to14.sh):
  python language_model/transformer_ntp_shape_sweep.py \\
      --param_groups 20000000 --depth_list 6,8,10,12,14 \\
      --agop_low_vram --d_model_max 2048 \\
      --only_depth 12 --result_shard d12 --device cuda:0

  python language_model/transformer_ntp_shape_sweep.py \\
      --data_dir ./data \\
      --param_groups 300000,1000000,3000000 \\
      --depth_list 1,2,3,4,5,6,8,10,12,16,20,24 \\
      --out_dir ./outputs/language_model/transformer_ntp_shape_sweep \\
      --device cuda

  # Regenerate all plots from an existing CSV (after appending new N):
  python language_model/transformer_ntp_shape_sweep.py \\
      --plot_only --out_dir ./outputs/language_model/transformer_ntp_shape_sweep
"""

from __future__ import annotations

import gc
import os
import csv
import math
import time
import random
import argparse
import dataclasses
from contextlib import nullcontext
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

import matplotlib
if os.environ.get("DISPLAY", "") == "":
    matplotlib.use("Agg")
import matplotlib.pyplot as plt

try:
    from .input_agop import estimate_input_agop_metrics_ntp
except ImportError:  # Direct ``python language_model/...py`` execution.
    from input_agop import estimate_input_agop_metrics_ntp


# ─────────────────────────────────────────────────────────────────────────────
# Constants
# ─────────────────────────────────────────────────────────────────────────────

VOCAB_SIZE = 256   # byte-level vocabulary
SEQ_LEN    = 256   # tokens (bytes) per sequence

# Output dimension for AGOP computation.
# We project vocab_size=256 logits down to AGOP_OUT=64 dims via a fixed
# random matrix before computing AGOP, for two reasons:
#   1. A 256×256 AGOP always has AOFE_ratio ≈ 1 - 1/256 ≈ 0.996 by construction
#      (255× more off-diagonal entries than diagonal) — no meaningful signal.
#   2. Matching the teacher-student experiment's 64×64 matrix makes metrics
#      directly comparable.
# The projection matrix is fixed (seed=42) across all shapes and N values.
AGOP_OUT = 64


# ─────────────────────────────────────────────────────────────────────────────
# Reproducibility
# ─────────────────────────────────────────────────────────────────────────────

def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def count_params(model: nn.Module) -> int:
    return int(sum(p.numel() for p in model.parameters()))


def symmetrize(M: torch.Tensor) -> torch.Tensor:
    return 0.5 * (M + M.T)


# ─────────────────────────────────────────────────────────────────────────────
# AOFE metric
# ─────────────────────────────────────────────────────────────────────────────

def agop_offdiag_metrics(agop: torch.Tensor) -> Tuple[float, float]:
    """
    Returns (AOFE, AOFE_ratio):
      AOFE       = ||AGOP||_F^2 - ||diag(AGOP)||_2^2
      AOFE_ratio = AOFE / ||AGOP||_F^2
    """
    agop = agop.float()
    fro2  = float((agop * agop).sum().item()) + 1e-12
    diag2 = float((torch.diag(agop) ** 2).sum().item())
    return max(fro2 - diag2, 0.0), max(fro2 - diag2, 0.0) / fro2


# ─────────────────────────────────────────────────────────────────────────────
# Statistics helpers
# ─────────────────────────────────────────────────────────────────────────────

def pearson_corr(x: np.ndarray, y: np.ndarray) -> float:
    x = np.asarray(x, dtype=np.float64)
    y = np.asarray(y, dtype=np.float64)
    if x.size < 2:
        return float("nan")
    x = x - x.mean()
    y = y - y.mean()
    return float(np.dot(x, y) / (np.linalg.norm(x) * np.linalg.norm(y) + 1e-12))


def spearman_corr(x: np.ndarray, y: np.ndarray) -> float:
    def _rank(a: np.ndarray) -> np.ndarray:
        a = np.asarray(a, dtype=np.float64)
        order = np.argsort(a)
        ranks = np.empty_like(order, dtype=np.float64)
        ranks[order] = np.arange(1, len(a) + 1, dtype=np.float64)
        return ranks
    return pearson_corr(_rank(x), _rank(y))


# ─────────────────────────────────────────────────────────────────────────────
# Data loading (local byte corpus)
# ─────────────────────────────────────────────────────────────────────────────

def load_corpus(data_dir: str) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Load train/validation/test splits as byte arrays.

    The scan intentionally accepts only locally supplied files named
    ``train.bin``, ``validation.bin``, and ``test.bin``.  Dataset acquisition
    is separate from the anonymous code release.

    Returns (train_bytes, val_bytes, test_bytes) as uint8 numpy arrays.
    """
    data_path = Path(data_dir)
    paths = {split: data_path / f"{split}.bin" for split in ("train", "validation", "test")}
    missing = [split for split, path in paths.items() if not path.exists()]
    if missing:
        raise FileNotFoundError(
            f"Local byte corpus under {data_path} is incomplete; missing {missing}. "
            "Expected train.bin, validation.bin, and test.bin."
        )
    splits = {
        split: np.frombuffer(path.read_bytes(), dtype=np.uint8).copy()
        for split, path in paths.items()
    }
    for split, values in splits.items():
        print(f"  [{split:10s}] loaded {len(values)/1e6:.1f}M bytes from {paths[split]}")
    return splits["train"], splits["validation"], splits["test"]


class RandomWindowDataset(torch.utils.data.Dataset):
    """Randomly sampled training windows; overlap is possible by construction."""

    def __init__(
        self,
        data: np.ndarray,
        seq_len: int,
        n_windows: int,
        seed: int = 0,
    ):
        rng = np.random.default_rng(seed)
        max_start = len(data) - seq_len - 1
        if max_start <= 0:
            raise ValueError(
                f"Corpus too short ({len(data)} bytes) for seq_len={seq_len}"
            )
        self.starts = rng.integers(0, max_start, size=n_windows)
        self.data   = data
        self.T      = seq_len

    def __len__(self) -> int:
        return len(self.starts)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        s = int(self.starts[idx])
        x = torch.from_numpy(self.data[s     : s + self.T    ].astype(np.int64))
        y = torch.from_numpy(self.data[s + 1 : s + self.T + 1].astype(np.int64))
        return x, y


class SequentialWindowDataset(torch.utils.data.Dataset):
    """Non-overlapping sequential windows for evaluation (deterministic)."""

    def __init__(self, data: np.ndarray, seq_len: int):
        self.data = data
        self.T    = seq_len
        self.n    = (len(data) - 1) // seq_len

    def __len__(self) -> int:
        return self.n

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        s = idx * self.T
        x = torch.from_numpy(self.data[s     : s + self.T    ].astype(np.int64))
        y = torch.from_numpy(self.data[s + 1 : s + self.T + 1].astype(np.int64))
        return x, y


class PrefixWindowDataset(torch.utils.data.Dataset):
    """A fixed, non-overlapping prefix of the corpus for Chinchilla data accounting."""

    def __init__(self, data: np.ndarray, seq_len: int, n_windows: int):
        available = (len(data) - 1) // seq_len
        if n_windows > available:
            raise ValueError(
                f"Corpus provides {available:,} non-overlapping windows but "
                f"{n_windows:,} were requested"
            )
        self.data = data
        self.T = seq_len
        self.n = n_windows

    def __len__(self) -> int:
        return self.n

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        s = idx * self.T
        x = torch.from_numpy(self.data[s : s + self.T].astype(np.int64))
        y = torch.from_numpy(self.data[s + 1 : s + self.T + 1].astype(np.int64))
        return x, y


# ─────────────────────────────────────────────────────────────────────────────
# Transformer model  (decoder-only, byte-level NTP)
# ─────────────────────────────────────────────────────────────────────────────

class CausalSelfAttention(nn.Module):
    def __init__(self, d_model: int, n_heads: int, dropout: float = 0.0):
        super().__init__()
        assert d_model % n_heads == 0
        self.n_heads = n_heads
        self.d_head  = d_model // n_heads
        self.dropout = dropout
        self.use_memory_efficient_attention = True
        self.qkv     = nn.Linear(d_model, 3 * d_model, bias=False)
        self.proj    = nn.Linear(d_model, d_model,     bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, T, d = x.shape
        q, k, v = self.qkv(x).chunk(3, dim=-1)

        def split(t: torch.Tensor) -> torch.Tensor:
            return t.view(B, T, self.n_heads, self.d_head).transpose(1, 2)

        q, k, v = split(q), split(k), split(v)
        # PyTorch dispatches to FlashAttention/memory-efficient kernels on
        # compatible CUDA hardware.  This is algebraically the same causal
        # attention as the explicit score matrix below, but avoids materializing
        # [B, H, T, T] scores for the 50M-parameter sweep.
        # CPU's flash path lacks the higher-order/JVP derivative needed by the
        # exact AGOP unit tests.  CPU therefore keeps the explicit reference
        # form, while CUDA training receives the memory-efficient kernel.
        if (
            self.use_memory_efficient_attention
            and x.device.type == "cuda"
            and hasattr(F, "scaled_dot_product_attention")
        ):
            out = F.scaled_dot_product_attention(
                q, k, v,
                dropout_p=(self.dropout if self.training else 0.0),
                is_causal=True,
            )
        else:  # pragma: no cover - compatibility fallback for old PyTorch.
            scores = (q @ k.transpose(-2, -1)) / math.sqrt(self.d_head)
            mask = torch.tril(torch.ones(T, T, device=x.device, dtype=torch.bool))
            scores = scores.masked_fill(~mask, float("-inf"))
            attn = F.softmax(scores, dim=-1)
            if self.dropout > 0 and self.training:
                attn = F.dropout(attn, p=self.dropout)
            out = attn @ v
        out = out.transpose(1, 2).contiguous().view(B, T, d)
        return self.proj(out)


class MLPBlock(nn.Module):
    def __init__(self, d_model: int, d_ff: int, dropout: float = 0.0):
        super().__init__()
        self.fc1     = nn.Linear(d_model, d_ff, bias=False)
        self.fc2     = nn.Linear(d_ff, d_model, bias=False)
        self.dropout = dropout

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = F.gelu(self.fc1(x))
        if self.dropout > 0 and self.training:
            x = F.dropout(x, p=self.dropout)
        return self.fc2(x)


class DecoderBlock(nn.Module):
    def __init__(self, d_model: int, n_heads: int, d_ff: int, dropout: float = 0.0):
        super().__init__()
        self.ln1  = nn.LayerNorm(d_model)
        self.attn = CausalSelfAttention(d_model, n_heads, dropout)
        self.ln2  = nn.LayerNorm(d_model)
        self.mlp  = MLPBlock(d_model, d_ff, dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.attn(self.ln1(x))
        x = x + self.mlp(self.ln2(x))
        return x


class TinyGPT_NTP(nn.Module):
    """
    Decoder-only Transformer for byte-level next-token prediction.

    forward(x: [B, T] int64) → logits [B, T, VOCAB_SIZE]

    The primary AGOP is computed with respect to the one-hot/simplex token
    input.  ``forward_from_embeddings`` is retained so the exact chain-rule
    maps through ``tok_emb.weight`` can be evaluated without materializing
    one-hot tensors during training.
    """

    def __init__(
        self,
        *,
        depth:      int,
        d_model:    int,
        n_heads:    int,
        d_ff:       int,
        seq_len:    int  = SEQ_LEN,
        vocab_size: int  = VOCAB_SIZE,
        dropout:    float = 0.0,
        pad_params: int  = 0,
    ):
        super().__init__()
        self.d_model    = d_model
        self.vocab_size = vocab_size
        self.seq_len    = seq_len
        self.depth      = depth

        self.tok_emb = nn.Embedding(vocab_size, d_model)
        self.pos_emb = nn.Embedding(seq_len,    d_model)
        self.drop    = nn.Dropout(dropout) if dropout > 0 else nn.Identity()
        self.blocks  = nn.ModuleList([
            DecoderBlock(d_model, n_heads, d_ff, dropout) for _ in range(depth)
        ])
        self.ln_f    = nn.LayerNorm(d_model)
        self.lm_head = nn.Linear(d_model, vocab_size, bias=False)

        self._pad = None
        if pad_params > 0:
            self._pad = nn.Parameter(torch.zeros(pad_params), requires_grad=True)

        self._init_weights()

    def _init_weights(self) -> None:
        for m in self.modules():
            if isinstance(m, (nn.Linear, nn.Embedding)):
                nn.init.normal_(m.weight, std=0.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.zeros_(m.bias)
        # Scale residual projections by 1/sqrt(depth) (GPT-2 style)
        for blk in self.blocks:
            nn.init.normal_(blk.attn.proj.weight, std=0.02 / math.sqrt(2 * self.depth))
            nn.init.normal_(blk.mlp.fc2.weight,   std=0.02 / math.sqrt(2 * self.depth))

    def forward_from_embeddings(self, e: torch.Tensor) -> torch.Tensor:
        """e: [B, T, d_model]  →  logits [B, T, vocab_size]"""
        x = self.drop(e)
        for blk in self.blocks:
            x = blk(x)
        return self.lm_head(self.ln_f(x))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """x: [B, T] int64  →  logits [B, T, vocab_size]"""
        B, T = x.shape
        pos  = torch.arange(T, device=x.device).unsqueeze(0).expand(B, T)
        e    = self.tok_emb(x) + self.pos_emb(pos)
        return self.forward_from_embeddings(e)


# ─────────────────────────────────────────────────────────────────────────────
# Legacy AGOP estimation (embedding-to-output projected matrix)
# ─────────────────────────────────────────────────────────────────────────────

def estimate_embedding_output_agop_ntp(
    model:        TinyGPT_NTP,
    data:         np.ndarray,
    *,
    proj_samples: int = 64,
    batch_size:   int = 128,
    n_batches:    int = 4,
    seed:         int = 1,
    agop_out:     int = AGOP_OUT,
    device:       torch.device,
    agop_microbatch: Optional[int] = None,
) -> torch.Tensor:
    """
    Legacy estimator retained only for paired comparison with committed results.

    Estimate E[J_P J_P^T] in projected *output* space via random JVPs.  This is
    not the input-space AGOP defined in Eq. (2) of the paper.

    J_P = P · J,  where J = d(logits[-1] ∈ R^{VOCAB}) / d(e_flat ∈ R^{T×d_model})
    and P ∈ R^{agop_out × VOCAB} is a fixed random projection matrix (seed=42).

    Projecting to agop_out=64 before computing AGOP is necessary because a
    VOCAB×VOCAB (256×256) AGOP always has AOFE_ratio ≈ 1 - 1/256 ≈ 0.996 by
    sheer count of off-diagonal entries, giving no discriminative signal.
    The 64×64 projection preserves structure (Johnson-Lindenstrauss) and matches
    the teacher-student experiment's dimensionality for direct comparison.

    Using E_u[Ju (Ju)^T] = J E[uu^T] J^T = J J^T  (u ~ N(0, I)),
    the outer product of JVP outputs is an unbiased estimator of AGOP.

    Memory: JVP is applied on the full (B, T, d_model) embedding tensor. For large
    B or d_model, set ``agop_microbatch`` to run JVP on slices of the batch
    dimension (mathematically equivalent to one JVP; accumulates (Ju^T Ju) / B).
    """
    model.eval()
    rng = np.random.default_rng(seed)
    T   = model.seq_len
    mb  = int(agop_microbatch) if agop_microbatch is not None and agop_microbatch > 0 else batch_size
    mb  = max(1, min(mb, batch_size))

    # Fixed projection matrix: same for all shapes and N values
    torch.manual_seed(42)
    proj = torch.randn(agop_out, model.vocab_size, device=device) / math.sqrt(agop_out)

    agop  = torch.zeros(agop_out, agop_out, device=device)
    count = 0

    for _ in range(n_batches):
        max_start = len(data) - T - 1
        if max_start <= 0:
            break
        starts = rng.integers(0, max_start, size=batch_size)
        x_np   = np.stack([data[s : s + T] for s in starts]).astype(np.int64)
        x      = torch.from_numpy(x_np).to(device)

        with torch.no_grad():
            pos = torch.arange(T, device=device).unsqueeze(0).expand(batch_size, T)
            e   = (model.tok_emb(x) + model.pos_emb(pos)).detach()
        bsz = int(e.shape[0])

        def fwd(e_in: torch.Tensor) -> torch.Tensor:
            logits = model.forward_from_embeddings(e_in)  # [B, T, vocab]
            return logits[:, -1, :] @ proj.T              # [B, agop_out]

        for _ in range(proj_samples):
            u = torch.randn_like(e)
            # Micro-batch the batch dimension to cap peak JVP memory.
            acc = torch.zeros(agop_out, agop_out, device=device)
            s0  = 0
            while s0 < bsz:
                s1 = min(s0 + mb, bsz)
                e_s, u_s = e[s0:s1], u[s0:s1]
                _, Ju = torch.autograd.functional.jvp(
                    fwd, (e_s,), (u_s,), create_graph=False
                )
                Ju = torch.nan_to_num(Ju.float(), nan=0.0, posinf=0.0, neginf=0.0)
                acc = acc + (Ju.T @ Ju)
                s0 = s1
                if device.type == "cuda":
                    torch.cuda.empty_cache()
            agop = agop + acc / float(bsz)
            if device.type == "cuda":
                torch.cuda.empty_cache()

        count += proj_samples
        if device.type == "cuda":
            torch.cuda.empty_cache()
        gc.collect()

    if count == 0:
        return agop
    return symmetrize(agop / float(count)).detach()


# ─────────────────────────────────────────────────────────────────────────────
# Training config
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class TrainCfg:
    lr:                float = 3e-4
    weight_decay:      float = 1e-2
    # data_ratio = D / N  (in bytes).  Default 60 ≈ Chinchilla's 20 BPE-tokens
    # (since ~3 bytes/BPE token for English text).
    data_ratio:        float = 60.0
    warmup_steps:      int   = 300
    batch_size:        int   = 64
    # BF16 plus PyTorch scaled-dot-product attention is used only in the large
    # sweep.  Keeping the default off preserves historical small-run behavior.
    amp_bf16:           bool  = False
    grad_clip:         float = 1.0
    eval_every:        int   = 200
    # 0 means evaluate every deterministic held-out window.  Large-corpus runs
    # use a fixed prefix so validation cost does not scale with training corpus.
    eval_max_batches:   int   = 0
    # Full CE on a Chinchilla-size sampled training set can dominate wall time
    # without adding information to the validation-based fit gate.  ``0``
    # preserves the historical full pass; positive values use a fixed prefix.
    final_train_eval_batches: int = 0
    evaluate_test_during_training: bool = True
    # head_dim controls n_heads = d_model // head_dim
    head_dim:          int   = 4
    dropout:           float = 0.0
    max_padding_ratio: float = 0.20
    max_train_factor:  float = 1.5
    fit_patience:      int   = 10
    agop_mode:         str   = "input"
    agop_split:        str   = "validation"
    agop_batch:        int   = 32
    agop_proj_samples: int   = 64
    agop_input_probes: int   = 16
    agop_output_probes: int  = 32
    agop_n_batches:    int   = 4
    agop_center_logits: bool = True
    agop_normalize_logits: bool = True
    agop_seed:         int   = 42
    # 0 = disabled (use full agop_batch). >0 caps AGOP micro-batches for VRAM.
    agop_microbatch:   int   = 0
    # Data windows and loader order are fixed across shapes and model seeds.
    data_seed:         int   = 314159
    # Frontier protocol controls.  The original sweep padded every model to its
    # nominal budget, leaving an unused parameter tensor.  That changes the
    # actual tokens/active-parameter ratio across shapes.  In strict mode, we
    # instead tune d_ff (while keeping it close to 4*d_model) so active counts
    # match the nominal budget, and train for D = data_ratio * active_N bytes.
    strict_active_match: bool = False
    active_match_tolerance: float = 0.005
    ffn_multiple:       int   = 4
    ffn_ratio_tolerance: float = 0.15
    data_by_active_params: bool = False
    # ``unique_prefix`` consumes each base-budget training target exactly once.
    # Repeating it after the Chinchilla base pass is explicit in
    # ``training_bytes_seen`` rather than being hidden by random resampling.
    train_window_mode:  str   = "random"
    # A fit certificate is deliberately separate from early stopping.  At the
    # end of the Chinchilla-counted schedule, a short low-LR restart challenge
    # tests whether validation loss can still materially improve.  A model that
    # fails this gate is kept in the scan table but excluded from the frontier.
    require_fit_certificate: bool = False
    fit_audit_fraction: float = 0.10
    fit_audit_lr:       float = 3e-5
    fit_max_audits:     int   = 3
    fit_rel_improve_tol: float = 5e-4
    fit_tail_rel_tol:   float = 5e-4
    fit_tail_evals:     int   = 4
    save_checkpoints:  bool  = False
    seed:              int   = 0
    d_model_min:       int   = 16
    d_model_max:       int   = 1024


def cosine_lr(step: int, base_lr: float, warmup: int, total: int) -> float:
    if step < warmup:
        return base_lr * (step + 1) / max(1, warmup)
    t = (step - warmup) / max(1, total - warmup)
    return base_lr * 0.5 * (1.0 + math.cos(math.pi * min(1.0, t)))


def autocast_context(device: torch.device, enabled: bool):
    """Return CUDA BF16 autocast only when explicitly requested."""
    if enabled and device.type == "cuda":
        return torch.autocast(device_type="cuda", dtype=torch.bfloat16)
    return nullcontext()


@torch.no_grad()
def evaluate_ntp(
    model:          TinyGPT_NTP,
    loader:         torch.utils.data.DataLoader,
    device:         torch.device,
    max_batches:    Optional[int] = None,
    log_every:      Optional[int] = None,
    log_label:      str = "eval",
    amp_bf16:       bool = False,
) -> float:
    """Per-token cross-entropy in nats."""
    model.eval()
    total_loss, total_n = 0.0, 0
    for i, (x, y) in enumerate(loader):
        if max_batches is not None and i >= max_batches:
            break
        if (
            log_every is not None
            and log_every > 0
            and i > 0
            and i % log_every == 0
        ):
            cur = total_loss / max(1, total_n)
            print(
                f"    [{log_label}] batch {i:,}  running_ce={cur:.4f} nats",
                flush=True,
            )
        x, y = x.to(device), y.to(device)
        with autocast_context(device, amp_bf16):
            logits = model(x)                        # [B, T, vocab]
            loss = F.cross_entropy(
                logits.view(-1, model.vocab_size), y.view(-1), reduction="sum"
            )
        total_loss += float(loss.item())
        total_n    += int(y.numel())
    return total_loss / max(1, total_n)


def train_one_model(
    model:        TinyGPT_NTP,
    train_loader: torch.utils.data.DataLoader,
    val_loader:   torch.utils.data.DataLoader,
    test_loader:  torch.utils.data.DataLoader,
    base_steps:   int,
    cfg:          TrainCfg,
    device:       torch.device,
) -> Tuple[Dict[str, float], List[Dict[str, float]]]:
    """
    Train via AdamW + cosine-LR with early stopping on val cross-entropy.
    Returns (metrics_dict, history_list).  When requested, the metrics include
    a convergence certificate based on a low-learning-rate restart challenge.
    """
    model.to(device).train()
    opt       = torch.optim.AdamW(
        model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay
    )
    loader_it = iter(train_loader)

    max_steps  = max(base_steps, int(math.ceil(cfg.max_train_factor * base_steps)))
    best_val   = float("inf")
    best_state: Optional[Dict] = None
    stale      = 0
    history: List[Dict[str, float]] = []
    t0 = time.time()
    user_interrupt = False
    eval_max_batches = None if cfg.eval_max_batches <= 0 else cfg.eval_max_batches

    for step in range(max_steps):
        try:
            try:
                x, y = next(loader_it)
            except StopIteration:
                loader_it = iter(train_loader)
                x, y = next(loader_it)

            x, y = x.to(device), y.to(device)
            lr   = cosine_lr(step, cfg.lr, cfg.warmup_steps, max_steps)
            for pg in opt.param_groups:
                pg["lr"] = lr

            with autocast_context(device, cfg.amp_bf16):
                logits = model(x)                        # [B, T, vocab]
                loss = F.cross_entropy(
                    logits.view(-1, model.vocab_size), y.view(-1)
                )
            opt.zero_grad(set_to_none=True)
            loss.backward()
            if cfg.grad_clip > 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.grad_clip)
            opt.step()

            if (step + 1) % cfg.eval_every == 0 or (step + 1) == max_steps:
                tr_ce = evaluate_ntp(
                    model, train_loader, device, max_batches=20,
                    amp_bf16=cfg.amp_bf16,
                )
                val_ce = evaluate_ntp(
                    model, val_loader, device, max_batches=eval_max_batches,
                    amp_bf16=cfg.amp_bf16,
                )
                te_ce  = (
                    evaluate_ntp(
                        model, test_loader, device, max_batches=eval_max_batches,
                        amp_bf16=cfg.amp_bf16,
                    )
                    if cfg.evaluate_test_during_training
                    else float("nan")
                )
                history.append({
                    "step": step + 1, "lr": lr,
                    "train_ce": tr_ce, "val_ce": val_ce, "test_ce": te_ce,
                    "phase": "main",
                })
                elapsed = time.time() - t0
                print(
                    f"    step {step+1:6d}/{max_steps}  lr={lr:.2e}  "
                    f"train={tr_ce:.4f}  val={val_ce:.4f}"
                    + (f"  test={te_ce:.4f}" if cfg.evaluate_test_during_training else "")
                    + f" nats  t={elapsed:.0f}s"
                )
                if val_ce + 1e-6 < best_val:
                    best_val   = val_ce
                    best_state = {
                        k: v.detach().cpu().clone()
                        for k, v in model.state_dict().items()
                    }
                    stale = 0
                else:
                    stale += 1
                if (step + 1) >= base_steps and stale >= cfg.fit_patience:
                    print(f"    [early-stop] patience={cfg.fit_patience} at step {step+1}")
                    break
                model.train()

        except KeyboardInterrupt:
            user_interrupt = True
            print(
                f"\n    [interrupt] Training stopped at step {step + 1}/{max_steps}. "
                "Restoring best val checkpoint then final eval + AOFE downstream.",
                flush=True,
            )
            break

    if best_state is not None:
        model.load_state_dict(best_state)

    main_steps_run = history[-1]["step"] if history else 0
    tail_values = [float(item["val_ce"]) for item in history[-cfg.fit_tail_evals:]]
    if len(tail_values) >= 2:
        fit_tail_relative_improvement = max(
            0.0,
            (tail_values[0] - tail_values[-1]) / max(abs(tail_values[0]), 1e-12),
        )
    else:
        fit_tail_relative_improvement = float("nan")

    fit_status = "not_requested"
    fit_audits_run = 0
    fit_initial_restart_relative_improvement = float("nan")
    fit_final_restart_relative_improvement = float("nan")
    if cfg.require_fit_certificate:
        # A zero-LR cosine endpoint makes a flat terminal curve insufficient
        # evidence of convergence.  Challenge the restored best checkpoint with
        # a small but nonzero LR on the same, predeclared training distribution.
        # These audit updates are counted in ``training_bytes_seen`` below.
        audit_steps = max(1, int(math.ceil(cfg.fit_audit_fraction * base_steps)))
        challenge_loader = iter(train_loader)
        before = evaluate_ntp(
            model, val_loader, device, max_batches=eval_max_batches,
            amp_bf16=cfg.amp_bf16,
        )
        best_audit_val = before
        best_audit_state = {
            k: v.detach().cpu().clone() for k, v in model.state_dict().items()
        }
        audit_opt = torch.optim.AdamW(
            model.parameters(), lr=cfg.fit_audit_lr, weight_decay=cfg.weight_decay
        )
        for audit_index in range(1, cfg.fit_max_audits + 1):
            model.train()
            for _ in range(audit_steps):
                try:
                    x, y = next(challenge_loader)
                except StopIteration:
                    challenge_loader = iter(train_loader)
                    x, y = next(challenge_loader)
                x, y = x.to(device), y.to(device)
                with autocast_context(device, cfg.amp_bf16):
                    logits = model(x)
                    loss = F.cross_entropy(
                        logits.view(-1, model.vocab_size), y.view(-1)
                    )
                audit_opt.zero_grad(set_to_none=True)
                loss.backward()
                if cfg.grad_clip > 0:
                    torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.grad_clip)
                audit_opt.step()

            after = evaluate_ntp(
                model, val_loader, device, max_batches=eval_max_batches,
                amp_bf16=cfg.amp_bf16,
            )
            relative_improvement = max(0.0, (before - after) / max(abs(before), 1e-12))
            if audit_index == 1:
                fit_initial_restart_relative_improvement = relative_improvement
            fit_final_restart_relative_improvement = relative_improvement
            fit_audits_run = audit_index
            history.append({
                "step": main_steps_run + audit_index * audit_steps,
                "lr": cfg.fit_audit_lr,
                "train_ce": float("nan"),
                "val_ce": after,
                "test_ce": float("nan"),
                "phase": f"fit_audit_{audit_index}",
            })
            print(
                f"    [fit-audit {audit_index}/{cfg.fit_max_audits}] "
                f"steps={audit_steps} val={after:.6f} "
                f"relative_improvement={relative_improvement:.3e}",
                flush=True,
            )
            if after < best_audit_val:
                best_audit_val = after
                best_audit_state = {
                    k: v.detach().cpu().clone() for k, v in model.state_dict().items()
                }
            if relative_improvement <= cfg.fit_rel_improve_tol:
                fit_status = "passed"
                break
            before = after

        if fit_status != "passed":
            fit_status = "failed_restart_improvement"
        model.load_state_dict(best_audit_state)

    # The fit certificate is validation based.  A fixed training prefix is a
    # sufficient descriptive train-loss diagnostic for large sampled corpora;
    # legacy runs can retain their full pass by leaving the cap at zero.
    print(
        "    [train] final CE on train/val/test "
        f"(train batches={'full' if cfg.final_train_eval_batches <= 0 else cfg.final_train_eval_batches}) …",
        flush=True,
    )
    tr_ce  = evaluate_ntp(
        model,
        train_loader,
        device,
        max_batches=(None if cfg.final_train_eval_batches <= 0 else cfg.final_train_eval_batches),
        log_every=500,
        log_label="final CE train",
        amp_bf16=cfg.amp_bf16,
    )
    val_ce = evaluate_ntp(
        model, val_loader, device, max_batches=eval_max_batches,
        log_every=500, log_label="final CE val", amp_bf16=cfg.amp_bf16,
    )
    te_ce = evaluate_ntp(
        model, test_loader, device, max_batches=eval_max_batches,
        log_every=500, log_label="final CE test", amp_bf16=cfg.amp_bf16,
    )
    print("    [train] final CE done.", flush=True)
    return {
        "train_ce": tr_ce, "val_ce": val_ce, "test_ce": te_ce,
        "steps_run": history[-1]["step"] if history else 0,
        "main_steps_run": main_steps_run,
        "fit_status": fit_status,
        "fit_audits_run": fit_audits_run,
        "fit_tail_relative_improvement": fit_tail_relative_improvement,
        "fit_initial_restart_relative_improvement": fit_initial_restart_relative_improvement,
        "fit_final_restart_relative_improvement": fit_final_restart_relative_improvement,
    }, history


# ─────────────────────────────────────────────────────────────────────────────
# Shape / parameter matching
# ─────────────────────────────────────────────────────────────────────────────

def active_parameter_count_for_shape(
    *,
    depth: int,
    d_model: int,
    d_ff: int,
    seq_len: int = SEQ_LEN,
    vocab_size: int = VOCAB_SIZE,
) -> int:
    """Exact active parameter count for ``TinyGPT_NTP`` without padding."""
    d = int(d_model)
    # token embedding + positional embedding + bias-free LM head + final LN.
    stem_and_head = d * (2 * vocab_size + seq_len + 2)
    # Per decoder block: two LNs (4d), qkv/proj (4d²), MLP (2*d*d_ff).
    per_block = 4 * d + 4 * d * d + 2 * d * int(d_ff)
    return int(stem_and_head + int(depth) * per_block)

def find_d_model_for_target_params(
    *,
    depth:         int,
    target_params: int,
    cfg:           TrainCfg,
    seq_len:       int = SEQ_LEN,
    vocab_size:    int = VOCAB_SIZE,
) -> Tuple[int, int, int, int]:
    """Binary-search for the largest d_model (multiple of head_dim) such that
    active_params ≤ target_params.

    Returns (d_model, n_heads, d_ff, active_params).
    Raises ValueError if even d_model_min exceeds target.
    """
    hd = cfg.head_dim
    lo = max(hd, (cfg.d_model_min // hd) * hd)
    hi = max(lo,  (cfg.d_model_max // hd) * hd)

    def n_active(d: int) -> int:
        return active_parameter_count_for_shape(
            depth=depth, d_model=d, d_ff=4 * d,
            seq_len=seq_len, vocab_size=vocab_size,
        )

    if n_active(lo) > target_params:
        raise ValueError(
            f"depth={depth}: d_model={lo} already exceeds "
            f"target_params={target_params:,}"
        )
    if n_active(hi) <= target_params:
        return hi, max(1, hi // hd), 4 * hi, n_active(hi)

    best_d, best_a = lo, n_active(lo)
    while lo <= hi:
        mid = ((lo + hi) // 2 // hd) * hd
        if mid < lo:
            break
        a = n_active(mid)
        if a <= target_params:
            best_d, best_a = mid, a
            lo = mid + hd
        else:
            hi = mid - hd

    candidates = []
    for d in [max(cfg.d_model_min, best_d - hd), best_d,
              min(cfg.d_model_max, best_d + hd)]:
        d = max(hd, (d // hd) * hd)
        a = n_active(d)
        if a <= target_params:
            candidates.append((abs(target_params - a), d, a))
    candidates.sort(key=lambda t: t[0])
    _, d_best, a_best = candidates[0]
    return d_best, max(1, d_best // hd), 4 * d_best, a_best


def find_budget_matched_shape(
    *,
    depth: int,
    target_params: int,
    cfg: TrainCfg,
    seq_len: int = SEQ_LEN,
    vocab_size: int = VOCAB_SIZE,
) -> Tuple[int, int, int, int]:
    """Find a shape with an *active* count close to ``target_params``.

    ``d_model`` remains a multiple of the fixed head dimension and ``d_ff`` a
    multiple of ``ffn_multiple``.  The search only permits a modest departure
    from the standard Transformer choice ``d_ff = 4*d_model``.  It replaces
    inactive padding in the frontier protocol, while preserving the original
    parameterization as the default path.
    """
    hd = cfg.head_dim
    d_lo = max(hd, (cfg.d_model_min // hd) * hd)
    d_hi = max(d_lo, (cfg.d_model_max // hd) * hd)
    ffn_multiple = max(1, int(cfg.ffn_multiple))
    candidates: List[Tuple[float, float, int, int, int]] = []

    # The closed-form active-count function makes a complete width search cheap
    # and avoids depth-dependent near-miss artifacts at the smallest budgets.
    candidate_widths = range(d_lo, d_hi + 1, hd)

    # The fixed non-FFN parameter count is exact for every candidate d_model;
    # constructing this tiny CPU model also protects against formula drift.
    for d_model in candidate_widths:
        n_heads = max(1, d_model // hd)
        base_params = active_parameter_count_for_shape(
            depth=depth,
            d_model=d_model,
            d_ff=0,
            seq_len=seq_len,
            vocab_size=vocab_size,
        )
        per_ffn = 2 * depth * d_model
        if per_ffn <= 0:
            continue
        ideal_ffn = (target_params - base_params) / per_ffn
        nominal_ffn = 4 * d_model
        min_ffn = max(
            ffn_multiple,
            int(math.floor((1.0 - cfg.ffn_ratio_tolerance) * nominal_ffn / ffn_multiple))
            * ffn_multiple,
        )
        max_ffn = max(
            min_ffn,
            int(math.ceil((1.0 + cfg.ffn_ratio_tolerance) * nominal_ffn / ffn_multiple))
            * ffn_multiple,
        )
        rounded = int(round(ideal_ffn / ffn_multiple)) * ffn_multiple
        for d_ff in {min_ffn, max_ffn, rounded - ffn_multiple, rounded, rounded + ffn_multiple}:
            if d_ff < min_ffn or d_ff > max_ffn:
                continue
            active = base_params + per_ffn * d_ff
            rel_error = abs(active - target_params) / target_params
            ffn_error = abs(d_ff / d_model - 4.0)
            candidates.append((rel_error, ffn_error, d_model, d_ff, active))

    if not candidates:
        raise ValueError(f"No shape candidates for depth={depth}, N={target_params:,}")
    candidates.sort(key=lambda row: (row[0], row[1], -row[4]))
    rel_error, _, d_model, d_ff, active = candidates[0]
    if rel_error > cfg.active_match_tolerance:
        raise ValueError(
            f"depth={depth}: closest active count {active:,} differs from "
            f"target {target_params:,} by {rel_error:.2%}, above "
            f"tolerance {cfg.active_match_tolerance:.2%}"
        )
    return d_model, max(1, d_model // hd), d_ff, active


def build_student(
    *,
    depth:         int,
    d_model:       int,
    n_heads:       int,
    d_ff:          int,
    target_params: int,
    cfg:           TrainCfg,
    seq_len:       int = SEQ_LEN,
    vocab_size:    int = VOCAB_SIZE,
    pad_to_target: bool = True,
) -> TinyGPT_NTP:
    tmp    = TinyGPT_NTP(
        depth=depth, d_model=d_model, n_heads=n_heads, d_ff=d_ff,
        seq_len=seq_len, vocab_size=vocab_size,
        dropout=cfg.dropout, pad_params=0,
    )
    active = count_params(tmp)
    if pad_to_target and active > target_params:
        raise ValueError(f"active {active:,} > target {target_params:,}")
    pad = int(target_params - active) if pad_to_target else 0
    return TinyGPT_NTP(
        depth=depth, d_model=d_model, n_heads=n_heads, d_ff=d_ff,
        seq_len=seq_len, vocab_size=vocab_size,
        dropout=cfg.dropout, pad_params=pad,
    )


# ─────────────────────────────────────────────────────────────────────────────
# Per-N shape sweep
# ─────────────────────────────────────────────────────────────────────────────

def run_shape_sweep_for_n(
    *,
    target_params: int,
    depths:        List[int],
    cfg:           TrainCfg,
    train_data:    np.ndarray,
    val_data:      np.ndarray,
    test_data:     np.ndarray,
    device:        torch.device,
    out_dir:       str,
    global_seed:   int,
) -> List[Dict]:
    """
    Run the full depth sweep for a single N value.
    Returns a list of result dicts (one per valid depth).
    """
    curve_dir = os.path.join(out_dir, f"curves_N{target_params}")
    os.makedirs(curve_dir, exist_ok=True)

    print(f"\n{'='*72}")
    print(f"Nominal N = {target_params:,}   D policy = {cfg.data_ratio:g} × "
          f"{'active_N' if cfg.data_by_active_params else 'target_N'} bytes")
    print(f"{'='*72}")

    # Fixed eval loaders (sequential windows on val / test)
    val_loader  = torch.utils.data.DataLoader(
        SequentialWindowDataset(val_data, SEQ_LEN),
        batch_size=cfg.batch_size, shuffle=False, num_workers=0, pin_memory=True,
    )
    test_loader = torch.utils.data.DataLoader(
        SequentialWindowDataset(test_data, SEQ_LEN),
        batch_size=cfg.batch_size, shuffle=False, num_workers=0, pin_memory=True,
    )

    results: List[Dict] = []

    for depth in depths:
        print(f"\n  ── depth={depth} ──")

        # 1. Find d_model
        try:
            if cfg.strict_active_match:
                d_model, n_heads, d_ff, active = find_budget_matched_shape(
                    depth=depth, target_params=target_params, cfg=cfg,
                )
            else:
                d_model, n_heads, d_ff, active = find_d_model_for_target_params(
                    depth=depth, target_params=target_params, cfg=cfg,
                )
        except ValueError as e:
            print(f"  [SKIP] {e}")
            continue

        pad_ratio = (target_params - active) / target_params
        alpha     = depth / d_model
        print(
            f"  d_model={d_model}  n_heads={n_heads}  d_ff={d_ff}  "
            f"active={active:,}  pad={pad_ratio:.1%}  α={alpha:.4f}"
        )
        if (not cfg.strict_active_match) and pad_ratio > cfg.max_padding_ratio:
            print(f"  [SKIP] padding ratio {pad_ratio:.1%} > {cfg.max_padding_ratio:.0%}")
            continue

        # 2. Build model
        set_seed(global_seed)
        model = build_student(
            depth=depth, d_model=d_model, n_heads=n_heads, d_ff=d_ff,
            target_params=target_params, cfg=cfg,
            pad_to_target=not cfg.strict_active_match,
        )
        actual_model_n = count_params(model)
        if cfg.strict_active_match and actual_model_n != active:
            raise RuntimeError(
                f"Strict active count mismatch: preview={active:,}, model={actual_model_n:,}"
            )
        print(f"  Total params ({'active' if cfg.strict_active_match else 'padded'}): "
              f"{actual_model_n:,}")

        # Count actual update-token exposure, not a nominal padding budget.
        data_reference_n = active if cfg.data_by_active_params else target_params
        D = int(cfg.data_ratio * data_reference_n)
        n_windows = max(cfg.batch_size, D // SEQ_LEN)
        base_steps = max(1, n_windows // cfg.batch_size)
        print(f"  D={D:,} bytes windows={n_windows:,} base_steps={base_steps:,}")

        # 3. Use identical training windows and shuffle order for every shape.
        # ``unique_prefix`` makes D an actual count of unique next-token targets
        # in the base pass; ``random`` remains for historical compatibility.
        if cfg.train_window_mode == "unique_prefix":
            train_dataset = PrefixWindowDataset(train_data, SEQ_LEN, n_windows)
        elif cfg.train_window_mode == "random":
            train_dataset = RandomWindowDataset(
                train_data, SEQ_LEN, n_windows, seed=cfg.data_seed,
            )
        else:
            raise ValueError(f"Unknown train_window_mode={cfg.train_window_mode!r}")
        loader_generator = torch.Generator()
        loader_generator.manual_seed(cfg.data_seed + 1)
        train_loader = torch.utils.data.DataLoader(
            train_dataset, batch_size=cfg.batch_size,
            shuffle=True, num_workers=0, pin_memory=True, drop_last=True,
            generator=loader_generator,
        )

        # 4. Train
        t_start = time.time()
        metrics, history = train_one_model(
            model, train_loader, val_loader, test_loader,
            base_steps, cfg, device,
        )
        elapsed = time.time() - t_start

        if cfg.save_checkpoints:
            checkpoint_dir = os.path.join(out_dir, "checkpoints")
            os.makedirs(checkpoint_dir, exist_ok=True)
            torch.save(
                {
                    "model_state_dict": model.state_dict(),
                    "target_n": target_params,
                    "active_n": active,
                    "pad_params": actual_model_n - active,
                    "model_total_n": actual_model_n,
                    "data_budget_bytes": D,
                    "depth": depth,
                    "d_model": d_model,
                    "n_heads": n_heads,
                    "d_ff": d_ff,
                    "seed": global_seed,
                    "data_budget_bytes": D,
                    "training_bytes_seen": int(metrics["steps_run"] * cfg.batch_size * SEQ_LEN),
                    "fit_status": metrics["fit_status"],
                    "fit_audits_run": metrics["fit_audits_run"],
                    "fit_initial_restart_relative_improvement": metrics[
                        "fit_initial_restart_relative_improvement"
                    ],
                    "fit_final_restart_relative_improvement": metrics[
                        "fit_final_restart_relative_improvement"
                    ],
                },
                os.path.join(
                    checkpoint_dir,
                    f"N{target_params}_depth{depth}_seed{global_seed}.pt",
                ),
            )

        # Save learning curve immediately after training so a later AGOP OOM
        # does not lose hours of run data.
        curve_path = os.path.join(curve_dir, f"depth{depth:04d}_d{d_model}.csv")
        curve_cols = (
            list(history[0].keys())
            if history
            else ["step", "lr", "train_ce", "val_ce", "test_ce"]
        )
        with open(curve_path, "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=curve_cols)
            w.writeheader()
            w.writerows(history)

        # 5. AGOP (free cached training blocks before Jacobian products)
        if device.type == "cuda":
            torch.cuda.empty_cache()
        gc.collect()

        split_map = {
            "train": train_data,
            "validation": val_data,
            "val": val_data,
            "test": test_data,
        }
        if cfg.agop_split not in split_map:
            raise ValueError(f"Unknown agop_split={cfg.agop_split!r}")
        agop_data = split_map[cfg.agop_split]
        micro = int(cfg.agop_microbatch) if cfg.agop_microbatch > 0 else None
        input_stats = None
        legacy_aofe = float("nan")
        legacy_aofe_ratio = float("nan")
        aofe = float("nan")
        aofe_ratio = float("nan")
        if cfg.agop_mode == "none":
            print("    [AGOP] deferred (training-only shape scan).", flush=True)
        elif cfg.agop_mode in {"input", "both"}:
            print(
                "    [AGOP] estimating full one-hot input-space energies "
                "(matrix-free Hutchinson) …",
                flush=True,
            )
            input_stats = estimate_input_agop_metrics_ntp(
                model,
                agop_data,
                input_probes=cfg.agop_input_probes,
                output_probes=cfg.agop_output_probes,
                batch_size=cfg.agop_batch,
                n_batches=cfg.agop_n_batches,
                seed=cfg.agop_seed,
                device=device,
                microbatch=micro,
                center_logits=cfg.agop_center_logits,
                normalize_logits=cfg.agop_normalize_logits,
            )
            aofe = input_stats.aofe
            aofe_ratio = input_stats.aofe_ratio
        elif cfg.agop_mode == "legacy":
            print("    [AGOP] estimating legacy embedding/output metric …", flush=True)
            legacy = estimate_embedding_output_agop_ntp(
                model,
                agop_data,
                proj_samples=cfg.agop_proj_samples,
                batch_size=cfg.agop_batch,
                n_batches=cfg.agop_n_batches,
                seed=cfg.agop_seed,
                device=device,
                agop_microbatch=micro,
            )
            aofe, aofe_ratio = agop_offdiag_metrics(legacy)
            legacy_aofe, legacy_aofe_ratio = aofe, aofe_ratio
        else:
            raise ValueError(f"Unknown agop_mode={cfg.agop_mode!r}")

        if cfg.agop_mode == "both":
            legacy = estimate_embedding_output_agop_ntp(
                model,
                agop_data,
                proj_samples=cfg.agop_proj_samples,
                batch_size=cfg.agop_batch,
                n_batches=cfg.agop_n_batches,
                seed=cfg.agop_seed,
                device=device,
                agop_microbatch=micro,
            )
            legacy_aofe, legacy_aofe_ratio = agop_offdiag_metrics(legacy)
        print("    [AGOP] finished.", flush=True)

        row = {
            "target_n":   target_params,
            "depth":      depth,
            "d_model":    d_model,
            "n_heads":    n_heads,
            "d_ff":       d_ff,
            "active_n":   active,
            "pad_ratio":  round(pad_ratio, 4),
            "data_budget_bytes": D,
            "data_reference": "active_n" if cfg.data_by_active_params else "target_n",
            "train_window_mode": cfg.train_window_mode,
            "training_bytes_seen": int(metrics["steps_run"] * cfg.batch_size * SEQ_LEN),
            "training_bytes_per_active_param": (
                metrics["steps_run"] * cfg.batch_size * SEQ_LEN / max(active, 1)
            ),
            "alpha":      round(alpha, 4),
            "train_ce":   round(metrics["train_ce"], 6),
            "val_ce":     round(metrics["val_ce"],   6),
            "test_ce":    round(metrics["test_ce"],  6),
            "aofe":          aofe,
            "aofe_ratio":    aofe_ratio,
            "agop_definition": (
                "onehot_input_full_matrix_free"
                if input_stats is not None
                else "legacy_embedding_output_projected"
                if cfg.agop_mode == "legacy"
                else "not_measured"
            ),
            "agop_split": cfg.agop_split,
            "agop_dim": input_stats.agop_dim if input_stats is not None else AGOP_OUT,
            "agop_examples": input_stats.num_examples if input_stats is not None else cfg.agop_batch * cfg.agop_n_batches,
            "agop_input_probes": input_stats.input_probes if input_stats is not None else 0,
            "agop_output_probes": input_stats.output_probes if input_stats is not None else cfg.agop_proj_samples,
            "agop_total_energy": input_stats.total_energy if input_stats is not None else float("nan"),
            "agop_diag_energy": input_stats.diag_energy if input_stats is not None else float("nan"),
            "agop_total_energy_se": input_stats.total_energy_se if input_stats is not None else float("nan"),
            "agop_diag_half_relative_l2": input_stats.diag_half_relative_l2 if input_stats is not None else float("nan"),
            "agop_trace": input_stats.agop_trace if input_stats is not None else float("nan"),
            "agop_logit_rms": input_stats.logit_rms if input_stats is not None else float("nan"),
            "agop_center_logits": cfg.agop_center_logits,
            "agop_normalize_logits": cfg.agop_normalize_logits,
            "legacy_aofe": legacy_aofe,
            "legacy_aofe_ratio": legacy_aofe_ratio,
            "model_seed": global_seed,
            "data_seed": cfg.data_seed,
            "steps_run":  metrics["steps_run"],
            "main_steps_run": metrics["main_steps_run"],
            "fit_status": metrics["fit_status"],
            "fit_audits_run": metrics["fit_audits_run"],
            "fit_tail_relative_improvement": metrics["fit_tail_relative_improvement"],
            "fit_initial_restart_relative_improvement": metrics["fit_initial_restart_relative_improvement"],
            "fit_final_restart_relative_improvement": metrics["fit_final_restart_relative_improvement"],
            "amp_bf16": cfg.amp_bf16,
            "eval_max_batches": cfg.eval_max_batches,
            "elapsed_s":  round(elapsed, 1),
        }
        results.append(row)
        print(
            f"  → test_ce={metrics['test_ce']:.4f} nats  "
            f"input_AOFE_ratio={aofe_ratio:.4f}  α={alpha:.4f}  "
            f"fit={metrics['fit_status']}  t={elapsed:.0f}s"
        )

    return results


# ─────────────────────────────────────────────────────────────────────────────
# Plotting
# ─────────────────────────────────────────────────────────────────────────────

def plot_per_n_results(
    results:       List[Dict],
    target_params: int,
    out_dir:       str,
) -> None:
    """Four-panel plot for a single N: loss and AOFE_ratio vs. depth and α."""
    if not results:
        return
    depths     = [r["depth"]      for r in results]
    alphas     = [r["alpha"]      for r in results]
    val_ces    = [r["val_ce"]     for r in results]
    test_ces   = [r["test_ce"]    for r in results]
    aofe_rats  = [r["aofe_ratio"] for r in results]

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    fig.suptitle(
        f"N = {target_params:,}  (byte-level next-token prediction)",
        fontsize=13, fontweight="bold",
    )

    # --- Loss vs depth ---
    ax = axes[0]
    ax.plot(depths, test_ces, "o-", color="tab:blue", lw=2, ms=6)
    best_idx = int(np.argmin(val_ces))
    ax.plot(depths[best_idx], test_ces[best_idx], "*", color="red", ms=14,
            zorder=5, label=f"val-selected depth={depths[best_idx]}")
    ax.set_xlabel("Depth", fontsize=11)
    ax.set_ylabel("Test cross-entropy (nats/byte)", fontsize=11)
    ax.set_title("Loss vs. depth", fontsize=11)
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.4)

    # --- AOFE_ratio vs depth (twin axes with loss) ---
    ax2 = axes[0].twinx()
    ax2.plot(depths, aofe_rats, "s--", color="tab:orange", lw=1.5, ms=5,
             alpha=0.7, label="AOFE_ratio")
    ax2.set_ylabel("AOFE_ratio", color="tab:orange", fontsize=10)
    ax2.tick_params(axis="y", labelcolor="tab:orange")

    # --- Loss vs aspect ratio α ---
    ax = axes[1]
    sorted_by_alpha = sorted(zip(alphas, test_ces, val_ces, aofe_rats))
    sa, sc, sv, sr = zip(*sorted_by_alpha)
    ax.plot(sa, sc, "o-", color="tab:blue", lw=2, ms=6)
    best_alpha_idx = int(np.argmin(sv))
    ax.plot(sa[best_alpha_idx], sc[best_alpha_idx], "*", color="red", ms=14,
            zorder=5, label=f"val-selected α*={sa[best_alpha_idx]:.4f}")
    ax.set_xlabel("Aspect ratio α = depth / d_model", fontsize=11)
    ax.set_ylabel("Test cross-entropy (nats/byte)", fontsize=11)
    ax.set_title("Loss vs. α  (shape)", fontsize=11)
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.4)

    ax3 = axes[1].twinx()
    ax3.plot(sa, sr, "s--", color="tab:orange", lw=1.5, ms=5, alpha=0.7,
             label="AOFE_ratio")
    ax3.set_ylabel("AOFE_ratio", color="tab:orange", fontsize=10)
    ax3.tick_params(axis="y", labelcolor="tab:orange")

    plt.tight_layout()
    fig.savefig(
        os.path.join(out_dir, f"N{target_params}_ntp_depth_alpha.png"),
        dpi=150, bbox_inches="tight",
    )
    plt.close(fig)


def plot_multi_n_summary(
    all_results: List[Dict],
    param_groups: List[int],
    out_dir: str,
) -> None:
    """
    Two-panel summary: left — scatter of projected AGOP **aofe_ratio** vs test CE; right — loss vs depth.
    Writes ``multi_N_ntp_summary.png`` plus ``optimal_alpha_vs_N.png``.
    """
    if not all_results:
        return

    try:
        cmap   = matplotlib.colormaps["tab10"]
    except AttributeError:
        cmap   = matplotlib.cm.get_cmap("tab10")   # matplotlib < 3.7 fallback
    colors = {n: cmap(i % 10) for i, n in enumerate(param_groups)}

    def _loss_vs_depth(ax: plt.Axes) -> None:
        for n in param_groups:
            rows = sorted(
                [r for r in all_results if r["target_n"] == n],
                key=lambda r: r["depth"],
            )
            if not rows:
                continue
            depths   = [r["depth"]   for r in rows]
            val_ces  = [r["val_ce"]  for r in rows]
            test_ces = [r["test_ce"] for r in rows]
            ax.plot(depths, test_ces, "o-", color=colors[n], lw=2, ms=5,
                    label=f"N={n/1e6:.1f}M")
            bi = int(np.argmin(val_ces))
            ax.plot(depths[bi], test_ces[bi], "*", color=colors[n], ms=14, zorder=5)
        ax.set_xlabel("Depth", fontsize=11)
        ax.set_ylabel("Test cross-entropy (nats/byte)", fontsize=11)
        ax.set_title("Loss vs. Depth  (★ = optimal)", fontsize=11)
        ax.legend(fontsize=10)
        ax.grid(True, alpha=0.4)

    def _scatter_vs_loss(
        ax: plt.Axes,
        xkey: str,
        xlabel: str,
        title: str,
    ) -> None:
        for n in param_groups:
            rows = [r for r in all_results if r["target_n"] == n]
            if not rows:
                continue
            xs = [r[xkey] for r in rows]
            ys = [r["test_ce"] for r in rows]
            ax.scatter(xs, ys, color=colors[n], s=60, alpha=0.8, zorder=3,
                       label=f"N={n/1e6:.1f}M")
            bi = int(np.argmin([r["val_ce"] for r in rows]))
            ax.scatter(xs[bi], ys[bi], color=colors[n], s=200, marker="*", zorder=5)
        ax.set_xlabel(xlabel, fontsize=11)
        ax.set_ylabel("Test cross-entropy (nats/byte)", fontsize=11)
        ax.set_title(title, fontsize=11)
        ax.legend(fontsize=10)
        ax.grid(True, alpha=0.4)

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    fig.suptitle(
        "NTP Shape Sweep — Byte-level LM",
        fontsize=13, fontweight="bold",
    )
    _scatter_vs_loss(
        axes[0],
        xkey="aofe_ratio",
        xlabel="AOFE_ratio (full one-hot input AGOP, matrix-free)",
        title="AOFE_ratio (AGOP) vs. Loss  (every (N, shape) pair)",
    )
    _loss_vs_depth(axes[1])
    plt.tight_layout()
    fig.savefig(
        os.path.join(out_dir, "multi_N_ntp_summary.png"),
        dpi=150, bbox_inches="tight",
    )
    plt.close(fig)

    # --- Additional: optimal α vs N (log-log) ---
    fig2, ax2 = plt.subplots(figsize=(7, 5))
    opt_alphas, opt_ns = [], []
    for n in param_groups:
        rows = [r for r in all_results if r["target_n"] == n]
        if not rows:
            continue
        best = min(rows, key=lambda r: r["val_ce"])
        opt_alphas.append(best["alpha"])
        opt_ns.append(n)
        ax2.scatter(n, best["alpha"], color=colors[n], s=120, zorder=5,
                    label=f"N={n/1e6:.1f}M,  α*={best['alpha']:.4f},  "
                          f"depth*={best['depth']},  d_model*={best['d_model']}")
    if len(opt_ns) >= 2:
        ax2.plot(opt_ns, opt_alphas, "--", color="gray", lw=1.5)
    ax2.set_xscale("log")
    ax2.set_xlabel("Parameter budget N", fontsize=11)
    ax2.set_ylabel("Optimal aspect ratio α* = depth / d_model", fontsize=11)
    ax2.set_title("Optimal Shape vs. N  (Chinchilla-style frontier)", fontsize=12)
    ax2.legend(fontsize=9)
    ax2.grid(True, alpha=0.4, which="both")
    fig2.tight_layout()
    fig2.savefig(
        os.path.join(out_dir, "optimal_alpha_vs_N.png"),
        dpi=150, bbox_inches="tight",
    )
    plt.close(fig2)


def print_summary_table(all_results: List[Dict], param_groups: List[int]) -> None:
    """Print a Chinchilla-style summary table to stdout."""
    print("\n" + "=" * 80)
    print("COMPLETE RESULTS: Transformer NTP × Shape Sweep")
    print("Task: byte-level next-token prediction")
    print("=" * 80)
    print(
        f"\n{'N':>8}  {'depth':>6}  {'d_model':>8}  {'α':>8}  "
        f"{'test_ce':>10}  {'AGOP_ratio':>12}"
    )
    print("-" * 68)

    for n in param_groups:
        rows = sorted(
            [r for r in all_results if r["target_n"] == n],
            key=lambda r: r["depth"],
        )
        if not rows:
            continue
        best_val = min(r["val_ce"] for r in rows)
        for r in rows:
            marker = " ← val-selected" if r["val_ce"] == best_val else ""
            print(
                f"  {n/1e6:>5.1f}M  {r['depth']:>6}  {r['d_model']:>8}  "
                f"{r['alpha']:>8.4f}  {r['test_ce']:>10.5f}  "
                f"{r['aofe_ratio']:>12.4f}{marker}"
            )

        ces      = np.array([r["test_ce"]        for r in rows])
        aofes    = np.array([r["aofe_ratio"]      for r in rows])
        pe_a  = pearson_corr(aofes, ces);  sp_a = spearman_corr(aofes, ces)
        best_row = min(rows, key=lambda r: r["val_ce"])
        print(
            f"        AGOP_ratio — Pearson={pe_a:.4f}  Spearman={sp_a:.4f}\n"
            f"        Optimal: depth*={best_row['depth']}  "
            f"d_model*={best_row['d_model']}  α*={best_row['alpha']:.4f}"
        )
        print()


# ─────────────────────────────────────────────────────────────────────────────
# Entry point
# ─────────────────────────────────────────────────────────────────────────────

def load_results_csv(csv_path: str) -> List[Dict]:
    """
    Load results_ntp_shape_sweep.csv with numeric types.
    If duplicate (target_n, depth) rows exist, keeps the last occurrence.
    """
    with open(csv_path, newline="") as f:
        rows = list(csv.DictReader(f))
    int_keys = {
        "target_n", "depth", "d_model", "n_heads", "d_ff", "active_n",
        "steps_run", "agop_dim", "agop_examples", "agop_input_probes",
        "agop_output_probes", "model_seed", "data_seed",
    }
    float_keys = {
        "pad_ratio", "alpha", "train_ce", "val_ce", "test_ce",
        "aofe", "aofe_ratio", "elapsed_s", "agop_total_energy",
        "agop_diag_energy", "agop_total_energy_se",
        "agop_diag_half_relative_l2", "agop_trace", "agop_logit_rms",
        "legacy_aofe", "legacy_aofe_ratio",
    }
    parsed: List[Dict] = []
    for r in rows:
        d = dict(r)
        for k in int_keys:
            if k in d and d[k] != "":
                d[k] = int(float(d[k]))
        for k in float_keys:
            if k in d and d[k] != "":
                d[k] = float(d[k])
        parsed.append(d)
    # Dedupe (target_n, depth): last row wins
    by_key: Dict[Tuple[int, int], Dict] = {}
    for r in parsed:
        key = (int(r["target_n"]), int(r["depth"]))
        by_key[key] = r
    return sorted(by_key.values(), key=lambda x: (x["target_n"], x["depth"]))


def main() -> None:
    parser = argparse.ArgumentParser(
        description="NTP shape sweep: byte-level next-token prediction.",
    )
    parser.add_argument(
        "--plot_only",
        action="store_true",
        help=(
            "Skip training and corpus load; read results_ntp_shape_sweep.csv under "
            "--out_dir, then regenerate per-N plots, multi-N summaries, and the stdout table. "
            "Use after adding new runs so figures include all N in the CSV."
        ),
    )
    parser.add_argument(
        "--data_dir", type=str, default="./data",
        help="Directory with local train.bin, validation.bin, and test.bin files.",
    )
    parser.add_argument(
        "--param_groups", type=str, default="300000,1000000,3000000",
        help="Comma-separated N values (model parameter budgets).",
    )
    parser.add_argument(
        "--depth_list", type=str, default="1,2,3,4,5,6,8,10,12,16,20,24",
        help="Comma-separated depth values to sweep.",
    )
    parser.add_argument(
        "--out_dir", type=str, default="./outputs/language_model/transformer_ntp_shape_sweep",
        help="Output directory for CSVs and plots.",
    )
    parser.add_argument("--device", type=str, default="cuda")
    # TrainCfg overrides
    parser.add_argument("--lr",            type=float, default=3e-4)
    parser.add_argument("--weight_decay",  type=float, default=1e-2)
    parser.add_argument("--data_ratio",    type=float, default=60.0,
                        help="D = data_ratio × N  bytes. Default 60 ≈ 20 BPE-tokens × 3.")
    parser.add_argument("--batch_size",    type=int,   default=64)
    parser.add_argument(
        "--amp_bf16",
        action="store_true",
        help="Use BF16 autocast for CUDA training/evaluation (large-scale runs).",
    )
    parser.add_argument("--eval_every",    type=int,   default=200)
    parser.add_argument(
        "--eval_max_batches", type=int, default=0,
        help="Cap deterministic validation/test evaluation batches (0 = all).",
    )
    parser.add_argument("--final_train_eval_batches", type=int, default=0)
    parser.add_argument(
        "--evaluate_test_during_training",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="For frontier protocol use --no-evaluate_test_during_training; test CE remains final-only.",
    )
    parser.add_argument("--warmup_steps",  type=int,   default=300)
    parser.add_argument("--head_dim",      type=int,   default=4,
                        help="n_heads = d_model // head_dim.")
    parser.add_argument("--dropout",       type=float, default=0.0)
    parser.add_argument("--max_padding_ratio", type=float, default=0.20)
    parser.add_argument("--max_train_factor",  type=float, default=1.5)
    parser.add_argument("--fit_patience",  type=int,   default=10)
    parser.add_argument(
        "--strict_active_match",
        action="store_true",
        help="Tune d_ff near 4*d_model and remove inactive padding; require active_N match.",
    )
    parser.add_argument("--active_match_tolerance", type=float, default=0.005)
    parser.add_argument("--ffn_multiple", type=int, default=4)
    parser.add_argument("--ffn_ratio_tolerance", type=float, default=0.15)
    parser.add_argument(
        "--data_by_active_params",
        action="store_true",
        help="Set D from active_N rather than nominal target_N.",
    )
    parser.add_argument(
        "--train_window_mode",
        choices=("random", "unique_prefix"),
        default="random",
        help="Random historical windows or a non-overlapping Chinchilla base prefix.",
    )
    parser.add_argument(
        "--require_fit_certificate",
        action="store_true",
        help="Run low-LR restart audits and record a pass/fail convergence certificate.",
    )
    parser.add_argument("--fit_audit_fraction", type=float, default=0.10)
    parser.add_argument("--fit_audit_lr", type=float, default=3e-5)
    parser.add_argument("--fit_max_audits", type=int, default=3)
    parser.add_argument("--fit_rel_improve_tol", type=float, default=5e-4)
    parser.add_argument("--fit_tail_rel_tol", type=float, default=5e-4)
    parser.add_argument("--fit_tail_evals", type=int, default=4)
    parser.add_argument(
        "--agop_mode",
        choices=("input", "legacy", "both", "none"),
        default="input",
        help="Primary one-hot input AGOP, old embedding/output metric, both, or deferred.",
    )
    parser.add_argument(
        "--agop_split",
        choices=("train", "validation", "val", "test"),
        default="validation",
        help="Corpus split used to estimate AGOP (fixed windows across shapes).",
    )
    parser.add_argument("--agop_batch",    type=int,   default=32)
    parser.add_argument("--agop_proj_samples", type=int, default=64)
    parser.add_argument("--agop_input_probes", type=int, default=16)
    parser.add_argument("--agop_output_probes", type=int, default=32)
    parser.add_argument("--agop_n_batches",    type=int, default=4)
    parser.add_argument(
        "--agop_center_logits",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Remove the output common-shift direction before AGOP (default: true).",
    )
    parser.add_argument(
        "--agop_normalize_logits",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Divide centered logits by a detached corpus RMS before AGOP.",
    )
    parser.add_argument("--agop_seed", type=int, default=42)
    parser.add_argument(
        "--agop_microbatch",
        type=int,
        default=0,
        help="Cap AGOP batch slices (0 = use full --agop_batch).",
    )
    parser.add_argument(
        "--agop_low_vram",
        action="store_true",
        help="Constrain AGOP probes and enable micro-batching.",
    )
    parser.add_argument(
        "--only_depth",
        type=int,
        default=None,
        help="If set, only run this layer depth (for parallel one-GPU-per-job). Ignores other entries in --depth_list.",
    )
    parser.add_argument(
        "--result_shard",
        type=str,
        default=None,
        help="If set, write rows to results_ntp_shape_sweep_{shard}.csv (parallel runs merge later).",
    )
    parser.add_argument("--d_model_min",   type=int,   default=16)
    parser.add_argument("--d_model_max",   type=int,   default=1024)
    parser.add_argument(
        "--data_seed",
        type=int,
        default=314159,
        help="Fixed training-window and loader-order seed, independent of shape/model seed.",
    )
    parser.add_argument("--save_checkpoints", action="store_true")
    parser.add_argument("--seed",          type=int,   default=0)
    args = parser.parse_args()

    if args.agop_low_vram:
        args.agop_batch = min(int(args.agop_batch), 32)
        args.agop_proj_samples = min(int(args.agop_proj_samples), 32)
        args.agop_input_probes = min(int(args.agop_input_probes), 8)
        args.agop_output_probes = min(int(args.agop_output_probes), 16)
        args.agop_n_batches = min(int(args.agop_n_batches), 3)
        if int(args.agop_microbatch) == 0:
            args.agop_microbatch = 8

    if args.plot_only:
        csv_path = os.path.join(args.out_dir, "results_ntp_shape_sweep.csv")
        if not os.path.isfile(csv_path):
            raise FileNotFoundError(
                f"{csv_path} not found. Run training first or set --out_dir correctly."
            )
        all_results = load_results_csv(csv_path)
        param_groups = sorted({int(r["target_n"]) for r in all_results})
        print(f"[plot_only] Loaded {len(all_results)} rows from {csv_path}")
        print(f"[plot_only] N values: {param_groups}\n")
        for n in param_groups:
            rows = sorted(
                [r for r in all_results if r["target_n"] == n],
                key=lambda r: r["depth"],
            )
            plot_per_n_results(rows, n, args.out_dir)
        plot_multi_n_summary(all_results, param_groups, args.out_dir)
        print_summary_table(all_results, param_groups)
        print(f"\n[plot_only] Plots and table regenerated under: {args.out_dir}")
        print(f"[plot_only] CSV unchanged: {csv_path}")
        return

    # Build config from args
    cfg = TrainCfg(**{
        f.name: getattr(args, f.name)
        for f in dataclasses.fields(TrainCfg)
        if hasattr(args, f.name)
    })

    param_groups  = [int(x) for x in args.param_groups.split(",")]
    depth_list    = [int(x) for x in args.depth_list.split(",")]
    if args.only_depth is not None:
        depth_list = [int(args.only_depth)]
    device        = torch.device(args.device if torch.cuda.is_available()
                                 else "cpu")

    os.makedirs(args.out_dir, exist_ok=True)
    print(f"Device : {device}")
    print(f"N      : {param_groups}")
    print(f"depths : {depth_list}")
    print(f"D=ratio×N: {cfg.data_ratio}×N bytes  (≈ {cfg.data_ratio/3:.0f}×N BPE-tokens)")
    print(
        f"AGOP: mode={cfg.agop_mode} split={cfg.agop_split} batch={cfg.agop_batch}  "
        f"input_probes={cfg.agop_input_probes} output_probes={cfg.agop_output_probes}  "
        f"n_batches={cfg.agop_n_batches} microbatch={cfg.agop_microbatch or 'full'}  "
        f"center={cfg.agop_center_logits} normalize={cfg.agop_normalize_logits}"
    )
    print(
        f"Frontier controls: strict_active_match={cfg.strict_active_match} "
        f"data_by_active={cfg.data_by_active_params} "
        f"train_windows={cfg.train_window_mode} "
        f"fit_certificate={cfg.require_fit_certificate} "
        f"amp_bf16={cfg.amp_bf16} eval_max_batches={cfg.eval_max_batches or 'all'}"
    )
    if args.result_shard:
        print(f"CSV shard: results_ntp_shape_sweep_{args.result_shard}.csv")
    print(f"Out    : {args.out_dir}\n")

    # Load corpus
    print("Loading byte-level corpus ...")
    train_data, val_data, test_data = load_corpus(args.data_dir)
    print(
        f"  train={len(train_data)/1e6:.1f}M  "
        f"val={len(val_data)/1e6:.1f}M  "
        f"test={len(test_data)/1e6:.1f}M bytes\n"
    )

    # ── Quick shape preview (no training) ────────────────────────────────────
    print("Shape preview (no training):")
    print(f"{'N':>10}  {'depth':>6}  {'d_model':>8}  {'active':>10}  "
          f"{'pad%':>6}  {'α':>8}")
    print("-" * 56)
    for n in param_groups:
        for depth in depth_list:
            try:
                if cfg.strict_active_match:
                    d, nh, dff, active = find_budget_matched_shape(
                        depth=depth, target_params=n, cfg=cfg,
                    )
                else:
                    d, nh, dff, active = find_d_model_for_target_params(
                        depth=depth, target_params=n, cfg=cfg,
                    )
                pad = (n - active) / n
                alpha = depth / d
                flag = " [PAD>20%]" if pad > cfg.max_padding_ratio else ""
                print(
                    f"  {n:>9,}  {depth:>6}  {d:>8}  {active:>10,}  "
                    f"{pad:>5.1%}  {alpha:>8.4f}{flag}"
                )
            except ValueError as e:
                print(f"  {n:>9,}  {depth:>6}  {'SKIP':>8}  {str(e)}")
    print()

    # ── Main sweep ────────────────────────────────────────────────────────────
    all_results: List[Dict] = []
    if args.result_shard:
        csv_name = f"results_ntp_shape_sweep_{args.result_shard}.csv"
    else:
        csv_name = "results_ntp_shape_sweep.csv"
    csv_path = os.path.join(args.out_dir, csv_name)
    fieldnames: Optional[List[str]] = None

    for n in param_groups:
        rows = run_shape_sweep_for_n(
            target_params=n,
            depths=depth_list,
            cfg=cfg,
            train_data=train_data,
            val_data=val_data,
            test_data=test_data,
            device=device,
            out_dir=args.out_dir,
            global_seed=cfg.seed,
        )
        all_results.extend(rows)

        # Plot per-N
        plot_per_n_results(rows, n, args.out_dir)

        # Append to CSV
        if rows:
            if fieldnames is None:
                fieldnames = list(rows[0].keys())
            mode = "a" if os.path.exists(csv_path) else "w"
            with open(csv_path, mode, newline="") as f:
                w = csv.DictWriter(f, fieldnames=fieldnames)
                if mode == "w":
                    w.writeheader()
                w.writerows(rows)

    # ── Final outputs ─────────────────────────────────────────────────────────
    if all_results:
        plot_multi_n_summary(all_results, param_groups, args.out_dir)
        print_summary_table(all_results, param_groups)

    print(f"\nAll results saved to: {args.out_dir}")
    print(f"CSV: {csv_path}")


if __name__ == "__main__":
    main()
