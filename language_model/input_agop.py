"""Matrix-free input-space AGOP metrics for byte-level language models.

The mathematical input is the per-position token simplex ``q`` rather than the
integer token id or the post-embedding activation.  For an embedding matrix E,

    e_t = q_t E + p_t,

so gradients and tangents are mapped exactly by the chain rule.  The full AGOP
has shape ``(T * V, T * V)`` and is intentionally never materialized here.
Hutchinson estimators recover its Frobenius and diagonal energies directly.
"""

from __future__ import annotations

import math
from contextlib import contextmanager
from dataclasses import dataclass
from typing import List, Optional, Sequence

import numpy as np
import torch


@contextmanager
def _explicit_attention_for_jvp(model: torch.nn.Module):
    """Temporarily disable fused attention kernels lacking JVP derivatives.

    CUDA Flash/memory-efficient attention is used for large-model training, but
    the exact AGOP matvec uses ``torch.autograd.functional.jvp``.  Current
    PyTorch kernels do not implement that derivative, so this narrow scope
    falls back to mathematically identical explicit causal attention only for
    the AGOP JVP.
    """
    modules = [
        module for module in model.modules()
        if hasattr(module, "use_memory_efficient_attention")
    ]
    states = [bool(module.use_memory_efficient_attention) for module in modules]
    try:
        for module in modules:
            module.use_memory_efficient_attention = False
        yield
    finally:
        for module, state in zip(modules, states):
            module.use_memory_efficient_attention = state


@dataclass
class InputAGOPResult:
    aofe: float
    aofe_ratio: float
    total_energy: float
    diag_energy: float
    raw_offdiag_energy: float
    total_energy_se: float
    diag_half_relative_l2: float
    agop_trace: float
    agop_dim: int
    num_examples: int
    input_probes: int
    output_probes: int
    logit_rms: float
    centered_logits: bool
    normalized_logits: bool


def _prepare_last_logits(
    model: torch.nn.Module,
    embeddings: torch.Tensor,
    *,
    center_logits: bool,
    normalize_logits: bool,
    logit_rms: float,
) -> torch.Tensor:
    logits = model.forward_from_embeddings(embeddings)[:, -1, :]
    if center_logits:
        logits = logits - logits.mean(dim=-1, keepdim=True)
    if normalize_logits:
        logits = logits / float(max(logit_rms, 1e-12))
    return logits


def embeddings_for_tokens(model: torch.nn.Module, x: torch.Tensor) -> torch.Tensor:
    """Return token plus position embeddings without building a one-hot tensor."""
    batch, seq_len = x.shape
    pos = torch.arange(seq_len, device=x.device).unsqueeze(0).expand(batch, seq_len)
    return model.tok_emb(x) + model.pos_emb(pos)


def token_input_vjp(
    model: torch.nn.Module,
    embeddings: torch.Tensor,
    output_cotangent: torch.Tensor,
    *,
    center_logits: bool = False,
    normalize_logits: bool = False,
    logit_rms: float = 1.0,
) -> torch.Tensor:
    """Compute ``J_q r`` for the one-hot/simplex token input ``q``.

    ``J_q`` follows the paper's convention: input coordinates by output
    coordinates.  The returned tensor has shape ``[B, T, V]``.
    """
    e = embeddings.detach().requires_grad_(True)
    logits = _prepare_last_logits(
        model,
        e,
        center_logits=center_logits,
        normalize_logits=normalize_logits,
        logit_rms=logit_rms,
    )
    r = output_cotangent
    if r.ndim == 1:
        r = r.unsqueeze(0).expand(logits.shape[0], -1)
    grad_e = torch.autograd.grad(
        logits,
        e,
        grad_outputs=r.to(device=logits.device, dtype=logits.dtype),
        create_graph=False,
        retain_graph=False,
        allow_unused=False,
    )[0]
    embedding_weight = model.tok_emb.weight.detach().to(dtype=grad_e.dtype)
    return grad_e @ embedding_weight.T


def token_input_agop_matvec(
    model: torch.nn.Module,
    embeddings: torch.Tensor,
    input_tangent: torch.Tensor,
    *,
    center_logits: bool = False,
    normalize_logits: bool = False,
    logit_rms: float = 1.0,
) -> torch.Tensor:
    """Compute per-example ``J_q J_q^T z`` without materializing ``J_q``."""
    embedding_weight = model.tok_emb.weight.detach().to(dtype=embeddings.dtype)
    z_q = input_tangent.to(device=embeddings.device, dtype=embeddings.dtype)
    if z_q.ndim == 2:
        z_e = z_q @ embedding_weight
        z_e = z_e.unsqueeze(0).expand(embeddings.shape[0], -1, -1)
    elif z_q.ndim == 3:
        z_e = z_q @ embedding_weight
    else:
        raise ValueError("input_tangent must have shape [T,V] or [B,T,V]")

    def fwd(e_in: torch.Tensor) -> torch.Tensor:
        return _prepare_last_logits(
            model,
            e_in,
            center_logits=center_logits,
            normalize_logits=normalize_logits,
            logit_rms=logit_rms,
        )

    with _explicit_attention_for_jvp(model):
        _, directional_output = torch.autograd.functional.jvp(
            fwd,
            (embeddings.detach(),),
            (z_e,),
            create_graph=False,
            strict=True,
        )
    return token_input_vjp(
        model,
        embeddings,
        directional_output.detach(),
        center_logits=center_logits,
        normalize_logits=normalize_logits,
        logit_rms=logit_rms,
    )


def _sample_token_batches(
    data: np.ndarray,
    *,
    seq_len: int,
    batch_size: int,
    n_batches: int,
    seed: int,
) -> List[torch.Tensor]:
    max_start = len(data) - seq_len - 1
    if max_start <= 0:
        raise ValueError(f"Corpus length {len(data)} is too short for seq_len={seq_len}")
    rng = np.random.default_rng(seed)
    batches: List[torch.Tensor] = []
    for _ in range(n_batches):
        starts = rng.integers(0, max_start, size=batch_size)
        x_np = np.stack([data[int(s) : int(s) + seq_len] for s in starts]).astype(np.int64)
        batches.append(torch.from_numpy(x_np))
    return batches


@torch.no_grad()
def _estimate_last_logit_rms(
    model: torch.nn.Module,
    token_batches: Sequence[torch.Tensor],
    *,
    device: torch.device,
    microbatch: int,
    center_logits: bool,
) -> float:
    total = 0.0
    count = 0
    for x_cpu in token_batches:
        for start in range(0, len(x_cpu), microbatch):
            x = x_cpu[start : start + microbatch].to(device)
            e = embeddings_for_tokens(model, x)
            logits = model.forward_from_embeddings(e)[:, -1, :].float()
            if center_logits:
                logits = logits - logits.mean(dim=-1, keepdim=True)
            total += float((logits * logits).sum().item())
            count += int(logits.numel())
    return max(math.sqrt(total / max(1, count)), 1e-12)


def _rademacher(shape: tuple[int, ...], generator: torch.Generator) -> torch.Tensor:
    values = torch.randint(0, 2, shape, generator=generator, dtype=torch.int8)
    return values.to(torch.float32).mul_(2.0).sub_(1.0)


def estimate_input_agop_metrics_ntp(
    model: torch.nn.Module,
    data: np.ndarray,
    *,
    input_probes: int = 16,
    output_probes: int = 32,
    batch_size: int = 32,
    n_batches: int = 4,
    seed: int = 42,
    device: torch.device,
    microbatch: Optional[int] = None,
    center_logits: bool = True,
    normalize_logits: bool = True,
) -> InputAGOPResult:
    """Estimate AOFE of the full one-hot input-space AGOP.

    Let ``G = E_x[J_q(x) J_q(x)^T]`` with dimension ``T*V``.  The total
    Frobenius energy is estimated as ``E_z ||Gz||^2`` using Rademacher input
    probes.  ``diag(G)`` is estimated independently in two halves with output
    probes and their dot product gives an unbiased split-sample estimate of
    ``||diag(G)||^2``.  Consequently, no input projection or full AGOP matrix is
    required.
    """
    if input_probes < 2:
        raise ValueError("input_probes must be at least 2")
    if output_probes < 4 or output_probes % 2:
        raise ValueError("output_probes must be an even integer of at least 4")
    if batch_size <= 0 or n_batches <= 0:
        raise ValueError("batch_size and n_batches must be positive")

    model.eval()
    seq_len = int(model.seq_len)
    vocab_size = int(model.vocab_size)
    mb = batch_size if microbatch is None or int(microbatch) <= 0 else int(microbatch)
    mb = max(1, min(mb, batch_size))
    token_batches = _sample_token_batches(
        data,
        seq_len=seq_len,
        batch_size=batch_size,
        n_batches=n_batches,
        seed=seed,
    )
    num_examples = sum(len(batch) for batch in token_batches)
    logit_rms = 1.0
    if normalize_logits:
        logit_rms = _estimate_last_logit_rms(
            model,
            token_batches,
            device=device,
            microbatch=mb,
            center_logits=center_logits,
        )

    generator = torch.Generator(device="cpu")
    generator.manual_seed(int(seed) + 10_003)

    total_probe_values: List[float] = []
    for _ in range(input_probes):
        z_q = _rademacher((seq_len, vocab_size), generator).to(device)
        gz = torch.zeros(seq_len, vocab_size, device=device, dtype=torch.float64)
        seen = 0
        for x_cpu in token_batches:
            for start in range(0, len(x_cpu), mb):
                x = x_cpu[start : start + mb].to(device)
                with torch.no_grad():
                    e = embeddings_for_tokens(model, x).detach()
                az = token_input_agop_matvec(
                    model,
                    e,
                    z_q,
                    center_logits=center_logits,
                    normalize_logits=normalize_logits,
                    logit_rms=logit_rms,
                )
                az = torch.nan_to_num(az.float(), nan=0.0, posinf=0.0, neginf=0.0)
                gz += az.double().sum(dim=0)
                seen += int(x.shape[0])
        gz /= float(max(1, seen))
        total_probe_values.append(float((gz * gz).sum().item()))
        del z_q, gz

    half = output_probes // 2
    diag_halves: List[torch.Tensor] = []
    for _group in range(2):
        diag_group = torch.zeros(seq_len, vocab_size, device=device, dtype=torch.float64)
        for _ in range(half):
            r = _rademacher((vocab_size,), generator).to(device)
            diag_probe = torch.zeros_like(diag_group)
            seen = 0
            for x_cpu in token_batches:
                for start in range(0, len(x_cpu), mb):
                    x = x_cpu[start : start + mb].to(device)
                    with torch.no_grad():
                        e = embeddings_for_tokens(model, x).detach()
                    jr = token_input_vjp(
                        model,
                        e,
                        r,
                        center_logits=center_logits,
                        normalize_logits=normalize_logits,
                        logit_rms=logit_rms,
                    )
                    jr = torch.nan_to_num(jr.float(), nan=0.0, posinf=0.0, neginf=0.0)
                    diag_probe += (jr.double() * jr.double()).sum(dim=0)
                    seen += int(x.shape[0])
            diag_group += diag_probe / float(max(1, seen))
            del r, diag_probe
        diag_halves.append(diag_group / float(half))

    total_values = np.asarray(total_probe_values, dtype=np.float64)
    total_energy = float(total_values.mean())
    total_energy_se = float(total_values.std(ddof=1) / math.sqrt(len(total_values)))
    diag_a, diag_b = diag_halves
    diag_energy = float((diag_a * diag_b).sum().item())
    raw_offdiag = total_energy - diag_energy
    offdiag = max(raw_offdiag, 0.0)
    ratio = min(max(offdiag / max(total_energy, 1e-30), 0.0), 1.0)
    diag_mean = 0.5 * (diag_a + diag_b)
    diag_half_relative_l2 = float(
        torch.linalg.vector_norm(diag_a - diag_b).item()
        / max(torch.linalg.vector_norm(diag_mean).item(), 1e-30)
    )
    trace = float(diag_mean.sum().item())

    return InputAGOPResult(
        aofe=offdiag,
        aofe_ratio=ratio,
        total_energy=total_energy,
        diag_energy=diag_energy,
        raw_offdiag_energy=raw_offdiag,
        total_energy_se=total_energy_se,
        diag_half_relative_l2=diag_half_relative_l2,
        agop_trace=trace,
        agop_dim=seq_len * vocab_size,
        num_examples=num_examples,
        input_probes=input_probes,
        output_probes=output_probes,
        logit_rms=logit_rms,
        centered_logits=center_logits,
        normalized_logits=normalize_logits,
    )
