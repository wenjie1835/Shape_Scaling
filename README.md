# Shape Scaling

Code for the final experiments in the accompanying anonymous submission.  The
repository contains no checkpoints, result tables, dataset copies, or links to
model-hosting accounts.  It is deliberately limited to the experiments
reported in the paper:

1. sparse tied-autoencoder reconstruction across the full double-descent sweep;
2. Gaussian minimum-norm regression, which isolates the shared fitted-noise
   amplification term; and
3. fixed-budget, byte-level Transformer depth--width scans with an input-AGOP
   Frobenius-norm estimator.

## Setup

```bash
python -m venv .venv
source .venv/bin/activate
pip install -U pip
pip install -r requirements.txt
```

The sparse and Gaussian experiments run on CPU.  The language-model scan is
GPU intensive and requires a CUDA-enabled PyTorch installation.

## Sparse reconstruction and double descent

```bash
python experiments/sparse_reconstruction.py --device cuda --output-dir outputs/sparse
```

The default protocol uses 22 training-set sizes, five model initializations,
3,000 AdamW updates per initialization, an input dimension of 1,000, a tied
two-dimensional bottleneck, and 5,000 independently drawn test examples.  It
writes per-run CSV data and a dual-axis figure of reconstruction loss and
`||G||_F` across the complete sweep.

## Gaussian minimum-norm regression

```bash
python experiments/gaussian_min_norm.py --output-dir outputs/gaussian
```

The default run uses `p=128`, ten sample sizes, three noise levels, and 128
paired repetitions.  It writes every fitted predictor, the identity checks,
and the figure used to inspect the interpolation peak.

## Language-model shape scan

Place pre-tokenized byte splits in a local directory:

```text
data/train.bin
data/validation.bin
data/test.bin
```

Each file is a flat `uint8` byte stream.  Data acquisition and any trained
models are intentionally outside this repository.  A small smoke run is:

```bash
python language_model/transformer_ntp_shape_sweep.py \
  --data_dir data --param_groups 300000 --depth_list 2,4 \
  --device cuda --data_ratio 6 --eval_max_batches 2 \
  --agop_low_vram --out_dir outputs/lm_smoke
```

For a full scan, increase `--data_ratio` to the paper protocol and select the
desired parameter budgets and depths.  Results are saved locally under
`--out_dir`; checkpoints are written only when `--save_checkpoints` is passed.

The AGOP estimator operates on the continuous one-hot input extension and
reports the Frobenius norm of the averaged operator.  Its probe count, output
centering, RMS normalization, and evaluation split are explicit command-line
arguments, so they can be locked before a sweep.

## Reproducibility boundaries

All output directories, raw corpora, checkpoints, cache files, and generated
figures are ignored by Git.  The repository records code and protocols only;
it does not provide external model locations or identities during review.
