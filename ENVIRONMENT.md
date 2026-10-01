# Environment

The environment that produced the POC results (git tag `poc-baseline`).
Exact Python package versions are in [`requirements.lock`](requirements.lock).

## Software

| Item | Value |
|---|---|
| Python | 3.14.4 (CPython, GCC 15.2.0 build, 2026-08-20) |
| OS | Ubuntu 26.04 LTS, Linux kernel 7.0.0-29-generic, x86_64 |
| torch / torchvision | 2.13.0+cpu / 0.28.0+cpu (CPU wheels from `download.pytorch.org/whl/cpu`) |
| torch build | `USE_CUDA=0`, BLAS and LAPACK from MKL, oneDNN 3.12.0, CPU capability AVX2 |
| open_clip_torch | 3.3.0 |
| scikit-learn | 1.9.0 |
| numpy / pandas | 2.5.2 / 3.0.5 |
| umap-learn | 0.5.12 |
| Pillow | 12.3.0 |
| Web front end | Node.js 22.22.1, pnpm 10.11.0, versions locked in `web/pnpm-lock.yaml` |

## Hardware

| Item | Value |
|---|---|
| Machine | KVM virtual machine (VPS) |
| CPU | Intel Core Processor (Haswell, no TSX), 6 vCPUs, 1 thread per core; AVX2 and FMA, no AVX-512 |
| RAM | 11 GiB |
| GPU | none. CPU only: `torch.cuda.is_available()` is `False` and no script selects a device |

Every model step runs on the CPU. The scripts and the API call
`torch.set_num_threads(os.cpu_count())`, i.e. 6 threads here
(`scripts/06_embed.py`, `scripts/12_rise.py`, `api/state.py`).

## Timings on this machine

| Step | Time |
|---|---|
| `scripts/06_embed.py`, 1,236 images | about 1.4 images per second |
| `scripts/12_rise.py`, one map (400 masks) | 255 to 262 s |
| `scripts/12_rise.py --gallery`, 10 maps | about 43 min |
| API start-up (model, probe, UMAP, tables) | about 30 to 40 s |

## Numerical reproducibility

Embeddings are computed in float32 on the CPU with oneDNN/MKL kernels.
A different CPU family, thread count or BLAS build can change the last bits
of an embedding; the probe's predictions are not expected to move, but
bit-identical embeddings are only guaranteed on the same software and a
CPU with the same instruction set.

Random seeds: 42 for the train/test split, UMAP and the API's map
subsample (`config.yaml: seed`); 0 for the 5-fold temperature-calibration
folds (`scripts/07_eval.py`) and for the RISE masks (`scripts/12_rise.py
--seed`). Inference uses no dropout and no sampling, so torch is not seeded.
