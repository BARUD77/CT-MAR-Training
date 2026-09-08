# CT Metal Artifact Reduction (MAR)

Deep-learning models for **reducing metal artifacts in CT images**. The project trains and
evaluates several architectures that map a metal-affected (MA) CT slice to a clean image,
optionally guided by a linear-interpolation (LI) prior and a soft artifact map
`A_MG = norm(relu(LI − MA))`.

The flagship model is a **SPADE-conditioned Swin-UNet v2**, which injects the soft artifact
map into every decoder stage via Spatially-Adaptive Denormalization (SPADE). Architecture
diagrams (TikZ) are in [`figures/`](figures/).

---

## Highlights

- **Model-agnostic trainer** — one entry point trains any registered architecture.
- **Multiple architectures**: vanilla U-Net, RED-CNN, Swin-UNet (+ feature-gating, 3-channel,
  mask-guided variants), Swin-UNet v2, **Swin-UNet v2 + multiscale SPADE**, and the
  MARMamba family (Mamba / ViT / PVT / MARformer).
- **Artifact-aware conditioning** via the soft artifact map, binary metal mask, or an
  oracle map (upper-bound analysis only).
- **Metal-masked losses & metrics** — the metal region is excluded from L1/SSIM/RMSE so the
  model is scored only on streak removal.
- **W&B logging**, cosine LR schedule with warmup, AMP (bf16/fp16), and early stopping.

---

## Main entry point

The primary trainer is **[`trainer_wandb_unclipped.py`](trainer_wandb_unclipped.py)**
(Weights & Biases logging, model-agnostic, HU clipping optional).

```bash
python trainer_wandb_unclipped.py --model <arch> [options]
```

### Supported `--model` values

| `--model` | Input | Notes |
|-----------|-------|-------|
| `unet` | MA or MA+LI | Vanilla U-Net generator |
| `redcnn` | MA or MA+LI | RED-CNN denoiser |
| `swinunet` | MA or MA+LI | Swin-UNet (requires `--model_cfg`) |
| `swinunet_fg` | MA (1ch) | Feature-gating; LI used to build `A_MG` |
| `swinunet_3ch` | `[MA, LI, A_MG]` | 3-channel input |
| `swinunet_v2` | MA or MA+LI | Swin-UNet v2 |
| `swinunet_v2_spade` | `[MA, LI]` + SPADE map | **Flagship**: multiscale SPADE conditioning |
| `marmamba` / `marvit` / `marpvt` / `marformer` | MA (1ch) | MARMamba family (internal residual; needs `mamba_ssm`) |

Swin models read hyper-parameters from a YAML config (`--model_cfg`); each variant ships one
under `models/<variant>/config.yaml`.

---

## Setup

```bash
# Python 3.10+ recommended
python -m venv .venv
.\.venv\Scripts\Activate.ps1        # Windows PowerShell
pip install -r requirements.txt
```

Notes:
- Install the correct **CUDA build of PyTorch** for your GPU (see the `torch`/`torchvision`
  lines in [`requirements.txt`](requirements.txt)).
- The MARMamba family imports `mamba_ssm` (Linux + CUDA); only needed if you select those models.
- `--loss marmamba` additionally requires the `lpips` package.
- Set up W&B once: `wandb login`.

---

## Data layout

The dataset ([`aapm_dataset.py`](aapm_dataset.py) → `CTMetalArtifactDataset`) expects parallel
directories of `.npy` slices with matching filenames:

```
--ma_dir     metal-affected (MA) inputs        (required)
--gt_dir     clean ground-truth targets        (required)
--li_dir     linear-interpolation (LI) prior   (required for ma_li / fg / 3ch / spade)
--mask_dir   binary metal masks                (optional; needed for --spade_map mask,
                                                 --metal_mask_on_gt)
```

Slices are HU-clipped to `[--hu_min, --hu_max]` and normalized to `[0, 1]` (`--no_clip` to
disable). A train/val split (90/10, seed 42) is created internally.

---

## Training examples

**Flagship — Swin-UNet v2 + multiscale SPADE (soft artifact map):**

```bash
python trainer_wandb_unclipped.py \
  --model swinunet_v2_spade \
  --model_cfg models/swin_unet_v2_SPADE_multiscale_soft_art_map/config.yaml \
  --ma_dir data/ma --li_dir data/li --gt_dir data/gt \
  --spade_map soft \
  --epochs 100 --batch_size 4 --learning_rate 1e-4 \
  --ssim_loss_weight 0.1 --amp --amp_dtype bf16 \
  --project ct-mar --run_name swin_v2_spade_soft
```

**SPADE conditioning variants** (`--spade_map`):
- `soft` — `A_MG = norm(relu(LI − MA))` (default, deployable)
- `mask` — binary metal mask (requires `--mask_dir`)
- `oracle` — `norm(relu(GT − MA))` (upper-bound only; leaks the target, not deployable)

**Vanilla U-Net, MA+LI input:**

```bash
python trainer_wandb_unclipped.py --model unet \
  --input_mode ma_li --ma_dir data/ma --li_dir data/li --gt_dir data/gt \
  --epochs 100 --batch_size 8 --amp
```

**MARMamba (MA-only):**

```bash
python trainer_wandb_unclipped.py --model marmamba \
  --input_mode ma --ma_dir data/ma --gt_dir data/gt \
  --loss marmamba --loss_alpha 0.8 --loss_beta 0.2 --amp
```

### Key flags

| Flag | Purpose |
|------|---------|
| `--input_mode {ma, ma_li}` | 1-channel MA vs. 2-channel `[MA, LI]` |
| `--residual` | Predict a correction added to MA (`pred = MA + f(x)`); identity warm start |
| `--loss {recon, marmamba}` | Masked L1 (+ optional masked SSIM) vs. Charbonnier + LPIPS |
| `--ssim_loss_weight` / `--ssim_loss_dilation` | SSIM loss term and metal-boundary dilation |
| `--amp` / `--amp_dtype {bf16, fp16}` | Mixed precision |
| `--warmup_epochs` / `--min_lr_ratio` | Linear warmup → cosine decay schedule |
| `--early_stop_patience` | Early stop on stalled val SSIM |
| `--metal_mask_on_gt` | Paint MA metal onto GT so only streaks must be removed (needs `--mask_dir`) |
| `--hu_min` / `--hu_max` / `--no_clip` | HU intensity window |
| `--log_dir` / `--project` / `--run_name` / `--entity` | Output dir & W&B run info |

Run `python trainer_wandb_unclipped.py --help` for the full list.

---

## Outputs

Checkpoints are written to `--log_dir` (default `./runs`):
- `last.pt` — overwritten every epoch.
- `best.pt` — best validation (masked) SSIM, with `monitor_metric` / `monitor_value`.

Each checkpoint stores `{"epoch", "model_state", ...}`. Metrics (loss, PSNR, SSIM, RMSE in HU)
are logged to Weights & Biases.

---

## Evaluation

Quantitative and qualitative evaluation live in the `eval_*.ipynb` / `qualitative_*.ipynb`
notebooks and helper scripts (e.g. [`metrics.py`](metrics.py),
[`evaluate_near_metal_roi_metrics.py`](evaluate_near_metal_roi_metrics.py)). Per-method metric
JSONs are collected under [`eval_outputs/`](eval_outputs/) and aggregated results under
[`evaluation_results/`](evaluation_results/).

---

## Repository structure

```
trainer_wandb_unclipped.py   # main trainer (this project's entry point)
trainer_wandb*.py / trainer*.py  # trainer variants (gradient accumulation, HP tuning, Lightning, ...)
aapm_dataset.py / dataset.py # dataset loaders
metrics.py                   # SSIM / PSNR / RMSE (+ masked variants)
models/                      # architectures + per-model config.yaml
  swin_unet_v2_SPADE_multiscale_soft_art_map/  # flagship SPADE Swin-UNet v2
  swin_unet*, unet.py, redcnn ...
MARMAMBA/                    # Mamba/ViT/PVT/MARformer models
figures/                     # TikZ architecture diagrams (SPADE Swin-UNet + SPADE block)
eval_*.ipynb / qualitative_*.ipynb  # evaluation & visualization
eval_outputs/ , evaluation_results/ , runs/  # metrics & checkpoints
requirements.txt
```

---

## Architecture diagrams

- [`figures/spade_swin_unet_architecture.tex`](figures/spade_swin_unet_architecture.tex) —
  U-shaped SPADE Swin-UNet v2 (encoder ↓, bottleneck, decoder ↑ with per-stage SPADE).
- [`figures/spade_block.tex`](figures/spade_block.tex) — internals of a SPADE block
  (`out = norm(x) · (1 + γ(A_MG)) + β(A_MG)`).

Compile with `pdflatex <file>.tex`.
