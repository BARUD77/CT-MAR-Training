# FIND-Net baseline — implementation notes

## Source
- Code: https://github.com/Farid-Tasharofi/FIND-Net, commit `505c326a29eaf3274ee87a614e680a0addd60286`
  (2025-09-02), vendored unchanged in `third_party/findnet/` (nested `.git` removed; `pretrained_models/*.pt`
  are git-ignored and not used). See `third_party/findnet/COMMIT.txt`.
- The repo contains **no training code**. The training recipe comes from:
  - FIND-Net paper (arXiv 2508.10617v1, Sec. 2 "Training Loss", Sec. 3.1 "Experimental Setup"):
    AdamW (β 0.9/0.999), lr 1e-4, weight decay 1e-5, linear warmup + cosine annealing, up to 200 epochs,
    batch size 6, 10 stages, A100 80 GB, early stopping on validation loss.
  - DICDNet `train_DICDNet.py` (hongwang01/DICDNet @ `6498998a`), which the paper says FIND-Net follows: the
    stage-wise loss and its weights.
  - FIND-Net `Dataset/dataset.py` / `test.sh`: preprocessing, augmentation, model hyperparameters, full 512×512.

## Files
| File | Purpose |
|---|---|
| `model_wrapper.py` | Imports the original `FINDNet` class (Gaussian filter ON), materializes aliased parameters |
| `dataset_adapter.py` | Wraps `baselines/common/data.py`; my [0,1] → FIND-Net units; non-metal mask = 1 − metal mask; flips |
| `config.yaml` | All hyperparameters, each tagged with its source |
| `train_findnet.py` | Training + per-epoch validation + checkpoints + W&B |

## Deviations from the original code / settings (and why)
1. **Gaussian filtering enabled without editing third_party.** `Model/ffc.py` defaults to
   `Gaussian_filter = False` (FIND-Net *No-GF*); the README says to edit the file for full FIND-Net. The wrapper
   reads `ffc.py`, replaces that single line in memory, and executes it as `Model.ffc`. (User decision D1.)
2. **Aliased parameters materialized** (`model.materialize_aliased_params`). `etaM_S`, `etaX_S`, `K_q`, every
   `Mnet.tau` are created from `.expand()`ed tensors and `K0`/`K` share one tensor. Current PyTorch cannot run
   `optimizer.step()` or `load_state_dict` on them. Each parameter is replaced by a contiguous clone with identical
   values (same initialization). The authors' released checkpoints show these entries were independent after
   their training. Architecture and layer shapes are unchanged; parameter count 1,111,036. (D2 — pending OK.)
3. **Loss weight `Al1 = 5e-4`.** DICDNet's code has `default=54-4` (= 50), almost certainly a typo for 5e-4.
   (User decision D3.) All other loss terms/weights are exactly as in DICDNet's code.
4. **Optimizer/schedule from the FIND-Net paper, not DICDNet's code** (Adam β1=0.5, lr 2e-4, MultiStepLR,
   batch 16, 64×64 patches). The warmup length and final LR are not given in the paper (config, to confirm).
5. **Preprocessing = my pipeline.** Data come from `baselines/common/data.py` (HU clipped to [-1024, 3072],
   normalized to [0,1], my train/val split, LI converted from μ to HU by `aapm_dataset.py`). The adapter then
   maps to FIND-Net's 0–255 scale (`adapter.normalization`, default `scale255` = 255·x). The original used the
   HU window [-1500, 3500] → `native_window` reproduces that mapping on top of my clipping.
   Predictions are converted back exactly (`NormConverter.to_norm`) for validation.
6. **Masks** come from the AAPM `metal_mask` files via the dataset; non-metal mask = 1 − metal mask (as the
   original). The GT is not metal-painted (as the original; irrelevant for the loss since every term is
   multiplied by the non-metal mask, and the validation metrics exclude the dilated metal region).
7. **Validation / model selection.** Metrics are `baselines/common/eval_utils.py` (masked SSIM, masked HU RMSE,
   PSNR) on the final stage `ListX[-1]` converted back to [0,1]; `best.pt` = highest val masked SSIM.
   The paper used early stopping on validation loss: the original loss on the val set is logged as `val/loss`;
   early stopping is off by default (`train.early_stop_patience`).
8. **Determinism / resume.** Shuffling uses a per-epoch seeded permutation so a mid-epoch resume continues the
   same order (DICDNet used `shuffle=True` with a random seed). The flip augmentation in DataLoader workers is not
   bit-reproducible across a resume.
9. **Numerics.** fp32 (no AMP); TF32 allowed on Ampere (`train.tf32`, as in trainer_wandb_unclipped.py);
   `cudnn.benchmark = True` (as DICDNet). Weight decay is applied to all parameters (the paper does not mention
   parameter groups). The last partial batch is kept (DICDNet default).
10. **Not used:** `Dataset/dataset.py` (needs CatSim/`gecatsim` and raw files), `test_FINDNet.py`, pretrained
    weights (trained on AAPM CT-MAR with the authors' own split — likely overlapping my val split).

## Dependencies
torch (with `torch.fft`), numpy, scipy (`loadmat`, metrics), scikit-learn (split), PyYAML, wandb.
All are already used by the project. **Do not** `pip install -r third_party/findnet/requirements.txt` (pins
torch 2.4.1, numpy 1.24.4, scipy 1.10.1, wandb 0.15.11 — conflicts with the other models).

## Launch on Colab
1. Log in to W&B in the notebook (Colab secret `WANDB_API_KEY` → `wandb.login`); the training subprocess reuses it.
2. From the repo root:
   ```
   python baselines/findnet/train_findnet.py --config baselines/findnet/config.yaml \
       --ma_dir /content/data/MA/MA_image --gt_dir /content/data/GT/GT \
       --li_dir /content/data/LI/LI --mask_dir /content/data/mask/metal_mask \
       --ckpt_dir /content/drive/MyDrive/MAR_project/runs/findnet --run_name gf
   ```
3. **Resume:** after a disconnect, re-run the setup cells and the **same command**. It loads
   `<ckpt_dir>/last.pt` (model, optimizer, scheduler, RNG, epoch, position inside the epoch, best SSIM) and resumes
   the same W&B run. `--no_resume` starts over (overwrites `last.pt`).
4. Checks: `--max_iters N` stops at N total optimizer steps (saving `last.pt`), `--subset N` trains on the first
   N train slices, `--val_subset N` validates on the first N val slices, `--wandb_mode offline|disabled`,
   `--log_every_iters N`, `--val_every_epochs k` (validation every k epochs and always after the last epoch).
5. Colab notebook: `FINDNet_training.ipynb` (cell 8 = training; rerun after a disconnect).
   Real-data checks: `colab/verify_common.py`, `colab/findnet_checks.py`.

## Checkpoint directory contents
`last.pt` (every epoch + every `checkpoint.save_every_iters` steps + on `--max_iters` stop), `best.pt` (on val
SSIM improvement), `config_resolved.yaml`, `wandb/`. Writes are atomic (local temp → `.tmp` → rename).
State dict keys are those of the original `FINDNet` class.
