I want to add FIND-Net (Tasharofi et al., MICCAI 2025) as a baseline in my CT
metal artifact reduction project. This task covers TRAINING AND VALIDATION
ONLY. Do not write test-set inference or touch the test set.

IMPORTANT ENVIRONMENT CONSTRAINT
You do NOT have access to my dataset or a GPU. The data is in a Google Cloud
Storage bucket, and training runs on Google Colab Pro+ (single 80 GB GPU,
sessions capped at 24 hours). Therefore:
- Test everything you can locally on CPU with SYNTHETIC data (random tensors
  or tiny fake files you create).
- For anything that needs real data or a GPU, write scripts and a Colab
  notebook that I will run. They must print a clear report I can paste back
  to you.
- Never hardcode data paths. All paths must come from arguments or the
  config file.
Work in stages and stop at each [STOP].

PROJECT CONTEXT
- Main trainer: trainer_wandb_unclipped.py. Dataset class: CTMetalArtifactDataset in
  aapm_dataset.py. Metrics: metrics.py.
- Dataset: AAPM CT-MAR, 512x512 slices, paired MA / GT, with LI images and
  metal masks, stored at gs://ctmar-dataset-uae/ in GCP project
  [PROJECT_ID]. Subfolders: GT, LI, MA_image, metal_mask.
- My reported models use: hu_min=[HU_MIN], hu_max=[HU_MAX], clip_hu=True,
  output_space="norm", region_policy=[REGION_POLICY], seed=42, val_size=0.1.
  FIND-Net must use exactly these so the train/val split and preprocessing
  are identical.
- Validation in trainer_wandb_unclipped.py: masked SSIM (metal dilated 2 px, filled to
  100 HU, data_range=1), masked RMSE in HU (metal dilated 1 px), PSNR =
  20*log10((hu_max - hu_min) / RMSE_HU). Best checkpoint selected by
  validation masked SSIM.
- Checkpoints must be saved to [CHECKPOINT_DIR, e.g. a mounted Google Drive
  folder
- Logging: Weights & Biases, project [PROJECT_NAME].
- I move code to Colab via [GIT REPO URL / Google Drive sync].

FOLDER STRUCTURE TO CREATE
baselines/
  common/        shared code used by ALL baselines (this and future ones)
  findnet/       FIND-Net-specific code only
colab/           notebooks and check scripts I run in Colab
Do NOT add FIND-Net to trainer_wandb.py.

STAGE 0: SET UP TASK FILES (so work survives across sessions)
1. Save this entire prompt, unchanged, to baselines/findnet/TASK.md.
2. Create baselines/findnet/PROGRESS.md with sections:
   - Current stage
   - Completed (what was done, files created/changed)
   - Decisions confirmed by me (exact values and dates)
   - Open questions
   - Next steps
   - Colab results (where I paste reports)
3. Create or update CLAUDE.md at the repo root with these permanent rules.
   If CLAUDE.md already exists, show me the proposed additions and ask
   before changing it:
   - Never use or read the test set unless a task explicitly says so.
   - Never modify anything in third_party/.
   - Never change the numerical behavior of trainer_wandb.py, metrics.py,
     or aapm_dataset.py without asking.
   - All baselines must build data via baselines/common/data.py and
     evaluate via baselines/common/eval_utils.py.
   - Dataset arguments: hu_min=-1024, hu_max=3072, clip_hu=True,
     output_space="norm", region_policy=all, seed=42,
     val_size=0.1.
   - No access to data or GPU locally: test with synthetic data; real-data
     checks go in colab/ scripts that I run.
   - Ask before installing packages that change versions used by other
     models.
WORKFLOW RULES FOR EVERY STAGE:
- At the start of each stage, re-read TASK.md, PROGRESS.md, and CLAUDE.md.
- At every [STOP], update PROGRESS.md before reporting to me.
- When I confirm a decision, record it in PROGRESS.md immediately, with the
  exact value.
[STOP] Show me the files created.

STAGE 1: GET THE ORIGINAL CODE AND REPORT
1. Clone https://github.com/Farid-Tasharofi/FIND-Net into
   third_party/findnet. Record the commit hash in
   third_party/findnet/COMMIT.txt.
2. Read the code and report:
   - Exact model inputs (MA image, LI image, non-metal mask, anything else),
     their shapes, value ranges, and argument order.
   - What the model returns (final image only, or per-stage outputs).
   - Original preprocessing and normalization.
   - Training recipe: loss (including any stage-wise weighting), optimizer,
     learning rate and schedule, batch size, patch size or full images,
     epochs/iterations, number of stages, augmentation, precision.
   - Dependencies and versions, and any conflicts with my requirements.
   - License, and whether pretrained weights exist and what data they used.
   - Whether any part requires CUDA (so I know what can run on CPU).
[STOP] Update PROGRESS.md and show me the report before writing any code.

STAGE 2: SHARED COMMON MODULE
Create baselines/common/ with:
- data.py: a function that builds train and val CTMetalArtifactDataset
  instances with exactly the arguments listed above, taking directory paths
  as arguments. All baselines will use this.
- eval_utils.py: move the validation metric logic from trainer_wandb.py
  (masked SSIM, masked HU RMSE, PSNR, per-image loop, mean and std) into a
  function like evaluate_predictions(pred, gt, metal_mask, hu_min, hu_max)
  returning per-image metric lists. Update trainer_wandb.py to call it.
- checkpoint.py: save and resume model, optimizer, scheduler, GradScaler (if
  used), RNG states (python, numpy, torch, cuda), epoch, global step, best
  metric, and W&B run id. Save "last.pt" every epoch and every [N]
  iterations, and "best.pt" when val masked SSIM improves. Write to a local
  temp file first, then copy/rename to the checkpoint directory, so a
  disconnect can't leave a corrupted checkpoint.
- logging_utils.py: W&B init helper with run names prefixed by the method
  name (e.g. "findnet_...") and resume support.

Local verification (you do this):
a) Metric equivalence: generate synthetic predictions, targets, and metal
   masks (random 512x512 images with a few random metal blobs), run the
   ORIGINAL metric code from trainer_wandb.py and the refactored
   eval_utils.py, and confirm per-image values match exactly.
b) Checkpoint round trip: save and reload with a tiny dummy model and
   optimizer; confirm states and RNG restore correctly.
c) If feasible, create a tiny fake dataset with the same folder structure
   and file naming my dataset uses (check aapm_dataset.py for the expected
   naming) and confirm common/data.py produces the same split as the
   original code path.

Colab verification (you write it, I run it):
Create colab/verify_common.py that, on my real data, prints:
- split sizes and the first 5 and last 5 file names of train and val from
  common/data.py AND from the original trainer_wandb.py code path, with a
  clear MATCH / MISMATCH line;
- validation metrics for an existing checkpoint ([PATH/TO/best.pt], model
  [MODEL_NAME], config [CONFIG_PATH]) computed with the original metric code
  and with eval_utils.py, with a clear MATCH / MISMATCH line.
[STOP] Update PROGRESS.md and report your local verification results.

STAGE 3: FIND-NET TRAINING SCRIPT
Create in baselines/findnet/:
- model_wrapper.py: imports the model from third_party/findnet without
  changing its architecture, stage count, kernel sizes, or any layer.
- dataset_adapter.py: wraps common/data.py and converts my samples into
  FIND-Net's expected inputs. Derive the non-metal mask as 1 - metal mask
  from my AAPM masks; never create masks by thresholding. If FIND-Net's
  native normalization differs from mine, convert inside the adapter, and
  provide a function converting predictions back to my [0, 1]
  normalization for validation.
- config.yaml: all hyperparameters, with comments marking which values come
  from the original code and which were set for my data. Data paths are
  config entries, not hardcoded.
- train_findnet.py: trains on my training split and validates on my val
  split every epoch using:
    * FIND-Net's ORIGINAL loss (including stage-wise terms), optimizer,
      learning rate schedule, and hyperparameters;
    * common/eval_utils.py on the FINAL stage output, after converting it
      back to my normalization;
    * common/checkpoint.py with automatic resume from the latest checkpoint;
    * common/logging_utils.py for W&B (train loss, val PSNR/SSIM/RMSE mean
      and std, learning rate, epoch).
  Support a --max_iters argument and a --subset N argument (train on only N
  slices) for the checks in Stage 4.
- IMPLEMENTATION_NOTES.md: commit hash, every deviation from the original
  code or settings and why, dependencies, and how to launch and resume
  training on Colab.

[STOP] Update PROGRESS.md. Before running anything, list every choice the
original code does not specify for my setting (e.g., 512x512 vs original
image size, patch sampling, batch size, epochs/iterations, normalization
conversion, mixed precision, whether metal is painted onto the GT target)
and ask me to confirm each one. Record my answers in PROGRESS.md.

STAGE 4: LOCAL SMOKE TESTS AND COLAB NOTEBOOK
Local smoke tests (you do this, on CPU with synthetic data):
1. Forward and backward pass through the wrapped model with synthetic
   inputs in the expected format: check shapes, value ranges, no NaNs, and
   that the conversion back to [0, 1] works. Use a small image size if 512
   is too slow on CPU, plus one 512x512 forward pass to confirm it runs.
2. Run train_findnet.py end to end on a tiny synthetic or fake-file dataset
   for a few iterations, including one validation pass and checkpoint save.
3. Resume test: stop after a few iterations, restart, and confirm it
   resumes at the correct step (use W&B offline mode for this test).

Colab notebook (you write it, I run it): create colab/findnet_train.ipynb
with clearly labeled cells:
1. Authenticate to Google Cloud (google.colab.auth) and set project
   [PROJECT_ID].
2. Copy the dataset from the gs:// bucket to the runtime's LOCAL disk using
   parallel copying (gcloud storage cp or gsutil -m cp), skipping the copy
   if the files are already present. Print the file count per folder and
   free disk space. Do NOT copy the test set.
3. Mount Google Drive if the checkpoint directory is on Drive.
4. Get the code (git clone/pull of [REPO], or from Drive) and install
   dependencies, printing installed versions of torch and key packages.
5. Log in to W&B.
6. Run colab/verify_common.py (Stage 2 checks).
7. Run colab/findnet_checks.py, which you also write, and which prints one
   report covering:
   - forward pass on 2 real training slices (shapes, value ranges, no NaNs);
   - overfit check: --subset 8, a few hundred iterations, with loss at
     start and end;
   - resume check: run ~50 iterations, restart, confirm correct resume;
   - timing: 300 iterations excluding the first 20 as warm-up, seconds per
     iteration, peak GPU memory, and estimated total time for the agreed
     budget.
   Each check ends with a clear PASS / FAIL line.
8. Launch full training (with automatic resume), with a note that I can
   rerun this cell after a disconnect.
[STOP] Update PROGRESS.md. Report local smoke test results and give me a
short checklist of what to run in Colab and what output to paste back.

GENERAL RULES
- Never use the test set in this task.
- If anything is ambiguous, or you'd need to change preprocessing,
  architecture, or the training recipe, stop and ask me.
- Don't install packages that change versions used by my other models
  without asking.
- Don't change the numerical behavior of trainer_wandb.py beyond the
  eval_utils refactor. for the colab training part just give me the notebook and I will run it myself so just focus on completing the trainer code
