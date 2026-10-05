# FIND-Net Baseline — Progress

Task spec: [TASK.md](TASK.md). Permanent rules: [CLAUDE.md](../../CLAUDE.md).

## Current stage
Stage 4 (partly): Colab notebook + colab/findnet_checks.py done and smoke-tested locally on fake data.
Waiting for the user to (a) confirm/adjust choices C1-C15 (currently the config defaults are used),
(b) push and run notebook cells 6-7 in Colab and paste both reports.

## Completed
### Stage 0 (2026-10-05)
- Created `baselines/findnet/TASK.md` (full task prompt, unchanged).
- Created `baselines/findnet/PROGRESS.md` (this file).
- Created `CLAUDE.md` at repo root (did not previously exist) with the permanent rules from TASK.md.
- No code written; no existing files modified.

### Stage 1 (2026-10-05)
- Cloned https://github.com/Farid-Tasharofi/FIND-Net into `third_party/findnet` at commit
  `505c326a29eaf3274ee87a614e680a0addd60286` (2025-09-02, "Update findnet.py"); recorded in
  `third_party/findnet/COMMIT.txt`. No files in third_party modified.
- Sources read: Model/findnet.py, Model/ProxNet.py, Model/ffc.py, Dataset/dataset.py, test_FINDNet.py,
  test.sh, README.md, requirements.txt, LICENSE; FIND-Net paper (arXiv 2508.10617v1, Sec. 2-3);
  DICDNet train_DICDNet.py (hongwang01/DICDNet @ 6498998a), since FIND-Net ships NO training script.
- CPU probes run (no files written): model builds/forwards/backwards on CPU; checkpoint key inspection.

#### Stage 1 report (key facts)
- **No training code in the repo** (test only). Recipe = paper Sec. 3.1 + DICDNet train script.
- **Model call:** `FINDNet(args)(CT_ma, LIct, Mask)`; all (B,1,H,W) float32; CT_ma/LIct in [0,255];
  `Mask` = NON-metal mask (1 - metal), {0,1}. Args: S=10, num_M=32, num_Q=32, T=3, etaM=1, etaX=5.
- **Returns** `(X0, ListX, ListA)`: X0 = LI-based init; ListX = 11 tensors (S+1), ListX[-1] = final;
  ListA = 10 artifact maps. Output not clamped.
- **Original preprocessing** (Dataset/dataset.py): HU -> (HU+1500)/5000 (i.e. window [-1500,3500]) ->
  bilinear resize to 512 (no-op) -> clip [0,1] -> x255. Mask from AAPM `metalonlymask` file, non_Mask =
  1 - Mask. GT not metal-painted (irrelevant: all loss terms are multiplied by non-metal mask).
  Train aug: random hflip p=0.5, random vflip p=0.5. Crop = full image when patchSize=512.
- **Loss** (DICDNet code; FIND-Net paper Eq. 6 same form, weights not given): with I=non-metal mask,
  L2X = sum_{j<S} 0.1*MSE(I*X_j, I*X) + MSE(I*X_final, I*X) + 0.1*MSE(I*X0, I*X);
  L1X = same with SUM |.| ; L1A = sum_{j<S} 0.1*SUM|I*A_j - I*(Y-X)| + SUM|I*A_last - I*(Y-X)|;
  loss = Xl2*L2X + Xl1*L1X + Al1*L1A with Xl2=1, Xl1=5e-4, **Al1 = `54-4` = 50 literally in code**
  (probable typo for 5e-4).
- **Optimizer/schedule (paper, FIND-Net):** AdamW(0.9,0.999), lr 1e-4, wd 1e-5, linear warmup + cosine
  annealing (warmup length not given), up to 200 epochs, batch 6, early stopping on val loss, A100 80GB,
  10 stages. (DICDNet code differs: Adam(0.5,0.999) lr 2e-4, MultiStepLR, batch 16, patch 64.)
- **Variant switch:** repo default `Gaussian_filter = False` in ffc.py => FIND-Net (No-GF). Full
  FIND-Net needs `Gaussian_filter = True` (README says edit the file). Params: GF 1,111,036; No-GF
  1,104,700. No-GF FourierUnit calls `rfft2(x, dim=2)` (1-D FFT over H) then `irfft2` over (H,W) —
  quirk; GF variant uses proper 2-D rfftn.
- **Aliased parameters bug:** etaM_S, etaX_S, K_q, Mnet.tau are nn.Parameters built from `.expand()`ed
  tensors, and K0/K wrap the same module-level tensor (shared storage, also shared across instances).
  With torch 2.7: AdamW.step() FAILS and load_state_dict FAILS on the unmodified model. Authors'
  pretrained checkpoints show these were independent during their training (10 distinct etaX values,
  K != K0, 288 distinct K_q values, 32 distinct tau). Cloning each param after construction (same init
  values, own storage) fixes step() and load_state_dict (verified on CPU).
- **Import quirks:** findnet.py loads `utils/init_kernel.mat` relative to CWD at import and uses
  `from Model.ProxNet import ...` => wrapper must put third_party/findnet on sys.path and chdir during
  import. Dataset/dataset.py needs `gecatsim` (CatSim) — not needed by us.
- **Dependencies:** model needs only torch (torch.fft) + scipy (loadmat). Works with local torch
  2.7.1 / scipy 1.13.1 / numpy 1.26.4. Repo's requirements.txt pins torch 2.4.1, numpy 1.24.4,
  scipy 1.10.1, wandb 0.15.11, torchvision 0.8.2+cu110 (inconsistent) — conflicts with user's
  requirements (numpy 2.3.1, scipy 1.16.0, torch 2.5/2.7.1); **do not install it**. No installs needed.
- **License:** Apache-2.0 (FFC parts also Apache-2.0, DICDNet-derived code attributed).
- **Pretrained weights:** pretrained_models/{FINDNet, FINDNet_no_GF, DICDNet}/checkpoint.pt (plain
  state_dicts, ~5-6 MB) trained on AAPM CT-MAR (authors' own split: 5500 train / 700 val / 700 test).
  Likely overlaps user's val split -> not usable for a fair baseline; user trains from scratch anyway.
- **CUDA:** model is device-agnostic, runs fwd+bwd on CPU (GF variant tested at 64x64). Only
  test_FINDNet.py hardcodes `.cuda()`.

### Stage 2 (2026-10-05)
Files created:
- `baselines/__init__.py`, `baselines/common/__init__.py`
- `baselines/common/data.py` — `build_datasets(ma_dir, gt_dir, li_dir=None, mask_dir=None)` with the
  fixed CLAUDE.md dataset args; `unpack_batch(batch, has_mask, has_li)` -> dict ma/gt/mask/li.
- `baselines/common/eval_utils.py` — `evaluate_predictions(pred, gt, metal_mask, hu_min, hu_max)` ->
  per-image lists {ssim, rmse, psnr}; `summarize_metrics(per_image)` -> mean + std (ddof=1).
- `baselines/common/checkpoint.py` — `save_checkpoint` / `load_checkpoint` / `find_resume_checkpoint` /
  `is_improvement`; model, optimizer, scheduler, scaler, RNG (python/numpy/torch/all CUDA), epoch,
  global_step, best_metric, wandb_run_id, extra. Atomic: local temp -> `<dest>.tmp` -> size check ->
  `os.replace`. LAST_NAME=last.pt, BEST_NAME=best.pt (policy applied by training scripts).
- `baselines/common/logging_utils.py` — `wandb_login(secret_name="WANDB_API_KEY")` (Colab secret ->
  env var, key never printed), `make_run_name(method, name)` (prefix e.g. `findnet_`),
  `init_wandb(method, project, ..., run_id=...)` (resume="allow" when run_id given; mode from arg or
  WANDB_MODE).
- `baselines/common/tests/original_metric_code.py` — frozen verbatim copy of the pre-refactor metric
  code (git cd22fab), `original_split_code.py` — frozen copy of the trainer's dataset construction.
- `baselines/common/tests/verify_local_stage2.py` — local checks a/b/c.
- `colab/verify_common.py` — real-data split + metric MATCH/MISMATCH report (LI and MA as stand-in
  predictions, no checkpoint needed).
Files changed:
- `trainer_wandb_unclipped.py` — import of eval_utils; inline per-image metric loop + mean/std
  replaced by `evaluate_predictions` + `summarize_metrics`; `metal_fill_norm` moved into eval_utils.
  Nothing else touched. Run it from the repo root (it imports `baselines.common`).
- `third_party/findnet/.git` removed (decision D4); probe `__pycache__` removed.
Local verification (CPU, synthetic) — 13/13 PASS:
- a0 frozen reference == git cd22fab trainer code (55 lines); a1 per-image SSIM/RMSE/PSNR exactly
  equal (10 imgs 512x512 with random metal blobs, 2 without mask, values outside [0,1]; max diff 0);
  a2 mean/std equal; a3 trainer calls eval_utils.
- b1 metadata, b2 model/optim/sched states, b3 RNG draws, b4 continued training bit-identical after
  resume, b5 simulated disconnect during copy leaves previous last.pt intact + no temp files,
  b6 improvement rule.
- c1 fake dataset (real naming `training_{body|head}_{metalart|nometal|li|metalonlymask}_img{ID}_HxWx1.npy`):
  data.py split == original trainer split (same order), c2 disjoint, c3 identical samples.
  Note: split depends on which dirs are passed (key intersection) — a missing LI file changes it.
- colab/verify_common.py dry-run on a fake dataset: ALL MATCH.
- logging_utils tested with a stub wandb module (wandb is NOT installed locally).

### Stage 3 (2026-10-05) — user asked to finish Stage 3 before the Colab notebook
Files created in `baselines/findnet/`:
- `__init__.py`
- `model_wrapper.py` — `build_findnet(...)` imports the original FINDNet (GF flag flipped in memory,
  third_party untouched), optional parameter materialization (D2), `final_output(out) = ListX[-1]`,
  `findnet_commit()`. Returns the original class (state_dict keys == released checkpoints).
- `dataset_adapter.py` — `NormConverter` (`scale255` | `native_window`, exact inverse `to_norm`),
  `FindNetAdapter` (dict: ma, li, gt in FIND-Net units; nonmetal = 1 - metal mask; metal_mask; gt_norm),
  original flips (train only), `build_findnet_datasets(..., subset, val_subset)`.
- `config.yaml` — all hyperparameters with source tags [orig]/[paper]/[dicd]/[user]/[mine](?).
- `train_findnet.py` — original loss (`findnet_loss`), AdamW + warmup/cosine per iteration, per-epoch
  validation via eval_utils on final stage converted to [0,1] (+ val/loss), best.pt on val SSIM,
  last.pt every epoch / every N iters / on --max_iters stop, auto-resume incl. mid-epoch position
  (per-epoch seeded permutation), W&B via logging_utils (define_metric: train/* vs global_step,
  val/* vs epoch, so resumed runs never hit W&B's monotonic-step limit), `--wandb_mode disabled`
  path that never imports wandb. CLI: --config, data dirs, --ckpt_dir, --max_iters, --subset,
  --val_subset, --epochs, --batch_size, --num_workers, --save_every_iters, --seed, --run_name,
  --project, --entity, --wandb_mode, --no_resume.
- `IMPLEMENTATION_NOTES.md` — commit, 10 deviations with reasons, dependencies, launch/resume.
Checks run: py_compile OK, config.yaml parses, `--help` OK, no Pylance errors. Nothing trained/run
(per Stage 3 rule). Smoke tests are Stage 4.
Security note: FINDNet_training.ipynb (untracked) contains a hardcoded W&B API key in plain text;
not in git history. Told user; notebook will switch to the Colab secret.

### Stage 4 (partial, 2026-10-05) — user asked for the Colab notebook without answering C1-C15
- `train_findnet.py`: added `--log_every_iters`, `--val_every_epochs` (config `validation.every_epochs`,
  default 1; validation always after the last epoch; early-stop patience counts validation rounds);
  `train/epoch_loss` now logged every epoch.
- `colab/findnet_checks.py` (new): one report with PASS/FAIL for 1) forward pass on 2 real train slices,
  2) overfit --subset 8 (300 iters, no val; PASS if last logged loss < 0.5 x first), 3) resume
  (stop@30, restart, expects resume@30, first log @40, stop@50, same W&B run id), 4) timing (300 iters
  after 20 warm-up; s/iter, peak GPU mem, val s/img, total hours for the config budget, epochs per
  23 h session). Checks 2-3 run train_findnet.py as a subprocess in --work_dir (local), W&B offline.
- `FINDNet_training.ipynb` rewritten (old swin cells and the plaintext W&B key removed): Settings,
  1 GCP auth + project ctmar-research, 2 skip-if-present parallel copy (gcloud storage cp) + unzip +
  file counts + free disk, 3 Drive mount, 4 git clone/pull + versions (no installs), 5 W&B login from
  Colab secret WANDB_API_KEY, 6 verify_common.py, 7 findnet_checks.py, 8 training (auto-resume;
  rerunnable), 9 checkpoint status.
- Local smoke test (fake 64x64 dataset; note: torch here sees a CUDA GPU, RTX 4080 Laptop, so it ran
  on GPU): forward PASS (shapes, 11/10 stage outputs, finite, round-trip 6e-8, 1,111,036 params),
  resume PASS (stop@30 -> resumed@30 -> first log @40 -> stop@50), timing PASS. Overfit FAIL on pure
  noise data (expected, nothing learnable: ratio 0.93); on a structured fake dataset (MA = GT +
  streaks + metal, LI ~ GT) overfit PASS: loss 171.3 -> 34.8 in 150 iters (ratio 0.20).

### Memory finding (2026-10-05) — BLOCKS full training with current config
- Measured on local RTX 4080 (12 GiB), fp32, forward + original loss + backward:
  128 px: 1.50 / 2.63 / 4.78 GiB for batch 1/2/4 (per image 1.14 GiB, fixed 0.36);
  256 px: 5.89 / 10.37 GiB for batch 1/2 (per image 4.49 GiB, fixed 1.40); 512 px batch 1: OOM.
  ~4x per doubling of side => 512x512 ~18 GiB per image => paper batch 6 needs ~110 GiB > 80 GB A100.
  The paper's "batch 6 on A100 80GB" is therefore only consistent with training on smaller patches
  (training patch size is NOT stated in the paper; the original Dataset supports random crops via
  patchSize; DICDNet trained on 64x64 patches) or a different memory regime.
- colab/findnet_checks.py: new check 2 "gpu memory" measures batch 1/2 at the real size, reports
  per-image/fixed memory and the largest batch that fits; checks 3-5 run with
  min(config batch, max fitting batch). Tested locally on fake 256 px data (max batch 2 on 12 GiB,
  resume PASS with batch 2 incl. epoch-end validation after a mid-epoch resume).
- Notebook cell 8 now warns not to start training before this decision.
- DECISION NEEDED (C3/C4): (a) full 512x512 with the largest batch that fits (likely 3-4);
  (b) random patches (e.g. 256x256) with batch 6, validation on full 512x512 (needs a crop option in
  the adapter); (c) 512x512 with gradient accumulation to an effective batch 6 (BatchNorm statistics
  per micro-batch differ from a true batch 6).
- Code not yet committed/pushed (all new files untracked; trainer_wandb_unclipped.py modified).

## Decisions confirmed by me
(Values given directly in TASK.md, recorded 2026-10-05.)
- Dataset arguments (from CLAUDE.md rules in TASK.md): hu_min=-1024, hu_max=3072, clip_hu=True,
  output_space="norm", region_policy=all, seed=42, val_size=0.1.
- Data bucket: gs://ctmar-dataset-uae/ with subfolders GT, LI, MA_image, metal_mask.
- Folder layout: baselines/common/, baselines/findnet/, colab/, third_party/findnet/.
- FIND-Net must NOT be added to trainer_wandb.py.
- Colab training is run by the user; agent delivers the notebook and trainer code only.
- Test set is not used in this task.

Answers to Stage 0 open questions (2026-10-05):
- Q1 Reference trainer: **trainer_wandb_unclipped.py** is used everywhere TASK.md mentions
  trainer_wandb.py (metric equivalence reference, original split code path, and the file refactored to
  call eval_utils.py).
- Q2 GCP project ID: **ctmar-research**.
- Q3 Checkpoint dir: a **Google Drive path** (exact path still to be set; config entry).
- Q4 W&B login: via API key stored as a **Colab secret** (read with google.colab.userdata), never
  hardcoded.
- Q5 Code transfer: git repo **https://github.com/BARUD77/CT-MAR-Training**.
- Q6 last.pt every N iterations: user not sure -> still open (see below).
- Q7 No existing checkpoint: user trains **from scratch**.
- Q8 Do a **fresh clone** into third_party/findnet; leave existing FIND-NET/ folder untouched.
- Q9 Test set is **not in the GCP bucket**; nothing to exclude in the copy cell.

Answers to Stage 1 decisions (2026-10-05):
- D1 Variant: **full FIND-Net with Gaussian filtering ON**, enabled by the wrapper loading
  third_party/findnet/Model/ffc.py with `Gaussian_filter = True` in memory (file on disk untouched).
- D2 Aliased params: user asked for advice. Agent advice: clone each aliased parameter after
  construction (same init values, own storage) — required to train/resume at all, matches the
  authors' checkpoints. **Pending explicit OK** (only affects Stage 3).
- D3 Loss weight **Al1 = 5e-4** (not the literal 50 from DICDNet's `54-4`). Xl2=1, Xl1=5e-4.
- D4 Third-party code: **option (b)** — nested .git removed (2026-10-05) and FIND-Net files vendored
  in the main repo; pretrained_models/*.pt are excluded by the existing `.gitignore` rule `*.pt`.
- D5 Still-open defaults (user did not specify; used as configurable defaults until told otherwise):
  last.pt every N=1000 iters, Drive dir /content/drive/MyDrive/ctmar_checkpoints/findnet, W&B secret
  name WANDB_API_KEY, verify_common.py uses LI and MA as stand-in predictions.

## Open questions
1. **Checkpoint frequency [N]** (user unsure). Proposal: N=1000 iterations, finalized after the Colab
   timing check (target: last.pt roughly every 10-15 min).
2. **Exact Drive checkpoint path.** Proposal: /content/drive/MyDrive/ctmar_checkpoints/findnet
   (config entry + CLI override).
3. **W&B project name** (placeholder [PROJECT_NAME]); W&B entity if any; Colab secret name for the key
   (proposal: WANDB_API_KEY).
4. **verify_common.py metric check without a checkpoint** (training from scratch). Proposal: compare the
   original metric code vs eval_utils.py using the LI image and the MA image as "predictions" on the real
   val split (no model needed).
5. **CLAUDE.md wording** — rule 3 names trainer_wandb.py; given Q1, propose adding
   trainer_wandb_unclipped.py to that rule (needs user OK, since CLAUDE.md now exists). User asked
   what the rule means (explained 2026-10-05); addition still awaiting OK.
6. **wandb not installed locally.** Stage 4 resume test needs W&B offline mode. Install `wandb` into
   the local anaconda env (not used by other models' numerics, but pip may bump shared deps such as
   protobuf)? Alternative: run the local resume test with W&B disabled/stubbed.
7. D2 (aliased-param clone) needs explicit OK before Stage 3.

### Stage 3 choices to confirm (current default in config.yaml)
- C1 D2 parameter materialization: ON (required to train at all).
- C2 Normalization conversion: `scale255` (255 * my_norm) vs `native_window` (FIND-Net's
  [-1500,3500] window applied after my clip).
- C3 Image size: full 512x512, no patches / no patch sampling (orig test + paper).
- C4 Batch size 6 (paper), val batch 6.
- C5 Budget: paper "up to 200 epochs" -> set after Colab timing; need number of train slices.
- C6 Warmup 5 epochs, cosine to 0 (min_lr_ratio 0).
- C7 Early stopping: off (best.pt by val SSIM) vs patience (on val SSIM or val loss).
- C8 Precision fp32, TF32 on, cudnn.benchmark on.
- C9 Metal NOT painted onto GT (original; no effect on loss/metrics).
- C10 Flip augmentation ON (original).
- C11 Weight decay on all params.
- C12 Validate every epoch on the full val split (FIND-Net inference ~1 s/img per paper).
- C13 ckpt_dir /content/drive/MyDrive/MAR_project/runs/findnet; last.pt every 1000 iters.
- C14 W&B project `ct-mar`, run name `findnet_gf`.
- C15 Seed 42 for init + shuffling; num_workers 8.

## Next steps
- User: confirm C1-C15 (and open items 5-6); record answers here. Until then the config.yaml
  defaults are what the notebook trains with.
- User: commit + push (baselines/, colab/, third_party/, CLAUDE.md, trainer_wandb_unclipped.py;
  FINDNet_training.ipynb now has no secrets), run notebook cells Settings, 1-7, paste both reports.
- Remaining Stage 4 local item: W&B offline-mode resume test (wandb not installed locally; the Colab
  check 3 covers it with W&B offline).

## Colab results
(Paste Colab reports here.)
