# Project rules (permanent)

- Never use or read the test set unless a task explicitly says so.
- Never modify anything in third_party/.
- Never change the numerical behavior of trainer_wandb.py, metrics.py, or aapm_dataset.py without asking.
- All baselines must build data via baselines/common/data.py and evaluate via baselines/common/eval_utils.py.
- Dataset arguments: hu_min=-1024, hu_max=3072, clip_hu=True, output_space="norm", region_policy=all,
  seed=42, val_size=0.1.
- No access to data or GPU locally: test with synthetic data; real-data checks go in colab/ scripts that
  the user runs.
- Ask before installing packages that change versions used by other models.
