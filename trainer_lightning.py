import argparse, os, json, torch, yaml, sys
from types import SimpleNamespace

# ---------------- your imports ----------------
from dataset import CTMetalArtifactDataset
from models.swin_unet.vision_transformer import SwinUnet
from models.unet import UnetGenerator
from metrics import compute_SSIM, compute_PSNR

# ---------------- lightning / wandb ----------------
import pytorch_lightning as pl
from pytorch_lightning import LightningModule, Trainer, seed_everything
from pytorch_lightning.loggers import WandbLogger
from pytorch_lightning.strategies import DDPStrategy
from pytorch_lightning.callbacks import ModelCheckpoint

# ======================= utilities (unchanged) =======================
def dict_to_namespace(d):
    if isinstance(d, dict):
        return SimpleNamespace(**{k: dict_to_namespace(v) for k, v in d.items()})
    return d

def load_yaml(path):
    with open(path, "r") as f:
        return yaml.safe_load(f)

def load_config_namespace(path):
    return dict_to_namespace(load_yaml(path))

def parse_json_or_none(s):
    if s is None:
        return None
    try:
        return json.loads(s)
    except json.JSONDecodeError as e:
        raise ValueError(f"--model_kwargs must be valid JSON. Error: {e}\nGot: {s}")

def build_model(name: str, in_ch: int, num_classes: int, device: torch.device,
                model_cfg_path: str | None = None, model_kwargs: dict | None = None):
    model_kwargs = model_kwargs or {}
    name = name.lower()

    if name in ("unet", "vanilla_unet", "u-net"):
        m = UnetGenerator(in_channels=in_ch, **model_kwargs)
        return m.to(device)

    elif name in ("swinunet", "swin_unet", "swin-u"):
        if not model_cfg_path:
            raise ValueError("SwinUnet requires --model_cfg <path to YAML>.")
        cfg_ns = load_config_namespace(model_cfg_path)
        try: cfg_ns.MODEL.SWIN.IN_CHANS = in_ch
        except AttributeError: pass
        m = SwinUnet(config=cfg_ns, **model_kwargs)
        return m.to(device)

    else:
        raise ValueError(f"Unknown model: {name}. Supported: unet, swinunet")

def param_groups(model, wd, nowd_names=('bias', 'bn', 'norm')):
    decay, no_decay = [], []
    for n, p in model.named_parameters():
        if not p.requires_grad:
            continue
        if any(k in n.lower() for k in nowd_names):
            no_decay.append(p)
        else:
            decay.append(p)
    return [{'params': decay, 'weight_decay': wd},
            {'params': no_decay, 'weight_decay': 0.0}]

# ======================= LightningModule =======================
class LitCTMAR(LightningModule):
    def __init__(self, hparams):
        super().__init__()
        self.save_hyperparameters(hparams)
        self.in_ch = 2 if self.hparams.input_mode == 'ma_li' else 1

        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = build_model(
            name=self.hparams.model,
            in_ch=self.in_ch,
            num_classes=self.hparams.num_classes,
            device=device,
            model_cfg_path=self.hparams.model_cfg,
            model_kwargs=parse_json_or_none(self.hparams.model_kwargs),
        )

        self.loss_fn = torch.nn.L1Loss()

        # dataset constants
        self.HU_MIN, self.HU_MAX = -1024.0, 3072.0

        self.train_ds = None
        self.val_ds   = None

    # --- data ---
    def setup(self, stage=None):
        # create datasets once per process
        self.train_ds = CTMetalArtifactDataset(
            ma_dir=self.hparams.ma_dir, gt_dir=self.hparams.gt_dir, mask_dir=None,
            split='train', hu_min=self.HU_MIN, hu_max=self.HU_MAX,
            head_policy="exclude", seed=42, val_size=0.1
        )
        self.val_ds = CTMetalArtifactDataset(
            ma_dir=self.hparams.ma_dir, gt_dir=self.hparams.gt_dir, mask_dir=None,
            split='val', hu_min=self.HU_MIN, hu_max=self.HU_MAX,
            head_policy="exclude", seed=42, val_size=0.1
        )

    def train_dataloader(self):
        pin = (self.device.type == 'cuda')
        return torch.utils.data.DataLoader(
            self.train_ds, batch_size=self.hparams.batch_size, shuffle=True,
            num_workers=4, pin_memory=pin
        )

    def val_dataloader(self):
        pin = (self.device.type == 'cuda')
        return torch.utils.data.DataLoader(
            self.val_ds, batch_size=self.hparams.batch_size, shuffle=False,
            num_workers=4, pin_memory=pin
        )

    # --- optimization ---
    def configure_optimizers(self):
        opt = torch.optim.AdamW(
            param_groups(self.model, wd=1e-4),
            lr=self.hparams.learning_rate, betas=(0.9, 0.999), eps=1e-8
        )
        return opt

    # --- steps ---
    def training_step(self, batch, batch_idx):
        x, y = batch
        pred = self.model(x)
        loss = self.loss_fn(pred, y)
        self.log("train/loss", loss, on_step=True, on_epoch=True, prog_bar=True, sync_dist=True)
        return loss

    def validation_step(self, batch, batch_idx):
        x, y = batch
        pred = self.model(x)

        pred_eval = torch.clamp(pred, 0, 1)
        gt_eval   = torch.clamp(y,   0, 1)

        # compute per-sample and average in this batch
        ssim = 0.0; psnr = 0.0; count = pred_eval.size(0)
        for i in range(count):
            ssim += compute_SSIM(pred_eval[i, 0], gt_eval[i, 0], data_range=1.0)
            psnr += compute_PSNR(pred_eval[i, 0], gt_eval[i, 0], data_range=1.0)

        ssim /= max(1, count)
        psnr /= max(1, count)

        # Lightning will reduce across GPUs when sync_dist=True
        self.log("val/ssim", ssim, prog_bar=True, sync_dist=True)
        self.log("val/psnr", psnr, prog_bar=False, sync_dist=True)

# ======================= CLI / entry =======================
def parse_args():
    p = argparse.ArgumentParser("Lightning trainer for CT-MAR")
    # model config
    p.add_argument('--model', type=str, choices=['unet', 'swinunet'], required=True)
    p.add_argument('--model_cfg', type=str, default=None)
    p.add_argument('--model_kwargs', type=str, default=None)

    # data
    p.add_argument('--ma_dir', type=str, required=True)
    p.add_argument('--li_dir', type=str, required=False)  # kept for compatibility
    p.add_argument('--gt_dir', type=str, required=True)
    p.add_argument('--input_mode', type=str, choices=['ma_li', 'ma'], default='ma')
    p.add_argument('--num_classes', type=int, default=1)

    # optimization
    p.add_argument('--batch_size', type=int, default=4)
    p.add_argument('--epochs', type=int, default=300)
    p.add_argument('--learning_rate', type=float, default=1e-4)

    # logging
    p.add_argument('--log_dir', type=str, default='./runs')
    p.add_argument('--run_name', type=str, default=None)
    p.add_argument('--project', type=str, default='ct-mar')
    p.add_argument('--entity', type=str, default=None)

    # trainer knobs (override if you want)
    p.add_argument('--devices', type=int, default=4)
    p.add_argument('--accumulate_grad_batches', type=int, default=2)
    p.add_argument('--precision', type=str, default="16-mixed")

    return p.parse_args()

def main():
    args = parse_args()
    os.makedirs(args.log_dir, exist_ok=True)
    seed_everything(42, workers=True)

    # --- LightningModule ---
    lit = LitCTMAR(args)

    # --- logger & checkpoints ---
    wb = WandbLogger(project=args.project, entity=args.entity, name=args.run_name,
                     save_dir=args.log_dir, log_model=True)
    ckpt_cb = ModelCheckpoint(
        dirpath=args.log_dir,
        filename=f"{args.model}" + "-{epoch:03d}-{val_ssim:.4f}",
        monitor="val/ssim",
        mode="max",
        save_last=True,
        save_top_k=1
    )

    # --- Trainer (Windows + gloo) ---
    trainer = Trainer(
        strategy=DDPStrategy(process_group_backend="gloo"),
        accelerator="gpu",
        devices=args.devices,
        max_epochs=args.epochs,
        accumulate_grad_batches=args.accumulate_grad_batches,
        precision=args.precision,
        gradient_clip_val=1.0,
        callbacks=[ckpt_cb],
        enable_progress_bar=True,
        log_every_n_steps=10,
        enable_model_summary=True,
        sync_batchnorm=True,
        logger=wb
    )

    trainer.fit(lit)

if __name__ == "__main__":
    main()
