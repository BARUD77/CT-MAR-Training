# trainer_wandb.py (patched)
import argparse
import os
import json
import math
import torch
import yaml
from types import SimpleNamespace
from torch.utils.data import DataLoader

# NOTE: import path must match your repo layout
from aapm_dataset import CTMetalArtifactDataset   # patched dataset that returns (x, y, li)
from models.swin_unet.vision_transformer import SwinUnet
from models.swin_unet_feature_gating.vision_transformer import SwinUnet as SwinUnetFG
from models.swin_unet_three_channel.vision_transformer import SwinUnet as SwinUnet3Ch
from models.swin_unet_v2.swin_unet_v2 import SwinTransformerSys as SwinUnetV2
from models.unet import UnetGenerator
# from models.redcnn import REDCNN
# from models.gan import Generator as GANGenerator, Discriminator as GANDiscriminator

import numpy as np
from metrics import compute_SSIM, compute_PSNR, compute_RMSE, compute_masked_SSIM, compute_masked_SSIM_per_image, compute_masked_RMSE_HU
import wandb


def _to_hu(x: torch.Tensor, hu_min: float, hu_max: float) -> torch.Tensor:
    return x * (float(hu_max) - float(hu_min)) + float(hu_min)

# ----------------------------- Utilities -----------------------------
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

# ----------------------------- Model Factory -----------------------------
def build_model(name: str,
                in_ch: int,
                num_classes: int,
                device: torch.device,
                model_cfg_path: str | None = None,
                model_kwargs: dict | None = None):
    """
    Instantiate model. For SwinUnet we expect a config YAML path (model_cfg_path).
    """
    model_kwargs = model_kwargs or {}
    name = name.lower()
    if name in ("unet", "vanilla_unet", "u-net"):
        m = UnetGenerator(
            in_channels=in_ch,
            **model_kwargs
        )
        return m.to(device)
    elif name in ("redcnn", "red_cnn", "red-cnn"):
        num_features = int(model_kwargs.pop("num_features", 96))
        ksize = int(model_kwargs.pop("ksize", 5))
        use_input_residual = bool(model_kwargs.pop("use_input_residual", True))
        m = REDCNN(
            in_channels=in_ch,
            out_channels=num_classes,
            num_features=num_features,
            ksize=ksize,
            use_input_residual=use_input_residual,
            **model_kwargs
        )
        return m.to(device)
    elif name in ("gan", "unet_gan", "patch_gan"):
        # GAN trainer uses Generator as the "model"; Discriminator is built
        # separately in main() when --model gan.
        bottleneck = str(model_kwargs.pop("bottleneck", "conv"))
        m = GANGenerator(
            in_channels=in_ch,
            out_channels=num_classes,
            bottleneck=bottleneck,
            **model_kwargs
        )
        return m.to(device)
    elif name in ("swinunet", "swin_unet", "swin-u"):
        if not model_cfg_path:
            raise ValueError("SwinUnet requires --model_cfg <path to YAML>.")
        cfg_ns = load_config_namespace(model_cfg_path)
        # try to inject in_ch / num_classes to the config if present
        try:
            cfg_ns.MODEL.SWIN.IN_CHANS = in_ch
        except Exception:
            # ignore if config layout differs
            pass
        try:
            cfg_ns.MODEL.SWIN.NUM_CLASSES = int(num_classes)
        except Exception:
            # ignore if config layout differs
            pass

        # The non-guided SwinUnet wrapper does not accept num_classes as a kwarg.
        # It reads NUM_CLASSES from the config instead.
        model_kwargs.pop("num_classes", None)
        m = SwinUnet(config=cfg_ns, **model_kwargs)
        return m.to(device)
    elif name in ("swinunet_fg", "swin_unet_fg", "swinunet-fg", "feature_gating"):
        if not model_cfg_path:
            raise ValueError("SwinUnet (feature gating) requires --model_cfg <path to YAML>.")
        cfg_ns = load_config_namespace(model_cfg_path)
        try:
            cfg_ns.MODEL.SWIN.IN_CHANS = in_ch
        except Exception:
            pass
        try:
            cfg_ns.MODEL.SWIN.NUM_CLASSES = int(num_classes)
        except Exception:
            pass
        # Wrapper reads NUM_CLASSES from the config; same shape as plain swinunet.
        model_kwargs.pop("num_classes", None)
        m = SwinUnetFG(config=cfg_ns, **model_kwargs)
        return m.to(device)
    elif name in ("swinunet_3ch", "swin_unet_3ch", "swinunet-3ch", "three_channel"):
        if not model_cfg_path:
            raise ValueError("SwinUnet (3-channel) requires --model_cfg <path to YAML>.")
        cfg_ns = load_config_namespace(model_cfg_path)
        try:
            cfg_ns.MODEL.SWIN.IN_CHANS = in_ch  # expected to be 3 for [MA, LI, A_MG]
        except Exception:
            pass
        try:
            cfg_ns.MODEL.SWIN.NUM_CLASSES = int(num_classes)
        except Exception:
            pass
        model_kwargs.pop("num_classes", None)
        m = SwinUnet3Ch(config=cfg_ns, **model_kwargs)
        return m.to(device)
    elif name in ("swinunet_v2", "swin_unet_v2", "swinunet-v2"):
        if not model_cfg_path:
            raise ValueError("SwinUnet V2 requires --model_cfg <path to YAML>.")
        cfg_ns = load_config_namespace(model_cfg_path)
        try:
            cfg_ns.MODEL.SWIN.IN_CHANS = in_ch
        except Exception:
            pass
        try:
            cfg_ns.MODEL.SWIN.NUM_CLASSES = int(num_classes)
        except Exception:
            pass
        model_kwargs.pop("num_classes", None)
        m = SwinUnetV2(
            img_size=int(cfg_ns.DATA.IMG_SIZE),
            patch_size=int(cfg_ns.MODEL.SWIN.PATCH_SIZE),
            in_chans=int(cfg_ns.MODEL.SWIN.IN_CHANS),
            num_classes=int(cfg_ns.MODEL.SWIN.NUM_CLASSES),
            embed_dim=int(cfg_ns.MODEL.SWIN.EMBED_DIM),
            depths=list(cfg_ns.MODEL.SWIN.DEPTHS),
            depths_decoder=list(cfg_ns.MODEL.SWIN.DECODER_DEPTHS),
            num_heads=list(cfg_ns.MODEL.SWIN.NUM_HEADS),
            window_size=int(cfg_ns.MODEL.SWIN.WINDOW_SIZE),
            drop_path_rate=float(cfg_ns.MODEL.DROP_PATH_RATE),
            final_upsample=str(cfg_ns.MODEL.SWIN.FINAL_UPSAMPLE),
            **model_kwargs
        )
        return m.to(device)
    else:
        raise ValueError(f"Unknown model: {name}. Supported: unet, redcnn, gan, swinunet, swinunet_fg, swinunet_3ch, swinunet_v2")

def param_groups(model, wd, nowd_names=('bias', 'bn', 'norm')):
    decay, no_decay = [], []
    for n, p in model.named_parameters():
        if not p.requires_grad:
            continue
        if any(k in n.lower() for k in nowd_names):
            no_decay.append(p)
        else:
            decay.append(p)
    return [
        {'params': decay, 'weight_decay': wd},
        {'params': no_decay, 'weight_decay': 0.0},
    ]


def apply_metal_on_gt(y_batch, x_batch, mask_batch):
    """Paint the metal region (mask==1) from the MA input onto the GT target.

    Keeps the metal present in both input and target so the model only has to
    remove streak artifacts rather than inpaint the metal itself.
    """
    if mask_batch is None:
        raise RuntimeError(
            "--metal_mask_on_gt is enabled but no metal mask is in the batch. "
            "Pass --mask_dir so the dataset provides masks."
        )
    return torch.where(mask_batch > 0.5, x_batch, y_batch)

# ----------------------------- Main -----------------------------
def main():
    parser = argparse.ArgumentParser(description="Training script (W&B only, model-agnostic)")
    # Model selection / config
    parser.add_argument('--model', type=str,
                        choices=['unet', 'redcnn', 'gan', 'swinunet', 'swinunet_fg', 'swinunet_3ch', 'swinunet_v2'],
                        required=True,
                        help="Pick an architecture from the registry.")
    parser.add_argument('--model_cfg', type=str, default=None,
                        help="Optional: path to a YAML config (required for swinunet).")
    parser.add_argument('--model_kwargs', type=str, default=None,
                        help='Optional: JSON dict of kwargs for the chosen model. '
                             'Example: \'{"base_channels":48,"bilinear":true}\'')

    # Data & run setup
    parser.add_argument('--ma_dir', type=str, required=True)
    parser.add_argument('--li_dir', type=str, required=False, help='Directory with LI files (required by the dataset)')
    parser.add_argument('--mask_dir', type=str, required=False, help='Optional: directory with metal masks (metalonlymask files)')
    parser.add_argument('--gt_dir', type=str, required=True)
    parser.add_argument('--input_mode', type=str, choices=['ma_li', 'ma'], default='ma',
                        help="Use 'ma_li' to concatenate MA+LI as 2-channel input, or 'ma' for MA-only input.")
    parser.add_argument('--num_classes', type=int, default=1, help='Output channels/classes')
    parser.add_argument('--batch_size', type=int, default=4)
    parser.add_argument('--num_workers', type=int, default=4,
                        help='DataLoader workers. In Colab, set --num_workers 0 if you hit multiprocessing issues.')
    parser.add_argument('--epochs', type=int, default=100)
    parser.add_argument('--early_stop_patience', type=int, default=0,
                        help='Stop training if val SSIM has not improved for this many consecutive epochs. '
                             '0 (default) disables early stopping.')
    parser.add_argument('--early_stop_min_delta', type=float, default=0.0,
                        help='Minimum SSIM improvement to count as progress (default 0.0).')
    parser.add_argument('--monitor_metric', type=str, default='ssim',
                        choices=['ssim'],
                        help='Validation metric used to select best.pt and drive early stopping. '
                             'Only the masked SSIM (dilated metal excluded) is reported.')
    parser.add_argument('--body_hu_thresh', type=float, default=-500.0,
                        help='HU threshold used to derive the body mask from GT (pixel > thresh = body).')
    parser.add_argument('--metal_mask_on_gt', action='store_true',
                        help='Paint the metal region (from the metal mask) of the MA input onto the GT target, '
                             'so the metal stays present in both input and target. Makes the MA->GT translation '
                             'easier (model only removes streak artifacts). Requires --mask_dir.')
    parser.add_argument('--learning_rate', type=float, default=1e-4)
    parser.add_argument('--warmup_epochs', type=float, default=5.0,
                        help='Linear LR warmup duration (in epochs) before cosine decay. '
                             'Set 0 to disable warmup. Cosine decay anneals LR toward ~0 over the remaining epochs.')
    parser.add_argument('--min_lr_ratio', type=float, default=0.0,
                        help='Final LR as a fraction of --learning_rate at the end of cosine decay (e.g. 0.01).')
    parser.add_argument('--log_dir', type=str, default='./runs', help='Directory to save logs and weights')
    parser.add_argument('--run_name', type=str, default=None, help='W&B run name')
    parser.add_argument('--project', type=str, default='ct-mar', help='W&B project name')
    parser.add_argument('--entity', type=str, default=None, help='W&B entity (team/user) if needed')
    parser.add_argument('--region_policy', type=str, default='all', choices=['all','body','head'],
                        help='Region filter passed to dataset (defaults to "all").')

    # Intensity window (HU) used for clipping/normalization inside the dataset
    parser.add_argument('--hu_min', type=float, default=-1024.0, help='Minimum HU for clipping before normalization')
    parser.add_argument('--hu_max', type=float, default=3072.0, help='Maximum HU for clipping before normalization')
    parser.add_argument('--no_clip', action='store_true', help='Disable HU clipping in the dataset (values may go outside [0,1] after normalization).')

    # GAN-specific (only used when --model gan)
    parser.add_argument('--gan_lambda_l1', type=float, default=100.0,
                        help='Weight of pixel L1 loss in the generator objective.')
    parser.add_argument('--gan_lambda_adv', type=float, default=1.0,
                        help='Weight of adversarial BCE loss in the generator objective.')
    parser.add_argument('--gan_d_lr', type=float, default=2e-4,
                        help='Learning rate for the GAN discriminator (Adam).')
    parser.add_argument('--gan_d_beta1', type=float, default=0.5,
                        help='Adam beta1 for the GAN discriminator.')

    args = parser.parse_args()

    # LI is required for ma_li mode and for the feature-gating Swin-UNet
    # (which uses LI to build the soft artifact map A_MG).
    is_swin_fg = args.model.lower() in ("swinunet_fg", "swin_unet_fg", "swinunet-fg", "feature_gating")
    is_swin_3ch = args.model.lower() in ("swinunet_3ch", "swin_unet_3ch", "swinunet-3ch", "three_channel")
    is_gan = args.model.lower() in ("gan", "unet_gan", "patch_gan")
    if args.input_mode == 'ma_li' and not args.li_dir:
        raise ValueError("input_mode='ma_li' requires --li_dir (LI files).")
    if is_swin_fg and not args.li_dir:
        raise ValueError("--model swinunet_fg requires --li_dir (LI files) to build the artifact map.")
    if is_swin_3ch and not args.li_dir:
        raise ValueError("--model swinunet_3ch requires --li_dir (LI files) to build the [MA, LI, A_MG] input.")

    os.makedirs(args.log_dir, exist_ok=True)

    # ---------------- W&B init ----------------
    run = wandb.init(
        project=args.project,
        entity=args.entity,
        name=args.run_name,
        config=vars(args),
        dir=args.log_dir,
        save_code=True,
        reinit=True,
    )

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # Decide input channels to pass to model builder
    # - swinunet_fg          -> single-stream MA input (1ch); LI is used as artifact_map
    # - swinunet_3ch         -> 3-channel input [MA, LI, A_MG]
    # - input_mode='ma_li'   -> 2-channel input [MA, LI] (vanilla UNet / plain SwinUnet)
    # - input_mode='ma'      -> 1-channel input [MA]
    if is_swin_fg:
        in_ch = 1
    elif is_swin_3ch:
        in_ch = 3
    else:
        in_ch = 2 if args.input_mode == 'ma_li' else 1

    # ---------------- Build model ----------------
    model = build_model(
        name=args.model,
        in_ch=in_ch,
        num_classes=args.num_classes,
        device=device,
        model_cfg_path=args.model_cfg,
        model_kwargs=parse_json_or_none(args.model_kwargs)
    )

    # track gradients/weights (optional)
    wandb.watch(model, log='gradients', log_freq=100)

    # ---------------- (GAN only) Build discriminator + its optimizer ----------------
    discriminator = None
    optimizer_D = None
    bce_loss = None
    if is_gan:
        discriminator = GANDiscriminator(in_channels=args.num_classes).to(device)
        optimizer_D = torch.optim.Adam(
            discriminator.parameters(),
            lr=args.gan_d_lr,
            betas=(args.gan_d_beta1, 0.999),
        )
        bce_loss = torch.nn.BCELoss()
        wandb.watch(discriminator, log='gradients', log_freq=100)
        print(f"[GAN] Built discriminator on {args.num_classes}-ch images. "
              f"lambda_l1={args.gan_lambda_l1}, lambda_adv={args.gan_lambda_adv}, "
              f"d_lr={args.gan_d_lr}, d_beta1={args.gan_d_beta1}")

    # ---------------- Dataset / loaders ----------------

    # Our CTMetalArtifactDataset (patched) requires li_dir when we want LI returned.
    # If input_mode == 'ma_li' we will request li_dir to be used.
    dataset_kwargs = dict(
        ma_dir=args.ma_dir,
        gt_dir=args.gt_dir,
        # LI is loaded whenever input_mode is ma_li OR we're running a model that needs LI.
        li_dir=args.li_dir if (args.input_mode == 'ma_li' or is_swin_fg or is_swin_3ch) else None,
        mask_dir=args.mask_dir,
        split='train',
        hu_min=float(args.hu_min),
        hu_max=float(args.hu_max),
        clip_hu=(not args.no_clip),
        output_space="norm",
        region_policy=args.region_policy,
        seed=42,
        val_size=0.1
    )

    # Build train/val datasets
    train_ds = CTMetalArtifactDataset(**{**dataset_kwargs, "split": "train"})
    # Normalized validation dataset for model I/O + normalized metrics
    val_ds   = CTMetalArtifactDataset(**{**dataset_kwargs, "split": "val"})
    # HU validation dataset for HU-domain metrics (no extra np.load inside the loop)
    val_ds_hu = CTMetalArtifactDataset(**{**dataset_kwargs, "split": "val", "output_space": "hu"})
    # Ensure perfect alignment between the two val loaders
    val_ds_hu.pairs = val_ds.pairs

    pin = (device.type == 'cuda')
    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers, pin_memory=pin)
    val_loader   = DataLoader(val_ds,   batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers, pin_memory=pin)
    val_loader_hu = DataLoader(val_ds_hu, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers, pin_memory=pin)

    print(f"Training on {len(train_ds)} samples, validating on {len(val_ds)} samples.")

    # ---------------- Optimizer / loss ----------------
    optimizer = torch.optim.AdamW(param_groups(model, wd=1e-4),
                              lr=args.learning_rate, betas=(0.9, 0.999), eps=1e-8)
    loss_fn = torch.nn.L1Loss()

    # ---------------- LR schedule: linear warmup + cosine decay (per-iteration) ----------------
    steps_per_epoch = max(1, len(train_loader))
    total_steps = max(1, args.epochs * steps_per_epoch)
    warmup_steps = max(0, int(round(args.warmup_epochs * steps_per_epoch)))
    min_lr_ratio = max(0.0, float(args.min_lr_ratio))

    def lr_lambda(current_step):
        # Linear warmup from 0 -> 1
        if warmup_steps > 0 and current_step < warmup_steps:
            return float(current_step + 1) / float(warmup_steps)
        # Cosine decay from 1 -> min_lr_ratio
        progress = float(current_step - warmup_steps) / float(max(1, total_steps - warmup_steps))
        progress = min(1.0, max(0.0, progress))
        cosine = 0.5 * (1.0 + math.cos(math.pi * progress))
        return min_lr_ratio + (1.0 - min_lr_ratio) * cosine

    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)

    best_ssim = -1.0
    best_ckpt_path = None
    epochs_since_improve = 0  # for early stopping

    # Training loop
    for epoch in range(1, args.epochs + 1):
        model.train()
        train_loss = 0.0
        train_loss_d = 0.0
        train_loss_g_adv = 0.0
        train_loss_g_l1 = 0.0
        if discriminator is not None:
            discriminator.train()

        for batch in train_loader:
            # unpack batch robustly: support (x,y), (x,y,li), (x,y,mask), (x,y,mask,li)
            x_batch = y_batch = li_batch = mask_batch = None
            if isinstance(batch, (list, tuple)):
                if len(batch) == 2:
                    x_batch, y_batch = batch
                elif len(batch) == 3:
                    third = batch[2]
                    # lightweight heuristic: if third is binary -> mask, else LI
                    is_binary = False
                    try:
                        uniq = torch.unique(third)
                        is_binary = (uniq.numel() <= 2) and torch.all((uniq == 0) | (uniq == 1))
                    except Exception:
                        is_binary = False
                    if is_binary:
                        x_batch, y_batch, mask_batch = batch
                    else:
                        x_batch, y_batch, li_batch = batch
                else:
                    # 4+ items -> assume (x,y,mask,li)
                    x_batch, y_batch, mask_batch, li_batch = batch[:4]
            else:
                x_batch, y_batch = batch

            # Move to device
            x_batch = x_batch.to(device, non_blocking=True)
            y_batch = y_batch.to(device, non_blocking=True)
            if li_batch is not None:
                li_batch = li_batch.to(device, non_blocking=True)
            if mask_batch is not None:
                mask_batch = mask_batch.to(device, non_blocking=True)

            # Optionally paint the metal region of MA onto the GT target.
            if args.metal_mask_on_gt:
                y_batch = apply_metal_on_gt(y_batch, x_batch, mask_batch)

            optimizer.zero_grad(set_to_none=True)

            # Build input tensor based on model + input_mode
            if is_swin_fg:
                # Single-stream MA input; pass A_MG = normalize(relu(LI - MA)) as artifact_map.
                if li_batch is None:
                    raise RuntimeError("swinunet_fg requires LI in the batch, but li_batch is None.")
                artifact_map = torch.relu(li_batch - x_batch)
                am_max = artifact_map.amax(dim=(2, 3), keepdim=True)
                artifact_map = artifact_map / (am_max + 1e-6)
                pred = model(x_batch, artifact_map=artifact_map)
            elif is_swin_3ch:
                # 3-channel input: [MA, LI, A_MG]
                if li_batch is None:
                    raise RuntimeError("swinunet_3ch requires LI in the batch, but li_batch is None.")
                artifact_map = torch.relu(li_batch - x_batch)
                am_max = artifact_map.amax(dim=(2, 3), keepdim=True)
                artifact_map = artifact_map / (am_max + 1e-6)
                x_in = torch.cat([x_batch, li_batch, artifact_map], dim=1)
                pred = model(x_in)
            else:
                if args.input_mode == 'ma_li':
                    if li_batch is None:
                        raise RuntimeError("input_mode='ma_li' requires LI in the batch, but li_batch is None.")
                    x_in = torch.cat([x_batch, li_batch], dim=1)
                else:
                    x_in = x_batch
                pred = model(x_in)

            if is_gan:
                # ---- Discriminator step ----
                optimizer_D.zero_grad(set_to_none=True)
                real_score = discriminator(y_batch)
                fake_score_d = discriminator(pred.detach())
                loss_d_real = bce_loss(real_score, torch.ones_like(real_score))
                loss_d_fake = bce_loss(fake_score_d, torch.zeros_like(fake_score_d))
                loss_d = 0.5 * (loss_d_real + loss_d_fake)
                loss_d.backward()
                optimizer_D.step()

                # ---- Generator step ----
                fake_score_g = discriminator(pred)
                loss_g_adv = bce_loss(fake_score_g, torch.ones_like(fake_score_g))
                loss_g_l1 = loss_fn(pred, y_batch)
                loss = args.gan_lambda_l1 * loss_g_l1 + args.gan_lambda_adv * loss_g_adv

                train_loss_d += float(loss_d.item())
                train_loss_g_adv += float(loss_g_adv.item())
                train_loss_g_l1 += float(loss_g_l1.item())
            else:
                loss = loss_fn(pred, y_batch)

            loss.backward()
            optimizer.step()
            scheduler.step()

            train_loss += loss.item()

        avg_loss = train_loss / max(1, len(train_loader))
        n_batches = max(1, len(train_loader))
        avg_loss_d = train_loss_d / n_batches
        avg_loss_g_adv = train_loss_g_adv / n_batches
        avg_loss_g_l1 = train_loss_g_l1 / n_batches

        # ---------------- Validation ----------------
        model.eval()
        ssim_list = []
        psnr_list = []
        rmse_list = []
        # normalized metal fill (100 HU) used to fill the dilated metal region for SSIM
        metal_fill_norm = (100.0 - float(args.hu_min)) / (float(args.hu_max) - float(args.hu_min))
        with torch.no_grad():
            for batch, batch_hu in zip(val_loader, val_loader_hu):
                # unpack batch robustly: support (x,y), (x,y,li), (x,y,mask), (x,y,mask,li)
                x_batch = y_batch = li_batch = mask_batch = None
                if isinstance(batch, (list, tuple)):
                    if len(batch) == 2:
                        x_batch, y_batch = batch
                    elif len(batch) == 3:
                        third = batch[2]
                        try:
                            uniq = torch.unique(third)
                            is_binary = (uniq.numel() <= 2) and torch.all((uniq == 0) | (uniq == 1))
                        except Exception:
                            is_binary = False
                        if is_binary:
                            x_batch, y_batch, mask_batch = batch
                        else:
                            x_batch, y_batch, li_batch = batch
                    else:
                        x_batch, y_batch, mask_batch, li_batch = batch[:4]
                else:
                    x_batch, y_batch = batch

                x_batch = x_batch.to(device, non_blocking=True)
                y_batch = y_batch.to(device, non_blocking=True)
                if li_batch is not None:
                    li_batch = li_batch.to(device, non_blocking=True)
                if mask_batch is not None:
                    mask_batch = mask_batch.to(device, non_blocking=True)

                # Unpack HU batch (same split/order; only used for HU-domain metrics)
                x_hu = y_hu = li_hu = mask_hu = None
                if isinstance(batch_hu, (list, tuple)):
                    if len(batch_hu) == 2:
                        x_hu, y_hu = batch_hu
                    elif len(batch_hu) == 3:
                        third_hu = batch_hu[2]
                        try:
                            uniq_hu = torch.unique(third_hu)
                            is_binary_hu = (uniq_hu.numel() <= 2) and torch.all((uniq_hu == 0) | (uniq_hu == 1))
                        except Exception:
                            is_binary_hu = False
                        if is_binary_hu:
                            x_hu, y_hu, mask_hu = batch_hu
                        else:
                            x_hu, y_hu, li_hu = batch_hu
                    else:
                        x_hu, y_hu, mask_hu, li_hu = batch_hu[:4]
                else:
                    x_hu, y_hu = batch_hu

                y_hu = y_hu.to(device, non_blocking=True)
                if li_hu is not None:
                    li_hu = li_hu.to(device, non_blocking=True)

                # Optionally paint the metal region onto the GT target (match training).
                if args.metal_mask_on_gt:
                    y_batch = apply_metal_on_gt(y_batch, x_batch, mask_batch)
                    if y_hu is not None and x_hu is not None and mask_hu is not None:
                        x_hu = x_hu.to(device, non_blocking=True)
                        y_hu = apply_metal_on_gt(y_hu, x_hu, mask_hu.to(device, non_blocking=True))

                if args.input_mode == 'ma_li':
                    if li_batch is None:
                        raise RuntimeError("input_mode='ma_li' requires LI in the batch, but li_batch is None.")
                    x_in = torch.cat([x_batch, li_batch], dim=1)
                else:
                    x_in = x_batch
                if is_swin_fg:
                    if li_batch is None:
                        raise RuntimeError("swinunet_fg requires LI in the batch, but li_batch is None.")
                    artifact_map = torch.relu(li_batch - x_batch)
                    am_max = artifact_map.amax(dim=(2, 3), keepdim=True)
                    artifact_map = artifact_map / (am_max + 1e-6)
                    pred = model(x_batch, artifact_map=artifact_map)
                elif is_swin_3ch:
                    if li_batch is None:
                        raise RuntimeError("swinunet_3ch requires LI in the batch, but li_batch is None.")
                    artifact_map = torch.relu(li_batch - x_batch)
                    am_max = artifact_map.amax(dim=(2, 3), keepdim=True)
                    artifact_map = artifact_map / (am_max + 1e-6)
                    x_in = torch.cat([x_batch, li_batch, artifact_map], dim=1)
                    pred = model(x_in)
                else:
                    pred = model(x_in)

                # Normalized metrics are computed on clamped [0,1] tensors.
                pred_eval = torch.clamp(pred, 0, 1)
                gt_eval   = torch.clamp(y_batch, 0, 1)

                # compute per-image metrics
                for i in range(pred_eval.size(0)):
                    img_pred = pred_eval[i, 0]
                    img_gt = gt_eval[i, 0]

                    # Extract per-image metal mask first (needed by all metrics below).
                    metalmask_arg = None
                    non_metal_mask = None
                    if mask_batch is not None:
                        m = mask_batch[i]
                        if m.dim() == 3 and m.size(0) == 1:
                            m = m.squeeze(0)
                        if m.dim() == 3 and m.size(0) == 1:
                            m = m.squeeze(0)
                        metalmask_arg = m
                        non_metal_mask = (m == 0).float()

                    # ---- SSIM: dilate metal by 2 px, fill dilated region to 100 HU,
                    #      average full SSIM map (data_range=1) over the non-dilated pixels.
                    if metalmask_arg is not None:
                        _, ssim_val = compute_masked_SSIM(
                            img_pred,
                            img_gt,
                            data_range=1.0,
                            mask=None,
                            metalmask=metalmask_arg,
                            metal_fill=metal_fill_norm,
                            dilate_iters=2,
                            hu_min=args.hu_min,
                            hu_max=args.hu_max,
                        )
                    else:
                        ssim_val = compute_SSIM(img_pred, img_gt, data_range=1.0)
                    ssim_list.append(float(ssim_val))

                    # ---- RMSE (HU) and PSNR: denormalize to HU, dilate metal by 1 px and
                    #      exclude it; RMSE in HU, PSNR = 20*log10(MAX/RMSE) with MAX=HU range.
                    rmse_hu = compute_masked_RMSE_HU(
                        img_pred,
                        img_gt,
                        metalmask=metalmask_arg,
                        hu_min=args.hu_min,
                        hu_max=args.hu_max,
                        dilate_iters=1,
                    )
                    rmse_list.append(float(rmse_hu))
                    psnr_max = float(args.hu_max) - float(args.hu_min)  # 4096 HU range
                    rmse_hu_safe = rmse_hu if rmse_hu > 0 else 1e-10
                    psnr_list.append(float(20.0 * np.log10(psnr_max / rmse_hu_safe)))

        avg_psnr = float(np.mean(psnr_list)) if psnr_list else 0.0
        avg_ssim = float(np.mean(ssim_list)) if ssim_list else 0.0
        avg_rmse = float(np.mean(rmse_list)) if rmse_list else 0.0
        std_psnr = float(np.std(psnr_list, ddof=1)) if len(psnr_list) > 1 else 0.0
        std_ssim = float(np.std(ssim_list, ddof=1)) if len(ssim_list) > 1 else 0.0
        std_rmse = float(np.std(rmse_list, ddof=1)) if len(rmse_list) > 1 else 0.0

        # ---------------- Log to W&B ----------------
        log_payload = {
                "loss": avg_loss,
                "psnr": avg_psnr,
                "psnr_std": std_psnr,
                "ssim": avg_ssim,
                "ssim_std": std_ssim,
                "rmse": avg_rmse,
                "rmse_std": std_rmse,
                "lr": optimizer.param_groups[0]["lr"],
                "epoch": epoch,
        }
        if is_gan:
            log_payload["gan/loss_d"] = avg_loss_d
            log_payload["gan/loss_g_adv"] = avg_loss_g_adv
            log_payload["gan/loss_g_l1"] = avg_loss_g_l1
        wandb.log(log_payload, step=epoch)
        print(
            f"[Epoch {epoch}] Loss: {avg_loss:.4f} "
            f"PSNR: {avg_psnr:.2f} \u00b1 {std_psnr:.2f} "
            f"SSIM: {avg_ssim:.4f} \u00b1 {std_ssim:.4f} "
            f"RMSE: {avg_rmse:.2f} \u00b1 {std_rmse:.2f} HU"
        )

        # ---------------- Save checkpoints ----------------
        # Save only a rolling "last.pt" (overwritten every epoch)
        last_ckpt_path = os.path.join(args.log_dir, "last.pt")
        torch.save({"epoch": epoch, "model_state": model.state_dict()}, last_ckpt_path)

        # Checkpoint saved locally (or to Google Drive if log_dir points there)
        # Skipped W&B upload via wandb.log_artifact()

        # Pick which metric drives best.pt / early stopping.
        _metric_lookup = {
            "ssim": avg_ssim,
        }
        monitored_value = _metric_lookup[args.monitor_metric]

        if monitored_value > best_ssim + args.early_stop_min_delta:
            best_ssim = monitored_value
            best_ckpt_path = os.path.join(args.log_dir, "best.pt")
            torch.save({"epoch": epoch, "model_state": model.state_dict(),
                        "monitor_metric": args.monitor_metric,
                        "monitor_value": monitored_value}, best_ckpt_path)
            # Checkpoint saved locally (or to Google Drive if log_dir points there)
            # Skipped W&B upload via wandb.log_artifact()
            epochs_since_improve = 0
        else:
            epochs_since_improve += 1

        # ---------------- Early stopping ----------------
        wandb.log({"early_stop/epochs_since_improve": epochs_since_improve,
                   "early_stop/best_ssim": best_ssim}, step=epoch)
        if args.early_stop_patience > 0 and epochs_since_improve >= args.early_stop_patience:
            print(f"⏹️  Early stopping at epoch {epoch}: "
                  f"no SSIM improvement for {epochs_since_improve} epochs "
                  f"(best={best_ssim:.4f}, patience={args.early_stop_patience}).")
            wandb.run.summary["early_stopped_epoch"] = epoch
            break

    wandb.finish()
    print("✅ Training complete!")


if __name__ == "__main__":
    main()
