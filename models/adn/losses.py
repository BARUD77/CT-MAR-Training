"""ADN losses and official hyper-parameters.

Ported from https://github.com/liaohaofu/adn (commit 10fd630): adn/networks/losses.py (GANLoss, adopted
from CycleGAN/pix2pix), adn/models/adn.py (ADNTrain.optimize) and config/adn.yaml + runs/adn.yaml.
Copyright (c) 2019, Haofu Liao; Copyright (c) 2017, Jun-Yan Zhu and Taesung Park. BSD licensed, see
models/adn/LICENSE.

All terms use mean reduction (nn.L1Loss / nn.MSELoss defaults in the original), so their scale does
not depend on the image or batch size.
"""
import torch
import torch.nn.functional as F

# config/adn.yaml (model: adn / nlayer), identical to MODEL_SPECS in the official demo.py.
ADN_G_KWARGS = dict(input_ch=1, base_ch=64, num_down=2, num_residual=4, num_sides=3, down_norm="instance",
                    res_norm="instance", up_norm="layer", fuse=True, shared_decoder=False)
ADN_D_KWARGS = dict(input_nc=1, ndf=64, n_layers=2, norm_layer="instance")

# config/adn.yaml `loss` (gt_art: True). Zero-weight terms (lh, lhl) are skipped, as in the original.
ADN_LOSS_WEIGHTS = dict(ll=20.0, lh=0.0, hh=20.0, lhl=0.0, hlh=20.0, art=20.0, gl=1.0, gh=1.0)

# config/adn.yaml `learn` (Adam with coupled L2 weight decay; StepLR stepped once per epoch, so with
# step_size 1e5 the learning rate never decays) and training length. runs/adn.yaml overrides the
# DeepLesion training batch size to 2 (default 4).
ADN_LEARN = dict(lr=1e-4, beta1=0.5, beta2=0.999, weight_decay=1e-4, gamma=0.5, step_size=1e5)
ADN_NUM_EPOCHS = 50
ADN_BATCH_SIZE = 2


def lsgan_loss(pred, target_is_real):
    """GANLoss('lsgan'): MSE against a constant 1 (real) / 0 (fake) map of the same size."""
    target = torch.ones_like(pred) if target_is_real else torch.zeros_like(pred)
    return F.mse_loss(pred, target)


def discriminator_loss(disc, fake, real):
    """get_d_loss: 0.5 * (MSE(D(real), 1) + MSE(D(fake.detach()), 0)). Returns (loss, D(real), D(fake))."""
    pred_real, pred_fake = disc(real), disc(fake.detach())
    return 0.5 * (lsgan_loss(pred_real, True) + lsgan_loss(pred_fake, False)), pred_real, pred_fake


def generator_adv_loss(disc, fake):
    """get_g_loss: MSE(D(fake), 1)."""
    return lsgan_loss(disc(fake), True)


def generator_losses(net, d_low, d_high, img_low, img_high, w, gt_art=True):
    """All weighted generator terms of ADNTrain.optimize for one batch.

    img_low: images with artifacts, img_high: clean images, both in [-1, 1]. `d_low`/`d_high` are the
    official model_dl/model_dh. Note the official naming: model_dl judges the output of the LOW branch
    (fake = pred_lh, real = clean images), model_dh the output of the HIGH branch (fake = pred_hl,
    real = artifact images). Returns (terms, preds): terms maps the original loss
    names (ll, lh, hh, lhl, hlh, art, gl_g, gh_g) of the non-zero-weight terms to weighted scalars;
    preds holds pred_ll, pred_lh, pred_hl, pred_hh (and pred_hlh / pred_lhl if used).
    """
    terms, preds = {}, {}
    pred_ll, pred_lh, art = net.forward1(img_low)
    pred_hl, pred_hh = net.forward2(img_high, art)
    preds.update(pred_ll=pred_ll, pred_lh=pred_lh, pred_hl=pred_hl, pred_hh=pred_hh)

    # low -> (low_l, low_h): adversarial on the artifact-reduced image vs real clean images.
    if w["gl"]:
        terms["gl_g"] = w["gl"] * generator_adv_loss(d_low, pred_lh)
    if w["lh"]:
        terms["lh"] = w["lh"] * F.l1_loss(pred_lh, img_high)
    if w["ll"]:
        terms["ll"] = w["ll"] * F.l1_loss(pred_ll, img_low)
    # high -> (high_l, high_h): adversarial on the artifact-transferred image vs real artifact images.
    if w["gh"]:
        terms["gh_g"] = w["gh"] * generator_adv_loss(d_high, pred_hl)
    if w["hh"]:
        terms["hh"] = w["hh"] * F.l1_loss(pred_hh, img_high)
    # low_h -> low_h_l (official call: forward_hl(pred_hl, pred_lh))
    if w["lhl"]:
        preds["pred_lhl"] = net.forward_hl(pred_hl, pred_lh)
        terms["lhl"] = w["lhl"] * F.l1_loss(preds["pred_lhl"], img_low)
    # high_l -> high_l_h (self-reduction)
    if w["hlh"]:
        preds["pred_hlh"] = net.forward_lh(pred_hl)
        terms["hlh"] = w["hlh"] * F.l1_loss(preds["pred_hlh"], img_high)
    # artifact consistency
    if w["art"]:
        ll = img_low if gt_art else pred_ll
        hh = img_high if gt_art else pred_hh
        terms["art"] = w["art"] * F.l1_loss(ll - pred_lh, pred_hl - hh)
    return terms, preds
