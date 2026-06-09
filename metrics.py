import torch
import scipy
import numpy as np
from math import exp
import torch.nn.functional as F
from scipy import ndimage


def compute_rec_metrics(pred, y, data_range, spatial_dims=2):
    pred_psnr = compute_PSNR(pred, y, data_range)
    pred_ssim = compute_SSIM(pred, y, data_range, spatial_dims=spatial_dims)
    pred_rmse = compute_RMSE(pred, y)
    return  pred_rmse, pred_psnr, pred_ssim


def compute_MSE(img1, img2):
    return ((img1 - img2) ** 2).mean()


def compute_RMSE(img1, img2):
    if type(img1) == torch.Tensor:
        return torch.sqrt(compute_MSE(img1, img2)).item()
    else:
        return np.sqrt(compute_MSE(img1, img2))


def compute_PSNR(img1, img2, data_range):
    eps = 1e-10
    mse_ = compute_MSE(img1, img2)
    if mse_ == 0:
        mse_ += eps
    if torch.is_tensor(img1):
        return 10 * torch.log10((data_range ** 2) / mse_).item()
    else:
        return 10 * np.log10((data_range ** 2) / mse_)


def compute_SSIM(img1, img2, data_range, window_size=11, channel=1, size_average=True, spatial_dims=2):
    # referred from https://github.com/Po-Hsun-Su/pytorch-ssim
    # default window_size 11
    if len(img1.size()) == 2:
        shape_ = img1.shape
        img1 = img1.view(1, 1, *shape_)
        img2 = img2.view(1, 1, *shape_)
    window = create_window(window_size, channel, spatial_dims=spatial_dims)
    window = window.type_as(img1)

    conv_op = F.conv2d if spatial_dims == 2 else F.conv3d

    mu1 = conv_op(img1, window, padding=window_size//2)
    mu2 = conv_op(img2, window, padding=window_size//2)
    mu1_sq, mu2_sq = mu1.pow(2), mu2.pow(2)
    mu1_mu2 = mu1*mu2

    sigma1_sq = conv_op(img1*img1, window, padding=window_size//2) - mu1_sq
    sigma2_sq = conv_op(img2*img2, window, padding=window_size//2) - mu2_sq
    sigma12 = conv_op(img1*img2, window, padding=window_size//2) - mu1_mu2

    C1, C2 = (0.01*data_range)**2, (0.03*data_range)**2
    #C1, C2 = 0.01**2, 0.03**2
    ssim_map = ((2*mu1_mu2+C1)*(2*sigma12+C2)) / ((mu1_sq+mu2_sq+C1)*(sigma1_sq+sigma2_sq+C2))
    if size_average:
        return ssim_map.mean().item()
    else:
        return ssim_map.mean(1).mean(1).mean(1).item()


def compute_masked_SSIM(img1, img2, data_range, mask=None, metalmask=None,
                        window_size=11, channel=1, size_average=True, spatial_dims=2,
                        metal_fill=None, hu_min=-1024.0, hu_max=3072.0, dilate_iters=0):
    """
    Tensor-based SSIM computation that mirrors `compute_SSIM` internals but
    returns the full SSIM map and the mean SSIM computed only over `mask`.
    Pixels inside `metalmask` are set to 100 in both images prior to SSIM,
    matching the AAPM scoring behavior.

    Returns: (ssim_map, ssim_masked_mean)
      - ssim_map: tensor of SSIM values (same shape as internal map)
      - ssim_masked_mean: python float average of ssim_map over mask (or
        global mean if mask is None or has no positive pixels)
    """
    # ensure tensors
    if not torch.is_tensor(img1):
        img1 = torch.from_numpy(np.asarray(img1)).float()
    if not torch.is_tensor(img2):
        img2 = torch.from_numpy(np.asarray(img2)).float()

    # reshape 2D to (1,1,H,W) as in compute_SSIM
    if len(img1.size()) == 2:
        shape_ = img1.shape
        img1 = img1.view(1, 1, *shape_)
        img2 = img2.view(1, 1, *shape_)

    # determine metal fill value
    if metal_fill is None:
        # if data_range==1.0 assume normalized images in [0,1] and compute normalized 100 HU
        if float(data_range) == 1.0:
            metal_fill = (100.0 - float(hu_min)) / (float(hu_max) - float(hu_min))
        else:
            metal_fill = 100.0

    # apply metal mask by setting metal pixels to 100 (match scoring utils).
    # Optionally dilate the metal mask first (scipy binary_dilation) so a margin
    # around the metal is also filled and (when mask is None) excluded from the mean.
    mm_dilated = None
    if metalmask is not None:
        if not torch.is_tensor(metalmask):
            metalmask = torch.from_numpy(np.asarray(metalmask))
        mm = metalmask.float()
        if len(mm.size()) == 2:
            mm = mm.view(1, 1, *mm.shape)
        if dilate_iters and int(dilate_iters) > 0:
            mm_np = (mm.detach().cpu().numpy() > 0.5)
            struct = ndimage.generate_binary_structure(2, 1)
            mm_out = np.zeros_like(mm_np)
            for b in range(mm_np.shape[0]):
                mm_out[b, 0] = ndimage.binary_dilation(
                    mm_np[b, 0], structure=struct, iterations=int(dilate_iters))
            mm = torch.from_numpy(mm_out.astype(np.float32)).to(img1.device)
        mm_dilated = mm
        img1 = img1.clone()
        img2 = img2.clone()
        img1[mm == 1] = float(metal_fill)
        img2[mm == 1] = float(metal_fill)

    window = create_window(window_size, channel, spatial_dims=spatial_dims)
    window = window.type_as(img1)
    conv_op = F.conv2d if spatial_dims == 2 else F.conv3d

    mu1 = conv_op(img1, window, padding=window_size // 2)
    mu2 = conv_op(img2, window, padding=window_size // 2)
    mu1_sq, mu2_sq = mu1.pow(2), mu2.pow(2)
    mu1_mu2 = mu1 * mu2

    sigma1_sq = conv_op(img1 * img1, window, padding=window_size // 2) - mu1_sq
    sigma2_sq = conv_op(img2 * img2, window, padding=window_size // 2) - mu2_sq
    sigma12 = conv_op(img1 * img2, window, padding=window_size // 2) - mu1_mu2

    C1, C2 = (0.01 * data_range) ** 2, (0.03 * data_range) ** 2
    ssim_map = ((2 * mu1_mu2 + C1) * (2 * sigma12 + C2)) / ((mu1_sq + mu2_sq + C1) * (sigma1_sq + sigma2_sq + C2))

    # compute masked mean
    if mask is None:
        if mm_dilated is not None:
            # average only over pixels outside the (dilated) metal region
            m = (mm_dilated == 0).float().to(ssim_map.device)
            masked_vals = ssim_map[m == 1]
            ssim_masked_mean = (float(masked_vals.mean().item())
                                if masked_vals.numel() > 0 else float(ssim_map.mean().item()))
        else:
            ssim_masked_mean = float(ssim_map.mean().item())
    else:
        if not torch.is_tensor(mask):
            mask = torch.from_numpy(np.asarray(mask))
        m = mask.float()
        if len(m.size()) == 2:
            m = m.view(1, 1, *m.shape)
        m = m.to(ssim_map.device)
        masked_vals = ssim_map[m == 1]
        if masked_vals.numel() == 0:
            ssim_masked_mean = float(ssim_map.mean().item())
        else:
            ssim_masked_mean = float(masked_vals.mean().item())

    return ssim_map, ssim_masked_mean


def compute_masked_RMSE_HU(pred, target, metalmask=None, hu_min=-1024.0, hu_max=3072.0,
                           dilate_iters=1):
    """Denormalize [0,1] images to HU and compute RMSE (in HU) over non-metal pixels.

    The metal mask is optionally dilated by `dilate_iters` (scipy binary_dilation)
    and those pixels are excluded from the average. Returns a python float RMSE in HU.
    """
    if not torch.is_tensor(pred):
        pred = torch.from_numpy(np.asarray(pred)).float()
    if not torch.is_tensor(target):
        target = torch.from_numpy(np.asarray(target)).float()
    dr = float(hu_max) - float(hu_min)
    pred_hu = pred.float() * dr + float(hu_min)
    target_hu = target.float() * dr + float(hu_min)
    se = (pred_hu - target_hu) ** 2

    if metalmask is not None:
        if not torch.is_tensor(metalmask):
            metalmask = torch.from_numpy(np.asarray(metalmask))
        mm2d = metalmask.float().squeeze()
        if dilate_iters and int(dilate_iters) > 0:
            mm_np = (mm2d.detach().cpu().numpy() > 0.5)
            struct = ndimage.generate_binary_structure(2, 1)
            mm_np = ndimage.binary_dilation(mm_np, structure=struct, iterations=int(dilate_iters))
            mm2d = torch.from_numpy(mm_np.astype(np.float32))
        valid = (mm2d.to(se.device).reshape(se.shape) == 0)
        se = se[valid]

    if se.numel() == 0:
        return 0.0
    mse = se.mean()
    return float(torch.sqrt(mse).item())


def compute_masked_SSIM_loss(pred, target, metal_mask=None, data_range=1.0,
                             window_size=11, channel=1, dilation=2, eps=1e-8):
    """Differentiable masked SSIM, suitable as a training loss term.

    Follows the "mask at the SSIM-map level" approach: the full per-pixel SSIM map
    is computed on the UNMODIFIED images (no zero/constant fill of the metal region,
    so no artificial boundary edge is injected), then averaged ONLY over valid
    pixels = those outside the (dilated) metal region. The metal mask is dilated with
    a GPU max-pool (fast, no SciPy/CPU round-trip) so SSIM windows that straddle the
    metal boundary are excluded from the mean.

    Returns a scalar tensor (mean SSIM over valid pixels), differentiable w.r.t. `pred`.
    Use `1 - compute_masked_SSIM_loss(...)` as the loss term.
    """
    if pred.dim() == 2:
        pred = pred.view(1, 1, *pred.shape)
        target = target.view(1, 1, *target.shape)

    window = create_window(window_size, channel).type_as(pred)
    pad = window_size // 2

    mu1 = F.conv2d(pred, window, padding=pad, groups=channel)
    mu2 = F.conv2d(target, window, padding=pad, groups=channel)
    mu1_sq, mu2_sq, mu1_mu2 = mu1 * mu1, mu2 * mu2, mu1 * mu2

    sigma1_sq = F.conv2d(pred * pred, window, padding=pad, groups=channel) - mu1_sq
    sigma2_sq = F.conv2d(target * target, window, padding=pad, groups=channel) - mu2_sq
    sigma12 = F.conv2d(pred * target, window, padding=pad, groups=channel) - mu1_mu2

    C1 = (0.01 * data_range) ** 2
    C2 = (0.03 * data_range) ** 2
    ssim_map = ((2 * mu1_mu2 + C1) * (2 * sigma12 + C2)) / \
               ((mu1_sq + mu2_sq + C1) * (sigma1_sq + sigma2_sq + C2))

    if metal_mask is not None:
        mm = metal_mask.type_as(ssim_map)
        if mm.dim() == 2:
            mm = mm.view(1, 1, *mm.shape)
        mm = (mm > 0.5).float()
        if dilation and int(dilation) > 0:
            k = 2 * int(dilation) + 1
            mm = F.max_pool2d(mm, kernel_size=k, stride=1, padding=int(dilation))
        valid = 1.0 - (mm > 0.5).float()
        denom = valid.sum().clamp_min(eps)
        return (ssim_map * valid).sum() / denom

    return ssim_map.mean()


def compute_masked_SSIM_per_image(img1, img2, data_range, mask=None, metalmask=None,
                                 window_size=11, channel=1, spatial_dims=2,
                                 metal_fill=None, hu_min=-1024.0, hu_max=3072.0):
    """Vectorized SSIM that returns a per-image masked mean.

    Inputs may be 2D (H,W) or 4D (B,1,H,W). Returns:
      - ssim_map: (B,1,H,W)
      - per_image: (B,) tensor of masked mean SSIM values
    """
    if not torch.is_tensor(img1):
        img1 = torch.from_numpy(np.asarray(img1)).float()
    if not torch.is_tensor(img2):
        img2 = torch.from_numpy(np.asarray(img2)).float()

    if img1.dim() == 2:
        img1 = img1.view(1, 1, *img1.shape)
        img2 = img2.view(1, 1, *img2.shape)

    if metal_fill is None:
        if float(data_range) == 1.0:
            metal_fill = (100.0 - float(hu_min)) / (float(hu_max) - float(hu_min))
        else:
            metal_fill = 100.0

    if metalmask is not None:
        if not torch.is_tensor(metalmask):
            metalmask = torch.from_numpy(np.asarray(metalmask))
        mm = metalmask.float()
        if mm.dim() == 2:
            mm = mm.view(1, 1, *mm.shape)
        if mm.dim() == 3:
            mm = mm.view(mm.size(0), 1, mm.size(1), mm.size(2))
        mm = mm.to(img1.device)
        img1 = img1.clone()
        img2 = img2.clone()
        img1[mm == 1] = float(metal_fill)
        img2[mm == 1] = float(metal_fill)

    window = create_window(window_size, channel, spatial_dims=spatial_dims).type_as(img1)
    conv_op = F.conv2d if spatial_dims == 2 else F.conv3d

    mu1 = conv_op(img1, window, padding=window_size // 2)
    mu2 = conv_op(img2, window, padding=window_size // 2)
    mu1_sq, mu2_sq = mu1.pow(2), mu2.pow(2)
    mu1_mu2 = mu1 * mu2

    sigma1_sq = conv_op(img1 * img1, window, padding=window_size // 2) - mu1_sq
    sigma2_sq = conv_op(img2 * img2, window, padding=window_size // 2) - mu2_sq
    sigma12 = conv_op(img1 * img2, window, padding=window_size // 2) - mu1_mu2

    C1, C2 = (0.01 * data_range) ** 2, (0.03 * data_range) ** 2
    ssim_map = ((2 * mu1_mu2 + C1) * (2 * sigma12 + C2)) / ((mu1_sq + mu2_sq + C1) * (sigma1_sq + sigma2_sq + C2))

    if mask is None:
        per_image = ssim_map.mean(dim=(1, 2, 3))
    else:
        if not torch.is_tensor(mask):
            mask = torch.from_numpy(np.asarray(mask))
        m = mask.float()
        if m.dim() == 2:
            m = m.view(1, 1, *m.shape)
        if m.dim() == 3:
            m = m.view(m.size(0), 1, m.size(1), m.size(2))
        m = m.to(ssim_map.device)

        masked_sum = (ssim_map * m).sum(dim=(1, 2, 3))
        denom = m.sum(dim=(1, 2, 3)).clamp_min(1.0)
        per_image = masked_sum / denom

    return ssim_map, per_image


def gaussian(window_size, sigma):
    gauss = torch.Tensor([exp(-(x - window_size // 2) ** 2 / float(2 * sigma ** 2)) for x in range(window_size)])
    return gauss / gauss.sum()


def create_window(window_size, channel, spatial_dims=2):
    _1D_window = gaussian(window_size, 1.5).unsqueeze(1)
    _2D_window = _1D_window.mm(_1D_window.t()).float().unsqueeze(0).unsqueeze(0)
    if spatial_dims == 2:
        window = _2D_window.expand(channel, 1, window_size, window_size).contiguous()
    else:
        window = _2D_window.expand(channel, 1, window_size, window_size, window_size).contiguous()
    return window
