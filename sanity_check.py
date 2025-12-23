# eval_sanity.py
import os
import re
import math
import argparse
import numpy as np
from skimage.metrics import structural_similarity as sk_ssim
from skimage.metrics import peak_signal_noise_ratio as sk_psnr

H, W = 512, 512
HU_MIN, HU_MAX = -2000.0, 6000.0   # window -> data_range = 4000 HU
DATA_RANGE_AFTER_01 = 1.0

# --- filename patterns you gave ---
RE_MA   = re.compile(r'^training_body_metalart_img(\d+)_512x512x1\.(npy|raw)$', re.IGNORECASE)
RE_MASK = re.compile(r'^training_body_metalonlymask_img(\d+)_512x512x1\.(npy|raw)$', re.IGNORECASE)
# RE_GT   = re.compile(r'^Simulated_Image_(\d+)_512x512\.(npy|raw)$', re.IGNORECASE)
RE_GT   = re.compile(r'^training_body_nometal_img(\d+)_512x512x1\.(npy|raw)$', re.IGNORECASE)

def load_img(path):
    ext = os.path.splitext(path)[1].lower()
    if ext == ".npy":
        arr = np.load(path).astype(np.float32)
        # accept (H,W) or (H,W,1)
        if arr.ndim == 3 and arr.shape[-1] == 1:
            arr = arr[..., 0]
    elif ext == ".raw":
        # float32 raw, exactly 512*512 values
        arr = np.fromfile(path, dtype=np.float32, count=H*W)
        if arr.size != H*W:
            raise ValueError(f"Unexpected size in {path}: {arr.size} != {H*W}")
        arr = arr.reshape(H, W)
    else:
        raise ValueError(f"Unsupported extension for {path}")
    if arr.shape != (H, W):
        raise ValueError(f"Unexpected shape {arr.shape} for {path}")
    return arr

def window_to_01(img, lo=HU_MIN, hi=HU_MAX):
    # window to [lo,hi], then scale to [0,1]
    img_w = np.clip(img, lo, hi)
    return (img_w - lo) / max(hi - lo, 1e-8)

def rmse(a, b):
    return float(np.sqrt(np.mean((a - b) ** 2, dtype=np.float64)))

def build_id_map(folder, regex):
    m = {}
    for f in os.listdir(folder):
        g = regex.match(f)
        if g:
            m[int(g.group(1))] = f
    return m

def main():
    ap = argparse.ArgumentParser(description="SSIM/PSNR/RMSE sanity check on first N matched pairs")
    ap.add_argument("--ma_dir", required=True, help="Folder with MA images (training_body_metalart_img{ID}_512x512x1.*)")
    ap.add_argument("--mask_dir", required=True, help="Folder with metal masks (training_body_metalonlymask_img{ID}_512x512x1.*)")
    ap.add_argument("--gt_dir", required=True, help="Folder with GT images (Simulated_Image_{ID}_512x512.*)")
    ap.add_argument("--count", type=int, default=20, help="How many IDs to evaluate (default: 20)")
    ap.add_argument("--add_hu_shift", type=float, default=0.0,
                   help="Optional constant to add to both MA and GT before windowing (default 0.0)")
    args = ap.parse_args()

    ma_map   = build_id_map(args.ma_dir, RE_MA)
    mask_map = build_id_map(args.mask_dir, RE_MASK)
    gt_map   = build_id_map(args.gt_dir, RE_GT)

    common_ids = sorted(set(ma_map).intersection(mask_map).intersection(gt_map))
    if not common_ids:
        raise RuntimeError("No common IDs across MA/Mask/GT. Check filenames and folders.")
    ids = common_ids[:args.count]

    print(f"Found {len(common_ids)} common IDs; evaluating first {len(ids)}: {ids[:10]}{'...' if len(ids)>10 else ''}")

    rows = []
    ssim_u_sum = psnr_u_sum = rmse_u_sum = 0.0
    ssim_m_sum = psnr_m_sum = rmse_m_sum = 0.0

    for i in ids:
        ma_path   = os.path.join(args.ma_dir,   ma_map[i])
        mask_path = os.path.join(args.mask_dir, mask_map[i])
        gt_path   = os.path.join(args.gt_dir,   gt_map[i])

        ma  = load_img(ma_path)
        gt  = load_img(gt_path)
        msk = load_img(mask_path)  # can be float; we binarize below

        if args.add_hu_shift != 0.0:
            ma = ma + args.add_hu_shift
            gt = gt + args.add_hu_shift

        # window & scale to [0,1]
        ma01 = window_to_01(ma)
        gt01 = window_to_01(gt)

        # --- Unmasked metrics (full image) ---
        ssim_u = float(sk_ssim(gt01, ma01, data_range=DATA_RANGE_AFTER_01, win_size=11, gaussian_weights=True))
        psnr_u = float(sk_psnr(gt01, ma01, data_range=DATA_RANGE_AFTER_01))
        rmse_u = rmse(gt01, ma01)

        # --- Masked metrics (exclude metal) ---
        # Expect mask to be >0.5 inside metal; set those pixels to 0 in both
        metal = msk > 0.5
        gt01_m = gt01.copy()
        ma01_m = ma01.copy()
        gt01_m[metal] = 0.0
        ma01_m[metal] = 0.0

        ssim_m = float(sk_ssim(gt01_m, ma01_m, data_range=DATA_RANGE_AFTER_01, win_size=11, gaussian_weights=True))
        psnr_m = float(sk_psnr(gt01_m, ma01_m, data_range=DATA_RANGE_AFTER_01))
        rmse_m = rmse(gt01_m, ma01_m)

        rows.append({
            "id": i,
            "ssim_unmasked": ssim_u, "psnr_unmasked": psnr_u, "rmse_unmasked": rmse_u,
            "ssim_masked":   ssim_m, "psnr_masked":   psnr_m, "rmse_masked":   rmse_m,
            "gt_path": gt_path, "ma_path": ma_path, "mask_path": mask_path
        })

        ssim_u_sum += ssim_u; psnr_u_sum += psnr_u; rmse_u_sum += rmse_u
        ssim_m_sum += ssim_m; psnr_m_sum += psnr_m; rmse_m_sum += rmse_m

    n = len(rows)
    print("\n=== Averages over", n, "samples ===")
    print(f"Unmasked: SSIM={ssim_u_sum/n:.6f}  PSNR={psnr_u_sum/n:.3f}  RMSE={rmse_u_sum/n:.6f}")
    print(f"Masked  : SSIM={ssim_m_sum/n:.6f}  PSNR={psnr_m_sum/n:.3f}  RMSE={rmse_m_sum/n:.6f}\n")

    # Write a results TXT next to the script (Windows-safe)
    out_name = "results_sanity.txt"
    with open(out_name, "w", encoding="utf-8") as f:
        f.write("# id ssim_unmasked psnr_unmasked rmse_unmasked ssim_masked psnr_masked rmse_masked gt_path ma_path mask_path\n")
        for r in rows:
            f.write(f"{r['id']:06d} {r['ssim_unmasked']:.6f} {r['psnr_unmasked']:.3f} {r['rmse_unmasked']:.6f} "
                    f"{r['ssim_masked']:.6f} {r['psnr_masked']:.3f} {r['rmse_masked']:.6f} "
                    f"{r['gt_path']} {r['ma_path']} {r['mask_path']}\n")
    print(f"Wrote per-sample results to {out_name}")
    print("Done.")

if __name__ == "__main__":
    main()
