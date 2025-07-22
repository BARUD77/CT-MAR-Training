import os, re
import numpy as np
from skimage.transform import radon, iradon
from skimage.filters import gaussian
from sklearn.cluster import KMeans

# ——— CONFIG ———
CT_FOLDER    = '/home/user/Desktop/Briya/CONVENTIONAL CT/dataset/body1/body1/npy files/baseline_npy'
MASK_FOLDER  = '/home/user/Desktop/Briya/CONVENTIONAL CT/dataset/body1/body1/npy files/MAsk_npy'
OUT_FOLDER_NMAR = '/home/user/Desktop/Briya/CONVENTIONAL CT/dataset/body1/body1/NMAR_CT'
MIU_AIR      = 0
MIU_WATER    = 0.19
os.makedirs(OUT_FOLDER_NMAR, exist_ok=True)

# ——— helper funcs (same as before) ———
def linear_attenuation(im, reverse=False):
    if not reverse:
        return MIU_WATER * (1 + im / 1000)
    else:
        return (im / MIU_WATER - 1) * 1000

def circle_mask(im):
    H, W = im.shape
    r = min(H, W)//2
    yy, xx = np.ogrid[:H, :W]
    return (yy - H//2)**2 + (xx - W//2)**2 > r**2

def proj_interp(proj, metal_trace):
    out = np.zeros_like(proj)
    for i in range(proj.shape[1]):
        row      = proj[:, i].copy()
        mt       = metal_trace[:, i]
        bad_idx  = np.nonzero(mt)[0]
        good_idx = np.where(mt == 0)[0]
        row[bad_idx] = np.interp(bad_idx, good_idx, row[good_idx])
        out[:, i]    = row
    return out

def nmar_proj_interp(proj, proj_prior, metal_trace):
    proj_prior[proj_prior < 0] = 0
    eps = 1e-6
    proj_prior = proj_prior + eps
    proj_norm = proj / proj_prior
    proj_norm_interp = proj_interp(proj_norm, metal_trace)
    proj_nmar = proj_norm_interp * proj_prior
    proj_nmar[metal_trace == 0] = proj[metal_trace == 0]
    return proj_nmar

# ——— gather & sort your CT filenames ———
ct_files = sorted(
    [f for f in os.listdir(CT_FOLDER) if f.endswith('.npy')],
    key=lambda fn: int(re.search(r'_img(\d+)_', fn).group(1))
)

# precompute your projection angles
theta = np.linspace(0., 180., 512, endpoint=False)  # assuming 512×512

for ct_fn in ct_files:
    mask_fn = ct_fn.replace('metalart', 'metalonlymask')
    ct_path   = os.path.join(CT_FOLDER, ct_fn)
    mask_path = os.path.join(MASK_FOLDER, mask_fn)

    # skip if no mask
    if not os.path.isfile(mask_path):
        print(f"⚠️  Skipping {ct_fn}: missing mask {mask_fn}")
        continue

    # load data
    im       = np.load(ct_path).astype(np.float32)
    metal_bw = np.load(mask_path).astype(bool)

    # build masks & raw sinogram
    cm     = circle_mask(im)
    im_raw = linear_attenuation(im);  im_raw[cm] = MIU_AIR
    sino       = radon(im_raw, theta=theta, circle=True)
    metal_sino = radon(metal_bw.astype(float), theta=theta, circle=True) > 0

    # LI step (needed to build the NMAR prior)
    sino_li = proj_interp(sino, metal_sino)
    im_li   = iradon(sino_li, theta=theta, circle=True, filter_name='ramp')

    # build NMAR prior from LI result
    im_prior = im_li.copy();  im_prior[metal_bw] = MIU_WATER
    im_smooth = gaussian(im_prior, sigma=1)

    # k-means to find air/water/bone
    flat   = im_prior.reshape(-1,1)
    kmeans = KMeans(n_clusters=3,
                    init=np.array([[MIU_AIR],[MIU_WATER],[2*MIU_WATER]]),
                    n_init=1).fit(flat)
    labels = kmeans.predict(flat).reshape(im.shape)
    centers= kmeans.cluster_centers_.flatten()
    order  = np.argsort(centers)   # 0=air,1=water,2=bone
    air_lbl, water_lbl, bone_lbl = order

    # set thresholds from the correct clusters
    thresh_water = np.min( im_prior[ labels == water_lbl ] )
    thresh_bone  = max(1.2*MIU_WATER,
                       np.min( im_prior[ labels == bone_lbl ] ))

    # create piecewise‐constant prior image
    prior_img = im_smooth.copy()
    prior_img[im_smooth <= thresh_water] = MIU_AIR
    mask_bone = (im_smooth > thresh_water) & (im_smooth < thresh_bone)
    prior_img[mask_bone]                = MIU_WATER
    prior_img[im_smooth >= thresh_bone] = 2*MIU_WATER

    # project prior and do NMAR interpolation
    proj_prior = radon(prior_img, theta=theta, circle=True)
    sino_nmar  = nmar_proj_interp(sino, proj_prior, metal_sino)

    # reconstruct and convert back to HU
    im_nmar = iradon(sino_nmar, theta=theta, circle=True, filter_name='ramp')
    recon_nmar = linear_attenuation(im_nmar, reverse=True)
    recon_nmar[metal_bw] = im[metal_bw]
    recon_nmar[cm]       = im[cm]


    # SAVE with the same basename as the CT input
    out_fn = ct_fn
    out_path = os.path.join(OUT_FOLDER_NMAR, out_fn)
    np.save(out_path, recon_nmar.astype(np.float32))
    print(f"Saved NMAR: {out_path}")
