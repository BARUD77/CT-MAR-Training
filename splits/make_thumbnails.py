"""Cache 64x64 GT thumbnails (same definition as colab/data_audit.py) for every training slice.

Output: splits/thumbs.npz with region, id, thumb (N, 64*64 float32, NCC-normalised). Only GT images are read;
the test set is not touched.
"""
import argparse
import os
import re
import sys
import time
from concurrent.futures import ProcessPoolExecutor

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(os.path.dirname(HERE), "colab"))
from data_audit import load2d, thumbnail  # noqa: E402  (no torch import at module level)

RX = re.compile(r'^training_(body|head)_nometal_img(\d+)_', re.IGNORECASE)


def work(path):
    return thumbnail(load2d(path).astype(np.float32)).ravel()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--gt_dir", default=r"D:\data_npy\GT")
    ap.add_argument("--out", default=os.path.join(HERE, "thumbs.npz"))
    ap.add_argument("--workers", type=int, default=4)
    a = ap.parse_args()
    files = []
    for f in sorted(os.listdir(a.gt_dir)):
        m = RX.match(f)
        if m:
            files.append((m.group(1).lower(), int(m.group(2)), os.path.join(a.gt_dir, f)))
    print("GT files:", len(files), flush=True)
    t0 = time.time()
    with ProcessPoolExecutor(a.workers) as ex:
        T = np.stack(list(ex.map(work, [f[2] for f in files], chunksize=64)))
    T = T - T.mean(axis=1, keepdims=True)
    T /= np.linalg.norm(T, axis=1, keepdims=True) + 1e-6
    np.savez_compressed(a.out, region=np.array([f[0] for f in files]), id=np.array([f[1] for f in files], np.int32),
                        thumb=T.astype(np.float32))
    print(f"saved {a.out} ({time.time() - t0:.0f} s)", flush=True)


if __name__ == "__main__":
    main()
