"""Build a scan/patient-wise ("grouped") train/val split for the AAPM CT-MAR training data, and check leakage.

Why: the original split (aapm_dataset.py: per-slice train_test_split, seed 42, 10%, stratified by region) puts
neighbouring slices of the same scan, and re-simulations of the same source slice, on both sides
(see colab/data_audit.py). No patient IDs are provided, so units are reconstructed from the images:

  * slice IDs follow scan order: consecutive IDs of one scan are near-identical (thumbnail NCC ~0.997)
  * HEAD (UCLH, ~10 patients): IDs form sweeps of 10-slice blocks whose mean images drift smoothly and drop
    sharply between sweeps. Unit = sweep, cut where adjacent block means have NCC < HEAD_CUT.
  * BODY (DeepLesion, ~1.2k patients): many short scans of varying length, patients' studies presumably
    stored next to each other. Unit = contiguous chunk of ~BODY_CHUNK IDs, each cut moved to the clearest
    scan break (lowest consecutive NCC) within +-BODY_SNAP IDs, so no scan is split.
  * Re-simulated twins (same anatomy, new metal/noise, at distant IDs; NCC >= TWIN_SIM, >= 2 links) force
    their units together (union-find).
Units are shuffled (seed 42) and added to validation until ~10% of each region (same ratio as before).

Leakage check (same measure as the data audit): for every validation slice, the most similar TRAINING slice
(thumbnail NCC, same region); share above the random-pair 99.9th percentile and at twin level (>= 0.999);
share of validation slices whose ID neighbour (+-1) is in training. Old split vs new split.

Inputs: splits/thumbs.npz (make_thumbnails.py), evaluation_results_data_audit/groups.csv (old split labels).
Outputs: splits/grouped_split_v1.csv (region,id,split,unit), *_report.json, *_leakage.png.
Only training-set GT images were used. The test set is not touched.
"""
import json
import os

import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.dirname(HERE)
NAME = "grouped_split_v1"
SEED, VAL_FRAC = 42, 0.10
SEG_SIM = 0.80        # consecutive IDs with NCC >= this are the same scan run
MIN_BLOCK = 5         # runs shorter than this are 'odd' slices, attached to the preceding run
HEAD_CUT = 0.50       # head: new sweep when adjacent block means have NCC < this
BODY_CHUNK, BODY_SNAP = 50, 15
TWIN_SIM, TWIN_MIN_LINKS, TWIN_MIN_GAP = 0.999, 1, 10


class UF:
    def __init__(self, n):
        self.p = np.arange(n)

    def find(self, a):
        while self.p[a] != a:
            self.p[a] = self.p[self.p[a]]
            a = self.p[a]
        return a

    def union(self, a, b):
        a, b = self.find(a), self.find(b)
        if a != b:
            self.p[max(a, b)] = min(a, b)


def consecutive_sim(ids, X):
    return np.array([float(X[k] @ X[k + 1]) if ids[k + 1] == ids[k] + 1 else -1.0 for k in range(len(ids) - 1)])


def head_units(ids, X):
    cons = consecutive_sim(ids, X)
    starts = np.concatenate([[0], np.where(cons < SEG_SIM)[0] + 1])
    ends = np.concatenate([starts[1:], [len(ids)]])
    unit = np.zeros(len(ids), int)
    u, prev_mean = 0, None
    for s, e in zip(starts, ends):
        if e - s >= MIN_BLOCK:
            m = X[s:e].mean(0)
            m = (m - m.mean()) / (np.linalg.norm(m - m.mean()) + 1e-6)
            if prev_mean is not None and float(m @ prev_mean) < HEAD_CUT:
                u += 1
            prev_mean = m
        unit[s:e] = u          # short 'odd' runs stay with the current sweep
    return unit


def body_units(ids, X):
    cons = consecutive_sim(ids, X)        # cons[k] links position k and k+1
    cuts, pos = [], BODY_CHUNK
    while pos < len(ids) - BODY_CHUNK // 2:
        lo, hi = max(1, pos - BODY_SNAP), min(len(ids) - 1, pos + BODY_SNAP)
        k = lo + int(np.argmin(cons[lo - 1:hi - 1]))   # cut before position k (between k-1 and k)
        cuts.append(k)
        pos = k + BODY_CHUNK
    unit = np.zeros(len(ids), int)
    for c in cuts:
        unit[c:] += 1
    return unit, cons, cuts


def twin_merge(unit, ids, X):
    import torch
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    Xt = torch.from_numpy(X).to(dev)
    links = {}
    for s in range(0, len(ids), 1024):
        S = (Xt[s:s + 1024] @ Xt.T).cpu().numpy()
        a, b = np.where(S >= TWIN_SIM)
        a = a + s
        far = np.abs(ids[a] - ids[b]) > TWIN_MIN_GAP
        for i, j in zip(a[far], b[far]):
            if unit[i] != unit[j]:
                key = (min(unit[i], unit[j]), max(unit[i], unit[j]))
                links[key] = links.get(key, 0) + 1
    uf = UF(unit.max() + 1)
    n_merge = 0
    for (p, q), c in links.items():
        if c >= 2 * TWIN_MIN_LINKS:      # each link is counted from both sides
            uf.union(p, q)
            n_merge += 1
    return np.array([uf.find(u) for u in unit]), n_merge


def assign(unit, rng):
    uniq, cnt = np.unique(unit, return_counts=True)
    target = VAL_FRAC * len(unit)
    order = rng.permutation(len(uniq))
    val_units, n = [], 0
    for k in order:                       # random units, skipping any that would overshoot by > 10%
        if n >= 0.97 * target:
            break
        if n + cnt[k] <= 1.10 * target:
            val_units.append(uniq[k])
            n += cnt[k]
    if n < 0.97 * target:                 # few large units (head): take the single unit closest to the target
        k = int(np.argmin(np.abs(cnt - target)))
        val_units, n = [uniq[k]], cnt[k]
    return np.isin(unit, val_units)


def leakage(regions, ids, X, is_val, null_p999):
    import torch
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    out = {}
    for r in ("body", "head"):
        sel = regions == r
        v, t = np.where(sel & is_val)[0], np.where(sel & ~is_val)[0]
        Tt = torch.from_numpy(X[t]).to(dev)
        nn = np.concatenate([(torch.from_numpy(X[v[s:s + 1024]]).to(dev) @ Tt.T).max(1).values.cpu().numpy()
                             for s in range(0, len(v), 1024)])
        key_train = set(ids[t].tolist())
        adj = np.mean([(i - 1 in key_train) or (i + 1 in key_train) for i in ids[v]])
        out[r] = {"n_val": int(len(v)), "n_train": int(len(t)), "median_nn_sim": float(np.median(nn)),
                  "frac_nn_above_null_p99.9": float(np.mean(nn > null_p999[r])),
                  "frac_nn_twin_level_0.999": float(np.mean(nn >= 0.999)),
                  "frac_val_with_id_neighbour_in_train": float(adj), "_nn": nn}
    return out


def main():
    z = np.load(os.path.join(HERE, "thumbs.npz"))
    regions, ids, X = z["region"], z["id"].astype(int), z["thumb"]
    old = pd.read_csv(os.path.join(REPO, "evaluation_results_data_audit", "groups.csv"))
    old_split = {(r, int(i)): s for r, i, s in zip(old.region, old.id, old.split)}
    audit = json.load(open(os.path.join(REPO, "evaluation_results_data_audit", "summary.json")))
    leak_key = next(k for k in audit if "leakage" in k)
    null_p999 = {r: v["p99.9"] for r, v in audit[leak_key]["null_random_pairs"].items()}

    rng = np.random.default_rng(SEED)
    is_val = np.zeros(len(ids), bool)
    unit_all = np.zeros(len(ids), int)
    report = {"params": dict(seed=SEED, val_frac=VAL_FRAC, seg_sim=SEG_SIM, min_block=MIN_BLOCK, head_cut=HEAD_CUT,
                             body_chunk=BODY_CHUNK, body_snap=BODY_SNAP, twin_sim=TWIN_SIM,
                             twin_min_links=TWIN_MIN_LINKS)}
    offset = 0
    for r in ("body", "head"):
        idx = np.where(regions == r)[0]
        idx = idx[np.argsort(ids[idx])]
        Xi, ii = X[idx], ids[idx]
        if r == "head":
            unit = head_units(ii, Xi)
            cut_info = {}
        else:
            unit, cons, cuts = body_units(ii, Xi)
            cut_sims = cons[np.array(cuts) - 1]
            cut_info = {"n_cuts": len(cuts), "cut_consecutive_sim_median": float(np.median(cut_sims)),
                        "frac_cuts_at_clear_break_lt_0.8": float(np.mean(cut_sims < 0.8))}
        n_before = len(np.unique(unit))
        unit, n_merge = twin_merge(unit, ii, Xi)
        _, inv = np.unique(unit, return_inverse=True)
        unit = inv
        v = assign(unit, rng)
        is_val[idx] = v
        unit_all[idx] = unit + offset
        offset += unit.max() + 1
        _, cnt = np.unique(unit, return_counts=True)
        report[r] = {"n": int(len(idx)), "units_before_twin_merge": int(n_before), "twin_merges": int(n_merge),
                     "units": int(len(cnt)), "unit_size_median": float(np.median(cnt)), "unit_size_max": int(cnt.max()),
                     "val_units": int(len(np.unique(unit[v]))), "val_slices": int(v.sum()),
                     "val_frac": float(v.mean()), **cut_info}
        if r == "head":
            starts = [int(ii[unit == u].min()) for u in np.unique(unit)]
            report[r]["unit_id_ranges"] = [[int(ii[unit == u].min()), int(ii[unit == u].max()), int((unit == u).sum()),
                                            bool(v[unit == u].all())] for u in np.unique(unit)]

    old_val = np.array([old_split[(r, i)] == "val" for r, i in zip(regions, ids)])
    L_old = leakage(regions, ids, X, old_val, null_p999)
    L_new = leakage(regions, ids, X, is_val, null_p999)
    report["leakage_old_per_slice_split"] = {r: {k: v for k, v in d.items() if k != "_nn"} for r, d in L_old.items()}
    report["leakage_new_grouped_split"] = {r: {k: v for k, v in d.items() if k != "_nn"} for r, d in L_new.items()}

    df = pd.DataFrame({"region": regions, "id": ids, "split": np.where(is_val, "val", "train"), "unit": unit_all})
    df = df.sort_values(["region", "id"])
    df.to_csv(os.path.join(HERE, f"{NAME}.csv"), index=False)
    with open(os.path.join(HERE, f"{NAME}_report.json"), "w") as f:
        json.dump(report, f, indent=2)

    # ---- print
    for r in ("body", "head"):
        print(f"\n=== {r} ===")
        print({k: v for k, v in report[r].items() if k != "unit_id_ranges"})
        if r == "head":
            print("  head units [first id, last id, n slices, in val]:", report[r]["unit_id_ranges"])
        for name, L in (("OLD per-slice split", L_old), ("NEW grouped split ", L_new)):
            d = L[r]
            print(f"  {name}: val {d['n_val']:5d} | nearest-train similarity median {d['median_nn_sim']:.4f} | "
                  f"above random-pair p99.9 {100 * d['frac_nn_above_null_p99.9']:5.1f}% | twin-level (>=0.999) "
                  f"{100 * d['frac_nn_twin_level_0.999']:5.1f}% | ID neighbour in train "
                  f"{100 * d['frac_val_with_id_neighbour_in_train']:5.1f}%")

    # ---- figure
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(1, 2, figsize=(11, 4), constrained_layout=True)
    for ax, r in zip(axes, ("body", "head")):
        bins = np.linspace(0.5, 1.0, 101)
        ax.hist(L_old[r]["_nn"], bins=bins, color="#eb6834", alpha=0.75, label="old: per-slice random split")
        ax.hist(L_new[r]["_nn"], bins=bins, color="#2a78d6", alpha=0.75, label="new: grouped (scan/patient) split")
        ax.axvline(null_p999[r], color="#898781", ls="--", lw=1.5, label="random pairs, 99.9th pct")
        ax.set_yscale("log")
        ax.set_title(f"{r}: validation slice vs most similar training slice", fontsize=10)
        ax.set_xlabel("thumbnail similarity (NCC)")
        ax.set_ylabel("validation slices (log)")
        for s in ("top", "right"):
            ax.spines[s].set_visible(False)
    axes[0].legend(frameon=False, fontsize=8, loc="upper left")
    fig.savefig(os.path.join(HERE, f"{NAME}_leakage.png"), dpi=150)
    print(f"\nsaved {NAME}.csv, {NAME}_report.json, {NAME}_leakage.png in {HERE}")


if __name__ == "__main__":
    main()
