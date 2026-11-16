"""Apply a saved train/val split file to CTMetalArtifactDataset instances without changing aapm_dataset.py.

The split file is a CSV with columns region, id, split ("train"/"val"), keyed by (region, image id), so it does
not depend on folder paths. The dataset still finds and loads files exactly as before; only the assignment
of slices to train/val changes. Usage:

    from splits.split_utils import apply_split_file
    apply_split_file(train_ds, val_ds, "splits/grouped_split_v1.csv")
"""
import csv
import os
import re

_RX = re.compile(r'(?:training|test)_(body|head)_metalart_img(\d+)_', re.IGNORECASE)


def _key(pair):
    m = _RX.search(os.path.basename(pair[0]))
    if not m:
        raise ValueError(f"cannot parse region/id from {pair[0]!r}")
    return m.group(1).lower(), int(m.group(2))


def read_split_file(path):
    with open(path, newline="") as f:
        rows = list(csv.DictReader(f))
    out = {(r["region"].lower(), int(r["id"])): r["split"] for r in rows}
    bad = {v for v in out.values()} - {"train", "val"}
    if bad:
        raise ValueError(f"{path}: unexpected split labels {bad}")
    return out


def apply_split_file(train_ds, val_ds, path, verbose=True):
    """Reassign train_ds.pairs / val_ds.pairs from the split file. Both datasets must have been built from the
    same folders (their pairs together cover every matched slice). Pairs are sorted by (region, id)."""
    split = read_split_file(path)
    pairs = {}
    for p in list(train_ds.pairs) + list(val_ds.pairs):
        pairs[_key(p)] = p
    missing = sorted(k for k in pairs if k not in split)
    if missing:
        raise ValueError(f"{len(missing)} slices are not in {path}, e.g. {missing[:5]}")
    keys = sorted(pairs)
    train_ds.pairs = [pairs[k] for k in keys if split[k] == "train"]
    val_ds.pairs = [pairs[k] for k in keys if split[k] == "val"]
    if verbose:
        def counts(ps):
            h = sum(1 for p in ps if "_head_" in os.path.basename(p[0]).lower())
            return len(ps) - h, h
        (tb, th), (vb, vh) = counts(train_ds.pairs), counts(val_ds.pairs)
        unused = len(split) - len(pairs)
        print(f"[split_file] {os.path.basename(path)}: train={len(train_ds.pairs)} (body {tb} / head {th}) | "
              f"val={len(val_ds.pairs)} (body {vb} / head {vh})"
              + (f" | {unused} listed slices not present in these folders" if unused else ""))
    return train_ds, val_ds
