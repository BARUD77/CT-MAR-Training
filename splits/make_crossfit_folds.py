"""Two cross-fitting folds of the grouped split v1 TRAIN set, for out-of-fold predictions (near-metal refiner).

The train part of grouped_split_v1.csv is divided into two halves BY UNIT (the leak-free groups), separately for
body and head, so the halves do not leak into each other. For fold A, half A is 'train' and everything else
(half B + the original validation set) is 'val'; fold B is the mirror. A base model trained on fold A predicts
half B out-of-fold, and vice versa. The original validation slices are never trained on in either fold.

Outputs: splits/grouped_split_v1_cfA.csv, splits/grouped_split_v1_cfB.csv (+ a printed balance check).
"""
import os
import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
df = pd.read_csv(os.path.join(HERE, "grouped_split_v1.csv"))
rng = np.random.default_rng(42)
half = pd.Series("", index=df.index)
for region in ("body", "head"):
    tr = df[(df.region == region) & (df.split == "train")]
    sizes = tr.groupby("unit").size()
    units = sizes.index.values[rng.permutation(len(sizes))]
    a = b = 0
    for u in units:                                   # greedy: give each unit to the smaller half
        if a <= b:
            half[tr.index[tr.unit == u]] = "A"; a += sizes[u]
        else:
            half[tr.index[tr.unit == u]] = "B"; b += sizes[u]
    print(f"{region}: {len(sizes)} train units -> half A {a} slices, half B {b} slices")
for fold, other in (("A", "B"), ("B", "A")):
    out = df[["region", "id", "unit"]].copy()
    out["split"] = np.where(half == fold, "train", "val")
    out["half"] = half.replace("", "orig_val")
    out[["region", "id", "split", "unit", "half"]].to_csv(os.path.join(HERE, f"grouped_split_v1_cf{fold}.csv"),
                                                          index=False)
# checks: halves are disjoint in units, original val never trained on
ua = set(df.unit[half == "A"]); ub = set(df.unit[half == "B"])
assert not (ua & ub), "a unit is in both halves"
assert (half[df.split == "val"] == "").all()
print("units disjoint between halves; original validation slices excluded from both training halves")
