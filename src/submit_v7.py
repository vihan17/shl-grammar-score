"""Submission v7: whisper-large-v3 embeddings (2560-d) via the v5 recipe.

  python submit_v7.py --embeds embeds_large.npz --out submission_v7_large.csv

Identical to submit_v5 (raw block + PCA32 block, xgb+cat 5-fold multi-seed,
OOF-chosen blend weights) except the embedding source file.
"""

import argparse
import json
import os
import sys

for _v in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS"):
    os.environ.setdefault(_v, "1")

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from exp3 import make_cat, make_xgb, prep  # noqa
from local_train import load  # noqa: E402
from submit_v5 import add_cols, run  # noqa: E402
from train_v3 import clip_pred, metrics  # noqa: E402

sys.stdout.reconfigure(encoding="utf-8", errors="replace")


def load_embeds_path(path):
    if not os.path.exists(path):
        raise SystemExit(f"{path} not found")
    z = np.load(path)
    X = z["X"] if "X" in z else z[list(z.keys())[0]]
    X = np.asarray(X, dtype=np.float32)
    if "done" in z:
        done = z["done"]
        if not done.all():
            raise SystemExit(f"{path}: only {int(done.sum())}/{len(done)} rows done")
    return X


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--embeds", default="embeds_large.npz")
    ap.add_argument("--pca-d", type=int, default=32)
    ap.add_argument("--out", default="submission_v7_large.csv")
    ap.add_argument("--summary", default="submit_v7_summary.json")
    ap.add_argument("--oof", default="oof_v7.csv")
    a = ap.parse_args()

    A_tr, A_te, y, ids = load()
    E = load_embeds_path(a.embeds)
    n_tr = len(A_tr)
    assert len(E) == n_tr + len(A_te), f"embeds {len(E)} != {n_tr + len(A_te)}"
    assert np.isfinite(E).all(), "nan/inf in embeds"
    print(f"embeds {E.shape} from {a.embeds} | base {A_tr.shape}")

    common = [c for c in A_tr.columns if c in A_te.columns and c != "filename"]
    base_tr, base_te = A_tr[common], A_te[common]

    blocks = {}
    raw_tr = pd.DataFrame(E[:n_tr],
                          columns=[f"emb_{i}" for i in range(E.shape[1])])
    raw_te = pd.DataFrame(E[n_tr:],
                          columns=[f"emb_{i}" for i in range(E.shape[1])])
    blocks["raw"] = (pd.concat([base_tr.reset_index(drop=True), raw_tr],
                               axis=1),
                     pd.concat([base_te.reset_index(drop=True), raw_te],
                               axis=1))
    from sklearn.decomposition import PCA
    pca = PCA(n_components=a.pca_d, random_state=0)
    ptr = pca.fit_transform(E[:n_tr])
    pte = pca.transform(E[n_tr:])
    print(f"PCA{a.pca_d} var explained {pca.explained_variance_ratio_.sum():.3f}")
    blocks[f"pca{a.pca_d}"] = add_cols(A_tr, A_te, ptr, pte, "emb_")

    oofs, preds = {}, {}
    for name, (a_tr, a_te) in blocks.items():
        X, X_te = prep(a_tr, a_te)
        print(f"\n=== {name} {X.shape} / {X_te.shape} ===", flush=True)
        oofs[name], preds[name] = run(X, y, X_te)

    names = list(oofs)
    best = (9.9, 1.0, "")
    grids = []
    if len(names) == 2:
        for w in np.arange(0.0, 1.01, 0.05):
            grids.append((w, {names[0]: w, names[1]: 1 - w}))
    else:
        grids = [(1.0, {n: 1.0 if n == names[0] else 0.0 for n in names})]
    for w, mix in grids:
        p = clip_pred(sum(mix[n] * oofs[n] for n in names))
        r = metrics(y, p)[0]
        if r < best[0]:
            best = (r, w, dict(mix))
    print(f"\nBEST BLOCK BLEND rmse={best[0]:.4f} mix={best[2]}")

    final = clip_pred(sum(best[2][n] * preds[n] for n in names))
    m = metrics(y, clip_pred(sum(best[2][n] * oofs[n] for n in names)))
    sub = pd.DataFrame({"filename": ids, "label": final})
    sub.to_csv(a.out, index=False)
    print("\nsubmission", sub.shape, "->", a.out)
    print(f"OOF rmse={m[0]:.4f} pearson={m[1]:.4f}")
    json.dump({"oof_rmse": round(m[0], 4), "oof_pearson": round(m[1], 4),
               "mix": best[2], "embeds": a.embeds, "out": a.out},
              open(a.summary, "w"), indent=1)
    pd.DataFrame({"filename": pd.read_csv("data/train.csv").filename,
                  "y_true": y.values,
                  "y_oof": clip_pred(sum(best[2][n] * oofs[n]
                                         for n in names))}).to_csv(
        a.oof, index=False)


if __name__ == "__main__":
    main()
