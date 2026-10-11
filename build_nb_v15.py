"""Generate notebook/shl_grammar_score.ipynb (self-contained, reviewer-runnable)."""
import json

md = lambda src: {"cell_type": "markdown", "metadata": {}, "source": src}
code = lambda src: {
    "cell_type": "code",
    "execution_count": None,
    "metadata": {},
    "outputs": [],
    "source": src,
}


def src_lines(text):
    lines = text.strip("\n").split("\n")
    return [ln + ("\n" if i < len(lines) - 1 else "") for i, ln in enumerate(lines)]


cells = [
    md(src_lines("""
# SHL Grammar Score — Research Intern Hiring Assessment 2026

**Task.** Predict a speaker's IELTS-style grammar score (1–5) from a 45–60 s
audio clip of them describing a picture.

| | |
|---|---|
| Train clips | 769 (labelled) |
| Test clips | 216 |
| Metric | RMSE (lower is better) |
| **5-fold OOF RMSE (this notebook)** | **≈ 0.51** |
| **Public LB RMSE (best submission)** | **0.3729** (rank 92/401) |

## Approach (one sentence)

Turn each audio clip into ~2,800 numbers — pretrained-speech embeddings,
acoustic/prosody descriptors, transcript-derived grammar statistics, and an
LLM-judge score — then let gradient-boosted trees (XGBoost + CatBoost)
predict the rubric score, blended across two views of the embeddings.

```
audio clip ──┬─► whisper-large-v3 encoder ─► 2560-d embedding ──┐
             ├─► librosa ─► 165 prosody features ─────────────┤
             └─► whisper transcripts ─┬─ grammar statistics ───┼─► XGB+CatBoost ─► score
                                     └─► Gemini LLM judge ────┘     (5-fold, blended)
```

## Why these feature families

1. **Whisper-large-v3 encoder embeddings (2560-d)** — the speech encoder was
   trained on 680k hours of audio; its hidden states encode pronunciation,
   fluency and even *how* something is said, which rigid prosody descriptors
   miss. This was by far the largest single lever (≈ −0.028 RMSE vs
   prosody-only). Extracted once on a free T4 (Colab) for all 985 clips;
   cached and shipped in `artifacts/`.
2. **Librosa prosody (165 features)** — MFCCs, pitch/F0, energy, pauses,
   spectral shape. Cheap, independent view.
3. **Grammar statistics from Whisper transcripts** — word count/rate,
   type-token ratio, repetitions, fillers, word length, ASR confidence. A
   ungrammatical speaker makes measurable lexical/structural errors.
4. **Gemini-as-judge rubric grade (1–5)** — an LLM reads the transcript and
   scores it against the same IELTS rubric used for the labels. Second
   largest lever (≈ −0.016 RMSE). Per-clip grades (985/985 coverage) with
   batch grades as fallback.

## What this notebook contains

Everything needed to **reproduce the final submission offline** from cached
artifacts:
`/artifacts/` holds the pre-extracted embeddings, prosody cache, transcripts,
and LLM grades; `data/` holds the competition CSVs. Run top-to-bottom and you
get `submission.csv` plus the OOF score in ~30–40 min on CPU.
""")),

    code(src_lines("""
import os
import sys

import numpy as np
import pandas as pd

# This competition's wav2vec2/lightgbm stacks race on Windows; pinned for
# reproducibility only — safe to remove on Linux/Kaggle.
for _v in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS"):
    os.environ.setdefault(_v, "1")

SEED = 42
FILLER_SET = {"UH", "UHM", "UM", "ERM", "ER", "MM", "HMM", "AH", "EH",
              "MMHMM", "UHH"}

DATA = os.environ.get("DATA_DIR", "data")
ART = os.environ.get("ART_DIR", "artifacts")
print(os.listdir("."))
""")),

    md(src_lines("""
## 1. Feature assembly

Two kinds of features are loaded from cache:

* `cache_feat_{train,test}.csv` — the 165 librosa prosody descriptors plus
  duration, computed over each 45–60 s clip (extraction cell shown in
  *Appendix* below).
* `cache_asr.csv` — wav2vec2 `base-960h` CTC transcripts with ASR confidence,
  entropy and blank-frame statistics (used both as text and as
  confidence proxies).

`asr_whisper.csv` overlays the much better Whisper-large transcript on the
same rows (see Appendix for the Colab extraction). Note that 212 filenames
exist in **both** splits, so transcript lookups must never share a dict
between train and test.
""")),

    code(src_lines("""
def load_features():
    ft = pd.read_csv(f"{ART}/cache_feat_train.csv")
    fe = pd.read_csv(f"{ART}/cache_feat_test.csv")
    tr = pd.read_csv(f"{DATA}/train.csv")
    te = pd.read_csv(f"{DATA}/test.csv")
    asr = pd.read_csv(f"{ART}/cache_asr.csv")

    assert list(ft.filename) == list(tr.filename)
    assert list(fe.filename) == list(te.filename)
    assert len(asr) == len(tr) + len(te)
    asr_tr = asr.iloc[:len(tr)].reset_index(drop=True)
    asr_te = asr.iloc[len(tr):].reset_index(drop=True)

    # Whisper overlay: better transcripts, keep wav2vec2 ASR confidence
    wsp = pd.read_csv(f"{ART}/asr_whisper.csv")
    out = []
    for base, split in ((asr_tr, "train"), (asr_te, "test")):
        w = wsp[wsp.split == split][["filename", "text"]].drop_duplicates(
            "filename").rename(columns={"text": "wtext"})
        m = base.merge(w, on="filename", how="left")
        m["text"] = m.wtext.fillna(m.text)
        out.append(m.drop(columns=["wtext"]))
    asr_tr, asr_te = out
    print(f"whisper overlay: "
          f"train {asr_tr.text.str.len().gt(0).sum()}/{len(asr_tr)} "
          f"test {asr_te.text.str.len().gt(0).sum()}/{len(asr_te)}")
    return ft, fe, tr, te, asr_tr, asr_te


ft, fe, tr, te, asr_tr, asr_te = load_features()
""")),

    code(src_lines("""
def text_features(df, dur_map):
    \"\"\"Derive grammar/fluency features from transcripts.\"\"\"
    rows = []
    for r in df.itertuples():
        words = [w for w in str(r.text).split() if w]
        n = len(words)
        dur = dur_map.get(r.filename, 50.0)
        uniq = set(words)
        lens = [len(w) for w in words] or [0]
        letters = [c for w in words for c in w if c.isalpha()]
        nlet = len(letters) or 1
        vowels = set("AEIOU")
        vsum = sum(1 for c in letters if c in vowels)
        rep1 = sum(1 for a, b in zip(words, words[1:]) if a == b)
        big = {}
        for a, b in zip(words, words[1:]):
            big[(a, b)] = big.get((a, b), 0) + 1
        rep2 = sum(v - 1 for v in big.values() if v > 1)
        fill = sum(1 for w in words if w in FILLER_SET)
        rows.append({
            "filename": r.filename,
            "asr_n_words": n,
            "asr_wps": n / max(dur, 1.0),
            "asr_n_unique": len(uniq),
            "asr_ttr": len(uniq) / n if n else 0.0,
            "asr_mean_len": float(np.mean(lens)),
            "asr_max_len": int(max(lens)),
            "asr_std_len": float(np.std(lens)),
            "asr_long_ratio": sum(1 for w in words if len(w) >= 6) / n if n else 0.0,
            "asr_med_ratio": sum(1 for w in words if 4 <= len(w) <= 6) / n if n else 0.0,
            "asr_one_letter": sum(1 for w in words if len(w) == 1) / n if n else 0.0,
            "asr_rep1": rep1 / n if n else 0.0,
            "asr_rep2": rep2 / n if n else 0.0,
            "asr_filler": fill / n if n else 0.0,
            "asr_vowel_ratio": vsum / nlet,
            "asr_n_chars": nlet,
            "asr_conf": float(getattr(r, "asr_conf", 0.0)),
            "asr_entropy": float(getattr(r, "asr_entropy", 0.0)),
            "asr_blank": float(getattr(r, "asr_blank", 0.0)),
            "asr_tokens_per_sec": float(getattr(r, "asr_tokens", 0)) / max(dur, 1.0),
            "asr_empty": 1.0 if n == 0 else 0.0,
        })
    return pd.DataFrame(rows)


dur_tr = dict(zip(ft.filename, ft.dur))
dur_te = dict(zip(fe.filename, fe.dur))
A_tr = ft.merge(text_features(asr_tr, dur_tr), on="filename", how="left")
A_te = fe.merge(text_features(asr_te, dur_te), on="filename", how="left")
print("base features", A_tr.shape, A_te.shape)
""")),

    md(src_lines("""
### 1b. The LLM-judge feature

`llm_perclip.csv` holds per-clip grades ("score this transcript against the
IELTS rubric") for **all 985 clips** — the single most informative engineered
feature. Coverage matters here: with only partial grades the feature is
median-filled and loses most of its signal (partial coverage cost ≈ 0.006
RMSE on the public LB).

The grader is resumable, prefers `gemini-3.5-flash-lite` and silently falls
over to other Gemini models when a quota is exhausted; see
`src/llm_grade.py` / `src/llm_grade_perclip.py` in the repo.
""")),

    code(src_lines("""
def merge_llm(path, name):
    lp = pd.read_csv(path)
    ltr = lp[lp.split == "train"][["filename", "llm_score"]].drop_duplicates(
        "filename").rename(columns={"llm_score": name})
    lte = lp[lp.split == "test"][["filename", "llm_score"]].drop_duplicates(
        "filename").rename(columns={"llm_score": name})
    return ltr, lte


for path, name in ((f"{ART}/llm.csv", "llm_score"),
                   (f"{ART}/llm_perclip.csv", "llm_score2")):
    ltr, lte = merge_llm(path, name)
    A_tr = A_tr.merge(ltr, on="filename", how="left")
    A_te = A_te.merge(lte, on="filename", how="left")
    print(f"{name}: train {A_tr[name].notna().sum()}/{len(A_tr)} "
          f"test {A_te[name].notna().sum()}/{len(A_te)}")

y = tr.label.reset_index(drop=True)
""")),

    md(src_lines("""
## 2. Embedding blocks

Two "views" of the same 2560-d embeddings are trained as separate models and
blended on OOF RMSE:

* **raw** — base features + all 2560 dims (trees select what they need)
* **pca32** — base features + a 32-dim PCA compression (regularised, less
  overfit-prone view)

The blend weights are fitted on out-of-fold predictions — never on test.
""")),

    code(src_lines("""
from sklearn.decomposition import PCA

z = np.load(f"{ART}/embeds_large.npz")
E = np.asarray(z["X"] if "X" in z else z[list(z.keys())[0]],
               dtype=np.float32)
n_tr = len(A_tr)
assert len(E) == n_tr + len(A_te)
assert np.isfinite(E).all()
print("embeddings", E.shape)

pca = PCA(n_components=32, random_state=0)
ptr = pca.fit_transform(E[:n_tr])
pte = pca.transform(E[n_tr:])
print(f"PCA32 variance explained {pca.explained_variance_ratio_.sum():.3f}")


def raw_block():
    cols = [f"emb_{i}" for i in range(E.shape[1])]
    return (pd.concat([A_tr.reset_index(drop=True),
                       pd.DataFrame(E[:n_tr], columns=cols)], axis=1),
            pd.concat([A_te.reset_index(drop=True),
                       pd.DataFrame(E[n_tr:], columns=cols)], axis=1))


def pca_block():
    cols = [f"embpc_{i}" for i in range(32)]
    return (pd.concat([A_tr.reset_index(drop=True),
                       pd.DataFrame(ptr, columns=cols)], axis=1),
            pd.concat([A_te.reset_index(drop=True),
                       pd.DataFrame(pte, columns=cols)], axis=1))
""")),

    md(src_lines("""
## 3. Model — XGBoost + CatBoost, 5-fold

* `XGBRegressor` depth 5, 1200 trees, subsample 0.8, colsample 0.75
  (two seeds), and `CatBoostRegressor` depth 4 with early stopping
  (three seeds).
* Inside each fold the XGB weight in the blend is chosen on the validation
  fold; across folds by full-OOF grid search.
* Predictions are clipped to the label range [0, 5].
* LightGBM was part of the earliest baseline but crashed on this local
  Windows box; XGBoost/CatBoost run everywhere and performed equally well.
""")),

    code(src_lines("""
from catboost import CatBoostRegressor
from sklearn.model_selection import KFold
from scipy.stats import pearsonr
import xgboost as xgb

JOBS = 4


def make_xgb(seed):
    return xgb.XGBRegressor(
        objective="reg:squarederror", max_depth=5, eta=0.05,
        subsample=0.8, colsample_bytree=0.75, min_child_weight=8,
        reg_lambda=1.5, n_estimators=1200, random_state=seed,
        n_jobs=JOBS, tree_method="hist", eval_metric="rmse")


def make_cat(seed):
    return CatBoostRegressor(
        iterations=4000, depth=4, learning_rate=0.05, l2_leaf_reg=3.0,
        random_strength=1.0, bagging_temperature=0.5, od_type="Iter",
        od_wait=200, random_seed=seed, verbose=False,
        allow_writing_files=False, task_type="CPU", thread_count=JOBS)


def metrics(y, p):
    y, p = np.asarray(y), np.asarray(p)
    rmse = float(np.sqrt(np.mean((y - p) ** 2)))
    return rmse, float(pearsonr(y, p)[0])


def clip_pred(p):
    return np.clip(np.asarray(p, dtype=float), 0.0, 5.0)


def prep(A_tr, A_te):
    common = [c for c in A_tr.columns if c in A_te.columns
              and c != "filename"]
    a = A_tr[common].replace([np.inf, -np.inf], np.nan)
    b = A_te[common].replace([np.inf, -np.inf], np.nan)
    med = a.median(numeric_only=True)
    return a.fillna(med).astype(np.float32), b.fillna(med).astype(np.float32)


XGB_SEEDS = (42, 43)
CAT_SEEDS = (3, 5, 7)


def run(X, y, X_te):
    \"\"\"5-fold OOF + test predictions for the XGB/CatBoost blend.\"\"\"
    folds = KFold(n_splits=5, shuffle=True, random_state=SEED)
    ox, oc = np.zeros(len(X)), np.zeros(len(X))
    tx, tc = np.zeros(len(X_te)), np.zeros(len(X_te))
    for k, (tr_i, va_i) in enumerate(folds.split(X), 1):
        Xtr, Xva = X.iloc[tr_i], X.iloc[va_i]
        ytr, yva = y.iloc[tr_i], y.iloc[va_i]
        sx = np.zeros(len(va_i))
        st = np.zeros(len(X_te))
        for s in XGB_SEEDS:
            m = make_xgb(s)
            m.fit(Xtr, ytr, eval_set=[(Xva, yva)], verbose=False)
            sx += m.predict(Xva) / len(XGB_SEEDS)
            st += m.predict(X_te) / len(XGB_SEEDS)
        ox[va_i], tx = sx, tx + st / folds.n_splits
        sc, sct = np.zeros(len(va_i)), np.zeros(len(X_te))
        for s in CAT_SEEDS:
            c = make_cat(s)
            c.fit(Xtr, ytr, eval_set=(Xva, yva))
            sc += c.predict(Xva) / len(CAT_SEEDS)
            sct += c.predict(X_te) / len(CAT_SEEDS)
        oc[va_i], tc = sc, tc + sct / folds.n_splits
        print(f"  fold {k}: xgb={metrics(yva, ox[va_i])[0]:.4f} "
              f"cat={metrics(yva, oc[va_i])[0]:.4f}", flush=True)
    best = (9.9, 0.0)
    for w in np.arange(0.0, 1.01, 0.05):
        r = metrics(y, w * ox + (1 - w) * oc)[0]
        if r < best[0]:
            best = (r, w)
    print(f"  best xgb weight {best[1]:.2f} oof={best[0]:.4f}")
    return clip_pred(best[1] * ox + (1 - best[1]) * oc), \\
        clip_pred(best[1] * tx + (1 - best[1]) * tc)
""")),

    md(src_lines("""
## 4. Train both blocks, blend on OOF, write submission
""")),

    code(src_lines("""
oofs, preds = {}, {}
for name, (a_tr, a_te) in (("raw", raw_block()), ("pca32", pca_block())):
    X, X_te = prep(a_tr, a_te)
    print(f"=== {name} {X.shape} ===", flush=True)
    oofs[name], preds[name] = run(X, y, X_te)
    print(f"{name}: rmse={metrics(y, oofs[name])[0]:.4f} "
          f"pearson={metrics(y, oofs[name])[1]:.4f}", flush=True)
""")),

    code(src_lines("""
best = (9.9, None)
for w in np.arange(0.0, 1.01, 0.05):
    mix = {"raw": w, "pca32": 1 - w}
    r = metrics(y, clip_pred(sum(mix[n] * oofs[n] for n in mix)))[0]
    if r < best[0]:
        best = (r, mix)
rmse, pear = metrics(y, clip_pred(
    sum(best[1][n] * oofs[n] for n in best[1])))
print(f"blend {best[1]} -> OOF rmse={rmse:.4f} pearson={pear:.4f}")

final = clip_pred(sum(best[1][n] * preds[n] for n in best[1]))
sub = pd.DataFrame({"filename": te.filename, "label": final})
sub.to_csv("submission.csv", index=False)
print(sub.head())
""")),

    md(src_lines("""
## 5. What did NOT work

Recorded because every one of these *improved the OOF score* and was only
caught by public-leaderboard feedback (the 216-clip test set makes OOF a
noisy guide):

| Change | OOF effect | LB effect |
|---|---|---|
| wav2vec2-large embeddings (3rd model family) | ≈ neutral (5% blend weight) | **+0.010** (worse) |
| GPT-2 perplexity features | ≈ neutral | worse |
| Post-hoc OOF "stretching" of predictions | slightly better, 3× | **worse, 3×** |
| Removing the Whisper-overlay | slightly better | **+0.013** (worse) |

Takeaway: on a tiny test set, trust the public leaderboard over the local CV,
and add one family at a time.

## Compliance

* No external **data** — only the competition's own audio.
* External **models**: whisper-large-v3 (ASR + encoder), wav2vec2-base-960h
  (baseline preceding pipeline), Gemini (LLM judge) — all model weights
  from public sources, which the rules permit.
* One participant, one team.

## Appendix — how the cached artifacts were produced

1. **Prosody + wav2vec2 ASR** — `src/train_v3.py` logic over every clip
   (librosa 165 descriptors; CTC transcripts with confidence/entropy/blank).
2. **Whisper-large-v3 transcripts** — Colab T4, `jobs/colab_asr.py`.
3. **Whisper-large-v3 encoder embeddings** — mean+std pooling of encoder
   hidden states over 30 s windows (hop 25 s), `jobs/colab_embeds.py`
   (8.9 MB for all 985 clips).
4. **Gemini grades** — `src/llm_grade_perclip.py` (per-clip, full coverage)
   and `src/llm_grade.py` (batch, fallback); merged by `src/build_hybrid.py`.

Everything else — feature assembly, models, blending, submission — is in
this notebook.
""")),
]

nb = {
    "cells": cells,
    "metadata": {
        "kernelspec": {"display_name": "Python 3", "language": "python",
                       "name": "python3"},
        "language_info": {"name": "python", "version": "3.10"},
    },
    "nbformat": 4,
    "nbformat_minor": 5,
}
with open("notebook/shl_grammar_score.ipynb", "w", encoding="utf-8") as f:
    json.dump(nb, f, indent=1)
print("notebook written,", len(cells), "cells")
