"""Local training/eval harness — same code path as the Kaggle notebook.

Feeds on the downloaded caches:
  cache_feat_train.csv / cache_feat_test.csv  (audio features, 165 cols)
  cache_asr.csv                              (rows 0:769 train, 769: test)
  data/train.csv, data/test.csv

Optional llm.csv (filename,llm_score) adds the LLM judge feature — it is
picked up automatically when present.

  python local_train.py                 # full model
"""

import argparse
import json
import os
import sys

# This box has 2 cores; LightGBM/OpenMP defaults race and die with a native
# access violation (reads address 0). Pin the thread pools before numpy loads.
for _v in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS"):
    os.environ.setdefault(_v, "1")

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                "nb"))
from train_v3 import text_features, run_model  # noqa: E402

sys.stdout.reconfigure(encoding="utf-8", errors="replace")


def _overlay_whisper(asr_tr, asr_te):
    """Replace `text` with Whisper transcripts when available.

    wav2vec2's own asr_conf/asr_entropy/asr_blank stay: they still describe
    how clean the audio is, while Whisper gives far better words.
    """
    path = "asr_whisper.csv"
    if not os.path.exists(path):
        return asr_tr, asr_te, ""
    wsp = pd.read_csv(path)
    if not {"split", "filename", "text"} <= set(wsp.columns):
        raise SystemExit(f"{path}: expected split,filename,text")
    out = []
    stats = []
    for base, split in ((asr_tr, "train"), (asr_te, "test")):
        w = wsp[wsp.split == split][["filename", "text"]].drop_duplicates(
            "filename").rename(columns={"text": "wtext"})
        m = base.merge(w, on="filename", how="left")
        n = int(m.wtext.notna().sum())
        m["text"] = m.wtext.fillna(m.text)
        m = m.drop(columns=["wtext"])
        out.append(m)
        stats.append(f"{split} {n}/{len(base)}")
    return out[0], out[1], "whisper overlay " + " ".join(stats)


def load():
    ft = pd.read_csv("cache_feat_train.csv")
    fe = pd.read_csv("cache_feat_test.csv")
    tr = pd.read_csv("data/train.csv")
    te = pd.read_csv("data/test.csv")
    asr = pd.read_csv("cache_asr.csv")

    assert list(ft.filename) == list(tr.filename), "feat/train order mismatch"
    assert list(fe.filename) == list(te.filename), "feat/test order mismatch"
    assert len(asr) == len(tr) + len(te), "asr length mismatch"
    asr_tr = asr.iloc[: len(tr)].reset_index(drop=True)
    asr_te = asr.iloc[len(tr):].reset_index(drop=True)
    assert list(asr_tr.filename) == list(tr.filename), "asr/train mismatch"
    assert list(asr_te.filename) == list(te.filename), "asr/test mismatch"
    asr_tr, asr_te, wnote = _overlay_whisper(asr_tr, asr_te)
    if wnote:
        print(wnote)

    # 212 filenames exist in both splits -> never share a lookup dict
    dur_tr = dict(zip(ft.filename, ft.dur))
    dur_te = dict(zip(fe.filename, fe.dur))

    txt_tr = text_features(asr_tr, dur_tr)
    txt_te = text_features(asr_te, dur_te)

    A_tr = ft.merge(txt_tr, on="filename", how="left")
    A_te = fe.merge(txt_te, on="filename", how="left")
    assert len(A_tr) == len(tr), f"A_tr {len(A_tr)} != {len(tr)}"

    if os.path.exists("llm.csv"):
        llm = pd.read_csv("llm.csv")
        before = len(A_tr)
        if "split" in llm.columns:
            llm_tr = llm[llm.split == "train"][["filename", "llm_score"]]
            llm_te = llm[llm.split == "test"][["filename", "llm_score"]]
        else:
            # legacy two-column file: treat as train-only grades
            llm_tr = llm_te = llm[["filename", "llm_score"]]
        A_tr = A_tr.merge(llm_tr.drop_duplicates("filename"),
                          on="filename", how="left")
        A_te = A_te.merge(llm_te.drop_duplicates("filename"),
                          on="filename", how="left")
        nan = (int(A_tr["llm_score"].isna().sum())
               if "llm_score" in A_tr else -1)
        if "llm_score" in A_tr and nan == len(A_tr):
            # no train grades yet: drop rather than feed an all-NaN column
            A_tr = A_tr.drop(columns=["llm_score"])
            A_te = A_te.drop(columns=["llm_score"])
            print("llm.csv present but no train grades yet — feature dropped")
        else:
            print(f"llm.csv merged: train {before}->{len(A_tr)}, nan={nan}, "
                  f"test nan="
                  f"{int(A_te['llm_score'].isna().sum()) if 'llm_score' in A_te else -1}")

    if os.path.exists("llm_perclip.csv"):
        lp = pd.read_csv("llm_perclip.csv")
        col = "llm_score2"
        if "split" in lp.columns:
            lp_tr = lp[lp.split == "train"][["filename", "llm_score"]]
            lp_te = lp[lp.split == "test"][["filename", "llm_score"]]
        else:
            lp_tr = lp_te = lp[["filename", "llm_score"]]
        lp_tr = lp_tr.drop_duplicates("filename").rename(
            columns={"llm_score": col})
        lp_te = lp_te.drop_duplicates("filename").rename(
            columns={"llm_score": col})
        A_tr = A_tr.merge(lp_tr, on="filename", how="left")
        A_te = A_te.merge(lp_te, on="filename", how="left")
        nan2 = int(A_tr[col].isna().sum())
        if nan2 == len(A_tr):
            A_tr = A_tr.drop(columns=[col])
            A_te = A_te.drop(columns=[col])
            print("llm_perclip.csv present but no train grades yet — dropped")
        else:
            print(f"llm_perclip merged as {col}: nan train={nan2}, "
                  f"test={int(A_te[col].isna().sum())}")

    if os.path.exists("text_feats.csv"):
        tf = pd.read_csv("text_feats.csv")
        tf = tf.rename(columns={"words": "tf_words",
                                "distinct2": "tf_distinct2",
                                "n_contr": "tf_contr"})
        keep = [c for c in ("ppl", "tf_words", "tf_distinct2", "tf_contr")
                if c in tf.columns]
        if "split" in tf.columns:
            tf_tr = tf[tf.split == "train"][["filename"] + keep]
            tf_te = tf[tf.split == "test"][["filename"] + keep]
        else:
            tf_tr = tf_te = tf[["filename"] + keep]
        A_tr = A_tr.merge(tf_tr.drop_duplicates("filename"),
                          on="filename", how="left")
        A_te = A_te.merge(tf_te.drop_duplicates("filename"),
                          on="filename", how="left")
        print(f"text_feats merged: {keep}, "
              f"nan train={int(A_tr[keep[0]].isna().sum()) if keep else -1}")

    if os.path.exists("llm_audio.csv"):
        la = pd.read_csv("llm_audio.csv")
        col = "llm_audio_score"
        if "split" in la.columns:
            la_tr = la[la.split == "train"][["filename", "llm_audio"]]
            la_te = la[la.split == "test"][["filename", "llm_audio"]]
        else:
            la_tr = la_te = la[["filename", "llm_audio"]]
        la_tr = la_tr.drop_duplicates("filename").rename(
            columns={"llm_audio": col})
        la_te = la_te.drop_duplicates("filename").rename(
            columns={"llm_audio": col})
        A_tr = A_tr.merge(la_tr, on="filename", how="left")
        A_te = A_te.merge(la_te, on="filename", how="left")
        n_nan = int(A_tr[col].isna().sum())
        if n_nan == len(A_tr):
            A_tr = A_tr.drop(columns=[col])
            A_te = A_te.drop(columns=[col])
            print("llm_audio.csv present but no train grades yet — dropped")
        else:
            print(f"llm_audio merged as {col}: nan train={n_nan}, "
                  f"test={int(A_te[col].isna().sum())}")

    y = tr.label.reset_index(drop=True)
    return A_tr, A_te, y, te.filename.values


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="submission_local.csv")
    ap.add_argument("--summary", default="local_summary.json")
    a = ap.parse_args()

    A_tr, A_te, y, ids = load()
    common = [c for c in A_tr.columns if c in A_te.columns
              and c != "filename"]
    A_tr = A_tr[common].replace([np.inf, -np.inf], np.nan)
    A_te = A_te[common].replace([np.inf, -np.inf], np.nan)
    med = A_tr.median(numeric_only=True)
    A_tr = A_tr.fillna(med)
    A_te = A_te.fillna(med)
    print("matrix train", A_tr.shape, "test", A_te.shape)

    sub, m, oof = run_model(A_tr, y, A_te, ids)
    sub.to_csv(a.out, index=False)
    print("\nsubmission", sub.shape, "->", a.out)
    print(sub.head(5).to_string())

    oof_df = pd.DataFrame({"filename": pd.read_csv(
        "data/train.csv").filename, "y_true": y.values, "y_oof": oof})
    oof_df["resid"] = oof_df.y_true - oof_df.y_oof
    oof_df.to_csv("oof_local.csv", index=False)

    summary = {"n_features": A_tr.shape[1], "oof_rmse": round(m[0], 4),
               "oof_pearson": round(m[1], 4)}
    json.dump(summary, open(a.summary, "w"), indent=1)
    print("\nTRAIN RMSE (5-fold OOF):", summary["oof_rmse"],
          "| pearson:", summary["oof_pearson"])

    # baseline: does the LLM score alone beat this?
    if "llm_score" in A_tr.columns:
        from scipy.stats import pearsonr
        s = A_tr["llm_score"].fillna(A_tr["llm_score"].median())
        rmse = float(np.sqrt(np.mean((y.values - s.values) ** 2)))
        r = float(pearsonr(y.values, s.values)[0])
        half = np.clip(np.round(s.values * 2) / 2, 0, 5)
        print(f"LLM score alone: rmse={rmse:.4f} pearson={r:.4f} "
              f"| rounded rmse={float(np.sqrt(np.mean((y.values-half)**2))):.4f}")


if __name__ == "__main__":
    main()
