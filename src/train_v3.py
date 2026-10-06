"""Grammar Score prediction v3: audio prosody + ASR text features, with
report artifacts (OOF predictions, evaluation plots, feature importances).

Offline on Kaggle. Weights are mounted from public datasets:
  - andgoogle/wav2vec2-base-960h-260507-1028  (fine-tuned, has lm_head)
  - warhero12/wav2vec2                         (config.json template)
Feature/ASR caches are reused if a mounted dataset provides them
(feat_train.csv, feat_test.csv, asr.csv), so re-runs are incremental.
"""

import json
import os
import shutil
import sys
import time
import traceback
import warnings
from concurrent.futures import ThreadPoolExecutor

warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd

BASE = "/kaggle/input/competitions/shl-hiring-assessment-2026/Dataset_Final"
WORK = "/kaggle/working"
OUT = f"{WORK}/asrmodel"
SMOKE = False
N_JOBS = 4
SEED = 42

np.random.seed(SEED)


def _mounted(name):
    """Find a cached artifact in any mounted input dataset."""
    for root, _dirs, files in os.walk("/kaggle/input"):
        if root.startswith(WORK):
            continue
        if name in files:
            return os.path.join(root, name)
    return None


# ==========================================================================
# 1. audio features
# ==========================================================================
def one_audio(path):
    import librosa

    try:
        y, sr = librosa.load(path, sr=16000, mono=True)
        if y.size < sr:
            return None
        y = y.astype(np.float32)
        dur = len(y) / sr
        f = {"dur": dur}

        hop, frame = 512, 2048
        rms = librosa.feature.rms(y=y, frame_length=frame, hop_length=hop)[0]
        rms = np.maximum(rms, 1e-8)

        f["rms_mean"] = float(rms.mean())
        f["rms_std"] = float(rms.std())
        f["rms_p10"] = float(np.percentile(rms, 10))
        f["rms_p50"] = float(np.percentile(rms, 50))
        f["rms_p90"] = float(np.percentile(rms, 90))
        f["crest"] = float(rms.max() / (rms.mean() + 1e-8))

        thr = max(1e-3, 0.12 * float(np.percentile(rms, 92)))
        silent = rms < thr
        f["silence_ratio"] = float(silent.mean())
        vr = (~silent).astype(np.int8)
        f["n_pause_runs"] = float(np.sum(np.diff(vr) == -1))

        zcr = librosa.feature.zero_crossing_rate(y, frame_length=frame,
                                                 hop_length=hop)[0]
        f["zcr_mean"] = float(zcr.mean())
        f["zcr_std"] = float(zcr.std())

        S = np.abs(librosa.stft(y, n_fft=frame, hop_length=hop))
        freqs = librosa.fft_frequencies(sr=sr, n_fft=frame)
        power = S ** 2
        psum = power.sum(axis=0) + 1e-12

        f["cent_mean"] = float(librosa.feature.spectral_centroid(
            S=S, sr=sr)[0].mean())
        f["cent_std"] = float(librosa.feature.spectral_centroid(
            S=S, sr=sr)[0].std())
        f["bw_mean"] = float(librosa.feature.spectral_bandwidth(
            S=S, sr=sr)[0].mean())
        f["bw_std"] = float(librosa.feature.spectral_bandwidth(
            S=S, sr=sr)[0].std())
        roll = librosa.feature.spectral_rolloff(S=S, sr=sr,
                                                roll_percent=0.85)[0]
        f["rolloff_mean"] = float(roll.mean())
        f["rolloff_std"] = float(roll.std())
        flat = librosa.feature.spectral_flatness(S=S)[0]
        f["flat_mean"] = float(flat.mean())
        f["flat_std"] = float(flat.std())

        for i, v in enumerate(librosa.feature.spectral_contrast(
                S=S, sr=sr, n_bands=6).mean(axis=1)):
            f[f"contrast_{i}"] = float(v)
        for i, v in enumerate(librosa.feature.chroma_stft(
                S=S, sr=sr).mean(axis=1)):
            f[f"chroma_{i}"] = float(v)

        mel = librosa.feature.melspectrogram(S=power, sr=sr, n_mels=24,
                                             power=1.0)
        lm = librosa.power_to_db(mel)
        for i in range(lm.shape[0]):
            f[f"mel_{i}"] = float(lm[i].mean())
            f[f"melv_{i}"] = float(lm[i].std())

        mf = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=20, n_fft=frame,
                                  hop_length=hop)
        for i in range(20):
            f[f"mfcc_{i}"] = float(mf[i].mean())
            f[f"mfccs_{i}"] = float(mf[i].std())
        d1 = librosa.feature.delta(mf, width=9)
        for i in range(13):
            f[f"dmfcc_{i}"] = float(d1[i].mean())
            f[f"dmfccs_{i}"] = float(d1[i].std())

        try:
            f0 = librosa.yin(y, fmin=55.0, fmax=400.0, sr=sr,
                             frame_length=frame, hop_length=hop)
            vmask = (rms > thr) if len(f0) == len(rms) else np.ones(
                len(f0), bool)
            if vmask.any():
                v = f0[vmask]
                f["f0_mean"] = float(v.mean())
                f["f0_std"] = float(v.std())
                f["f0_p25"] = float(np.percentile(v, 25))
                f["f0_p75"] = float(np.percentile(v, 75))
            else:
                f["f0_mean"] = f["f0_std"] = 0.0
                f["f0_p25"] = f["f0_p75"] = 0.0
            f["voiced_ratio"] = float(vmask.mean())
        except Exception:
            f["f0_mean"] = f["f0_std"] = f["f0_p25"] = f["f0_p75"] = 0.0
            f["voiced_ratio"] = 0.0

        try:
            on = librosa.onset.onset_detect(y=y, sr=sr, hop_length=hop,
                                            units="time")
            f["onsets_per_sec"] = float(len(on) / dur)
            if len(on) > 1:
                ioi = np.diff(on)
                f["ioi_med"] = float(np.median(ioi))
                f["ioi_cv"] = float(ioi.std() / (ioi.mean() + 1e-8))
            else:
                f["ioi_med"] = f["ioi_cv"] = 0.0
        except Exception:
            f["onsets_per_sec"] = f["ioi_med"] = f["ioi_cv"] = 0.0

        env = librosa.onset.onset_strength(y=y, sr=sr, hop_length=hop)
        f["onset_str_mean"] = float(env.mean())
        f["onset_str_std"] = float(env.std())
        try:
            f["tempo"] = float(np.atleast_1d(librosa.feature.rhythm.tempo(
                onset_envelope=env, sr=sr, hop_length=hop))[0])
        except Exception:
            f["tempo"] = 0.0

        cent_frames = (freqs[:, None] * power).sum(axis=0) / psum
        f["spec_flat_dyn"] = float(cent_frames.std())
        return f
    except Exception:
        return None


def extract_audio(paths, names):
    t0 = time.time()
    rows = []
    with ThreadPoolExecutor(max_workers=N_JOBS) as ex:
        for i, r in enumerate(ex.map(one_audio, paths, chunksize=4), 1):
            rows.append(r)
            if i % 100 == 0 or i == len(paths):
                print(f"  audio {i}/{len(paths)} {time.time()-t0:.0f}s",
                      flush=True)
    keep = [n for n, r in zip(names, rows) if r is not None]
    ok = [r for r in rows if r is not None]
    df = pd.DataFrame(ok)
    if len(df):
        df.insert(0, "filename", keep)
    print(f"audio feats {len(ok)}/{len(paths)} failures="
          f"{len(rows)-len(ok)} {time.time()-t0:.0f}s")
    return df


# ==========================================================================
# 2. ASR
# ==========================================================================
FILLERS = {"UH", "UHM", "UM", "ERM", "ER", "MM", "HMM", "AH", "EH"}


def find_asr_dirs():
    cands = []
    for root, _dirs, files in os.walk("/kaggle/input"):
        if "pytorch_model.bin" in files or "model.safetensors" in files:
            cands.append((root, files))
    asr = tmpl = None
    for root, files in cands:
        if "vocab.json" in files and "model.safetensors" in files:
            asr = root
            break
    if asr is None:
        for root, files in cands:
            if "vocab.json" in files:
                asr = root
                break
    for root, files in cands:
        if "config.json" in files and "vocab.json" in files and root != asr:
            tmpl = root
            break
    if tmpl is None:
        for root, files in cands:
            if "config.json" in files and root != asr:
                tmpl = root
                break
    return asr, tmpl


def prepare_asr_model():
    asr, tmpl = find_asr_dirs()
    print("asr src:", asr)
    print("cfg tmpl:", tmpl)
    if asr is None:
        raise RuntimeError("no wav2vec2 checkpoint mounted")
    shutil.rmtree(OUT, ignore_errors=True)
    os.makedirs(OUT, exist_ok=True)
    for f in sorted(os.listdir(asr)):
        if f == "tf_model.h5":
            continue
        shutil.copy(os.path.join(asr, f), os.path.join(OUT, f))
    if "config.json" not in os.listdir(OUT):
        if tmpl is None:
            raise RuntimeError("no config template")
        cfg = json.load(open(f"{tmpl}/config.json"))
        for junk in ("_name_or_path", "transformers_version", "torch_dtype"):
            cfg.pop(junk, None)
        cfg["architectures"] = ["Wav2Vec2ForCTC"]
        json.dump(cfg, open(f"{OUT}/config.json", "w"), indent=1)

    import torch
    from transformers import (Wav2Vec2CTCTokenizer, Wav2Vec2FeatureExtractor,
                              Wav2Vec2ForCTC)
    torch.set_num_threads(1)
    tok = Wav2Vec2CTCTokenizer(f"{OUT}/vocab.json")
    fe = Wav2Vec2FeatureExtractor.from_pretrained(OUT)
    model = Wav2Vec2ForCTC.from_pretrained(OUT)
    model.eval()
    n = sum(p.numel() for p in model.parameters()) / 1e6
    print(f"ASR model loaded ({n:.1f}M params)")
    return model, fe, tok


_AS = {}


def init_asr():
    if not _AS:
        _AS["m"], _AS["fe"], _AS["tok"] = prepare_asr_model()


def one_asr(item):
    import soundfile as sf
    import torch

    fid, path = item
    try:
        y, sr = sf.read(path, dtype="float32")
        if y.ndim > 1:
            y = y.mean(axis=1)
        if sr != 16000:
            import librosa
            y = librosa.resample(y, orig_sr=sr, target_sr=16000)
        with torch.no_grad():
            inp = _AS["fe"](y, sampling_rate=16000, return_tensors="pt")
            logits = _AS["m"](**inp).logits[0]
            lp = torch.log_softmax(logits, dim=-1)
            pr = lp.exp()
            conf = pr.max(dim=-1).values
            ent = -(pr * lp).sum(dim=-1)
            am = logits.argmax(dim=-1)
            blank = (am == 0).float().mean()
            text = _AS["tok"].decode(am).strip()
        return {
            "filename": fid,
            "text": text,
            "asr_conf": float(conf.mean()),
            "asr_entropy": float(ent.mean()),
            "asr_blank": float(blank),
            "asr_tokens": int(am.numel()),
        }
    except Exception:
        return {"filename": fid, "text": "", "asr_conf": 0.0,
                "asr_entropy": 0.0, "asr_blank": 0.0, "asr_tokens": 0}


def run_asr(items):
    """items: list of (filename, path). Returns DataFrame, resumable."""
    fp = f"{WORK}/asr.csv"
    done = {}
    if os.path.exists(fp):
        prev = pd.read_csv(fp)
        done = {r.filename: r for r in prev.itertuples()}
        print(f"resuming ASR: {len(done)} already done")
    todo = [it for it in items if it[0] not in done]
    print(f"ASR todo: {len(todo)}/{len(items)}")
    if not todo:
        return pd.DataFrame(done.values())

    init_asr()
    t0 = time.time()
    rows = list(done.values())
    with ThreadPoolExecutor(max_workers=N_JOBS) as ex:
        for i, r in enumerate(ex.map(one_asr, todo, chunksize=1), 1):
            rows.append(r)
            if i % 10 == 0 or i == len(todo):
                el = time.time() - t0
                eta = el / i * (len(todo) - i)
                print(f"  asr {i}/{len(todo)} {el:.0f}s elapsed "
                      f"eta {eta:.0f}s", flush=True)
                pd.DataFrame(rows).to_csv(fp, index=False)
    pd.DataFrame(rows).to_csv(fp, index=False)
    print(f"ASR done {len(rows)} clips {time.time()-t0:.0f}s")
    return pd.DataFrame(rows)


FILLER_SET = {"UH", "UHM", "UM", "ERM", "ER", "MM", "HMM", "AH", "EH",
              "MMHMM", "UHH"}


def text_features(df, dur_map):
    """Derive grammar/fluency features from transcripts."""
    rows = []
    for r in df.itertuples():
        words = [w for w in str(r.text).split() if w]
        n = len(words)
        dur = dur_map.get(r.filename, 50.0)
        uniq = set(words)
        lens = [len(w) for w in words] or [0]
        vowels = set("AEIOU")
        letters = [c for w in words for c in w if c.isalpha()]
        nlet = len(letters) or 1
        vsum = sum(1 for c in letters if c in vowels)

        rep1 = sum(1 for a, b in zip(words, words[1:]) if a == b)
        big = {}
        for a, b in zip(words, words[1:]):
            big[(a, b)] = big.get((a, b), 0) + 1
        rep2 = sum(v - 1 for v in big.values() if v > 1)

        fill = sum(1 for w in words if w in FILLER_SET)
        one = sum(1 for w in words if len(w) == 1)
        longw = sum(1 for w in words if len(w) >= 6)
        medw = sum(1 for w in words if 4 <= len(w) <= 6)

        rows.append({
            "filename": r.filename,
            "asr_n_words": n,
            "asr_wps": n / max(dur, 1.0),
            "asr_n_unique": len(uniq),
            "asr_ttr": len(uniq) / n if n else 0.0,
            "asr_mean_len": float(np.mean(lens)),
            "asr_max_len": int(max(lens)),
            "asr_std_len": float(np.std(lens)),
            "asr_long_ratio": longw / n if n else 0.0,
            "asr_med_ratio": medw / n if n else 0.0,
            "asr_one_letter": one / n if n else 0.0,
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


# ==========================================================================
# 3. model
# ==========================================================================
def metrics(y, p):
    from scipy.stats import pearsonr
    y, p = np.asarray(y), np.asarray(p)
    rmse = float(np.sqrt(np.mean((y - p) ** 2)))
    r = float(pearsonr(y, p)[0]) if len(set(np.round(y, 6))) > 1 else 0.0
    return rmse, r


def clip_pred(p):
    return np.clip(np.asarray(p, dtype=float), 0.0, 5.0)


def run_model(X, y, X_test, ids_test):
    import lightgbm as lgb
    import xgboost as xgb
    from sklearn.model_selection import KFold

    lgb_p = dict(
        objective="regression", metric="rmse", learning_rate=0.04,
        num_leaves=31, min_child_samples=25, feature_fraction=0.75,
        bagging_fraction=0.8, bagging_freq=1, lambda_l2=1.0,
        n_estimators=1500, random_state=SEED, n_jobs=4, verbosity=-1,
    )
    xgb_p = dict(
        objective="reg:squarederror", max_depth=5, eta=0.05,
        subsample=0.8, colsample_bytree=0.75, min_child_weight=8,
        reg_lambda=1.5, n_estimators=1200, random_state=SEED,
        n_jobs=4, tree_method="hist", eval_metric="rmse",
    )

    oof_l = np.zeros(len(X))
    oof_x = np.zeros(len(X))
    te_l = np.zeros(len(X_test))
    te_x = np.zeros(len(X_test))
    folds = KFold(n_splits=5, shuffle=True, random_state=SEED)

    for k, (tr_i, va_i) in enumerate(folds.split(X), 1):
        Xtr, Xva = X.iloc[tr_i], X.iloc[va_i]
        ytr, yva = y.iloc[tr_i], y.iloc[va_i]

        m1 = lgb.LGBMRegressor(**lgb_p)
        m1.fit(Xtr, ytr, eval_set=[(Xva, yva)],
               callbacks=[lgb.early_stopping(150, verbose=False)])
        oof_l[va_i] = m1.predict(Xva)
        te_l += m1.predict(X_test) / folds.n_splits

        m2 = xgb.XGBRegressor(**xgb_p)
        m2.fit(Xtr, ytr, eval_set=[(Xva, yva)], verbose=False)
        oof_x[va_i] = m2.predict(Xva)
        te_x += m2.predict(X_test) / folds.n_splits

        print(f"  fold {k}: lgb={metrics(yva, oof_l[va_i])[0]:.4f} "
              f"xgb={metrics(yva, oof_x[va_i])[0]:.4f}")

    best_w, best = 1.0, 1e9
    for w in np.arange(0.0, 1.01, 0.05):
        r = metrics(y, w * oof_l + (1 - w) * oof_x)[0]
        if r < best:
            best, best_w = r, w
    print(f"  best blend w(lgb)={best_w:.2f} rmse={best:.4f}")

    oof = clip_pred(best_w * oof_l + (1 - best_w) * oof_x)
    pred = clip_pred(best_w * te_l + (1 - best_w) * te_x)

    m = metrics(y, oof)
    print(f"\nOOF  rmse={m[0]:.4f}  pearson={m[1]:.4f}")
    sub = pd.DataFrame({"filename": ids_test, "label": pred})
    return sub, m, oof


def main():
    t0 = time.time()
    tr = pd.read_csv(f"{BASE}/train.csv")
    te = pd.read_csv(f"{BASE}/test.csv")
    print(f"train {tr.shape}  test {te.shape}")
    print(tr.label.value_counts().sort_index().to_string())

    tr_paths = [f"{BASE}/train/{x}" for x in tr.filename]
    te_paths = [f"{BASE}/test/{x}" for x in te.filename]
    if SMOKE:
        tr = tr.iloc[:10].reset_index(drop=True)
        tr_paths = tr_paths[:10]
        te = te.iloc[:6].reset_index(drop=True)
        te_paths = te_paths[:6]

    # ---- audio -------------------------------------------------------
    fp_tr, fp_te = f"{WORK}/feat_train.csv", f"{WORK}/feat_test.csv"
    m_tr, m_te = _mounted("feat_train.csv"), _mounted("feat_test.csv")
    if m_tr and m_te:
        atr, ate = pd.read_csv(m_tr), pd.read_csv(m_te)
        print("audio cache (mounted)", atr.shape, ate.shape)
    elif os.path.exists(fp_tr) and not SMOKE:
        atr = pd.read_csv(fp_tr)
        ate = pd.read_csv(fp_te)
        print("audio cache", atr.shape, ate.shape)
    else:
        print("extracting audio features...")
        atr = extract_audio(tr_paths, tr.filename.tolist())
        ate = extract_audio(te_paths, te.filename.tolist())
        if not SMOKE:
            atr.to_csv(fp_tr, index=False)
            ate.to_csv(fp_te, index=False)
    atr["label"] = tr.label.values[:len(atr)]

    # ---- ASR ---------------------------------------------------------
    print("\ntranscribing...")
    ma = _mounted("asr.csv")
    if ma and not os.path.exists(f"{WORK}/asr.csv"):
        shutil.copy(ma, f"{WORK}/asr.csv")
        print("asr cache mounted from", ma)
    all_items = [(fn, p) for fn, p in zip(tr.filename, tr_paths)]
    all_items += [(fn, p) for fn, p in zip(te.filename, te_paths)]
    asr = run_asr(all_items)

    dur_map = dict(zip(atr.filename, atr.dur))
    dur_map.update(dict(zip(ate.filename, ate.dur)))

    txt = text_features(asr, dur_map)
    print("text feats", txt.shape)

    A_tr = atr.merge(txt, on="filename", how="left")
    A_te = ate.merge(txt, on="filename", how="left")
    A_tr = A_tr.drop(columns=["label"]) if "label" in A_tr else A_tr

    if SMOKE:
        print("SMOKE train", A_tr.shape, "test", A_te.shape)
        print("nan", int(A_tr.isna().sum().sum()),
              int(A_te.isna().sum().sum()))
        print(A_tr[["asr_n_words", "asr_conf", "asr_entropy", "asr_ttr"]]
              .describe().to_string())
        print("SMOKE_OK")
        return

    y = atr.label.iloc[:len(A_tr)].reset_index(drop=True)
    ids = te.filename.values

    common = [c for c in A_tr.columns if c in A_te.columns]
    A_tr = A_tr[common].replace([np.inf, -np.inf], np.nan)
    A_te = A_te[common].replace([np.inf, -np.inf], np.nan)
    med = A_tr.median(numeric_only=True)
    A_tr = A_tr.fillna(med)
    A_te = A_te.fillna(med)
    print("matrix", A_tr.shape)

    sub, m, oof = run_model(A_tr, y, A_te, ids)
    sub.to_csv(f"{WORK}/submission.csv", index=False)
    print("\nsubmission", sub.shape)
    print(sub.head(8).to_string())

    import lightgbm as lgb
    fi = lgb.LGBMRegressor(objective="regression", n_estimators=900,
                           learning_rate=0.05, random_state=SEED, n_jobs=4,
                           verbosity=-1)
    fi.fit(A_tr, y)
    imp = pd.Series(fi.feature_importances_,
                    index=A_tr.columns).sort_values(ascending=False)
    imp.rename("importance").to_csv(f"{WORK}/feature_importance.csv")
    print("\ntop features:\n" + imp.head(30).to_string())

    pd.DataFrame({"filename": tr.filename.iloc[:len(oof)],
                  "y_true": y.values, "y_oof": oof,
                  "resid": y.values - oof}).to_csv(f"{WORK}/oof.csv",
                                                   index=False)
    _report_plots(y.values, oof, imp)

    summary = {"n_train": len(A_tr), "n_test": len(A_te),
               "n_features": A_tr.shape[1], "oof_rmse": round(m[0], 4),
               "oof_pearson": round(m[1], 4),
               "elapsed_s": int(time.time() - t0)}
    json.dump(summary, open(f"{WORK}/summary.json", "w"), indent=1)
    print("\nTRAIN RMSE (5-fold OOF):", round(m[0], 4))
    print("SUMMARY", json.dumps(summary))
    print(f"DONE {time.time()-t0:.0f}s")


def _report_plots(y, p, imp):
    """Evaluation figures for the submission report."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from scipy.stats import pearsonr

    r = pearsonr(y, p)[0]
    fig, axes = plt.subplots(2, 2, figsize=(13, 10))

    ax = axes[0, 0]
    ax.scatter(y, p, s=14, alpha=0.55, edgecolor="k", linewidth=0.3)
    ax.plot([0, 5], [0, 5], "r--", lw=1)
    ax.set(xlabel="True Grammar Score", ylabel="OOF prediction",
           title=f"OOF predictions  (RMSE {np.sqrt(np.mean((y-p)**2)):.3f}, "
                 f"r={r:.3f})")
    ax.grid(alpha=0.3)

    ax = axes[0, 1]
    resid = y - p
    ax.hist(resid, bins=36, color="steelblue", edgecolor="white")
    ax.axvline(0, color="r", lw=1)
    ax.axvline(resid.mean(), color="k", ls="--", lw=1,
               label=f"mean {resid.mean():+.3f}")
    ax.set(xlabel="residual (y - prediction)", title="Residual distribution")
    ax.legend()
    ax.grid(alpha=0.3)

    ax = axes[1, 0]
    bins = np.clip(np.round(y * 2) / 2, 0, 5)
    order = sorted(set(bins))
    mp = [p[bins == b].mean() if (bins == b).any() else np.nan for b in order]
    sd = [p[bins == b].std() if (bins == b).any() else 0.0 for b in order]
    ax.errorbar(order, mp, yerr=sd, fmt="o-", capsize=4)
    ax.plot([0, 5], [0, 5], "r--", lw=1)
    ax.set(xlabel="true score", ylabel="mean predicted",
           title="Calibration by true label")
    ax.grid(alpha=0.3)

    ax = axes[1, 1]
    imp.head(20).iloc[::-1].plot.barh(ax=ax, color="darkorange")
    ax.set(title="Top 20 feature importances", xlabel="importance")
    ax.grid(alpha=0.3, axis="x")

    fig.suptitle("Grammar Score model — out-of-fold evaluation", fontsize=13)
    fig.tight_layout()
    fig.savefig(f"{WORK}/report.png", dpi=110)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(8, 4))
    pd.Series(y).value_counts().sort_index().plot.bar(ax=ax,
                                                      color="slategray")
    ax.set(title="Training label distribution", xlabel="Grammar Score",
           ylabel="count")
    fig.tight_layout()
    fig.savefig(f"{WORK}/label_dist.png", dpi=110)
    plt.close(fig)
    print("plots written: report.png, label_dist.png")


if __name__ == "__main__":
    try:
        main()
    except Exception:
        traceback.print_exc()
        sys.exit(1)
