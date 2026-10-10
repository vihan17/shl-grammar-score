"""Audio-native Gemini batch grading (prosody/disfluency from raw audio).

  python llm_grade_audio.py --batches 1 --batch-size 3   # probe
  python llm_grade_audio.py                               # full run

Sends base64 16k-mono WAV clips inline (5/request ≈ 13MB, under limits).
Quota: gemini-3.5-flash-lite 500 req/day -> 985 clips at batch 5 = 197 req.
Writes llm_audio.csv (split,filename,llm_audio), resumable.
"""
import argparse
import base64
import io
import json
import os
import re
import sys
import time
import urllib.error
import urllib.request

import numpy as np
import pandas as pd

sys.stdout.reconfigure(encoding="utf-8", errors="replace")
import llm_grade as G  # noqa: E402  (RUBRIC, fewshot)

TOK = os.environ.get("GEMINI_API_KEY", "")
MODELS = ["gemini-3.5-flash-lite", "gemini-3.5-flash",
          "gemini-3.7-flash", "gemini-3.8-flash"]
URL = "https://generativelanguage.googleapis.com/v1beta/models/{}:generateContent"
SR = 16000
RPM_SLEEP = 6.0

DEAD = set()


def wav_b64(path):
    import struct
    import wave
    with wave.open(path, "rb") as w:
        nch, sw, sr = w.getnchannels(), w.getsampwidth(), w.getframerate()
        raw = w.readframes(w.getnframes())
    if nch > 1 and sw == 2:
        samples = struct.unpack(f"<{len(raw)//2}h", raw)
        raw = struct.pack(f"<{len(samples)//nch}h", *samples[::nch])
        nch = 1
    buf = io.BytesIO()
    with wave.open(buf, "wb") as w:
        w.setnchannels(nch)
        w.setsampwidth(sw)
        w.setframerate(sr)
        w.writeframes(raw)
    return base64.b64encode(buf.getvalue()).decode()


def _dead_model(model, err_msg):
    if model in DEAD:
        return True
    m = re.search(r"retry in (\d+)h (\d+)m (\d+)s", err_msg)
    if m:
        secs = int(m.group(1)) * 3600 + int(m.group(2)) * 60 + int(m.group(3))
        if secs > 600:
            DEAD.add(model)
            print(f"  DEAD {model} ({secs//3600}h) -> next", flush=True)
            return True
    return False


def call_parts(parts, max_tokens=8192):
    last = None
    for model in MODELS:
        if model in DEAD:
            continue
        body = json.dumps({
            "contents": [{"role": "user", "parts": parts}],
            "generationConfig": {"temperature": 0.0,
                                 "maxOutputTokens": max_tokens,
                                 "responseMimeType": "application/json"},
        }).encode()
        for t in range(3):
            req = urllib.request.Request(
                URL.format(model), data=body,
                headers={"x-goog-api-key": TOK,
                         "Content-Type": "application/json"})
            try:
                d = json.load(urllib.request.urlopen(req, timeout=300))
                txt = "".join(p.get("text", "")
                              for p in d["candidates"][0]["content"]["parts"])
                if txt.strip():
                    return txt
                last = f"{model}: empty text"
            except urllib.error.HTTPError as e:
                body_txt = e.read().decode()[:300]
                last = f"{model}: HTTP {e.code} {body_txt}"
                if e.code == 429 and _dead_model(model, body_txt):
                    break
                if e.code in (404, 429, 500, 503, 504):
                    break
                raise RuntimeError(last)
            except Exception as e:
                last = f"{model}: {type(e).__name__} {str(e)[:120]}"
                break
            time.sleep(5)
        print(f"  model fail -> {last}", flush=True)
    raise RuntimeError("all models failed: " + str(last))


def build_parts(shots, batch, b64s):
    parts = [{"text": G.RUBRIC + "\n\nFew-shot calibration examples:"}]
    for i, (lab, txt) in enumerate(shots):
        parts.append({"text": f"\n[example {i}] Score {lab}\n"
                              f"Transcript: {txt[:600]}"})
    parts.append({"text": "\nNow listen to EACH audio clip below "
                          "(grammar, vocabulary, coherence, fluency, "
                          "pronunciation). Reply with JSON only:\n"
                          '{"scores": [{"filename": "...", "score": <number>}]}'
                          "\nClips follow: filename tag then its audio."})
    for (sp, fn), b64 in zip(batch, b64s):
        parts.append({"text": f'\n<item split="{sp}" filename="{fn}">'})
        parts.append({"inline_data": {"mime_type": "audio/wav", "data": b64}})
        parts.append({"text": "</item>"})
    return parts


def parse_scores(raw, expected):
    m = None
    try:
        obj = json.loads(raw)
        if isinstance(obj, dict):
            m = obj.get("scores") or obj.get("items")
    except json.JSONDecodeError:
        g = re.search(r"\{.*\}", raw, re.S)
        if g:
            try:
                obj = json.loads(g.group(0))
                m = obj.get("scores")
            except json.JSONDecodeError:
                pass
    out = {}
    if isinstance(m, list):
        for e in m:
            if not isinstance(e, dict):
                continue
            fn = str(e.get("filename", ""))
            try:
                s = float(e.get("score"))
            except (TypeError, ValueError):
                continue
            out[fn] = max(0.0, min(5.0, s))
    return {k: v for k, v in out.items() if k in expected}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--in", dest="inp", default="asr_whisper.csv")
    ap.add_argument("--out", dest="out", default="llm_audio.csv")
    ap.add_argument("--batch-size", type=int, default=5)
    ap.add_argument("--batches", type=int, default=0)
    ap.add_argument("--base", default="data")
    a = ap.parse_args()
    if not os.path.exists(a.inp):
        alt = a.inp + ".disabled" if not a.inp.endswith(".disabled") \
            else a.inp.replace(".disabled", "")
        if os.path.exists(alt):
            a.inp = alt

    asr = pd.read_csv(a.inp)
    tr = pd.read_csv(os.path.join(a.base, "train.csv"))
    lab = dict(zip(tr.filename, tr.label))
    items = [(r.split, r.filename) for r in asr.itertuples()]
    print(f"items {len(items)} model0 {MODELS[0]} batch {a.batch_size}")

    done = {}
    if os.path.exists(a.out):
        prev = pd.read_csv(a.out)
        done = {(r.split, r.filename): r.llm_audio for r in prev.itertuples()}
        print("resuming:", len(done))
    todo = [i for i in items if i not in done]
    print("todo", len(todo))
    if not todo:
        return

    tr_map = {}
    for sp, fn in items:
        if sp == "train":
            row = asr[(asr.split == "train") & (asr.filename == fn)]
            tr_map[fn] = "" if row.empty or pd.isna(row.text.iloc[0]) \
                else str(row.text.iloc[0])
    tr_items = list(tr_map.items())
    shots = G.fewshot(tr, tr_items)

    rows = list(done.items())
    bs = a.batch_size
    t0 = time.time()
    n_batches = (len(todo) + bs - 1) // bs
    if a.batches:
        n_batches = min(n_batches, a.batches)
    ok = bad = 0
    for bi in range(n_batches):
        batch = todo[bi * bs:(bi + 1) * bs]
        expected = {fn for _, fn in batch}
        try:
            b64s = []
            for sp, fn in batch:
                for path in (os.path.join("audio", "Dataset_Final", sp, fn),
                             os.path.join(a.base, sp, fn),
                             os.path.join(a.base, fn)):
                    if os.path.exists(path):
                        break
                else:
                    raise FileNotFoundError(f"audio not found: {sp}/{fn}")
                b64s.append(wav_b64(path))
            raw = call_parts(build_parts(shots, batch, b64s))
            got = parse_scores(raw, expected)
        except Exception as e:
            print(f"batch {bi+1}/{n_batches} FAILED: {e}", flush=True)
            break
        missing = [fn for _, fn in batch if fn not in got]
        for sp, fn in batch:
            if fn in got:
                rows.append(((sp, fn), got[fn]))
                ok += 1
            else:
                bad += 1
        pd.DataFrame(
            [((sp, fn), sc) for (sp, fn), sc in rows],
            columns=["key", "llm_audio"]).assign(
                split=lambda d: [x[0] for x in d.key],
                filename=lambda d: [x[1] for x in d.key],
            ).drop(columns=["key"]).to_csv(a.out, index=False)
        print(f"batch {bi+1}/{n_batches}: {len(got)}/{len(batch)} parsed "
              f"(missing {len(missing)}) | total {ok} | "
              f"{time.time()-t0:.0f}s", flush=True)
        time.sleep(RPM_SLEEP)

    scored = [(lab[fn], sc) for (sp, fn), sc in rows
              if sp == "train" and fn in lab and sc is not None]
    if len(scored) > 20:
        from scipy.stats import pearsonr
        y = np.array([x for x, _ in scored])
        p = np.array([x for _, x in scored])
        print(f"VAL n={len(y)} rmse={float(np.sqrt(np.mean((y-p)**2))):.4f} "
              f"pearson={pearsonr(y, p)[0]:.4f}", flush=True)
    if bad or ok == 0:
        sys.exit(1)


if __name__ == "__main__":
    main()
