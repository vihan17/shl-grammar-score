"""Per-clip Gemini grading of whisper transcripts (high quality path).

  python llm_grade_perclip.py --limit 150   # quick VAL check
  python llm_grade_perclip.py               # full 985

Writes llm_perclip.csv (split,filename,llm_score), resumable.
"""
import argparse
import json
import os
import sys
import threading
import time
import urllib.error
import urllib.request
from concurrent.futures import ThreadPoolExecutor, as_completed

import pandas as pd

sys.stdout.reconfigure(encoding="utf-8", errors="replace")
import llm_grade as G  # noqa: E402  (RUBRIC, fewshot, grade_one)

TOK = os.environ.get("GEMINI_API_KEY", "")
# strict preference order: lite first (500/day), then flashes (20/day each)
MODELS = ["gemini-3.5-flash-lite", "gemini-3.5-flash",
          "gemini-3.7-flash", "gemini-3.8-flash"]
URL = ("https://generativelanguage.googleapis.com/v1beta/models/"
       "{}:generateContent")
_lk = threading.Lock()
DEAD = set()  # models whose daily quota is gone (retry delay > 10 min)

# free tier: 15 req/min per model -> keep 14/min per model, rotate models
_RPM = 10
_times = {m: [] for m in MODELS}
_rr = [0]


def _acquire(model):
    while True:
        with _lk:
            q = _times[model]
            now = time.time()
            while q and now - q[0] >= 60.0:
                q.pop(0)
            if len(q) < _RPM:
                q.append(now)
                return
            wait = q[0] + 60.0 - now + 0.05
        time.sleep(min(wait, 5.0))


def call_model(msgs, temperature=0.0):
    sys_msg = " ".join(m["content"] for m in msgs if m["role"] == "system")
    contents = []
    for m in msgs:
        if m["role"] == "system":
            continue
        role = "model" if m["role"] == "assistant" else "user"
        if contents and contents[-1]["role"] == role:
            contents[-1]["parts"][0]["text"] += "\n" + m["content"]
        else:
            contents.append({"role": role, "parts": [{"text": m["content"]}]})
    body = {"contents": contents,
            "generationConfig": {"temperature": temperature,
                                 "maxOutputTokens": 64,
                                 "responseMimeType": "application/json"}}
    if sys_msg:
        body["systemInstruction"] = {"parts": [{"text": sys_msg}]}
    last = None
    with _lk:
        avail = [m for m in MODELS if m not in DEAD]
    if not avail:
        raise RuntimeError("all models dead (daily quota exhausted)")
    order = avail  # strict preference; no rotation (flashes have tiny quotas)
    for model in order:
        if model in DEAD:
            continue
        if "lite" not in model:
            body["generationConfig"]["thinkingConfig"] = {"thinkingBudget": 0}
        else:
            body["generationConfig"].pop("thinkingConfig", None)
        _acquire(model)
        attempts = 0
        waits = 0
        while attempts < 6:
            attempts += 1
            req = urllib.request.Request(
                URL.format(model), data=json.dumps(body).encode(),
                headers={"x-goog-api-key": TOK,
                         "Content-Type": "application/json"})
            try:
                d = json.load(urllib.request.urlopen(req, timeout=90))
                txt = "".join(p.get("text", "")
                              for p in d["candidates"][0]["content"]["parts"])
                if txt.strip():
                    return txt
                last = f"{model}: empty"
            except urllib.error.HTTPError as e:
                body_txt = e.read().decode()
                last = f"{model}: HTTP {e.code} {body_txt[:100]}"
                if e.code == 429:
                    import re as _re
                    m = _re.search(r"retry in (?:(\d+)h)?(?:(\d+)m)?([0-9.]+)s",
                                   body_txt)
                    delay = 45.0
                    if m and (m.group(1) or m.group(2)):
                        delay = (int(m.group(1) or 0) * 3600
                                 + int(m.group(2) or 0) * 60
                                 + float(m.group(3)))
                    elif m:
                        delay = float(m.group(3))
                    if delay > 600:
                        with _lk:
                            DEAD.add(model)
                        print(f"  {model} daily quota gone — disabled",
                              flush=True)
                        break
                    time.sleep(delay + 1)
                    waits += 1
                    if waits > 8:
                        break
                    attempts -= 1  # quota waits don't consume attempts
                    continue
                if e.code in (404, 500, 503, 504):
                    break
                raise RuntimeError(last)
            except Exception as e:
                last = f"{model}: {type(e).__name__} {str(e)[:100]}"
                break
        time.sleep(1)
    raise RuntimeError(str(last))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--in", dest="inp", default="asr_whisper.csv")
    ap.add_argument("--out", dest="out", default="llm_perclip.csv")
    ap.add_argument("--limit", type=int, default=0,
                    help="only first N train rows (quick VAL)")
    ap.add_argument("--workers", type=int, default=6)
    ap.add_argument("--base", default="data")
    a = ap.parse_args()

    asr = pd.read_csv(a.inp)
    tr = pd.read_csv(os.path.join(a.base, "train.csv"))
    lab = dict(zip(tr.filename, tr.label))
    items = [(r.split, r.filename,
              "" if pd.isna(r.text) else str(r.text))
             for r in asr.itertuples()]
    if a.limit:
        train = [i for i in items if i[0] == "train"][:a.limit]
        test = [i for i in items if i[0] == "test"][:0]
        items = train + test
    print(f"items {len(items)}", flush=True)

    done = {}
    if os.path.exists(a.out):
        prev = pd.read_csv(a.out)
        done = {(r.split, r.filename): r.llm_score for r in prev.itertuples()}
        print("resuming:", len(done), flush=True)
    todo = [i for i in items if (i[0], i[1]) not in done]
    print("todo", len(todo), flush=True)
    if not todo:
        return

    tr_items = [(f, t) for s, f, t in items if s == "train"]
    shots = G.fewshot(tr, tr_items)

    lock = threading.Lock()
    rows = list(done.items())
    t0 = time.time()
    ok = bad = 0
    rate_limited = 0

    def work(it):
        nonlocal rate_limited
        sp, fn, txt = it
        try:
            sc = G.grade_one(call_model, txt, shots, 1, 0.0)
        except Exception as e:
            with _lk:
                rate_limited += 1
            print("ERR", sp, fn, str(e)[:120], flush=True)
            return (sp, fn), None
        return (sp, fn), sc

    with ThreadPoolExecutor(max_workers=a.workers) as ex:
        futs = [ex.submit(work, it) for it in todo]
        for i, fut in enumerate(as_completed(futs), 1):
            (sp, fn), sc = fut.result()
            with lock:
                if sc is None:
                    bad += 1
                else:
                    ok += 1
                    rows.append(((sp, fn), sc))
                if i % 25 == 0 or i == len(todo):
                    el = time.time() - t0
                    eta = el / i * (len(todo) - i)
                    print(f"  {i}/{len(todo)} ok={ok} bad={bad} "
                          f"{el:.0f}s eta {eta:.0f}s", flush=True)
                    pd.DataFrame(
                        [((sp, fn), sc) for (sp, fn), sc in rows],
                        columns=["key", "llm_score"]).assign(
                            split=lambda d: [x[0] for x in d.key],
                            filename=lambda d: [x[1] for x in d.key],
                        ).drop(columns=["key"]).to_csv(a.out, index=False)

    pd.DataFrame(
        [((sp, fn), sc) for (sp, fn), sc in rows],
        columns=["key", "llm_score"]).assign(
            split=lambda d: [x[0] for x in d.key],
            filename=lambda d: [x[1] for x in d.key],
        ).drop(columns=["key"]).to_csv(a.out, index=False)
    print(f"DONE ok={ok} bad={bad} {time.time()-t0:.0f}s", flush=True)

    scored = [(lab[fn], sc) for (sp, fn), sc in rows
              if sp == "train" and fn in lab and sc is not None]
    if len(scored) > 20:
        import numpy as np
        from scipy.stats import pearsonr
        y = np.array([x for x, _ in scored])
        p = np.array([x for _, x in scored])
        print(f"VAL n={len(y)} rmse={float(np.sqrt(np.mean((y-p)**2))):.4f} "
              f"pearson={pearsonr(y, p)[0]:.4f}", flush=True)


if __name__ == "__main__":
    main()
