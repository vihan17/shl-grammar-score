"""Re-grade with whisper-large-v3 transcripts (split-aware, resumable).

  python llm_grade_wsp.py --limit 8      # quota probe
  python llm_grade_wsp.py                # full 985

Writes llm_whisper.csv with split,filename,llm_score keyed on (split,filename)
because 212 test filenames reuse train names.
"""
import argparse
import os
import sys
import threading
import time
from concurrent.futures import ThreadPoolExecutor

import pandas as pd

sys.stdout.reconfigure(encoding="utf-8", errors="replace")
import llm_grade as G  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--in", dest="inp", default="asr_whisper.csv")
    ap.add_argument("--out", dest="out", default="llm_whisper.csv")
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--model", default="gemini-2.5-flash")
    ap.add_argument("--workers", type=int, default=6)
    ap.add_argument("--base", default="data")
    a = ap.parse_args()

    k, kname = G.key()
    print(f"key={kname} (len {len(k)}) model={a.model}")

    def call(msgs, temperature=0.0):
        return G.chat(a.model, msgs, k, temperature=temperature)

    asr = pd.read_csv(a.inp)
    tr = pd.read_csv(os.path.join(a.base, "train.csv"))
    lab = dict(zip(tr.filename, tr.label))
    items = [(r.split, r.filename,
              "" if pd.isna(r.text) else str(r.text))
             for r in asr.itertuples()]
    print("items", len(items))

    if a.limit:
        train = [i for i in items if i[0] == "train"][:a.limit]
        test = [i for i in items if i[0] == "test"][:max(1, a.limit // 4)]
        items = train + test
        print("limited to", len(items))

    done = {}
    if os.path.exists(a.out):
        prev = pd.read_csv(a.out)
        done = {(r.split, r.filename): r.llm_score for r in prev.itertuples()}
        print("resuming:", len(done), "already graded")

    todo = [i for i in items if (i[0], i[1]) not in done]
    print("todo", len(todo))
    if not todo:
        return

    tr_items = [(f, t) for s, f, t in items if s == "train"]
    shots = G.fewshot(tr, tr_items)
    print("shots:", [(l, t[:30]) for l, t in shots])

    lock = threading.Lock()
    rows = list(done.items())
    t0 = time.time()
    ok = bad = 0

    def work(it):
        s, f, t = it
        try:
            sc = G.grade_one(call, t, shots, 1, 0.0)
        except Exception as e:
            print("ERR", s, f, type(e).__name__, str(e)[:140], flush=True)
            return (s, f), None
        return (s, f), sc

    with ThreadPoolExecutor(max_workers=a.workers) as ex:
        for i, ((sp, fn), sc) in enumerate(ex.map(work, todo), 1):
            with lock:
                if sc is None:
                    bad += 1
                else:
                    ok += 1
                    rows.append(((sp, fn), sc))
                if i % 10 == 0 or i == len(todo):
                    el = time.time() - t0
                    eta = el / i * (len(todo) - i)
                    print(f"  {i}/{len(todo)} ok={ok} bad={bad} "
                          f"eta {eta:.0f}s", flush=True)
                    pd.DataFrame(
                        [((sp, fn), sc) for (sp, fn), sc in rows],
                        columns=["key", "llm_score"]
                    ).assign(
                        split=lambda d: [x[0] for x in d.key],
                        filename=lambda d: [x[1] for x in d.key],
                    ).drop(columns=["key"]).to_csv(a.out, index=False)

    pd.DataFrame(
        [((sp, fn), sc) for (sp, fn), sc in rows],
        columns=["key", "llm_score"]
    ).assign(
        split=lambda d: [x[0] for x in d.key],
        filename=lambda d: [x[1] for x in d.key],
    ).drop(columns=["key"]).to_csv(a.out, index=False)
    print(f"DONE ok={ok} bad={bad} {time.time()-t0:.0f}s")

    scored = [(lab[fn], sc) for (sp, fn), sc in rows
              if sp == "train" and fn in lab and sc is not None]
    if len(scored) > 20:
        import numpy as np
        from scipy.stats import pearsonr
        y = np.array([x for x, _ in scored])
        p = np.array([x for _, x in scored])
        print(f"VAL n={len(y)} rmse={float(np.sqrt(np.mean((y-p)**2))):.4f} "
              f"pearson={pearsonr(y, p)[0]:.4f}")


if __name__ == "__main__":
    main()
