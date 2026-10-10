"""Grammar-fluency text features from GPT-2 perplexity + simple stats.

  python add_ppl.py            # writes text_feats.csv (985 rows)

Reads whisper transcripts (asr_whisper.csv.disabled), scores each under
GPT-2 (local CPU), plus length/distinct-n/contraction stats.
Resumable: skips rows already present in output.
"""
import argparse
import os
import sys
import time

import torch
from transformers import GPT2LMHeadModel, GPT2TokenizerFast

import numpy as np
import pandas as pd

sys.stdout.reconfigure(encoding="utf-8", errors="replace")

INP = "asr_whisper.csv.disabled"
if not os.path.exists(INP) and os.path.exists("asr_whisper.csv"):
    INP = "asr_whisper.csv"
OUT = "text_feats.csv"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--limit", type=int, default=0,
                    help="smoke: only first N todo items")
    a = ap.parse_args()

    asr = pd.read_csv(INP)
    items = [(r.split, r.filename,
              "" if pd.isna(r.text) else str(r.text))
             for r in asr.itertuples()]
    print(f"items {len(items)}")

    done = set()
    if os.path.exists(OUT):
        prev = pd.read_csv(OUT)
        done = set(zip(prev.split, prev.filename))
        print("resuming:", len(done))
    todo = [i for i in items if (i[0], i[1]) not in done]
    if a.limit:
        todo = todo[:a.limit]
    print("todo", len(todo))
    if not todo:
        return

    tok = GPT2TokenizerFast.from_pretrained("gpt2")
    model = GPT2LMHeadModel.from_pretrained("gpt2")
    model.eval()

    rows = []
    t0 = time.time()
    with torch.no_grad():
        for k, (sp, fn, txt) in enumerate(todo):
            t = " ".join(txt.split())
            if not t:
                rows.append(dict(split=sp, filename=fn, ppl=np.nan,
                                 words=0, distinct2=np.nan, n_contr=0))
                continue
            enc = tok(t, return_tensors="pt", truncation=True, max_length=512)
            out = model(**enc, labels=enc["input_ids"])
            ppl = float(torch.exp(out.loss))
            words = t.split()
            bw = [tuple(words[i:i + 2]) for i in range(len(words) - 1)]
            d2 = len(set(bw)) / len(bw) if bw else np.nan
            n_contr = sum(t.lower().count(w) for w in
                          ["n't", "'re", "'ve", "'ll", "'d", "'m", "'s"])
            rows.append(dict(split=sp, filename=fn, ppl=ppl,
                             words=len(words), distinct2=d2,
                             n_contr=n_contr))
            if (k + 1) % 100 == 0:
                pd.DataFrame(rows).to_csv(OUT, index=False,
                                          mode="a", header=not os.path.exists(OUT))
                rows = []
                el = time.time() - t0
                print(f"  {k+1}/{len(todo)} {el/(k+1):.2f}s/clip", flush=True)
    if rows:
        pd.DataFrame(rows).to_csv(OUT, index=False,
                                  mode="a", header=not os.path.exists(OUT))
    el = time.time() - t0
    df = pd.read_csv(OUT)
    print(f"PPL DONE {len(df)}/{len(items)} in {el/60:.1f} min")

    tr = pd.read_csv(os.path.join("data", "train.csv"))
    lab = dict(zip(tr.filename, tr.label))
    m = df[df.split == "train"].copy()
    m["label"] = m.filename.map(lab)
    m = m.dropna(subset=["ppl", "label"])
    if len(m) > 30:
        from scipy.stats import pearsonr
        for c in ["ppl", "words", "distinct2", "n_contr"]:
            v = m.dropna(subset=[c])
            print(f"  {c}: pearson(label)={pearsonr(v.label, v[c])[0]:+.4f}")


if __name__ == "__main__":
    main()
