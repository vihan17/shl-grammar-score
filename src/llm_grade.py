"""Grade ASR transcripts against the Grammar Score rubric with an LLM.

Usage:
  python llm_grade.py --in asr.csv --out llm.csv --limit 60
  python llm_grade.py --in asr.csv --out llm.csv

Reads OPENROUTER_API_KEY (or OPENAI_API_KEY). Resume-safe: existing
filenames in --out are skipped.
"""

import argparse
import json
import os
import re
import sys
import threading
import time
import urllib.error
import urllib.request
from concurrent.futures import ThreadPoolExecutor

sys.stdout.reconfigure(encoding="utf-8", errors="replace")

BASE = "/kaggle/input/competitions/shl-hiring-assessment-2026/Dataset_Final"
LOCAL_BASE = "data"
RUBRIC = """\
You are grading SPEAKER grammar for the SHL Grammar Scoring Engine task.
The ground truth is an MOS Likert Grammar Score in 0-5 (continuous).

Official rubric (per-score descriptors):

1 - The person's speech struggles with proper sentence structure and
    syntax, displaying limited control over simple grammatical structures
    and memorized sentence patterns.
2 - The person has a limited understanding of sentence structure and
    syntax. Although they use simple structures, they consistently make
    basic sentence structure and grammatical mistakes. They might leave
    sentences incomplete.
3 - The person demonstrates a decent grasp of sentence structure but
    makes errors in grammatical structure, or they show a decent grasp of
    grammatical structure but make errors in sentence syntax and
    structure.
4 - The person displays a strong understanding of sentence structure and
    syntax. They consistently show good control of grammar. While
    occasional errors may occur, they are generally minor and do not lead
    to misunderstandings; the person can correct most of them.
5 - Overall, the person showcases high grammatical accuracy and adept
    control of complex grammar. They use grammar accurately and
    effectively, seldom making noticeable mistakes. Additionally, they
    handle complex language structures well and correct themselves when
    necessary.

Half points (e.g. 2.5, 3.5, 4.5) are allowed and encouraged when the
speaker sits between two descriptors.

Score GRAMMAR ONLY: sentence structure, syntax, agreement, tense, article
and preposition use, completeness of sentences. Do NOT penalize for:
topic interest, pronunciation, hesitations or filler words (fluency),
vocabulary sophistication, or content correctness. Where the transcript
shows a clear ASR decoding artifact rather than a real speaker error,
ignore that token rather than counting it against grammar.

Reply with JSON only: {"score": <number from 0 to 5>}"""


def key():
    for name in ("GEMINI_API_KEY", "GOOGLE_API_KEY", "OPENROUTER_API_KEY",
                 "OPENAI_API_KEY"):
        k = os.environ.get(name) or ""
        if len(k) >= 20:
            return k, name
    raise SystemExit("no usable API key in env (set GEMINI_API_KEY)")


GEMINI_BASE = ("https://generativelanguage.googleapis.com/v1beta/models/")


def gemini_call(model, messages, k, temperature=0.0, max_tokens=64):
    """OpenAI-style chat messages -> Gemini generateContent."""
    sys_msg = " ".join(m["content"] for m in messages
                       if m["role"] == "system")
    contents = []
    for m in messages:
        if m["role"] == "system":
            continue
        role = "model" if m["role"] == "assistant" else "user"
        if contents and contents[-1]["role"] == role:
            contents[-1]["parts"][0]["text"] += "\n" + m["content"]
        else:
            contents.append({"role": role,
                             "parts": [{"text": m["content"]}]})
    body = {"contents": contents,
            "generationConfig": {"temperature": temperature,
                                 "maxOutputTokens": max_tokens,
                                 "responseMimeType": "application/json"}}
    if sys_msg:
        body["systemInstruction"] = {"parts": [{"text": sys_msg}]}
    url = f"{GEMINI_BASE}{model}:generateContent"
    for t in range(5):
        req = urllib.request.Request(
            url, data=json.dumps(body).encode(),
            headers={"x-goog-api-key": k,
                     "Content-Type": "application/json"})
        try:
            d = json.load(urllib.request.urlopen(req, timeout=120))
            parts = (d.get("candidates", [{}])[0].get("content", {})
                     .get("parts", []))
            txt = "".join(p.get("text", "") for p in parts)
            if txt:
                return txt
            if t == 4:
                raise RuntimeError("empty response: " + json.dumps(d)[:200])
        except urllib.error.HTTPError as e:
            msg = e.read().decode()[:200]
            if e.code in (429, 500, 503) and t < 4:
                time.sleep(2 * (t + 1))
                continue
            raise RuntimeError(f"HTTP {e.code}: {msg}")
        time.sleep(1.5)
    raise RuntimeError("unreachable")


def chat(model, messages, k, tries=5, base="https://openrouter.ai",
         temperature=0.0, max_tokens=64):
    body = json.dumps({"model": model, "messages": messages,
                       "max_tokens": max_tokens,
                       "temperature": temperature}).encode()
    for t in range(tries):
        try:
            req = urllib.request.Request(
                base + "/api/v1/chat/completions", data=body,
                headers={"Authorization": "Bearer " + k,
                         "Content-Type": "application/json",
                         "X-Title": "SHL Grammar Score"})
            d = json.load(urllib.request.urlopen(req, timeout=120))
            return d["choices"][0]["message"]["content"]
        except urllib.error.HTTPError as e:
            msg = e.read().decode()[:200]
            if e.code in (429, 500, 502, 503) and t < tries - 1:
                time.sleep(3 * (t + 1))
                continue
            raise RuntimeError(f"HTTP {e.code}: {msg}")
        except Exception:
            if t < tries - 1:
                time.sleep(2 * (t + 1))
                continue
            raise
    raise RuntimeError("unreachable")


def parse_score(text):
    m = re.search(r'"score"\s*:\s*([0-9.]+)', text)
    if m:
        return float(m.group(1))
    m = re.search(r'(-?\d+(?:\.\d+)?)', text)
    if m:
        v = float(m.group(1))
        if 0 <= v <= 5:
            return v
        return max(0.0, min(5.0, v))
    return None


def fewshot(tr, asr, n=6):
    """Pick train examples spread across the score range."""
    lab = dict(zip(tr.filename, tr.label))
    rows = [(f, lab[f], t) for f, t in asr if f in lab]
    rows = [r for r in rows if r[1] is not None]
    rows.sort(key=lambda r: r[1])
    picks = []
    if len(rows) >= n:
        for i in range(n):
            picks.append(rows[int((i + 0.5) * len(rows) / n) - 1])
    else:
        picks = rows
    seen, out = set(), []
    for f, l, t in picks:
        if f in seen:
            continue
        seen.add(f)
        out.append((l, t))
    return out


def grade_one(call, transcript, shots, samples=1, temperature=0.0):
    msgs = [{"role": "system", "content": RUBRIC}]
    for label, text in shots:
        msgs.append({"role": "user",
                     "content": "Transcript:\n" + text[:4000]})
        msgs.append({"role": "assistant",
                     "content": json.dumps({"score": label})})
    msgs.append({"role": "user",
                 "content": "Transcript:\n" + (transcript or "")[:4000]
                            + "\nReply with JSON only."})
    scores = []
    for _ in range(max(1, samples)):
        try:
            raw = call(msgs, temperature)
        except Exception:
            if len(scores):
                continue
            raise
        s = parse_score(raw)
        if s is not None:
            scores.append(s)
    if not scores:
        return None
    return float(sum(scores) / len(scores))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--in", dest="inp", default="asr.csv")
    ap.add_argument("--out", dest="out", default="llm.csv")
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--model", default="gemini-2.5-flash")
    ap.add_argument("--provider", default="auto",
                    choices=["auto", "gemini", "openrouter"])
    ap.add_argument("--workers", type=int, default=6)
    ap.add_argument("--samples", type=int, default=1)
    ap.add_argument("--temperature", type=float, default=0.0)
    ap.add_argument("--base", default="data")
    args = ap.parse_args()

    k, kname = key()
    provider = args.provider
    if provider == "auto":
        provider = "gemini" if kname in ("GEMINI_API_KEY",
                                         "GOOGLE_API_KEY") else "openrouter"
    if provider == "gemini":
        def call(msgs, temperature=0.0):
            return gemini_call(args.model, msgs, k, temperature, 64)
    else:
        def call(msgs, temperature=0.0):
            return chat(args.model, msgs, k, temperature=temperature)
    print(f"provider={provider} model={args.model} key={kname} "
          f"(len {len(k)})")
    import pandas as pd
    asr = pd.read_csv(args.inp)
    tr = pd.read_csv(os.path.join(args.base, "train.csv"))
    print("asr", asr.shape, "train", tr.shape)

    items = [(r.filename, "" if pd.isna(r.text) else str(r.text))
             for r in asr.itertuples()]
    if args.limit:
        train_fn = set(tr.filename)
        tr_items = [i for i in items if i[0] in train_fn][:args.limit]
        te_items = [i for i in items if i[0] not in train_fn]
        items = tr_items + te_items[: max(0, args.limit // 4)]
        print("limited to", len(items), "clips")

    done = {}
    if os.path.exists(args.out):
        prev = pd.read_csv(args.out)
        done = {r.filename: r.llm_score for r in prev.itertuples()}
        print("resuming:", len(done), "already graded")

    todo = [(f, t) for f, t in items if f not in done]
    print("todo", len(todo), "| model", args.model)
    if not todo:
        return

    shots = fewshot(tr, items)
    print("few-shot examples:", [(l, t[:40]) for l, t in shots])

    lock = threading.Lock()
    rows = list(done.items())
    t0 = time.time()
    ok = bad = 0

    def work(it):
        f, t = it
        try:
            s = grade_one(call, t, shots, args.samples, args.temperature)
        except Exception as e:
            print("ERR", f, type(e).__name__, str(e)[:160], flush=True)
            return f, None
        return f, s

    with ThreadPoolExecutor(max_workers=args.workers) as ex:
        for i, (f, s) in enumerate(ex.map(work, todo), 1):
            with lock:
                if s is None:
                    bad += 1
                else:
                    ok += 1
                    rows.append((f, s))
                if i % 10 == 0 or i == len(todo):
                    el = time.time() - t0
                    eta = el / i * (len(todo) - i)
                    print(f"  graded {i}/{len(todo)} ok={ok} bad={bad} "
                          f"{el:.0f}s eta {eta:.0f}s", flush=True)
                    pd.DataFrame(rows, columns=["filename", "llm_score"]).to_csv(
                        args.out, index=False)
    pd.DataFrame(rows, columns=["filename", "llm_score"]).to_csv(
        args.out, index=False)
    print(f"DONE {ok} ok {bad} failed {time.time()-t0:.0f}s")

    # validation against labels where available
    lab = dict(zip(tr.filename, tr.label))
    scored = [(lab[f], s) for f, s in rows if f in lab and s is not None]
    if len(scored) > 20:
        import numpy as np
        from scipy.stats import pearsonr
        y = np.array([a for a, _ in scored])
        p = np.array([b for _, b in scored])
        rmse = float(np.sqrt(np.mean((y - p) ** 2)))
        print(f"VAL on {len(y)} train clips: rmse={rmse:.4f} "
              f"pearson={pearsonr(y, p)[0]:.4f}")
        b = np.clip(np.round(p * 2) / 2, 0, 5)
        print(f"VAL rounded: rmse={float(np.sqrt(np.mean((y-b)**2))):.4f}")


if __name__ == "__main__":
    main()
