"""Build hybrid llm_perclip.csv: batch grades (full coverage) overlaid with
per-clip grades where available (higher quality).

Writes llm_perclip.csv (split,filename,llm_score) covering all 983 rows.
"""
import pandas as pd

batch = pd.read_csv("llm_whisper.csv")[["split", "filename", "llm_score"]]
perc = pd.read_csv("llm_perclip_465.csv")[
    ["split", "filename", "llm_score"]].drop_duplicates(["split", "filename"])

h = batch.merge(perc, on=["split", "filename"], how="left",
                suffixes=("_batch", "_perc"))
h["llm_score"] = h.llm_score_perc.combine_first(h.llm_score_batch)
h = h[["split", "filename", "llm_score"]]
assert len(h) == len(batch), (len(h), len(batch))

h.to_csv("llm_perclip.csv", index=False)
n_p = len(perc)
print(f"hybrid rows={len(h)} train={(h.split=='train').sum()} "
      f"test={(h.split=='test').sum()}")
print("premium(perclip) rows:",
      len(batch.merge(perc, on=['split', 'filename'], how='inner')))
print("nan:", int(h.llm_score.isna().sum()), "premium:", n_p)
