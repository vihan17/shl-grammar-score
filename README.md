# SHL Grammar Score — Research Intern Hiring Assessment 2026

Kaggle competition: [shl-hiring-assessment-2026](https://www.kaggle.com/competitions/shl-hiring-assessment-2026)

**Task.** Predict the `Grammar Score` of a speaker from a 45–60 s audio clip
(the speaker describes a picture). The score is an IELTS-style Likert rating
(rubric 1–5).

| | |
|---|---|
| Train clips | 769 (labelled) |
| Test clips | 216 |
| Evaluation | Pearson correlation + RMSE (lower RMSE is better) |
| **Train RMSE (5-fold OOF)** | **0.7608** |
| Train Pearson (5-fold OOF) | 0.7892 |

## Approach

Two feature families feeding a blended gradient-boosting model.

### 1. Prosody / acoustic features (164)

Extracted with `librosa` from each clip:

- MFCCs (20) + deltas, mean and std; mel bands (24) mean/std
- chroma, spectral contrast (7 bands), rolloff, flatness, centroid, bandwidth
- ZCR, RMS envelope percentiles, crest factor
- F0 via `yin` (mean/std/quartiles, voiced ratio)
- onset rate, inter-onset interval statistics, onset-strength stats, tempo
- silence ratio, pause runs, spectral-flux dynamics

### 2. Grammar features from speech content (21)

Transcripts come from a **fine-tuned `wav2vec2-base-960h` CTC model**, loaded
fully offline on Kaggle (weights mounted from a public Kaggle dataset;
`config.json` templated from a second one because the published checkpoint
ships without it). Per rules: **external models are allowed, external data is
not** — no external data is used anywhere.

From the transcripts we derive grammar/fluency proxies:

- word count, words-per-second, unique words, type-token ratio
- mean/max/std word length, long-word and medium-word ratios
- immediate repetition (1-gram and 2-gram), hesitation fillers
- vowel ratio, one-letter-word ratio
- ASR mean confidence, token entropy, blank rate, tokens/sec, empty flag

### 3. Model

- LightGBM (1500 trees, early stopping) + XGBoost (1200 trees)
- 5-fold KFold, blend weight selected on out-of-fold RMSE
- predictions clipped to `[0, 5]`
- Rounding to the 0.5 label grid was tested and **rejected** — it hurt OOF
  RMSE (0.7608 → 0.7730) and Pearson (0.7892 → 0.7813)

## Files

| Path | What it is |
|---|---|
| `notebook/shl_grammar_score.ipynb` | Submission notebook: documented pipeline + train RMSE + evaluation plots |
| `src/train_v3.py` | Same pipeline as a plain script (the notebook is generated from it) |
| `submission/submission.csv` | Predictions for the 216 test clips (`filename,label`) |

## Reproducing

The notebook runs on Kaggle with no internet access:

1. Competition data is pre-mounted at
   `/kaggle/input/competitions/shl-hiring-assessment-2026/Dataset_Final`
2. ASR weights are mounted from two public datasets:
   `andgoogle/wav2vec2-base-960h-260507-1028` (weights) and
   `warhero12/wav2vec2` (`config.json` template)
3. Run the notebook — it prints `TRAIN RMSE (5-fold OOF)` and writes
   `submission.csv`, `oof.csv`, `report.png`, `label_dist.png`,
   `feature_importance.csv`

Full pipeline (audio features + ASR + training) takes ≈2.5 h on 4 CPUs;
feature and transcript caches make re-runs incremental.

## Compliance

- No external data (per competition rules)
- External **model** (`wav2vec2-base-960h`) used, which the rules permit
- Single participant, single team
