# SHL Grammar Score — Research Intern Hiring Assessment 2026

Kaggle competition: [shl-hiring-assessment-2026](https://www.kaggle.com/competitions/shl-hiring-assessment-2026)

**Task.** Predict the `Grammar Score` of a speaker from a 45–60 s audio clip
(the speaker describes a picture). The score is an IELTS-style Likert rating
(rubric 1–5, labels present 0–5).

| | |
|---|---|
| Train clips | 769 (labelled) |
| Test clips | 216 |
| Evaluation | Pearson correlation + RMSE (lower RMSE better) |
| **Train RMSE (5-fold OOF)** | **0.5090** |
| **Public leaderboard RMSE (best)** | **0.3729** (rank 92/401, 2026-10-10) |
| Leader | 0.3064 |

## Approach

Three feature families → XGBoost + CatBoost, blended on OOF RMSE.

### 1. Whisper-large-v3 encoder embeddings (2560-d)

Mean + std pooling of the whisper-large-v3 encoder hidden states over 30 s
windows (hop 25 s), extracted once on a free T4 (Colab) for all 985 clips —
audio content in a dense, model-learned form. This was the single biggest
lever (≈ −0.028 RMSE vs prosody-only).

### 2. Prosody / acoustics (164 librosa descriptors)

MFCCs + deltas, mel bands, chroma, spectral contrast/rolloff/flatness/
centroid/bandwidth, ZCR, RMS percentiles, crest factor, F0 (yin), onset
statistics, tempo, silence & pause structure.

### 3. Grammar features from speech content

- **Transcripts** from `whisper-large-v3` (external model; no external
  data) → word count/rate, type-token ratio, repetition, fillers, word
  length stats, ASR confidence proxies.
- **Gemini rubric grades** — an LLM scores each clip against the IELTS-style
  rubric (1–5). Two streams, merged into one feature:
  - per-clip grades (higher quality, full 985-clip coverage)
  - batch grades (full coverage) as fallback
  This LLM-judge feature was the second biggest lever (≈ −0.016).

### 4. Model

- XGBoost + CatBoost, 5-fold KFold, predictions clipped [0, 5]
- Block blend (raw embeddings + PCA-32 blocks) weights fitted on OOF RMSE
- OOF→LB transfer checked against the public leaderboard every round
  (OOF ranking proved unreliable on the 216-row test set — leaderboard
  feedback drove all major decisions)

## Files

| Path | What it is |
|---|---|
| `src/submit_v7.py` | Best recipe (embeddings + prosody + grades, 2-block blend) |
| `src/local_train.py` | Feature loading: overlays whisper transcripts, merges Gemini grades |
| `src/llm_grade.py` | Gemini per-clip rubric grading (resumable, multi-model fallback) |
| `src/llm_grade_wsp.py` | Whisper-transcript wrapper around the grader |
| `src/llm_grade_perclip.py` | Premium per-clip grading pass |
| `src/llm_grade_audio.py` | Audio-native grading (prosody/disfluency from raw WAV) |
| `src/build_hybrid.py` | Merges premium + batch grade streams |
| `src/add_ppl.py` | GPT-2 perplexity text features (tried; not in best model) |
| `src/train_v3.py` | Legacy baseline (LightGBM + wav2vec2 CTC, OOF 0.7608) — history |
| `notebook/shl_grammar_score.ipynb` | **Final documented pipeline** — run top-to-bottom to reproduce `submission.csv` and the OOF score from cached artifacts |
| `artifacts/` | Cached artifacts (whisper embeddings, prosody cache, transcripts, LLM grades) so the notebook runs offline |
| `submission/submission.csv` | Best public-LB predictions (0.3729) |

## What did NOT help (kept for the record)

- wav2vec2-large embeddings — 5% blend weight, hurt LB (+0.010)
- GPT-2 perplexity features — negatively correlated with label, hurt LB
- Post-hoc OOF distribution stretching — helped OOF twice, hurt LB every time

## Compliance

- No external data (per competition rules)
- External **models** used (whisper-large-v3, Gemini), which the rules permit
- Single participant, single team
