---
title: Khaansi AI
emoji: 🩺
colorFrom: green
colorTo: gray
sdk: streamlit
sdk_version: 1.63.0
app_file: app.py
pinned: false
---

# Khaansi AI 🩺

**Urdu voice-based TB screening assistant** — built for the Alibaba Cloud AI Hackathon Pakistan 2026.

Khaansi (کھانسی = "cough") combines cough-audio analysis with a conversational symptom interview, delivered entirely by voice in Pakistani Urdu, to produce a TB risk score and a plain-language explanation with next-step guidance.

> ⚠️ **Screening aid only — not a medical diagnosis.** This is a hackathon prototype.

---

## Why

Pakistan is among the world's highest TB-burden countries, and a large fraction of cases go undiagnosed. Barriers include limited clinic access and low awareness of symptoms. A voice assistant that anyone can talk to in their own language — describe their cough, answer a few spoken questions, and get clear guidance — lowers the barrier to early screening.

## How it works

```
              cough audio                       spoken Urdu answers
                  │                                    │
                  ▼                                    ▼
        ┌──────────────────┐                 ┌──────────────────┐
        │  HeAR embedding  │                 │  Whisper (STT)   │
        │  google/hear     │                 │  faster-whisper  │
        └────────┬─────────┘                 └────────┬─────────┘
                 │                                     │
                 │                            ┌────────▼─────────┐
                 │                            │  Qwen (DashScope)│
                 │                            │  symptom Q&A in  │
                 │                            │  Urdu, parses    │
                 │                            │  answers         │
                 │                            └────────┬─────────┘
                 │                                     │
        ┌────────▼─────────────────────────────────────▼────────┐
        │   fused vector: HeAR embedding + 16 symptom fields    │
        │   unknown fields → NaN → learned imputer (no fake     │
        │   defaults)                                           │
        └────────┬──────────────────────────────────────────────┘
                 ▼
        ┌──────────────────┐        ┌──────────────────┐
        │  TB classifier   │───────▶│  Qwen explains   │
        │  risk score      │        │  the result in   │
        │  (AUC 0.825)     │        │  simple Urdu     │
        └──────────────────┘        └────────┬─────────┘
                                             ▼
                                   ┌──────────────────┐
                                   │  Edge-TTS speaks │
                                   │  it back (ur-PK  │
                                   │  voices)         │
                                   └──────────────────┘
```

### Why a hybrid voice stack

Alibaba's Qwen real-time voice models (ASR/TTS) do **not** officially support Urdu (confirmed in their documentation). We therefore keep Qwen for what it does excel at — text reasoning and dialogue — and pair it with Urdu-capable components:

| Stage | Component | Why |
|---|---|---|
| STT (speech → text) | OpenAI Whisper (`faster-whisper`) | strong Urdu transcription |
| Reasoning (text) | **Qwen** via DashScope | excellent Urdu dialogue + JSON extraction |
| TTS (text → speech) | Edge-TTS | free, natural Pakistani Urdu voices (`ur-PK-AsadNeural`, `ur-PK-UzmaNeural`) |

### Model

- Cough audio → **HeAR** embeddings (Google's `google/hear` health-acoustic model, 16 kHz, 2 s clips).
- A 16-field symptom vector (demographics, TB history, classic TB symptoms) is fused with the embedding and fed to a trained classifier.
- Decision threshold **0.317**, tuned for **80% sensitivity** — appropriate for a screening tool where missed cases cost more than false alarms.
- Fields a patient doesn't know (height, weight, heart rate, temperature) are passed as NaN and filled by a **learned imputer** — never hardcoded defaults.

## Validation

- Held-out test set: **AUC 0.825**.
- Spot-checked against real labeled clips from the CODA-TB DREAM Challenge test set (not redistributed in this repo):
  - True negative: predicted 9.1% (correctly below threshold)
  - True positives: predicted 27.2%, 41%, 33% (2 of 3 above threshold — consistent with the 80%-sensitivity tuning, not a bug)
- Whisper Urdu round-trip: Edge-TTS-spoken Urdu sentences transcribed back with near-verbatim accuracy (see `test_whisper.py`).

## Setup

Requirements: Python 3.12, [ffmpeg](https://ffmpeg.org/download.html) on PATH, a HuggingFace token, and a DashScope (Alibaba Cloud) API key for voice mode.

```bash
git clone <repo-url>
cd khaansi-ai
python -m venv venv
venv\Scripts\activate          # Windows
pip install -r requirements.txt

copy .env.example .env         # add HF_TOKEN (required) and DASHSCOPE_API_KEY (for voice mode)
streamlit run app.py
```

- `HF_TOKEN` is required for both modes because the cough classifier downloads the gated `google/hear` model from HuggingFace.
- `DASHSCOPE_API_KEY` is only required for **Conversational voice (Urdu)** mode; the manual form works without it once `HF_TOKEN` is set.

Note: `scikit-learn` is pinned to `1.6.1` to match the training environment — other versions fail to unpickle the classifier.

## Usage

Two modes, selectable at the top of the app:

1. **🎙️ Conversational voice (Urdu)** — the full experience. Record a cough, then answer spoken questions in Urdu. The assistant acknowledges each answer, asks follow-ups, and finally speaks the risk explanation. Multi-answer turns are handled (answering several questions in one breath skips ahead), unclear answers are re-asked once.
2. **📝 Manual form (English)** — fallback/clinic mode: upload or record a cough, fill the symptom form, get the score.

## Repo layout

```
app.py                  Streamlit app (both modes)
voice_agent.py          Whisper STT + Qwen + Edge-TTS helpers, Urdu dialogue strings
tb_classifier.joblib    trained fused classifier
tb_symptom_imputer.joblib, tb_threshold.joblib, tb_symptom_cols.joblib
sanitycheck.py          CLI pipeline sanity check
test_edge_tts.py        standalone Edge-TTS Urdu voice test
test_whisper.py         standalone Whisper Urdu round-trip test
test_qwen.py            standalone Qwen symptom-parsing + explanation test
.env.example            API key template
```

## Data & privacy

- Trained on the [CODA-TB DREAM Challenge](https://www.synapse.org/coda_tb) dataset. Per dataset access terms, **no patient audio is redistributed in this repo**.
- The repo contains no real patient data, no API keys (`.env` is gitignored).
- Any demo audio included is self-recorded and clearly labeled as non-clinical.
