# Klinik Chong Malay–Chinese Chatbot

Final Year Project framework for a text-based Malaysian clinic assistant. The
system combines Malay–Chinese–Pinyin language handling, intent classification,
grounded retrieval, SQLite appointment data, clinical information collection,
conversation state, emotion-aware responses and emergency-call support.

## Main components

- mBERT token-level language detection
- HMM Pinyin-to-Hanzi conversion
- XLM-R eight-class intent recognition
- ChromaDB retrieval for the clinic handbook
- SQLite retrieval for live doctors, schedules and appointments
- FastAPI backend with a browser-based user and developer interface
- Optional OpenAI generation and optional Twilio emergency-call demonstration

## Security and data policy

This repository contains no API keys, Twilio credentials, trained-model files,
ChromaDB index, SQLite clinic database or patient records. Each user must supply
their own services, model artefacts and authorised data. Do not commit `.env`.

## Setup

1. Clone the repository and create a virtual environment.
2. Install dependencies:

   ```bash
   pip install -r requirements_ver22.txt
   ```

3. Copy the example configuration and enter your own values:

   ```bash
   cp .env.example .env
   ```

4. Place the required model artefacts outside Git and configure their absolute
   paths with `MBERT_MODEL_PATH`, `PINYIN2HANZI_HMM_PATH` and
   `XLMR_INTENT_MODEL_PATH`. The runtime expects:

   - `mBert_language_detect_model/best_model`
   - `pinyin2hanzi_hmm_model/pinyin2hanzi_hmm_model.joblib`
   - `xlmr`

5. Provide a ChromaDB directory and an authorised SQLite database through
   `KLINIK_CHONG_CHROMA_PATH` and `KLINIK_CHONG_DB_PATH`.
6. Start the application:

   ```bash
   python model_server_ver22.py --host 127.0.0.1 --port 8000
   ```

7. Open `http://127.0.0.1:8000/`.

For interface-only testing without loading the trained language models, add
`--mock`. Mock mode is not a model-performance test.

## API keys

`OPENAI_API_KEY` is required for LLM-backed extraction, conversation-state and
emotion-aware generation. Twilio variables are optional and are used only when
the emergency-call demonstration is enabled. The application fails safely when
Twilio is not configured; ordinary chatbot functions remain available.

## Notebooks

The `notebooks/` directory contains output-cleared Colab notebooks for the RAG
pipeline and V22 launcher. They read secrets from Colab Secrets or environment
variables and do not contain real credentials.

## Important limitation

This is an academic prototype and a non-diagnostic support tool. It must not be
used as a replacement for professional medical advice or official emergency
services. Deployers are responsible for privacy, access control, consent,
security review and compliance with applicable healthcare and data-protection
requirements.
