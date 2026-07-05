# 🎧 AETHER — Complete Project Handoff Document
# For seamless continuation of development

---

## 🔑 CRITICAL CONTEXT: Read This First

You are continuing development of **Project Aether** — an Emotion-Aware Music Intelligence Platform. This document contains EVERYTHING about the project from its inception to its current state. Two phases (1A and 1B) are already complete and trained. Development continues from **Phase 1C** onwards.

**The developer is Arnav**, a student building this as a portfolio project for internship interviews. The project must be:
1. Technically excellent (production-quality code, SOTA models)
2. Real-world useful (not just academic — people should actually want to use it)
3. Interview-defensible (every technical choice must be justifiable)
4. Complete end-to-end (from ML models to deployed website)

---

## 📋 PROJECT OVERVIEW

### What is Aether?

Aether is an AI platform that deeply understands how you feel from your **voice and words** (in English and Hindi), and uses that understanding to curate highly precise music playlists that resonate with your emotional state.

### Two Core Features

#### 🎯 Main Feature: Emotion-Aware Playlist Curator
- User types or speaks a prompt describing their emotional state (English or Hindi)
- System analyzes the emotional nuance using multi-modal AI (text + voice emotion detection)
- Returns a highly precise, curated playlist matching that exact emotional state
- User can play the playlist directly on the website with seamless crossfade transitions
- User can export the playlist to Spotify or Apple Music

#### 🎶 Fun Feature: Live Emotion Music Player (Secondary, fully built — NOT a demo)
- Real-time emotion-based music playback with continuous adaptation
- User speaks or types while music is playing
- System detects emotional changes and smoothly transitions to matching songs
- Seamless crossfade transitions (Apple Music-style)
- A polished secondary experience — think of it like Instagram Stories vs Instagram Notes

### Why This Matters
- **Spotify's Daylist** (2024) and **Apple Music's Personal Station** prove market demand for mood-based music
- Aether's differentiator: FREE-FORM natural language input (not just listening history), voice emotion detection, Hindi support, and 15 nuanced emotions vs generic "happy/sad"
- No existing platform lets you *describe* a complex emotional state and get a precisely matching playlist

---

## 🏗️ ARCHITECTURE & PHASES

### Complete Build Order
```
Phase 1A: Text Emotion Model               ✅ COMPLETE
Phase 1B: Voice Emotion Model (EN + HI)    ✅ COMPLETE  
Phase 1C: Unified Fusion Layer             ← RESUME HERE
Phase 2:  Music Dataset Processing
Phase 3:  Emotion → Music Mapping  
Phase 4:  Recommendation & Playlist Engine (+ Spotify/Apple Music API)
Phase 5:  RAG Layer (Explainable Recommendations)
Phase 6:  Agentic AI Layer (Autonomous Curation)
Phase 7:  Seamless Transitions & Crossfade System
Phase 8:  Full-Stack Website (FastAPI + Next.js)
```

---

## ✅ PHASE 1A: Text Emotion Model — COMPLETE

### What It Does
Takes input text → outputs a probability distribution across 15 Aether emotions.

### Technical Details
- **Model**: Fine-tuned `distilroberta-base` from HuggingFace
- **Datasets**: 
  - 20-Emotion Text Classification Dataset (2025) — 79,595 sentences, 20 emotions
  - GoEmotions — additional training data
  - Combined and mapped to 15 Aether categories
- **Training**: Google Colab T4 GPU, ~1726 seconds
- **Output**: 15-dimensional emotion probability vector with intensity

### Files
- **Notebook**: `phase_1a_text_emotion/Aether_Phase_1A.ipynb` (6 cells, all executed successfully)
- **Trained Model**: Saved to Google Drive at `Aether_models/text_emotion/`
- **Inference Utility**: `phase_1a_text_emotion/inference.py`

### Results
- Successfully trained and evaluated
- Saved to Google Drive permanently

---

## ✅ PHASE 1B: Voice Emotion Model — COMPLETE

### What It Does
Takes input audio → outputs:
1. Acoustic emotion probabilities (15 Aether emotions) from voice tone/prosody
2. Transcribed text via Whisper (for dual-signal processing)

### Technical Details
- **Feature Extractor**: emotion2vec+ large (`iic/emotion2vec_plus_large`) — SOTA 2024/2025 foundation model for Speech Emotion Recognition, used FROZEN (no fine-tuning of the large model)
- **Classification Head**: MLP (1024 → 256 → 128 → 15) with BatchNorm, Dropout(0.3), ReLU
- **Parameters**: 299,791 (all trainable — only the MLP head trains)
- **Speech-to-Text**: OpenAI Whisper (supports English + Hindi natively)

### Datasets
- **RAVDESS** (`xbgoose/ravdess` on HuggingFace) — 1,440 speech samples
  - 8 emotions: neutral, calm, happy, sad, angry, fearful, disgust, surprised
  - String labels in `emotion` column
- **CREMA-D** (`AbstractTTS/CREMA-D` on HuggingFace) — 7,442 speech samples
  - 6 emotions: anger, disgust, fear, happy, sad, neutral
  - String labels in `major_emotion` column
- **Combined**: 8,882 total samples, 0 skipped

### Label Mapping (Source → Aether)
```
RAVDESS:  neutral→calm, calm→calm, happy→happy, sad→sad, angry→angry,
          fearful→anxious, disgust→frustrated, surprised→dreamy

CREMA-D:  anger→angry, disgust→frustrated, fear→anxious,
          happy→happy, sad→sad, neutral→calm
```

### Training Results
- **Accuracy**: 87.17%
- **F1 (weighted)**: 0.8714
- **F1 (macro)**: 0.8794
- **Per-emotion F1**: happy 0.90, angry 0.93, calm 0.91, frustrated 0.86, dreamy 0.93
- **Training**: 16 epochs (early stopped at epoch 16, best at epoch 9), 52 seconds
- **Train/Val split**: 7,549 train / 1,333 validation (85/15, stratified)

### Label Distribution in Training Data
```
happy:      1,463 (16.5%)
sad:        1,463 (16.5%)  
angry:      1,463 (16.5%)
calm:       1,375 (15.5%)
anxious:    1,463 (16.5%)
frustrated: 1,463 (16.5%)
dreamy:       192 ( 2.2%)
(other 8 Aether emotions: 0 samples — covered by text model only)
```
Class weights applied: dreamy got 6.616x weight to handle imbalance.

### Files
- **Training Script**: `phase_1b_voice_emotion/train_voice_emotion.py` (1,168 lines, 15 sections)
- **Notebook**: `phase_1b_voice_emotion/Aether_phase_1B.ipynb` (6 cells, all executed)
- **Trained Model**: Google Drive at `Aether_models/voice_emotion/final/voice_emotion_head.pt`
- **Config**: Google Drive at `Aether_models/voice_emotion/model_config.json`
- **Inference Utility**: Google Drive at `Aether_models/voice_inference.py`

### Important Technical Notes for Phase 1B
- emotion2vec+ large outputs **1024-dim** embeddings (NOT 768 — the base model is 768, large is 1024)
- `torchcodec` must be installed in Colab for HuggingFace `datasets` v5.0 audio decoding
- The `AbstractTTS/CREMA-D` dataset requires the `major_emotion` column (not `label` or `emotion`)
- PyTorch 2.2+ deprecated `verbose=True` in `ReduceLROnPlateau` — don't use it
- All label mapping uses a unified lowercase map to prevent case-sensitivity bugs

---

## ❌ PHASE 1C: Unified Fusion Layer — NOT YET BUILT (START HERE)

### What It Should Do
Take emotion signals from text (Phase 1A) and voice (Phase 1B) and fuse them into one unified 15-emotion score.

### Planned Design
- Dynamic weighting based on which inputs are active:
  - Text + Voice: `0.60 × text + 0.40 × voice`
  - Text only: `1.0 × text`
  - Voice only: `0.5 × voice_acoustic + 0.5 × voice_transcription_text`
- The fusion layer adapts to whatever is available
- Weights are tunable
- Output: One unified 15-emotion probability distribution

### Files Location
- `phase_1c_fusion/` (currently empty)

---

## 📂 REMAINING PHASES (NOT YET BUILT)

### Phase 2: Music Dataset Processing
- Primary: Almost Million Songs Dataset 2025 (Kaggle, ~1M Spotify tracks)
- Supplementary: MTG-Jamendo (55K+ tracks with mood tags), Spotify Global 2009-2025
- Features: tempo, energy, valence, danceability, acousticness, key, mode, etc.
- Location: `phase_2_music_data/` (empty)

### Phase 3: Emotion → Music Mapping
- Map 15 emotions to target musical characteristics
- Rule-based first, then learnable neural mapping
- Location: `phase_3_emotion_music_mapping/` (empty)

### Phase 4: Recommendation & Playlist Engine
- Cosine similarity core matching
- Intelligent playlist sequencing (flow, energy arc, variety)
- Spotify/Apple Music API integration (metadata, export)
- Location: `phase_4_recommendation/` (empty)

### Phase 5: RAG Layer
- Vector database (ChromaDB/Pinecone) of songs with lyric embeddings, mood tags
- Natural language queries
- Explainable recommendations
- Location: `phase_5_rag/` (empty)

### Phase 6: Agentic AI Layer
- LangChain/LangGraph agent
- Perceive → Plan → Use Tools → Reflect loop
- Autonomous playlist arc planning
- Location: `phase_6_agentic_ai/` (empty)

### Phase 7: Seamless Transitions & Crossfade
- Applies to BOTH features (playlist playback + live emotion player)
- BPM-matched transitions, key-compatible song selection
- 3-5 second crossfade, Apple Music-style
- Location: `phase_7_drift_crossfade/` (empty)

### Phase 8: Full-Stack Website
- Backend: FastAPI (Python)
- Frontend: Next.js, TailwindCSS, Framer Motion, Three.js
- Design: Apple.com-inspired, premium, dark mode
- Deployment: Vercel (frontend) + Render/Railway (backend)
- Location: `phase_8_website/` (empty)

---

## 🎨 15 AETHER EMOTION CATEGORIES

```python
AETHER_EMOTIONS = [
    "happy",        # 0  - joyful, excited, grateful — upbeat pop, feel-good
    "sad",          # 1  - grieving, disappointed — slow ballads, minor key
    "angry",        # 2  - rage, hatred, outrage — heavy rock, aggressive beats
    "calm",         # 3  - peaceful, serene, relaxed — ambient, lo-fi, soft
    "anxious",      # 4  - worried, nervous, panicked — tense, building, unsettling
    "energetic",    # 5  - pumped, thrilled, wild — EDM, dance, high tempo
    "focused",      # 6  - concentrated, determined — lo-fi beats, minimal, study
    "nostalgic",    # 7  - remembering, bittersweet — retro, acoustic, warm
    "romantic",     # 8  - loving, tender, intimate — love songs, R&B, slow jams
    "melancholic",  # 9  - deep sorrow, existential — dark, layered, minor key depth
    "confident",    # 10 - empowered, bold, proud — powerful, bass-heavy, anthems
    "hopeful",      # 11 - optimistic, looking forward — uplifting, building, major key
    "frustrated",   # 12 - irritated, stuck, blocked — hard rock, punk, dissonant
    "lonely",       # 13 - isolated, missing, empty — minimal, sparse, echo-heavy
    "dreamy",       # 14 - ethereal, imaginative — synth, atmospheric, ambient
]
```

---

## 📁 PROJECT FOLDER STRUCTURE

```
Aether/
├── config.py                          # Central configuration (emotions, paths, constants)
├── implementation_plan.md             # Complete build plan (updated with new pivot)
├── requirements.txt                   # Python dependencies
├── README.md                          # Project readme
├── Aether_Project_Overview.docx       # Original project overview doc
│
├── phase_1a_text_emotion/             # ✅ COMPLETE
│   └── Aether_Phase_1A.ipynb          # Executed Colab notebook
│
├── phase_1b_voice_emotion/            # ✅ COMPLETE
│   ├── train_voice_emotion.py         # Full training script (1,168 lines)
│   └── Aether_phase_1B.ipynb          # Executed Colab notebook
│
├── phase_1c_fusion/                   # ← NEXT: Fusion layer
├── phase_2_music_data/                # Music dataset processing
├── phase_3_emotion_music_mapping/     # Emotion → music feature mapping
├── phase_4_recommendation/            # Playlist engine + Spotify/Apple API
├── phase_5_rag/                       # RAG for explainable recommendations
├── phase_6_agentic_ai/                # Agentic AI layer
├── phase_7_drift_crossfade/           # Seamless transitions system
├── phase_8_website/                   # Full-stack website (FastAPI + Next.js)
│
├── models/                            # Local model storage
│   ├── text_emotion/
│   ├── voice_emotion/
│   └── fusion/
│
└── data/                              # Dataset storage
    └── processed/
```

**Google Drive Model Storage:**
```
Google Drive/
└── Aether_models/
    ├── text_emotion/                  # Phase 1A trained model
    ├── voice_emotion/
    │   ├── final/
    │   │   └── voice_emotion_head.pt  # Phase 1B trained model weights
    │   └── model_config.json          # Phase 1B model config
    └── voice_inference.py             # Phase 1B inference utility
```

---

## 🛠️ TECH STACK

| Layer | Technology |
|-------|-----------|
| ML / Models | Python, PyTorch, HuggingFace Transformers, emotion2vec+ (FunASR), Whisper, Librosa, Scikit-learn |
| RAG | ChromaDB or Pinecone, LangChain, OpenAI/HuggingFace embeddings |
| Agentic AI | LangChain / LangGraph |
| Backend | FastAPI (Python) |
| Frontend | Next.js, TailwindCSS, Framer Motion, Three.js, Web Audio API |
| APIs | Spotify Web API, Apple Music API |
| Database | MongoDB or Supabase |
| Deployment | Vercel (frontend), Render/Railway (backend) |
| Training | Google Colab (T4 GPU) |

---

## 📐 CODE QUALITY STANDARDS

The following standards have been established in Phase 1A and 1B and MUST be maintained:

1. **Defensive coding**: Always handle edge cases, missing data, API failures gracefully
2. **Comprehensive error handling**: try/except with proper cleanup (especially temp files)
3. **Clear section-based structure**: Each phase has numbered sections with descriptive headers
4. **Detailed docstrings**: Every function has a docstring explaining what it does, inputs, outputs
5. **Progress logging**: Print statements showing progress, percentages, checkmarks
6. **Case-insensitive label handling**: All string comparisons use `.lower().strip()`
7. **Type safety**: Explicit type checking before operations (isinstance checks)
8. **Clean imports**: Organized at top of file or at section level for Colab compatibility
9. **Model saving**: Always save model config JSON alongside weights for reproducibility
10. **Google Drive backup**: Always include a Drive save section for Colab notebooks

---

## ⚠️ KNOWN ISSUES & GOTCHAS

1. **emotion2vec+ large outputs 1024-dim embeddings**, NOT 768 (the base model). Always use 1024.
2. **`torchcodec`** is required by `datasets` v5.0+ for audio decoding. Include in pip install.
3. **`AbstractTTS/CREMA-D`** uses `major_emotion` column, not `label` or `emotion`.
4. **`MahiA/CREMA-D`** has NO audio data — only text paths. Do not use.
5. **`mu-llama/CREMA-D`** was deleted from HuggingFace. Do not reference.
6. **PyTorch 2.2+** deprecated `verbose=True` in `ReduceLROnPlateau`.
7. **Colab free tier** disconnects after ~1.5 hours idle. Always save to Drive immediately after training.
8. **8 of 15 Aether emotions** (energetic, focused, nostalgic, romantic, melancholic, confident, hopeful, lonely) have NO voice training data — only text model detects them. The fusion layer must handle this gracefully.

---

## 🎯 IMMEDIATE NEXT STEP

**Build Phase 1C: Unified Emotion Fusion Layer**
- Combines Phase 1A (text) and Phase 1B (voice) outputs
- Dynamic weighting based on available inputs
- Location: `phase_1c_fusion/`
- Should follow the same code quality standards as Phase 1A and 1B
