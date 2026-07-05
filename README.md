# 🎧 Aether — Emotion-Aware Music Intelligence

> **Real-Time Multimodal Emotion Detection → Emotion-Aware Playlist Curator & Live Music Player**

Aether is an AI platform that deeply understands how you feel from your **voice and words** (in English and Hindi), and uses that understanding to curate highly precise music playlists that resonate with your emotional state. Built with RAG for explainable recommendations and Agentic AI for autonomous curation.

## 🎯 Core Features

1. **Emotion-Aware Playlist Curator (Main)**: Type or speak a prompt describing your complex emotional state (e.g., "nostalgic but hopeful"). Aether returns a precisely curated playlist matching that exact nuance.
2. **Live Emotion Music Player (Fun/Secondary)**: Real-time emotion-based music playback. The system tracks emotional drift while you type/speak and smoothly transitions to matching songs with Apple Music-style crossfades.

## 🏗 Project Structure & Build Phases

```
Aether/
├── config.py                          # Central configuration (emotions, paths, constants)
├── implementation_plan.md             # Complete build plan
├── Aether_Complete_Handoff.md         # Comprehensive Claude handoff document
├── requirements.txt                   # Python dependencies
│
├── phase_1a_text_emotion/             # ✅ COMPLETE (Trained & saved)
├── phase_1b_voice_emotion/            # ✅ COMPLETE (Trained & saved)
├── phase_1c_fusion/                   # Unified Emotion Fusion Layer (Text + Voice)
├── phase_2_music_data/                # Music dataset processing
├── phase_3_emotion_music_mapping/     # Emotion → music feature mapping
├── phase_4_recommendation/            # Playlist engine + Spotify/Apple API
├── phase_5_rag/                       # RAG for explainable recommendations
├── phase_6_agentic_ai/                # Agentic AI layer
├── phase_7_drift_crossfade/           # Seamless transitions system
├── phase_8_website/                   # Full-stack website (FastAPI + Next.js)
```

## 🧠 Core Emotions (15 Categories)

Aether classifies into 15 highly distinct emotions, each with its own musical profile:
`happy`, `sad`, `angry`, `calm`, `anxious`, `energetic`, `focused`, `nostalgic`, `romantic`, `melancholic`, `confident`, `hopeful`, `frustrated`, `lonely`, `dreamy`

## 📊 Datasets

| Component | Dataset | Size |
|-----------|---------|------|
| Text Emotion | 20-Emotion (2025) + GoEmotions | ~80k+ samples |
| Voice Emotion | RAVDESS + CREMA-D | 8,882 samples |
| Music Library | Almost Million Songs 2025 + MTG-Jamendo | ~1M tracks |

## 🛠 Tech Stack

- **ML**: PyTorch, HuggingFace Transformers, emotion2vec+, Whisper, Librosa
- **RAG**: ChromaDB/Pinecone, LangChain, Embeddings
- **Agentic AI**: LangChain / LangGraph
- **Backend**: FastAPI (Python)
- **Frontend**: Next.js, TailwindCSS, Framer Motion, Three.js, Web Audio API
- **APIs**: Spotify Web API, Apple Music API
