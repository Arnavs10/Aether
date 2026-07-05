# 🎧 Project Aether — Complete Project Overview & Build Plan

## What is Aether?

Aether is an **Emotion-Aware Music Intelligence Platform** — an AI system that deeply understands how you feel from your voice and words (in English and Hindi), and uses that understanding to curate highly precise music playlists that truly resonate with your emotional state.

### The Core Idea

Most music platforms (Spotify, Apple Music) recommend songs based on **what you listened to before** (collaborative filtering + listening history). Aether is fundamentally different — it lets you **express your complex emotional state in natural language** (text or voice, English or Hindi), and returns a perfectly curated playlist that captures exactly what you're feeling.

> **Example**: A user says *"I'm feeling nostalgic about college but also hopeful about what's coming next"* — and Aether returns a playlist that blends bittersweet nostalgia with uplifting, forward-looking tracks. Not a generic "happy" playlist. Not a generic "sad" playlist. A nuanced, emotionally intelligent selection.

### The Two Core Features

#### 🎯 Main Feature: Emotion-Aware Playlist Curator
- User types or speaks a prompt describing how they feel (English or Hindi)
- System analyzes the emotional nuance using multi-modal AI (text + voice emotion detection)
- Returns a highly precise, curated playlist matching that exact emotional state
- User can play the playlist directly on the website with seamless transitions
- User can export the playlist to Spotify or Apple Music

#### 🎶 Fun Feature: Live Emotion Music Player
- Real-time emotion-based music playback with continuous adaptation
- User speaks or types while music is playing
- System detects emotional changes in real-time and smoothly transitions to matching songs
- Seamless crossfade transitions (Apple Music-style, no hard cuts)
- A "chill mode" / interactive experience — fully built, not a demo

### The Emotional Intelligence Angle

This is what makes Aether genuinely special and research-level:

- **Multimodal Emotion Detection**: Aether doesn't just read keywords. It understands emotion from text *and* voice simultaneously. It fuses both signals into a unified emotional understanding — like how a human friend reads your tone *and* words to know how you really feel.

- **Nuanced Emotion Understanding**: 15 distinct emotion categories with intensity levels, not just "happy/sad/angry." Aether can distinguish between "melancholic" and "sad," between "confident" and "energetic."

- **Natural Language Prompts**: Users describe their feelings naturally — complex, mixed, contradictory emotions — and the system understands. In both English and Hindi.

- **Explainable Recommendations**: Using RAG, Aether can explain *why* it chose each song — not a black box.

- **Active Curation via AI Agent**: An Agentic AI layer actively plans your musical journey — perceiving your mood, strategizing a playlist arc, using tools to search and filter, and reflecting on whether its choices are working.

### Why It Matters (Real-World Value)

- **Spotify's Daylist & Apple Music's Personal Station** prove the market demand for mood-based music. Aether goes further with free-form natural language input and voice emotion detection.
- **Mental Health**: Music therapy is a growing field. Emotionally intelligent music curation has measurable benefits for anxiety, focus, and mood regulation.
- **The Gap**: No existing platform lets you *describe* a complex emotional state and get a perfectly matching playlist. Users currently have to manually search, browse mood playlists that are too generic, or rely on algorithms that don't understand *how they feel right now*.
- **Bilingual**: Hindi support makes this relevant for 600M+ Hindi speakers globally.

### How It's Different From Similar Projects

| Aspect | Typical Projects | Aether |
|--------|-----------------|--------|
| Input | Listening history | Natural language prompt (text + voice) |
| Language | English only | English + Hindi |
| Emotion depth | Basic mood (happy/sad) | 15 nuanced emotions with intensity |
| Operation | One-time recommendation | Smart curation + live adaptation |
| Music response | Generic mood playlists | Precisely curated, emotionally matched |
| Intelligence | Cosine similarity only | RAG + Agentic AI + ML models |
| Explainability | None ("you might like this") | Natural language explanations |
| Playback | External player | Built-in player with seamless crossfade |
| Export | None | Spotify + Apple Music integration |
| Interface | Basic dashboard | Premium, Apple-inspired experience |

---

## The Technology Domains Aether Touches

- **NLP** (Natural Language Processing) — text emotion classification
- **Speech Emotion Recognition** — voice-based affect detection using emotion2vec+
- **Affective Computing** — emotion modeling, intensity levels, drift detection
- **Recommender Systems** — cosine similarity, embedding-based matching
- **RAG** (Retrieval-Augmented Generation) — vector databases, explainable retrieval
- **Agentic AI** — autonomous agents with tool use, planning, and memory
- **Full-Stack Engineering** — FastAPI + Next.js + audio processing
- **API Integration** — Spotify API, Apple Music API

---

## Build Phases — Complete Roadmap

### Phase 1: Emotion Detection Models (The Brain)

Everything downstream depends on this being accurate.

#### Phase 1A — Text Emotion Model ✅ COMPLETE

- **Datasets Used**:
  - **20-Emotion Text Classification Dataset (2025)** — 79,595 sentences, 20 distinct emotions (HuggingFace, MIT License)
  - **GoEmotions** — Additional training data
  - Combined and mapped to 15 Aether emotion categories
- **Model**: Fine-tuned `distilroberta-base` from HuggingFace
- **Architecture**: DistilRoBERTa → 15-class classification head
- **Results**: Successfully trained and saved to Google Drive
- **Output**: Input text → 15 emotion probability distribution
- **Files**: 
  - `phase_1a_text_emotion/Aether_Phase_1A.ipynb` (executed Colab notebook)
  - Model saved to Google Drive: `Aether_models/text_emotion/`

#### Phase 1B — Voice Emotion Model (Bilingual: English + Hindi) ✅ COMPLETE

- **Feature Extractor**: **emotion2vec+ large** (iic/emotion2vec_plus_large) — SOTA 2024/2025 foundation model for Speech Emotion Recognition, used as frozen feature extractor
- **Datasets Used**:
  - **RAVDESS** (xbgoose/ravdess) — 1,440 speech samples, 8 emotions
  - **CREMA-D** (AbstractTTS/CREMA-D) — 7,442 speech samples, 6 emotions
  - Combined: 8,882 samples → mapped to 15 Aether categories
- **Classification Head**: MLP (1024 → 256 → 128 → 15) with BatchNorm, Dropout, ReLU
- **Speech-to-Text**: OpenAI Whisper (supports English + Hindi) for dual-signal output
- **Results**:
  - Accuracy: 87.17%
  - F1 (weighted): 0.8714
  - F1 (macro): 0.8794
  - Per-emotion F1: happy 0.90, angry 0.93, calm 0.91, dreamy 0.93
- **Output**: Input audio → [Acoustic Emotion Probabilities] + [Transcribed Text]
- **Files**:
  - `phase_1b_voice_emotion/train_voice_emotion.py` (complete training script, 1,168 lines)
  - `phase_1b_voice_emotion/Aether_phase_1B.ipynb` (executed Colab notebook)
  - Model saved to Google Drive: `Aether_models/voice_emotion/`

#### Phase 1C — Unified Emotion Fusion Layer (Text + Voice) — NOT YET BUILT

This is the brain's brain. It takes emotion signals from **all available inputs** and fuses them into one unified emotion score.

- **Dynamic weighting based on which inputs are active**:

| Active Inputs | Weights |
|---------------|--------|
| Text + Voice | `0.60 × text + 0.40 × voice` |
| Text only | `1.0 × text` |
| Voice only | `0.5 × voice_acoustic + 0.5 × voice_transcription_text` |

- **Why dynamic?** Because on the unified interface, the user might be:
  - Speaking → voice (acoustic + transcribed text)
  - Typing → text only
  - Speaking AND typing → both
- The fusion layer adapts to whatever is available — no modality is required
- Weights are tunable and can be learned over time
- **Output**: One unified 15-emotion score that drives everything downstream

---

### Phase 2: Music Dataset Processing

> [!TIP]
> Using multiple complementary music datasets for maximum coverage.

- **Primary Dataset**: **Almost Million Songs Dataset 2025** (Kaggle) — ~1 million Spotify tracks with 16 key attributes: tempo, energy, valence, danceability, acousticness, key, mode, popularity, speechiness, instrumentalness, liveness, loudness, and more
- **Supplementary Datasets**:
  - **MTG-Jamendo** — 55,000+ full-length tracks with 195 tags (genre, instrument, **mood/theme**). The mood tags are especially valuable for emotion-music mapping
  - **Spotify Global Music Dataset (2009–2025)** — longitudinal dataset covering 15+ years of music trends
  - **Spotify Wrapped 2025 / Top Songs** — latest trending tracks for freshness
- **Features**: tempo, energy, valence, danceability, loudness, key, mode, acousticness, instrumentalness, speechiness, liveness, popularity
- **Process**: Load in pandas → Clean → Merge across datasets → Normalize features → Select relevant features
- **Output**: A comprehensive, feature-rich music database ready for matching

---

### Phase 3: Emotion → Music Mapping

- Map each of the 15 emotions to target musical characteristics
- Example: `Calm → {tempo: 80, energy: 0.3, valence: 0.5}`
- Version 1: Rule-based mapping dictionary
- Version 2: Small neural network (emotion embedding → music feature vector)
- **Output**: Any detected emotion → target music feature profile

---

### Phase 4: Recommendation & Playlist Engine

- **Algorithm**: Cosine Similarity for core matching
- Convert emotion targets to feature vectors
- Compare against music dataset
- Return top-N closest matching songs as a curated playlist
- **Playlist logic**: Not just top-N random matches, but intelligent sequencing (smooth flow, energy arc, genre variety)
- **Spotify/Apple Music API Integration**:
  - Fetch song metadata, album art, preview URLs
  - Enable playlist export to user's Spotify/Apple Music account
- **Output**: Emotion prompt → ranked, sequenced playlist of best-matching songs

---

### Phase 5: RAG Layer (Retrieval-Augmented Generation)

- **What it adds**: Intelligence and explainability to recommendations
- **Build a vector database** (ChromaDB or Pinecone) of songs with:
  - Lyric embeddings
  - Mood tags
  - Audio feature metadata
- **Enable natural language queries**: "find me something like this but more calm"
- **Generate explanations**: "This song was chosen because its harmonic progression matches tracks that resonate with reflective moods"
- **Key concepts**: Embeddings, vector similarity search, retrieval pipelines, prompt engineering

---

### Phase 6: Agentic AI Layer

- **What it adds**: Proactive, autonomous music curation
- **The agent** (built with LangChain/LangGraph):
  - **Perceives** → reads current emotion state from user prompt
  - **Plans** → strategizes a playlist arc (e.g., "user is feeling mixed nostalgia + hope, plan a journey from bittersweet to uplifting")
  - **Uses tools** → search songs, fetch lyrics, adjust BPM filters, explain choices
  - **Reflects** → evaluates if the recommendation worked, learns
- **Key concepts**: Agent loops, tool use, memory, planning, ReAct pattern

---

### Phase 7: Seamless Transitions & Crossfade System

- **For Main Feature (Playlist Curator)**: When playing the curated playlist on-site, songs transition with smooth crossfades
- **For Fun Feature (Live Emotion Player)**:
  - Track emotion history over time (sliding window)
  - Detect when `emotion_distance > threshold` → drift detected
  - Find transition-compatible next song (similar key, close BPM)
  - Smooth crossfade: current track volume ↓ while next track volume ↑
  - Duration: 3–5 seconds, Apple Music-style
- **Output**: Seamless, professional-quality music playback in both features

---

### Phase 8: Full-Stack Website

#### Backend (FastAPI + Python)
- API endpoints:
  - `POST /predict-text` — text emotion detection
  - `POST /predict-voice` — voice emotion detection
  - `POST /fuse-emotion` — fusion layer
  - `POST /curate-playlist` — main feature: generate emotion-matched playlist
  - `POST /live-recommend` — fun feature: real-time song recommendation
  - `POST /rag-query` — RAG-powered natural language queries
  - `POST /agent-curate` — trigger agentic curation
  - `POST /export-playlist` — export to Spotify/Apple Music

#### Frontend (Next.js + Premium UI)
- **Design direction**: Apple.com-inspired, modern, premium
- **Color palette**: Deep blacks, rich blues, white accents, blue shade gradients
- **Animations**: Smooth micro-interactions, Framer Motion, glassmorphism
- **Audio visualizations**: Three.js / Web Audio API

##### 🎯 Main Page: Playlist Curator
```
┌─────────────────────────────────────────────────┐
│                   AETHER                        │
├─────────────────────────────────────────────────┤
│                                                 │
│   ┌──────────────────────────────────────┐      │
│   │  💬 "Tell me how you feel..."        │      │
│   │  [Text input] or [🎤 Speak]         │      │
│   │  (English / Hindi)                   │      │
│   └──────────────────────────────────────┘      │
│                                                 │
│   ┌──────────────────────────────────────┐      │
│   │  EMOTION ANALYSIS                    │      │
│   │  (animated emotion visualization)    │      │
│   │  "You're feeling nostalgic + hopeful"│      │
│   └──────────────────────────────────────┘      │
│                                                 │
│   ┌──────────────────────────────────────┐      │
│   │  🎵 YOUR CURATED PLAYLIST            │      │
│   │  (list of matched songs)             │      │
│   │  [▶ Play All] [Export to Spotify]    │      │
│   └──────────────────────────────────────┘      │
│                                                 │
│   "Why these songs?" (RAG explanation)          │
│                                                 │
└─────────────────────────────────────────────────┘
```

##### 🎶 Fun Feature Page: Live Emotion Player
```
┌─────────────────────────────────────────────────┐
│                   AETHER LIVE                   │
├─────────────────────────────────────────────────┤
│                                                 │
│   ┌──────────────────────────────────────┐      │
│   │       EMOTION VISUALIZER             │      │
│   │    (animated mood orb / waveform)    │      │
│   └──────────────────────────────────────┘      │
│                                                 │
│   ┌──────────────────────────────────────┐      │
│   │  🎵 NOW PLAYING — Music Player       │      │
│   │  (crossfade transitions)             │      │
│   └──────────────────────────────────────┘      │
│                                                 │
│   ┌──────────────────────────────────────┐      │
│   │  💬 Type how you feel...    [Enter]  │      │
│   │  🎤 [Speak] (English / Hindi)        │      │
│   └──────────────────────────────────────┘      │
│                                                 │
│   "You seem reflective — here's why we          │
│    chose this track..." (RAG explanation)        │
│                                                 │
└─────────────────────────────────────────────────┘
```

##### Other Pages / Features
- Landing page with project explanation
- Mood history timeline
- Explainable recommendation cards
- Settings (choose language, adjust sensitivity)

#### Deployment
- Frontend → Vercel
- Backend → Render or Railway

---

## Tech Stack Summary

| Layer | Technology |
|-------|-----------| 
| ML / Models | Python, PyTorch, HuggingFace, emotion2vec+, Whisper, Librosa, Scikit-learn |
| RAG | ChromaDB or Pinecone, LangChain, OpenAI/HuggingFace embeddings |
| Agentic AI | LangChain / LangGraph |
| Backend | FastAPI (Python) |
| Frontend | Next.js, TailwindCSS, Framer Motion, Three.js, Web Audio API |
| APIs | Spotify Web API, Apple Music API |
| Database | MongoDB or Supabase |
| Deployment | Vercel (frontend), Render/Railway (backend) |
| Dev Tools | VS Code, Google Colab (model training), GitHub |

---

## 15 Aether Emotion Categories

| # | Emotion | Musical Profile | Source Datasets |
|---|---------|----------------|-----------------|
| 0 | happy | Upbeat pop, feel-good, major key | Text + Voice |
| 1 | sad | Slow ballads, minor key | Text + Voice |
| 2 | angry | Heavy rock, aggressive beats | Text + Voice |
| 3 | calm | Ambient, lo-fi, soft | Text + Voice |
| 4 | anxious | Tense, building, unsettling | Text + Voice |
| 5 | energetic | EDM, dance, high tempo | Text only |
| 6 | focused | Lo-fi beats, minimal, study | Text only |
| 7 | nostalgic | Retro, acoustic, warm | Text only |
| 8 | romantic | Love songs, R&B, slow jams | Text only |
| 9 | melancholic | Dark, layered, minor key depth | Text only |
| 10 | confident | Powerful, bass-heavy, anthems | Text only |
| 11 | hopeful | Uplifting, building, major key | Text only |
| 12 | frustrated | Hard rock, punk, dissonant | Text + Voice |
| 13 | lonely | Minimal, sparse, echo-heavy | Text only |
| 14 | dreamy | Synth, atmospheric, ambient | Text + Voice |

> **Note**: Emotions marked "Text only" don't have voice training data (RAVDESS/CREMA-D don't have those emotions). The text model detects all 15. The fusion layer handles this gracefully.

---

## Build Order (What We Do First → Last)

```
Phase 1A: Text Emotion Model               ✅ COMPLETE
Phase 1B: Voice Emotion Model (EN + HI)    ✅ COMPLETE
Phase 1C: Unified Fusion Layer             ← NEXT
Phase 2:  Music Dataset Processing
Phase 3:  Emotion → Music Mapping
Phase 4:  Recommendation & Playlist Engine
Phase 5:  RAG Layer
Phase 6:  Agentic AI Layer
Phase 7:  Seamless Transitions & Crossfade
Phase 8:  Full-Stack Website
          - Backend API (FastAPI)
          - Main Feature: Playlist Curator
          - Fun Feature: Live Emotion Player
          - Premium Apple-inspired design
```
