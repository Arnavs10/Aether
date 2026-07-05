"""
═══════════════════════════════════════════════════════════════════
AETHER — Central Configuration
All shared constants, emotion definitions, model paths, and settings.
═══════════════════════════════════════════════════════════════════
"""

import os
from pathlib import Path

# ──────────────────────────────────────────────
# Project Paths
# ──────────────────────────────────────────────
PROJECT_ROOT = Path(__file__).parent
MODELS_DIR = PROJECT_ROOT / "models"
DATA_DIR = PROJECT_ROOT / "data"
PROCESSED_DATA_DIR = DATA_DIR / "processed"

TEXT_MODEL_DIR = MODELS_DIR / "text_emotion"
VOICE_MODEL_DIR = MODELS_DIR / "voice_emotion"
FUSION_MODEL_DIR = MODELS_DIR / "fusion"

# ──────────────────────────────────────────────
# Aether Core Emotions — 15 Categories
# ──────────────────────────────────────────────
# Each emotion maps to a genuinely different
# musical profile. This is NOT arbitrary —
# every category produces distinct music.
# ──────────────────────────────────────────────

AETHER_EMOTIONS = [
    "happy",        # joyful, excited, grateful — upbeat pop, feel-good
    "sad",          # grieving, disappointed — slow ballads, minor key
    "angry",        # rage, hatred, outrage — heavy rock, aggressive beats
    "calm",         # peaceful, serene, relaxed — ambient, lo-fi, soft
    "anxious",      # worried, nervous, panicked — tense, building, unsettling
    "energetic",    # pumped, thrilled, wild — EDM, dance, high tempo
    "focused",      # concentrated, determined — lo-fi beats, minimal, study
    "nostalgic",    # remembering, bittersweet past — retro, acoustic, warm
    "romantic",     # loving, tender, intimate — love songs, R&B, slow jams
    "melancholic",  # deep sorrow, existential — dark, layered, minor key depth
    "confident",    # empowered, bold, proud — powerful, bass-heavy, anthems
    "hopeful",      # optimistic, looking forward — uplifting, building, major key
    "frustrated",   # irritated, stuck, blocked — hard rock, punk, dissonant
    "lonely",       # isolated, missing, empty — minimal, sparse, echo-heavy
    "dreamy",       # ethereal, imaginative, floating — synth, atmospheric, ambient
]

NUM_EMOTIONS = len(AETHER_EMOTIONS)

EMOTION_TO_ID = {emotion: idx for idx, emotion in enumerate(AETHER_EMOTIONS)}
ID_TO_EMOTION = {idx: emotion for idx, emotion in enumerate(AETHER_EMOTIONS)}

# ──────────────────────────────────────────────
# Emotion Intensity Levels
# Modifies music parameters per emotion
# ──────────────────────────────────────────────
INTENSITY_LEVELS = {
    0: "neutral",     # no strong emotion detected
    1: "mild",        # slight emotional signal
    2: "moderate",    # clear emotional state
    3: "intense",     # strong emotional signal
}

# ──────────────────────────────────────────────
# Fine-Grained → Core Emotion Mapping
# Maps 100+ granular emotion labels from any
# dataset (SemEval-2025, 20-Emotion, GoEmotions)
# to Aether's 15 core categories
# ──────────────────────────────────────────────

FINE_TO_CORE_EMOTION_MAP = {
    # ─── happy ───
    "joy": "happy", "happiness": "happy", "amusement": "happy",
    "excitement": "happy", "love": "happy", "gratitude": "happy",
    "admiration": "happy", "approval": "happy", "pride": "happy",
    "relief": "happy", "happy": "happy", "elation": "happy",
    "contentment": "happy", "enthusiasm": "happy", "delight": "happy",
    "cheerfulness": "happy", "satisfaction": "happy", "euphoria": "happy",
    "bliss": "happy", "pleasure": "happy", "ecstasy": "happy",
    "glee": "happy", "jubilation": "happy", "exuberance": "happy",

    # ─── sad ───
    "sadness": "sad", "sad": "sad", "grief": "sad",
    "disappointment": "sad", "remorse": "sad", "sorrow": "sad",
    "despair": "sad", "guilt": "sad", "regret": "sad",
    "heartbreak": "sad", "misery": "sad", "anguish": "sad",
    "dejection": "sad", "gloom": "sad", "unhappiness": "sad",
    "hurt": "sad", "suffering": "sad",

    # ─── angry ───
    "anger": "angry", "angry": "angry", "rage": "angry",
    "hate": "angry", "hostility": "angry", "outrage": "angry",
    "fury": "angry", "wrath": "angry", "contempt": "angry",
    "resentment": "angry", "bitterness": "angry", "aggression": "angry",
    "indignation": "angry", "vengeance": "angry",

    # ─── calm ───
    "calm": "calm", "serenity": "calm", "peace": "calm",
    "relaxation": "calm", "tranquility": "calm", "composure": "calm",
    "patience": "calm", "stillness": "calm", "harmony": "calm",
    "balance": "calm", "acceptance": "calm", "equanimity": "calm",
    "neutral": "calm", "indifference": "calm",

    # ─── anxious ───
    "fear": "anxious", "anxious": "anxious", "anxiety": "anxious",
    "nervousness": "anxious", "worry": "anxious", "panic": "anxious",
    "apprehension": "anxious", "dread": "anxious", "unease": "anxious",
    "tension": "anxious", "distress": "anxious", "alarm": "anxious",
    "trepidation": "anxious", "paranoia": "anxious", "hysteria": "anxious",

    # ─── energetic ───
    "energetic": "energetic", "excitement": "energetic",
    "thrill": "energetic", "exhilaration": "energetic",
    "adrenaline": "energetic", "vigor": "energetic",
    "vitality": "energetic", "zeal": "energetic",
    "dynamism": "energetic", "liveliness": "energetic",
    "spirited": "energetic", "fired_up": "energetic",

    # ─── focused ───
    "focused": "focused", "concentration": "focused",
    "determination": "focused", "attentiveness": "focused",
    "resolve": "focused", "dedication": "focused",
    "mindfulness": "focused", "absorption": "focused",
    "engagement": "focused", "vigilance": "focused",
    "contemplation": "focused", "deliberation": "focused",

    # ─── nostalgic ───
    "nostalgia": "nostalgic", "nostalgic": "nostalgic",
    "reminiscence": "nostalgic", "wistfulness": "nostalgic",
    "sentimentality": "nostalgic", "yearning": "nostalgic",
    "homesickness": "nostalgic", "longing_past": "nostalgic",
    "bittersweet": "nostalgic", "remembrance": "nostalgic",

    # ─── romantic ───
    "romantic": "romantic", "romance": "romantic",
    "affection": "romantic", "tenderness": "romantic",
    "intimacy": "romantic", "passion": "romantic",
    "devotion": "romantic", "adoration": "romantic",
    "infatuation": "romantic", "desire": "romantic",
    "longing": "romantic", "warmth": "romantic",
    "caring": "romantic", "fondness": "romantic",

    # ─── melancholic ───
    "melancholy": "melancholic", "melancholic": "melancholic",
    "pensiveness": "melancholic", "gloominess": "melancholic",
    "existential": "melancholic", "somber": "melancholic",
    "mournful": "melancholic", "brooding": "melancholic",
    "dismal": "melancholic", "morose": "melancholic",
    "heavyhearted": "melancholic", "desolation": "melancholic",

    # ─── confident ───
    "confidence": "confident", "confident": "confident",
    "empowerment": "confident", "boldness": "confident",
    "courage": "confident", "assertiveness": "confident",
    "self_assurance": "confident", "strength": "confident",
    "power": "confident", "dominance": "confident",
    "triumph": "confident", "victory": "confident",
    "invincibility": "confident",

    # ─── hopeful ───
    "hope": "hopeful", "hopeful": "hopeful",
    "optimism": "hopeful", "anticipation": "hopeful",
    "aspiration": "hopeful", "faith": "hopeful",
    "expectation": "hopeful", "encouragement": "hopeful",
    "inspiration": "hopeful", "possibility": "hopeful",
    "hopefulness": "hopeful", "promising": "hopeful",

    # ─── frustrated ───
    "frustration": "frustrated", "frustrated": "frustrated",
    "annoyance": "frustrated", "irritation": "frustrated",
    "exasperation": "frustrated", "impatience": "frustrated",
    "vexation": "frustrated", "agitation": "frustrated",
    "disapproval": "frustrated", "displeasure": "frustrated",
    "dissatisfaction": "frustrated", "aggravation": "frustrated",

    # ─── lonely ───
    "loneliness": "lonely", "lonely": "lonely",
    "isolation": "lonely", "solitude": "lonely",
    "abandonment": "lonely", "alienation": "lonely",
    "emptiness": "lonely", "disconnection": "lonely",
    "forsaken": "lonely", "desolate": "lonely",
    "excluded": "lonely", "forgotten": "lonely",

    # ─── dreamy ───
    "dreamy": "dreamy", "wonder": "dreamy",
    "imagination": "dreamy", "fantasy": "dreamy",
    "ethereal": "dreamy", "surreal": "dreamy",
    "whimsical": "dreamy", "mystical": "dreamy",
    "enchantment": "dreamy", "reverie": "dreamy",
    "daydreaming": "dreamy", "trance": "dreamy",
    "curiosity": "dreamy", "awe": "dreamy",
    "fascination": "dreamy", "amazement": "dreamy",
    "surprise": "dreamy", "confusion": "dreamy",
    "realization": "dreamy", "epiphany": "dreamy",
    "mysterious": "dreamy",
}


def map_label_to_aether(label: str) -> str | None:
    """Map any fine-grained emotion label to one of 15 Aether core emotions."""
    label_lower = label.lower().strip().replace(" ", "_").replace("-", "_")
    return FINE_TO_CORE_EMOTION_MAP.get(label_lower, None)


# ──────────────────────────────────────────────
# Emotion → Music Feature Targets
# Each of 15 emotions maps to a DISTINCT
# musical profile. Intensity scales these values.
# ──────────────────────────────────────────────

EMOTION_MUSIC_TARGETS = {
    "happy":       {"tempo": 120, "energy": 0.80, "valence": 0.85, "danceability": 0.75, "acousticness": 0.20, "instrumentalness": 0.05},
    "sad":         {"tempo": 70,  "energy": 0.20, "valence": 0.15, "danceability": 0.25, "acousticness": 0.65, "instrumentalness": 0.20},
    "angry":       {"tempo": 135, "energy": 0.92, "valence": 0.18, "danceability": 0.55, "acousticness": 0.05, "instrumentalness": 0.10},
    "calm":        {"tempo": 80,  "energy": 0.22, "valence": 0.50, "danceability": 0.28, "acousticness": 0.80, "instrumentalness": 0.40},
    "anxious":     {"tempo": 105, "energy": 0.62, "valence": 0.22, "danceability": 0.38, "acousticness": 0.30, "instrumentalness": 0.25},
    "energetic":   {"tempo": 142, "energy": 0.95, "valence": 0.72, "danceability": 0.88, "acousticness": 0.05, "instrumentalness": 0.15},
    "focused":     {"tempo": 92,  "energy": 0.35, "valence": 0.42, "danceability": 0.18, "acousticness": 0.45, "instrumentalness": 0.70},
    "nostalgic":   {"tempo": 88,  "energy": 0.38, "valence": 0.48, "danceability": 0.32, "acousticness": 0.72, "instrumentalness": 0.15},
    "romantic":    {"tempo": 85,  "energy": 0.42, "valence": 0.68, "danceability": 0.45, "acousticness": 0.55, "instrumentalness": 0.10},
    "melancholic": {"tempo": 62,  "energy": 0.15, "valence": 0.08, "danceability": 0.12, "acousticness": 0.75, "instrumentalness": 0.35},
    "confident":   {"tempo": 118, "energy": 0.85, "valence": 0.78, "danceability": 0.72, "acousticness": 0.10, "instrumentalness": 0.08},
    "hopeful":     {"tempo": 112, "energy": 0.65, "valence": 0.82, "danceability": 0.55, "acousticness": 0.35, "instrumentalness": 0.12},
    "frustrated":  {"tempo": 128, "energy": 0.82, "valence": 0.12, "danceability": 0.42, "acousticness": 0.08, "instrumentalness": 0.20},
    "lonely":      {"tempo": 72,  "energy": 0.18, "valence": 0.18, "danceability": 0.12, "acousticness": 0.82, "instrumentalness": 0.55},
    "dreamy":      {"tempo": 82,  "energy": 0.28, "valence": 0.58, "danceability": 0.22, "acousticness": 0.60, "instrumentalness": 0.65},
}

# Intensity modifiers — scale the base targets
INTENSITY_MODIFIERS = {
    0: {"tempo": 0.0, "energy": 0.0, "valence": 0.0},   # neutral
    1: {"tempo": 0.9, "energy": 0.85, "valence": 0.9},   # mild — softer
    2: {"tempo": 1.0, "energy": 1.0, "valence": 1.0},    # moderate — as defined
    3: {"tempo": 1.1, "energy": 1.15, "valence": 1.1},   # intense — amplified
}


# ──────────────────────────────────────────────
# Fusion Layer Weights
# Dynamic weighting based on active modalities
# ──────────────────────────────────────────────
FUSION_WEIGHTS = {
    "text_voice":      {"text": 0.60, "voice": 0.40},
    "text_only":       {"text": 1.0},
    "voice_only":      {"voice_acoustic": 0.50, "voice_text": 0.50},
}

# ──────────────────────────────────────────────
# Model Configuration
# ──────────────────────────────────────────────
TEXT_MODEL_NAME = "distilroberta-base"
VOICE_MODEL_NAME = "emotion2vec+ large"

TEXT_TRAINING_CONFIG = {
    "learning_rate": 2e-5,
    "batch_size": 32,
    "num_epochs": 5,
    "warmup_ratio": 0.1,
    "weight_decay": 0.01,
    "max_length": 128,
    "eval_strategy": "epoch",
    "save_strategy": "epoch",
    "load_best_model_at_end": True,
    "metric_for_best_model": "f1",
}

VOICE_TRAINING_CONFIG = {
    "learning_rate": 1e-4,
    "batch_size": 16,
    "num_epochs": 10,
    "max_audio_length_sec": 10,
    "sample_rate": 16000,
}



# ──────────────────────────────────────────────
# Emotion Drift Detection
# ──────────────────────────────────────────────
DRIFT_WINDOW_SIZE = 5
DRIFT_THRESHOLD = 0.35
CROSSFADE_DURATION_SEC = 4

# ──────────────────────────────────────────────
# API & RAG Configuration
# ──────────────────────────────────────────────
API_HOST = "0.0.0.0"
API_PORT = 8000
CHROMA_COLLECTION_NAME = "aether_songs"
EMBEDDING_MODEL = "all-MiniLM-L6-v2"
RAG_TOP_K = 10
