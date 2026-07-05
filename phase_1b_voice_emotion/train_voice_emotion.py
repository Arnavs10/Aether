"""
═══════════════════════════════════════════════════════════════════
AETHER — Phase 1B: Voice Emotion Model Training Script
═══════════════════════════════════════════════════════════════════

Uses emotion2vec+ (SOTA 2024/2025) as frozen feature extractor,
then trains a classification head mapping acoustic features to
15 Aether emotion categories.

Also integrates Whisper for speech-to-text (EN + HI), providing
a DUAL-SIGNAL output: acoustic emotion + transcribed text.

Architecture:
  Audio → emotion2vec+ (frozen) → 768-dim embedding
       → MLP head → 15 Aether emotions

Datasets (auto-downloaded from HuggingFace):
  - RAVDESS (via xbgoose/ravdess) — 1,440 speech samples
  - CREMA-D (via AbstractTTS/CREMA-D) — 7,442 speech samples
  Combined: ~8,800+ samples across 6-8 source emotions
  → mapped to 15 Aether categories

Designed for:
  - Google Colab T4 GPU (recommended)
  - MacBook Air M2 (MPS)
  - CPU (slower but works)

Usage:
  Colab: Upload → Run each section in order
  Local: python phase_1b_voice_emotion/train_voice_emotion.py
═══════════════════════════════════════════════════════════════════
"""

# ═══════════════════════════════════════════════
# SECTION 1: Install & Import
# ═══════════════════════════════════════════════
# Colab first cell:
# !pip install -q funasr torch torchaudio torchcodec datasets scikit-learn librosa soundfile numpy

import os
import sys
import json
import time
import random
import warnings
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from pathlib import Path
from collections import Counter

warnings.filterwarnings("ignore")

# Detect environment
if torch.cuda.is_available():
    DEVICE = "cuda"
    print(f"✅ GPU: {torch.cuda.get_device_name(0)} | VRAM: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")
elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
    DEVICE = "mps"
    print("✅ Apple Silicon MPS")
else:
    DEVICE = "cpu"
    print("⚠️  CPU mode — training will be slower")

# MLP head always trains on CPU or CUDA (MPS has some op gaps)
TRAIN_DEVICE = "cuda" if DEVICE == "cuda" else "cpu"
print(f"   PyTorch {torch.__version__} | Device: {DEVICE} | Training on: {TRAIN_DEVICE}")


# ═══════════════════════════════════════════════
# SECTION 2: Aether 15 Emotion System
# ═══════════════════════════════════════════════
# Same emotion system as Phase 1A — ensures
# consistency across all modalities.

AETHER_EMOTIONS = [
    "happy", "sad", "angry", "calm", "anxious",
    "energetic", "focused", "nostalgic", "romantic",
    "melancholic", "confident", "hopeful", "frustrated",
    "lonely", "dreamy",
]
NUM_LABELS = len(AETHER_EMOTIONS)  # 15

EMOTION_TO_ID = {e: i for i, e in enumerate(AETHER_EMOTIONS)}
ID_TO_EMOTION = {i: e for i, e in enumerate(AETHER_EMOTIONS)}


# ═══════════════════════════════════════════════
# SECTION 3: Voice Emotion → Aether Mapping
# ═══════════════════════════════════════════════
# Maps source dataset emotion labels to Aether's
# 15 core categories. These datasets have 6-8
# raw emotions which map to a subset of our 15.

# RAVDESS emotions (from filename encoding):
# 01=neutral, 02=calm, 03=happy, 04=sad,
# 05=angry, 06=fearful, 07=disgust, 08=surprised
RAVDESS_TO_AETHER = {
    "neutral": "calm",
    "calm": "calm",
    "happy": "happy",
    "sad": "sad",
    "angry": "angry",
    "fearful": "anxious",
    "disgust": "frustrated",
    "surprised": "dreamy",
}

# CREMA-D emotions:
# ANG=angry, DIS=disgust, FEA=fear, HAP=happy, NEU=neutral, SAD=sad
CREMAD_TO_AETHER = {
    "ANG": "angry",
    "DIS": "frustrated",
    "FEA": "anxious",
    "HAP": "happy",
    "NEU": "calm",
    "SAD": "sad",
    "anger": "angry",
    "disgust": "frustrated",
    "fear": "anxious",
    "happy": "happy",
    "happiness": "happy",
    "neutral": "calm",
    "sad": "sad",
    "sadness": "sad",
}

# emotion2vec+ native labels (for direct inference)
EMOTION2VEC_TO_AETHER = {
    "angry": "angry",
    "disgusted": "frustrated",
    "fearful": "anxious",
    "happy": "happy",
    "neutral": "calm",
    "other": "calm",
    "sad": "sad",
    "surprised": "dreamy",
    "unknown": "calm",
}


# Unified lowercase mapping to prevent any case-sensitive lookup bugs
UNIFIED_AETHER_MAP = {}
for mapping in [RAVDESS_TO_AETHER, CREMAD_TO_AETHER, EMOTION2VEC_TO_AETHER]:
    for k, v in mapping.items():
        UNIFIED_AETHER_MAP[k.lower()] = v

def map_voice_label_to_aether(label: str) -> str | None:
    """Map any voice emotion label to one of 15 Aether core emotions."""
    if not isinstance(label, str):
        return None
    label_lower = label.lower().strip().replace(" ", "_").replace("-", "_")
    
    # Try the unified mapping
    if label_lower in UNIFIED_AETHER_MAP:
        return UNIFIED_AETHER_MAP[label_lower]
        
    # Direct match
    if label_lower in EMOTION_TO_ID:
        return label_lower
    return None


# Model & training config
EMBEDDING_DIM = 1024  # emotion2vec+ large output dimension
HIDDEN_DIM = 256
DROPOUT = 0.3
BATCH_SIZE = 32
NUM_EPOCHS = 30
LEARNING_RATE = 1e-3
WEIGHT_DECAY = 1e-4
PATIENCE = 7  # early stopping patience
SAMPLE_RATE = 16000
MAX_AUDIO_SEC = 10

OUTPUT_DIR = Path("./models/voice_emotion")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

print(f"\n📋 Aether Voice Emotion Model Config:")
print(f"   Feature Extractor: emotion2vec+ large (frozen)")
print(f"   Embedding dim: {EMBEDDING_DIM}")
print(f"   Classification head: MLP ({EMBEDDING_DIM}→{HIDDEN_DIM}→{NUM_LABELS})")
print(f"   Emotions: {NUM_LABELS} categories")
print(f"   Categories: {AETHER_EMOTIONS}")
print(f"   Batch size: {BATCH_SIZE}")
print(f"   Epochs: {NUM_EPOCHS} (early stopping patience: {PATIENCE})")
print(f"   Output: {OUTPUT_DIR}")


# ═══════════════════════════════════════════════
# SECTION 4: Load emotion2vec+ Feature Extractor
# ═══════════════════════════════════════════════
# emotion2vec+ is the SOTA (2024/2025) foundation
# model for Speech Emotion Recognition. We use it
# as a FROZEN feature extractor — no fine-tuning
# of the large model itself. Only our MLP head
# learns.

from funasr import AutoModel as FunASRAutoModel

print("\n🧠 Loading emotion2vec+ large (frozen feature extractor)...")

e2v_model = FunASRAutoModel(model="iic/emotion2vec_plus_large", hub="hf")

print("✅ emotion2vec+ large loaded")
print(f"   Output: {EMBEDDING_DIM}-dim embeddings per utterance")
print(f"   Sample rate: {SAMPLE_RATE} Hz")


# ═══════════════════════════════════════════════
# SECTION 5: Load & Combine Datasets
# ═══════════════════════════════════════════════
# Strategy: Load RAVDESS + CREMA-D from HuggingFace
# (auto-downloaded, no manual download needed),
# extract emotion2vec+ embeddings from each audio,
# map labels to Aether's 15 categories.

from datasets import load_dataset, Audio
import librosa
import soundfile as sf

print("\n📥 Loading datasets (auto-downloaded from HuggingFace)...")

datasets_loaded = []

# ── Dataset 1: RAVDESS ──
# xbgoose/ravdess has string 'emotion' column: angry, calm, etc.
print("\n   Loading RAVDESS...")
ravdess_repos = [
    "xbgoose/ravdess",
    "narad/ravdess",
    "xevict/ravdess",
    "Aniemore/ravdess"
]

ds_ravdess = None
for repo in ravdess_repos:
    try:
        ds = load_dataset(repo, split="train")
        # Ensure audio column exists and is cast to target sample rate
        if "audio" not in ds.column_names:
            print(f"   ⚠️  {repo} has no 'audio' column, skipping")
            continue
        ds_ravdess = ds.cast_column("audio", Audio(sampling_rate=SAMPLE_RATE))
        print(f"   ✅ RAVDESS ({repo}): {len(ds_ravdess):,} samples")
        # label_names=None because xbgoose/ravdess uses string 'emotion' column directly
        datasets_loaded.append(("RAVDESS", ds_ravdess, None))
        break
    except Exception as e:
        print(f"   ⚠️  Could not load {repo}: {e}")

# ── Dataset 2: CREMA-D ──
# AbstractTTS/CREMA-D has 'audio' + 'major_emotion' columns with actual audio data
print("\n   Loading CREMA-D...")
cremad_repos = [
    "AbstractTTS/CREMA-D",
    "Zahra99/CREMA-D",
    "CheonggyeMountain-Sherpa/CREMA-D",
    "Aniemore/crema_d"
]

ds_cremad = None
for repo in cremad_repos:
    try:
        ds = load_dataset(repo, split="train")
        # Only accept repos that actually contain audio data
        if "audio" not in ds.column_names:
            print(f"   ⚠️  {repo} has no 'audio' column, skipping")
            continue
        ds_cremad = ds.cast_column("audio", Audio(sampling_rate=SAMPLE_RATE))
        print(f"   ✅ CREMA-D ({repo}): {len(ds_cremad):,} samples")
        # label_names=None — labels detected automatically from column values
        datasets_loaded.append(("CREMA-D", ds_cremad, None))
        break
    except Exception as e:
        print(f"   ⚠️  Could not load {repo}: {e}")

if not datasets_loaded:
    print("❌ Could not load any dataset!")
    print("   Please check your internet connection and try again.")
    sys.exit(1)

print(f"\n📊 Loaded: {' + '.join([name for name, _, _ in datasets_loaded])}")

# ═══════════════════════════════════════════════
# SECTION 6: Extract Embeddings
# ═══════════════════════════════════════════════
# Run each audio through emotion2vec+ to get
# 768-dim embeddings. This is the heavy part
# but only needs to be done once.

import tempfile

print("\n🔄 Extracting emotion2vec+ embeddings...")
print("   This may take a while on first run (~10-30 min depending on GPU/CPU)")


def get_label_name(dataset_name, example, label_names):
    """Extract the emotion label string from a dataset example.
    
    Handles both integer-indexed labels (with label_names lookup)
    and string labels (returned directly).
    """
    # Try common column names in priority order
    for col in ["emotion", "major_emotion", "label", "classname", "labels", "sentiment", "class"]:
        if col in example:
            val = example[col]
            # Integer label → look up name
            if isinstance(val, int) and label_names is not None:
                if 0 <= val < len(label_names):
                    return label_names[val]
            # String label → return directly
            elif isinstance(val, str) and val.strip():
                return val.strip()
    return None


def extract_embedding_safe(audio_array, sr):
    """Extract embedding from a single audio clip using emotion2vec+.
    
    Writes audio to a temp .wav file (required by funasr),
    runs emotion2vec+ inference, and returns the 768-dim embedding.
    Returns None on any failure (short audio, bad data, etc.).
    """
    tmp_path = None
    try:
        # Ensure float32 numpy array
        if not isinstance(audio_array, np.ndarray):
            audio_array = np.array(audio_array, dtype=np.float32)
        audio_array = audio_array.astype(np.float32)

        # Ensure correct sample rate
        if sr != SAMPLE_RATE:
            audio_array = librosa.resample(audio_array, orig_sr=sr, target_sr=SAMPLE_RATE)
            sr = SAMPLE_RATE

        # Trim to max length
        max_samples = MAX_AUDIO_SEC * SAMPLE_RATE
        if len(audio_array) > max_samples:
            audio_array = audio_array[:max_samples]

        # Skip very short audio (< 0.3 seconds)
        if len(audio_array) < int(SAMPLE_RATE * 0.3):
            return None

        # Save to temp file (funasr requires a file path)
        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp:
            tmp_path = tmp.name
            sf.write(tmp_path, audio_array, SAMPLE_RATE)

        # Extract embedding
        res = e2v_model.generate(
            input=tmp_path,
            granularity="utterance",
            extract_embedding=True,
        )

        if res and len(res) > 0 and "feats" in res[0]:
            feats = res[0]["feats"]
            if isinstance(feats, np.ndarray):
                return feats.flatten()
            elif isinstance(feats, list):
                return np.array(feats).flatten()
            elif hasattr(feats, 'numpy'):
                return feats.numpy().flatten()

        return None

    except Exception:
        return None

    finally:
        # Always clean up temp file
        if tmp_path is not None and os.path.exists(tmp_path):
            try:
                os.unlink(tmp_path)
            except OSError:
                pass


all_embeddings = []
all_labels = []
skipped = 0
processed = 0

for ds_name, ds, label_names in datasets_loaded:
    count_before = len(all_embeddings)
    ds_skipped = 0

    print(f"\n   Processing {ds_name} ({len(ds):,} samples)...")

    # Detect column names
    columns = ds.column_names
    print(f"   Columns: {columns}")

    # Detect label column in priority order
    label_col = None
    for col in ["emotion", "major_emotion", "label", "classname", "labels", "sentiment", "class"]:
        if col in columns:
            label_col = col
            break

    if label_col is None:
        print(f"   ⚠️  No label column found in {ds_name}, skipping")
        continue

    print(f"   Using label column: '{label_col}'")

    # If label_names not provided, try to get them from features
    if label_names is None:
        feat = ds.features[label_col]
        if hasattr(feat, 'names'):
            label_names = feat.names
            print(f"   Label names from features: {label_names}")

    for i, example in enumerate(ds):
        # Progress — print every 200 samples (flush for Colab)
        if (i + 1) % 200 == 0 or i == 0:
            print(f"   [{ds_name}] {i+1}/{len(ds)} ({(i+1)/len(ds)*100:.1f}%)")

        # Get label
        raw_label = get_label_name(ds_name, example, label_names)
        if raw_label is None:
            ds_skipped += 1
            continue

        # Map to Aether
        aether_label = map_voice_label_to_aether(raw_label)
        if aether_label is None:
            ds_skipped += 1
            continue

        aether_id = EMOTION_TO_ID[aether_label]

        # Get audio
        audio_data = example.get("audio")
        if audio_data is None:
            ds_skipped += 1
            continue

        audio_array = audio_data["array"]
        sr = audio_data["sampling_rate"]

        # Extract embedding
        embedding = extract_embedding_safe(audio_array, sr)
        if embedding is None:
            ds_skipped += 1
            continue

        all_embeddings.append(embedding)
        all_labels.append(aether_id)
        processed += 1

    count_after = len(all_embeddings)
    skipped += ds_skipped
    print(f"   ✅ {ds_name}: {count_after - count_before:,} embeddings extracted (skipped {ds_skipped:,})")

print(f"\n   Total embeddings: {len(all_embeddings):,}")
print(f"   Total skipped: {skipped:,}")

if len(all_embeddings) < 100:
    print("❌ Too few embeddings extracted! Check dataset loading.")
    sys.exit(1)

# ═══════════════════════════════════════════════
# SECTION 7: Prepare Training Data
# ═══════════════════════════════════════════════

print("\n📊 Preparing training data...")

X = np.array(all_embeddings)
y = np.array(all_labels)

# Show distribution
print(f"\n   Embedding shape: {X.shape}")
print(f"\n📊 Label distribution:")
label_counts = Counter(y)
total = len(y)
for eid in range(NUM_LABELS):
    count = label_counts.get(eid, 0)
    pct = count / total * 100 if total > 0 else 0
    bar = "█" * int(pct)
    emotion_name = ID_TO_EMOTION[eid]
    print(f"   {emotion_name:>12}: {count:>6} ({pct:>5.1f}%) {bar}")

# Train/validation split
from sklearn.model_selection import train_test_split

X_train, X_val, y_train, y_val = train_test_split(
    X, y, test_size=0.15, random_state=42, stratify=y,
)

print(f"\n   📊 Final: Train {len(X_train):,} | Validation {len(X_val):,}")


# ═══════════════════════════════════════════════
# SECTION 8: Handle Class Imbalance
# ═══════════════════════════════════════════════

from sklearn.utils.class_weight import compute_class_weight

print("\n⚖️  Computing class weights for imbalanced data...")

present_classes = np.unique(y_train)
class_weights = compute_class_weight(
    class_weight="balanced",
    classes=present_classes,
    y=y_train,
)

weight_tensor = torch.ones(NUM_LABELS)
for cls, w in zip(present_classes, class_weights):
    weight_tensor[cls] = w
    print(f"   {ID_TO_EMOTION[cls]:>12}: weight = {w:.3f}")

weight_tensor = weight_tensor.to(TRAIN_DEVICE)
print("✅ Class weights computed — rare emotions get higher weight during training")


# ═══════════════════════════════════════════════
# SECTION 9: MLP Classification Head
# ═══════════════════════════════════════════════
# Lightweight but effective: 768 → 256 → 15
# with BatchNorm, dropout, and residual-like
# structure for stable training.

class VoiceEmotionHead(nn.Module):
    """
    MLP classification head for voice emotion detection.

    Takes 768-dim emotion2vec+ embeddings and classifies
    into 15 Aether emotion categories.

    Architecture:
        768 → BatchNorm → 256 (ReLU, Dropout)
            → BatchNorm → 128 (ReLU, Dropout)
            → 15 (logits)
    """

    def __init__(self, input_dim=EMBEDDING_DIM, hidden_dim=HIDDEN_DIM,
                 num_classes=NUM_LABELS, dropout=DROPOUT):
        super().__init__()

        self.net = nn.Sequential(
            # Layer 1: 768 → 256
            nn.BatchNorm1d(input_dim),
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),

            # Layer 2: 256 → 128
            nn.BatchNorm1d(hidden_dim),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(dropout),

            # Output: 128 → 15
            nn.Linear(hidden_dim // 2, num_classes),
        )

    def forward(self, x):
        return self.net(x)


model = VoiceEmotionHead().to(TRAIN_DEVICE)

total_params = sum(p.numel() for p in model.parameters())
trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
print(f"\n🧠 MLP Classification Head:")
print(f"   Architecture: {EMBEDDING_DIM} → {HIDDEN_DIM} → {HIDDEN_DIM//2} → {NUM_LABELS}")
print(f"   Parameters: {total_params:,} (all trainable)")
print(f"   Dropout: {DROPOUT}")


# ═══════════════════════════════════════════════
# SECTION 10: Training Setup
# ═══════════════════════════════════════════════

from torch.utils.data import TensorDataset, DataLoader

print("\n⚙️  Configuring training...")

# Create tensors
X_train_t = torch.FloatTensor(X_train).to(TRAIN_DEVICE)
y_train_t = torch.LongTensor(y_train).to(TRAIN_DEVICE)
X_val_t = torch.FloatTensor(X_val).to(TRAIN_DEVICE)
y_val_t = torch.LongTensor(y_val).to(TRAIN_DEVICE)

train_dataset = TensorDataset(X_train_t, y_train_t)
val_dataset = TensorDataset(X_val_t, y_val_t)

train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE * 2, shuffle=False)

# Loss with class weights
criterion = nn.CrossEntropyLoss(weight=weight_tensor)

# Optimizer with weight decay
optimizer = optim.AdamW(
    model.parameters(),
    lr=LEARNING_RATE,
    weight_decay=WEIGHT_DECAY,
)

# Learning rate scheduler (verbose is deprecated in newer PyTorch)
scheduler = optim.lr_scheduler.ReduceLROnPlateau(
    optimizer, mode="max", factor=0.5, patience=3
)

print(f"   Train batches: {len(train_loader)}")
print(f"   Val batches: {len(val_loader)}")
print(f"   Optimizer: AdamW (lr={LEARNING_RATE}, wd={WEIGHT_DECAY})")
print(f"   Scheduler: ReduceLROnPlateau (patience=3)")
print("✅ Training configured")


# ═══════════════════════════════════════════════
# SECTION 11: Train 🚀
# ═══════════════════════════════════════════════

from sklearn.metrics import accuracy_score, f1_score

print("\n" + "═" * 60)
print("🚀 TRAINING — Aether Voice Emotion Model (15 emotions)")
print("═" * 60 + "\n")

best_val_f1 = 0.0
best_epoch = 0
patience_counter = 0
train_start = time.time()

history = {"train_loss": [], "val_loss": [], "val_acc": [], "val_f1": []}

for epoch in range(NUM_EPOCHS):
    # ── Train ──
    model.train()
    train_loss = 0.0
    train_batches = 0

    for batch_x, batch_y in train_loader:
        optimizer.zero_grad()
        logits = model(batch_x)
        loss = criterion(logits, batch_y)
        loss.backward()

        # Gradient clipping for stability
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)

        optimizer.step()
        train_loss += loss.item()
        train_batches += 1

    avg_train_loss = train_loss / train_batches

    # ── Validate ──
    model.eval()
    val_loss = 0.0
    val_batches = 0
    all_preds = []
    all_true = []

    with torch.no_grad():
        for batch_x, batch_y in val_loader:
            logits = model(batch_x)
            loss = criterion(logits, batch_y)
            val_loss += loss.item()
            val_batches += 1

            preds = torch.argmax(logits, dim=-1)
            all_preds.extend(preds.cpu().numpy())
            all_true.extend(batch_y.cpu().numpy())

    avg_val_loss = val_loss / val_batches
    val_acc = accuracy_score(all_true, all_preds)
    val_f1 = f1_score(all_true, all_preds, average="weighted", zero_division=0)

    history["train_loss"].append(avg_train_loss)
    history["val_loss"].append(avg_val_loss)
    history["val_acc"].append(val_acc)
    history["val_f1"].append(val_f1)

    # Learning rate scheduling
    scheduler.step(val_f1)

    # Print progress
    marker = " ← best" if val_f1 > best_val_f1 else ""
    print(f"   Epoch {epoch+1:>2}/{NUM_EPOCHS} │ "
          f"Loss: {avg_train_loss:.4f} / {avg_val_loss:.4f} │ "
          f"Acc: {val_acc:.4f} │ F1: {val_f1:.4f}{marker}")

    # Early stopping check
    if val_f1 > best_val_f1:
        best_val_f1 = val_f1
        best_epoch = epoch + 1
        patience_counter = 0

        # Save best model
        torch.save({
            "model_state_dict": model.state_dict(),
            "embedding_dim": EMBEDDING_DIM,
            "hidden_dim": HIDDEN_DIM,
            "num_classes": NUM_LABELS,
            "dropout": DROPOUT,
            "best_val_f1": best_val_f1,
            "best_epoch": best_epoch,
        }, OUTPUT_DIR / "best_model.pt")
    else:
        patience_counter += 1
        if patience_counter >= PATIENCE:
            print(f"\n   ⏹️  Early stopping at epoch {epoch+1} (no improvement for {PATIENCE} epochs)")
            break

train_time = time.time() - train_start

print(f"\n" + "═" * 60)
print("✅ TRAINING COMPLETE")
print("═" * 60)
print(f"   Time: {train_time:.0f}s")
print(f"   Best epoch: {best_epoch}")
print(f"   Best F1 (weighted): {best_val_f1:.4f}")


# ═══════════════════════════════════════════════
# SECTION 12: Evaluate
# ═══════════════════════════════════════════════

from sklearn.metrics import (
    classification_report, confusion_matrix,
    precision_score, recall_score, f1_score as sk_f1_score,
)

print("\n📈 Final Evaluation...")

# Load best model
checkpoint = torch.load(OUTPUT_DIR / "best_model.pt", map_location=TRAIN_DEVICE, weights_only=True)
model.load_state_dict(checkpoint["model_state_dict"])
model.eval()

# Run on validation set
all_preds = []
all_true = []

with torch.no_grad():
    for batch_x, batch_y in val_loader:
        logits = model(batch_x)
        preds = torch.argmax(logits, dim=-1)
        all_preds.extend(preds.cpu().numpy())
        all_true.extend(batch_y.cpu().numpy())

all_preds = np.array(all_preds)
all_true = np.array(all_true)

# Overall metrics
final_acc = accuracy_score(all_true, all_preds)
final_f1_w = sk_f1_score(all_true, all_preds, average="weighted", zero_division=0)
final_f1_m = sk_f1_score(all_true, all_preds, average="macro", zero_division=0)
final_prec = precision_score(all_true, all_preds, average="weighted", zero_division=0)
final_rec = recall_score(all_true, all_preds, average="weighted", zero_division=0)

print(f"\n   Accuracy:        {final_acc:.4f}")
print(f"   F1 (weighted):   {final_f1_w:.4f}")
print(f"   F1 (macro):      {final_f1_m:.4f}")
print(f"   Precision:       {final_prec:.4f}")
print(f"   Recall:          {final_rec:.4f}")

# Per-emotion report
present_in_test = sorted(set(all_true))
target_names = [ID_TO_EMOTION[i] for i in present_in_test]

print("\n📋 Per-Emotion Classification Report:")
report = classification_report(
    all_true, all_preds,
    labels=present_in_test,
    target_names=target_names,
    digits=4,
)
print(report)

report_dict = classification_report(
    all_true, all_preds,
    labels=present_in_test,
    target_names=target_names,
    digits=4,
    output_dict=True,
)


# ═══════════════════════════════════════════════
# SECTION 13: Save Model & Config
# ═══════════════════════════════════════════════

print("\n💾 Saving model...")

final_path = OUTPUT_DIR / "final"
final_path.mkdir(parents=True, exist_ok=True)

# Save the best model checkpoint to final dir
torch.save({
    "model_state_dict": model.state_dict(),
    "embedding_dim": EMBEDDING_DIM,
    "hidden_dim": HIDDEN_DIM,
    "num_classes": NUM_LABELS,
    "dropout": DROPOUT,
    "best_val_f1": best_val_f1,
    "best_epoch": best_epoch,
}, final_path / "voice_emotion_head.pt")

# Save comprehensive config (matches Phase 1A format)
model_config = {
    "project": "Aether",
    "phase": "1B",
    "component": "Voice Emotion Detection",
    "feature_extractor": "emotion2vec+ large (iic/emotion2vec_plus_large)",
    "feature_extractor_hub": "hf",
    "embedding_dim": EMBEDDING_DIM,
    "classification_head": {
        "architecture": f"MLP ({EMBEDDING_DIM}→{HIDDEN_DIM}→{HIDDEN_DIM//2}→{NUM_LABELS})",
        "hidden_dim": HIDDEN_DIM,
        "dropout": DROPOUT,
    },
    "num_labels": NUM_LABELS,
    "emotions": AETHER_EMOTIONS,
    "emotion_to_id": EMOTION_TO_ID,
    "id_to_emotion": {str(k): v for k, v in ID_TO_EMOTION.items()},
    "sample_rate": SAMPLE_RATE,
    "max_audio_sec": MAX_AUDIO_SEC,
    "dataset": " + ".join([name for name, _, _ in datasets_loaded]),
    "voice_label_mappings": {
        "RAVDESS": RAVDESS_TO_AETHER,
        "CREMA-D": CREMAD_TO_AETHER,
        "emotion2vec": EMOTION2VEC_TO_AETHER,
    },
    "training": {
        "epochs_trained": best_epoch,
        "max_epochs": NUM_EPOCHS,
        "batch_size": BATCH_SIZE,
        "learning_rate": LEARNING_RATE,
        "weight_decay": WEIGHT_DECAY,
        "early_stopping_patience": PATIENCE,
        "weighted_loss": True,
    },
    "evaluation": {
        "accuracy": final_acc,
        "f1_weighted": final_f1_w,
        "f1_macro": final_f1_m,
        "precision": final_prec,
        "recall": final_rec,
        "per_emotion": report_dict,
    },
    "dual_signal": {
        "acoustic": "emotion2vec+ embeddings → MLP → 15 emotions",
        "text": "Whisper transcription → Phase 1A text model → 15 emotions",
        "fusion_ready": True,
    },
}

with open(OUTPUT_DIR / "model_config.json", "w") as f:
    json.dump(model_config, f, indent=2)

print(f"✅ Model → {final_path}")
print(f"✅ Config → {OUTPUT_DIR / 'model_config.json'}")


# ═══════════════════════════════════════════════
# SECTION 14: Inference Utility Class
# ═══════════════════════════════════════════════

print("\n📦 Saving inference utility...")

inference_code = '''"""
═══════════════════════════════════════════════
Aether — Voice Emotion Inference Utility
Loads emotion2vec+ and the trained MLP head to
predict emotions from audio with intensity levels.

Dual-signal output:
  1. Acoustic emotion (from voice tone/prosody)
  2. Transcribed text (from Whisper) — ready for
     Phase 1A text emotion model

This connects to Phase 1C (Fusion) which will
combine text + voice signals.
═══════════════════════════════════════════════
"""

import os
import json
import tempfile
import numpy as np
import torch
import torch.nn as nn
from pathlib import Path


class VoiceEmotionHead(nn.Module):
    """MLP classification head for voice emotion detection."""

    def __init__(self, input_dim=1024, hidden_dim=256, num_classes=15, dropout=0.3):
        super().__init__()
        self.net = nn.Sequential(
            nn.BatchNorm1d(input_dim),
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.BatchNorm1d(hidden_dim),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim // 2, num_classes),
        )

    def forward(self, x):
        return self.net(x)


class VoiceEmotionDetector:
    """
    Detects emotion from audio input using Aether's voice emotion pipeline.

    Returns:
      - Acoustic emotion: 15-emotion probability distribution from voice
      - Transcribed text: Speech-to-text via Whisper (ready for Phase 1A)

    Emotions: happy, sad, angry, calm, anxious, energetic, focused,
              nostalgic, romantic, melancholic, confident, hopeful,
              frustrated, lonely, dreamy
    """

    EMOTIONS = [
        "happy", "sad", "angry", "calm", "anxious",
        "energetic", "focused", "nostalgic", "romantic",
        "melancholic", "confident", "hopeful", "frustrated",
        "lonely", "dreamy",
    ]

    INTENSITY_THRESHOLDS = {
        "neutral": (0.0, 0.15),
        "mild": (0.15, 0.35),
        "moderate": (0.35, 0.65),
        "intense": (0.65, 1.0),
    }

    def __init__(self, model_dir: str = None, load_whisper: bool = True):
        if model_dir is None:
            model_dir = str(Path(__file__).parent.parent / "models" / "voice_emotion")

        self.model_dir = Path(model_dir)
        self.sample_rate = 16000

        # Load config
        config_path = self.model_dir / "model_config.json"
        if config_path.exists():
            with open(config_path) as f:
                self.config = json.load(f)

        # Load emotion2vec+ feature extractor
        from funasr import AutoModel as FunASRAutoModel
        self.e2v = FunASRAutoModel(model="iic/emotion2vec_plus_large", hub="hf")

        # Load MLP head
        checkpoint_path = self.model_dir / "final" / "voice_emotion_head.pt"
        checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=True)

        self.head = VoiceEmotionHead(
            input_dim=checkpoint.get("embedding_dim", 768),
            hidden_dim=checkpoint.get("hidden_dim", 256),
            num_classes=checkpoint.get("num_classes", 15),
            dropout=checkpoint.get("dropout", 0.3),
        )
        self.head.load_state_dict(checkpoint["model_state_dict"])
        self.head.eval()

        # Load Whisper (optional — for dual-signal)
        self.whisper_model = None
        if load_whisper:
            try:
                import whisper
                self.whisper_model = whisper.load_model("base")
                print("✅ Whisper loaded (speech-to-text ready)")
            except ImportError:
                print("⚠️  Whisper not installed. Text transcription disabled.")
                print("   Install with: pip install openai-whisper")

    def _get_intensity(self, confidence: float) -> str:
        """Map confidence score to intensity level."""
        for level, (low, high) in self.INTENSITY_THRESHOLDS.items():
            if low <= confidence < high:
                return level
        return "intense"

    def _extract_embedding(self, audio_path: str) -> np.ndarray | None:
        """Extract emotion2vec+ embedding from an audio file."""
        try:
            res = self.e2v.generate(
                input=audio_path,
                granularity="utterance",
                extract_embedding=True,
            )
            if res and len(res) > 0 and "feats" in res[0]:
                feats = res[0]["feats"]
                if isinstance(feats, np.ndarray):
                    return feats.flatten()
                elif isinstance(feats, list):
                    return np.array(feats).flatten()
                elif hasattr(feats, "numpy"):
                    return feats.numpy().flatten()
            return None
        except Exception:
            return None

    def predict(self, audio_path: str) -> dict:
        """
        Predict emotion from an audio file.

        Args:
            audio_path: Path to .wav audio file (16kHz recommended)

        Returns:
            {
                "acoustic_emotion": str,       # top predicted emotion
                "confidence": float,            # 0-1 score
                "intensity": str,               # neutral/mild/moderate/intense
                "probabilities": dict,          # all 15 emotion scores
                "top_3": list,                  # top 3 emotions with scores
                "transcription": str | None,    # Whisper text (for Phase 1A)
            }
        """
        # Extract embedding
        embedding = self._extract_embedding(audio_path)
        if embedding is None:
            return {"error": "Could not extract embedding from audio"}

        # Run through MLP head
        with torch.no_grad():
            x = torch.FloatTensor(embedding).unsqueeze(0)
            logits = self.head(x)
            probs = torch.softmax(logits, dim=-1).squeeze().numpy()

        # Build results
        prob_dict = {self.EMOTIONS[i]: round(float(probs[i]), 4) for i in range(len(self.EMOTIONS))}
        sorted_emotions = sorted(prob_dict.items(), key=lambda x: x[1], reverse=True)

        top_emotion = sorted_emotions[0][0]
        top_conf = sorted_emotions[0][1]
        top_3 = [{"emotion": e, "score": s} for e, s in sorted_emotions[:3]]

        # Whisper transcription (dual-signal)
        transcription = None
        if self.whisper_model is not None:
            try:
                result = self.whisper_model.transcribe(audio_path)
                transcription = result.get("text", "").strip()
            except Exception:
                pass

        return {
            "acoustic_emotion": top_emotion,
            "confidence": round(top_conf, 4),
            "intensity": self._get_intensity(top_conf),
            "probabilities": prob_dict,
            "top_3": top_3,
            "transcription": transcription,
        }

    def predict_from_array(self, audio_array: np.ndarray, sr: int = 16000) -> dict:
        """Predict emotion from a numpy audio array."""
        import soundfile as sf

        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp:
            sf.write(tmp.name, audio_array, sr)
            result = self.predict(tmp.name)
            os.unlink(tmp.name)
            return result

    def get_emotion_vector(self, audio_path: str) -> np.ndarray:
        """
        Returns emotion as a 15-dim vector for the fusion layer.
        Order matches EMOTIONS list.
        """
        result = self.predict(audio_path)
        if "error" in result:
            return np.zeros(len(self.EMOTIONS))
        return np.array([result["probabilities"].get(e, 0.0) for e in self.EMOTIONS])


if __name__ == "__main__":
    print("Aether Voice Emotion Detector — Test Mode")
    print("=" * 50)
    detector = VoiceEmotionDetector(load_whisper=False)
    print("\\n✅ Detector loaded successfully!")
    print("   Use detector.predict('audio.wav') to classify audio")
    print("   Use detector.get_emotion_vector('audio.wav') for fusion layer")
'''

inference_path = Path("./phase_1b_voice_emotion/inference.py")
inference_path.parent.mkdir(parents=True, exist_ok=True)

with open(inference_path, "w") as f:
    f.write(inference_code)

print(f"✅ Inference utility → {inference_path}")


# ═══════════════════════════════════════════════
# COMPLETE ✅
# ═══════════════════════════════════════════════

DATASET_NAME = " + ".join([name for name, _, _ in datasets_loaded])

print("\n" + "═" * 60)
print("🎉 PHASE 1B COMPLETE — Aether Voice Emotion Model")
print("═" * 60)
print(f"""
   ✅ 15 Emotion Categories (with intensity levels)
   ✅ Feature Extractor: emotion2vec+ large (SOTA, frozen)
   ✅ Classification Head: MLP ({EMBEDDING_DIM}→{HIDDEN_DIM}→{HIDDEN_DIM//2}→{NUM_LABELS})
   ✅ Dataset: {DATASET_NAME}
   ✅ Accuracy: {final_acc:.4f}
   ✅ F1 (weighted): {final_f1_w:.4f}
   ✅ F1 (macro): {final_f1_m:.4f}
   ✅ Model saved: {final_path}
   ✅ Inference utility: {inference_path}

   🔌 Dual-Signal Output:
      Acoustic: emotion2vec+ → MLP → 15 emotions
      Text:     Whisper transcription → Phase 1A text model

   Next → Phase 1C: Unified Fusion Layer (text + voice)
""")


# ═══════════════════════════════════════════════
# SECTION 15: Save to Google Drive (Colab only)
# ═══════════════════════════════════════════════
# Colab's runtime disk is wiped on disconnect.
# Run this section to make your model permanent.

import shutil
from pathlib import Path

try:
    from google.colab import drive
    print("\n☁️  Saving to Google Drive...")
    drive.mount("/content/drive", force_remount=False)

    drive_path = Path("/content/drive/MyDrive/Aether_models/voice_emotion")
    drive_path.mkdir(parents=True, exist_ok=True)

    # Output paths used earlier in the script
    config_path = OUTPUT_DIR / "model_config.json"

    # Copy trained model files
    shutil.copytree(final_path,  drive_path / "final", dirs_exist_ok=True)
    shutil.copy2(config_path,    drive_path / "model_config.json")

    # Copy inference utility
    shutil.copy2(inference_path, drive_path.parent / "voice_inference.py")

    print(f"✅ Model weights → {drive_path / 'final'}")
    print(f"✅ Config        → {drive_path / 'model_config.json'}")
    print(f"✅ Inference     → {drive_path.parent / 'voice_inference.py'}")
    print("\n🔒 Permanently saved to Google Drive — safe to disconnect.")

except ImportError:
    print("\n💻 Not running on Colab — skipping Drive save.")
    print(f"   Model is saved locally at: {OUTPUT_DIR.resolve()}")
