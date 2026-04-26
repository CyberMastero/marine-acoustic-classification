import os
import numpy as np
import librosa
import matplotlib.pyplot as plt
from tensorflow.keras.models import load_model

# ======================================
# CONFIG
# ======================================
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

MODEL_PATH = os.path.join(BASE_DIR, "model", "cnn_mel_model.h5")
CLIPS_DIR = os.path.join(BASE_DIR, "test_audio_clips")
OUTPUT_DIR = os.path.join(BASE_DIR, "static", "outputs")

SR = 20000
N_MELS = 64
EXPECTED_FRAMES = 9
CLIP_DURATION = 0.25  # seconds
THRESHOLD = 0.60

os.makedirs(OUTPUT_DIR, exist_ok=True)

# ======================================
# LOAD MODEL
# ======================================
model = load_model(MODEL_PATH)

# ======================================
# PREPROCESS FUNCTION
# ======================================
def preprocess_clip(filepath):
    y, _ = librosa.load(filepath, sr=SR)

    if np.max(np.abs(y)) > 0:
        y = y / np.max(np.abs(y))

    mel = librosa.feature.melspectrogram(
        y=y,
        sr=SR,
        n_fft=1024,
        hop_length=512,
        n_mels=N_MELS
    )

    mel_db = librosa.power_to_db(mel, ref=np.max)

    frames = mel_db.shape[1]
    if frames > EXPECTED_FRAMES:
        mel_db = mel_db[:, :EXPECTED_FRAMES]
    else:
        mel_db = np.pad(
            mel_db,
            ((0, 0), (0, EXPECTED_FRAMES - frames)),
            mode="constant"
        )

    mel_db = mel_db[np.newaxis, ..., np.newaxis]
    return mel_db

# ======================================
# LOAD CLIPS
# ======================================
clips = sorted([
    f for f in os.listdir(CLIPS_DIR)
    if f.endswith(".wav")
])

marine_probs = []

for clip in clips:
    clip_path = os.path.join(CLIPS_DIR, clip)
    X = preprocess_clip(clip_path)
    probs = model.predict(X, verbose=0)[0]

    marine_probs.append(probs[1])  # index 1 = Marine Biological Sound

marine_probs = np.array(marine_probs)

# ======================================
# TIME AXIS
# ======================================
time_axis = np.arange(len(marine_probs)) * CLIP_DURATION

# ======================================
# PLOT
# ======================================
plt.figure(figsize=(12, 4))

plt.plot(
    time_axis,
    marine_probs,
    color="#00e5ff",
    linewidth=1.5,
    label="Marine Probability"
)

plt.axhline(
    y=THRESHOLD,
    color="red",
    linestyle="--",
    linewidth=1.5,
    label="Detection Threshold (0.6)"
)

plt.ylim(0, 1.05)
plt.xlabel("Time (seconds)")
plt.ylabel("Marine Probability")
plt.title("Marine Biological Activity Timeline")
plt.grid(alpha=0.3)
plt.legend()

# ======================================
# SAVE + SHOW
# ======================================
output_path = os.path.join(OUTPUT_DIR, "marine_activity_timeline.png")
plt.savefig(output_path, dpi=150, bbox_inches="tight")
plt.show()

print(f"✅ Timeline saved at: {output_path}")
