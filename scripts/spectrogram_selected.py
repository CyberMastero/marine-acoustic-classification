import os
import librosa
import librosa.display
import matplotlib.pyplot as plt
import numpy as np

# =============================
# CONFIG
# =============================
CLIP_DIR = "test_audio_clips"
OUTPUT_DIR = "static/outputs/selected_clips"
SR = 20000
N_MELS = 64

SELECTED_CLIPS = [
    "clip_3.wav",
    "clip_27.wav",
    "clip_83.wav",
    "clip_150.wav"
]

# =============================
# SETUP
# =============================
os.makedirs(OUTPUT_DIR, exist_ok=True)

# =============================
# PROCESS
# =============================
for clip in SELECTED_CLIPS:
    clip_path = os.path.join(CLIP_DIR, clip)

    if not os.path.exists(clip_path):
        print(f"❌ Missing file: {clip}")
        continue

    y, sr = librosa.load(clip_path, sr=SR)

    # Normalize
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

    plt.figure(figsize=(6, 4))
    librosa.display.specshow(
        mel_db,
        sr=SR,
        hop_length=512,
        x_axis="time",
        y_axis="mel"
    )
    plt.colorbar(format="%+2.0f dB")
    plt.title(f"Mel Spectrogram – {clip}")
    plt.tight_layout()

    out_path = os.path.join(
        OUTPUT_DIR,
        clip.replace(".wav", ".png")
    )
    plt.savefig(out_path)
    plt.close()

    print(f"✅ Saved: {out_path}")

print("🎯 All selected spectrograms generated.")
