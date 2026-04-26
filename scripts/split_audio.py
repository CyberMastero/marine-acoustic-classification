import librosa
import soundfile as sf
import os

# =============================
# CONFIG (MATCHES YOUR FILE)
# =============================
INPUT_WAV = "humpback-whale-megaptera-novaeangliae.wav"
OUT_DIR = "test_audio_clips"
CLIP_DURATION = 0.25   # seconds
SR = 20000

# =============================
# CHECK INPUT
# =============================
if not os.path.exists(INPUT_WAV):
    raise FileNotFoundError(
        f"❌ Input audio '{INPUT_WAV}' not found in current directory."
    )

# =============================
# LOAD AUDIO
# =============================
y, sr = librosa.load(INPUT_WAV, sr=SR)

samples_per_clip = int(CLIP_DURATION * sr)

# =============================
# SPLIT
# =============================
os.makedirs(OUT_DIR, exist_ok=True)

count = 0
for i in range(0, len(y) - samples_per_clip, samples_per_clip):
    clip = y[i:i + samples_per_clip]
    sf.write(os.path.join(OUT_DIR, f"clip_{count}.wav"), clip, sr)
    count += 1

print(f"✅ Done: {count} clips created in '{OUT_DIR}'")
