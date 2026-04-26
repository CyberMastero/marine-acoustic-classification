import os
import librosa
import numpy as np
import pandas as pd
from tensorflow.keras.models import load_model

MODEL = load_model("model/cnn_mel_model.h5")

SR = 20000
N_MELS = 64
EXPECTED_FRAMES = 9

rows = []

for i, file in enumerate(sorted(os.listdir("test_audio_clips"))):
    if not file.endswith(".wav"):
        continue

    y, _ = librosa.load(f"test_audio_clips/{file}", sr=SR)
    mel = librosa.feature.melspectrogram(y=y, sr=SR, n_mels=N_MELS)
    mel_db = librosa.power_to_db(mel, ref=1.0)
    mel_db = np.clip(mel_db, -80, -20)

    if mel_db.shape[1] < EXPECTED_FRAMES:
        mel_db = np.pad(mel_db, ((0,0),(0,EXPECTED_FRAMES-mel_db.shape[1])))
    else:
        mel_db = mel_db[:, :EXPECTED_FRAMES]

    X = mel_db[np.newaxis, ..., np.newaxis]
    prob = MODEL.predict(X, verbose=0)[0][1]

    rows.append({
        "time_sec": i * 0.25,
        "marine_prob": prob
    })

pd.DataFrame(rows).to_csv("batch_predictions.csv", index=False)
print("✅ batch_predictions.csv created")
