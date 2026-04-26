import os
import numpy as np
import librosa
import joblib

from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline

DATASET_ROOT = "data/kaggle/DDDDD"
MODEL_DIR = "model"
MODEL_PATH = os.path.join(MODEL_DIR, "marine_audio_classifier.pkl")

os.makedirs(MODEL_DIR, exist_ok=True)

def extract_mfcc(file_path):
    try:
        y, sr = librosa.load(file_path, sr=22050)
        mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=13)
        return np.mean(mfcc, axis=1)
    except Exception as e:
        print(f"Skipping {file_path}: {e}")
        return None

def collect_data():
    X, y = [], []

    # ---------- MARINE (label = 1) ----------
    marine_root = os.path.join(DATASET_ROOT, "Marine Animals")
    for root, _, files in os.walk(marine_root):
        for f in files:
            if f.lower().endswith(".wav"):
                path = os.path.join(root, f)
                feat = extract_mfcc(path)
                if feat is not None:
                    X.append(feat)
                    y.append(1)

    # ---------- NOISE (label = 0) ----------
    noise_folders = [
        "Natural Sounds",
        "Vessels",
        "Other anthropogenic"
    ]

    for folder in noise_folders:
        noise_root = os.path.join(DATASET_ROOT, folder)
        for root, _, files in os.walk(noise_root):
            for f in files:
                if f.lower().endswith(".wav"):
                    path = os.path.join(root, f)
                    feat = extract_mfcc(path)
                    if feat is not None:
                        X.append(feat)
                        y.append(0)

    return np.array(X), np.array(y)

def train_and_save():
    print("Collecting data...")
    X, y = collect_data()

    print(f"Total samples: {len(X)}")
    print(f"Marine: {np.sum(y == 1)} | Noise: {np.sum(y == 0)}")

    model = Pipeline([
        ("scaler", StandardScaler()),
        ("clf", LogisticRegression(max_iter=1000))
    ])

    print("Training model...")
    model.fit(X, y)

    joblib.dump(model, MODEL_PATH)
    print(f"Model saved to {MODEL_PATH}")

if __name__ == "__main__":
    train_and_save()
