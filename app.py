import os
import uuid
import numpy as np
import librosa
import librosa.display
import joblib
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from flask import Flask, render_template, request, Response, jsonify
from werkzeug.utils import secure_filename

app = Flask(__name__)

UPLOAD_FOLDER = "uploads"
STATIC_FOLDER = "static"

os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(STATIC_FOLDER, exist_ok=True)

ALLOWED_EXTENSIONS = {"wav", "mp3"}

CLASSES = ["Background / Noise", "Marine Biological Sound"]

# ===== SAFE MODEL LOAD =====
try:
    ML_MODEL = joblib.load("model/marine_audio_classifier.pkl")
except Exception as e:
    ML_MODEL = None
    print("⚠ Model not loaded:", e)

history = []

# ================= FILE VALIDATION =================
def allowed_file(filename):
    return "." in filename and filename.rsplit(".", 1)[1].lower() in ALLOWED_EXTENSIONS

# ================= AUDIO =================
@app.route('/audio/<filename>')
def serve_audio(filename):
    path = os.path.join(UPLOAD_FOLDER, filename)

    if not os.path.exists(path):
        return "File not found", 404

    mime = "audio/mpeg" if filename.endswith(".mp3") else "audio/wav"
    return Response(open(path, "rb"), mimetype=mime)

def load_audio(path):
    y, sr = librosa.load(path, sr=22050, duration=10)
    y, _ = librosa.effects.trim(y)

    if np.max(np.abs(y)) != 0:
        y = y / np.max(np.abs(y))

    return y, sr

# ================= EVENTS =================
def extract_events(y, sr):
    rms = librosa.feature.rms(y=y)[0]
    times = librosa.times_like(rms, sr=sr)

    threshold = np.mean(rms) + 0.3 * np.std(rms)

    events = []
    active = False

    for t, r in zip(times, rms):
        if r > threshold and not active:
            start = t
            active = True
        elif r <= threshold and active:
            end = t
            events.append((round(start,2), round(end,2), round(end-start,2)))
            active = False

    return events

# ================= VISUALS =================
def generate_waveform(y, sr, events, uid):
    t = np.linspace(0, len(y)/sr, len(y))

    plt.figure(figsize=(10,4))
    plt.plot(t, y)

    for s,e,_ in events:
        plt.axvspan(s, e, color='red', alpha=0.3)

    path = os.path.join(STATIC_FOLDER, f"wave_{uid}.png")
    plt.savefig(path)
    plt.close()

    return path

def generate_spectrogram(y, sr, uid):
    mel = librosa.feature.melspectrogram(y=y, sr=sr)
    mel_db = librosa.power_to_db(mel, ref=np.max)

    plt.figure(figsize=(10,4))
    librosa.display.specshow(mel_db, sr=sr, x_axis='time', y_axis='mel')
    plt.colorbar()

    path = os.path.join(STATIC_FOLDER, f"mel_{uid}.png")
    plt.savefig(path)
    plt.close()

    return path

# ================= ML =================
def ml_predict(y, sr):
    if ML_MODEL is None:
        return "Model not loaded", 0, np.zeros(13)

    mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=13)
    features = np.mean(mfcc, axis=1)

    probs = ML_MODEL.predict_proba(features.reshape(1,-1))[0]
    idx = np.argmax(probs)

    return CLASSES[idx], float(probs[idx]), features

# ================= XAI =================
def extract_xai(y, sr, events):
    fft = np.fft.fft(y)
    freqs = np.fft.fftfreq(len(fft), 1/sr)

    dominant_freq = abs(freqs[np.argmax(np.abs(fft))])
    duration = len(y)/sr

    density = len(events)/duration if duration else 0
    energy = np.mean(y**2)

    return round(dominant_freq,2), round(density,2), round(energy,4)

# ================= MONITORING =================
def compute_health():
    if not history:
        return "No Data", 0, "Low", []

    confs = [h["confidence"] for h in history]
    marine = sum(1 for h in history if "Marine" in h["prediction"])

    percent = (marine/len(history))*100
    stability_val = np.std(confs)

    stability = "High" if stability_val < 0.05 else "Moderate" if stability_val < 0.15 else "Low"
    health = "GOOD" if percent > 60 else "MODERATE" if percent > 30 else "LOW"

    alerts = []
    if percent > 70:
        alerts.append("🚨 High marine activity")
    if stability == "Low":
        alerts.append("⚠ Model unstable")

    return health, round(percent,2), stability, alerts

# ================= DRIFT =================
def detect_drift(current_feat):
    if len(history) < 3:
        return "No Data", 0

    past = [np.array(h["features"]) for h in history[:-1]]
    dist = np.mean([np.linalg.norm(current_feat - p) for p in past])

    level = "LOW" if dist < 20 else "MODERATE" if dist < 50 else "HIGH"

    return level, round(dist,2)

# ================= SOS =================
def detect_sos(events, density, energy):
    score = len(events)*0.4 + density*30 + energy*100

    if score > 120:
        return "HIGH RISK", "🚨 Possible underwater anomaly"
    elif score > 70:
        return "MODERATE RISK", "⚠ Unusual activity"
    else:
        return "LOW RISK", "Normal"

# ================= GLOBAL CONTEXT =================
@app.context_processor
def inject_global_data():
    health, marine_percent, stability, alerts = compute_health()

    drift_level, drift_score = ("No Data", 0)
    sos_level, sos_message = None, None

    if history:
        last_feat = np.array(history[-1]["features"])
        drift_level, drift_score = detect_drift(last_feat)

        last = history[-1]
        sos_level = last.get("sos_level")
        sos_message = last.get("sos_message")

    return dict(
        monitoring={
            "health": health,
            "marine_percent": marine_percent,
            "stability": stability,
            "alerts": alerts
        },
        drift={
            "level": drift_level,
            "score": drift_score
        },
        sos_level=sos_level,
        sos_message=sos_message
    )

# ================= ROUTES =================
@app.route("/")
def dashboard():
    return render_template("dashboard.html")

@app.route("/analysis", methods=["GET","POST"])
def analysis():
    results = []
    error_msg = None

    if request.method == "POST":
        files = request.files.getlist("audio")

        for file in files:
            if file and allowed_file(file.filename):
                try:
                    uid = str(uuid.uuid4())
                    filename = secure_filename(file.filename)
                    ext = filename.rsplit(".", 1)[1].lower()
                    fname = f"{uid}.{ext}"

                    path = os.path.join(UPLOAD_FOLDER, fname)
                    file.save(path)

                    y, sr = load_audio(path)
                    events = extract_events(y, sr)

                    waveform = generate_waveform(y, sr, events, uid)
                    spec = generate_spectrogram(y, sr, uid)

                    pred, conf, feat = ml_predict(y, sr)
                    freq, density, energy = extract_xai(y, sr, events)

                    drift_level, drift_score = detect_drift(feat)
                    sos_level, sos_msg = detect_sos(events, density, energy)

                    history.append({
                        "prediction": pred,
                        "confidence": conf,
                        "events": len(events),
                        "features": feat.tolist(),
                        "file": fname,
                        "sos_level": sos_level,
                        "sos_message": sos_msg
                    })

                    results.append({
                        "file": fname,
                        "prediction": pred,
                        "confidence": round(conf*100,2),
                        "waveform": waveform,
                        "spectrogram": spec,
                        "events": events,
                        "drift_level": drift_level,
                        "sos_level": sos_level
                    })

                except Exception as e:
                    error_msg = str(e)

    return render_template("analysis.html", results=results, error=error_msg)

@app.route("/realtime", methods=["POST"])
def realtime():
    file = request.files.get("audio")

    if not file:
        return jsonify({"error": "No file"})

    try:
        uid = str(uuid.uuid4())
        path = os.path.join(UPLOAD_FOLDER, f"{uid}.wav")
        file.save(path)

        y, sr = load_audio(path)
        pred, conf, feat = ml_predict(y, sr)

        history.append({
            "prediction": pred,
            "confidence": conf,
            "features": feat.tolist(),
            "events": 0
        })

        return jsonify({
            "prediction": pred,
            "confidence": round(conf * 100, 2)
        })

    except:
        return jsonify({"error": "Processing failed"})

@app.route("/reports")
def reports():
    total = len(history)
    marine = sum(1 for h in history if "Marine" in h["prediction"])
    noise = total - marine

    avg_conf = round(np.mean([h["confidence"] for h in history]),2) if history else 0

    bar_labels = list(range(1, total+1))
    bar_values = [h.get("events", 0) for h in history]

    return render_template(
        "reports.html",
        total=total,
        marine=marine,
        noise=noise,
        avg_conf=avg_conf,
        bar_labels=bar_labels,
        bar_values=bar_values
    )

@app.route("/insights")
def insights():
    return render_template("insights.html", history=history)

@app.route("/monitoring")
def monitoring_page():
    return render_template("monitoring.html")

@app.route("/drift")
def drift_page():
    if not history:
        return render_template("drift.html", level="No Data", score=0)

    last_feat = np.array(history[-1]["features"])
    level, score = detect_drift(last_feat)

    return render_template("drift.html", level=level, score=score)

if __name__ == "__main__":
    app.run(debug=True)