# Marine Acoustic Classification System

## Overview
This project is a web-based system for underwater acoustic signal classification and event detection. It combines signal processing, machine learning (Random Forest), and deep learning (CNN) to analyze marine audio and identify meaningful acoustic patterns.

The system is designed for applications in marine monitoring, environmental analysis, and bioacoustic research.

---

## Key Highlights
- Hybrid ML + DL approach (Random Forest + CNN)
- RMS-based acoustic event detection
- Interactive web dashboard for visualization
- Real-time audio analysis capability (extendable)

---

## Features
- Marine sound classification
- Acoustic event detection using RMS energy
- Waveform and spectrogram visualization
- Confidence score prediction
- User-friendly web interface

---

## Methodology
1. User uploads an audio file via web interface  
2. Audio preprocessing (noise reduction, normalization, segmentation)  
3. Feature extraction:
   - MFCC (for Random Forest)
   - Spectrogram (for CNN)
4. Classification:
   - Random Forest for structured features  
   - CNN for deep feature learning  
5. RMS-based detection identifies acoustic events  
6. Results displayed with visualizations and confidence scores  

---

## Tech Stack
- Python, Flask  
- TensorFlow, Scikit-learn  
- Librosa, OpenCV  
- HTML, CSS, JavaScript  

---

## Project Structure
module5_dashboard/
│-- app.py
│-- train_model.py
│-- templates/
│-- static/
│-- scripts/
│-- requirements.txt
│-- README.md

---

## How to Run
```bash
pip install -r requirements.txt
python app.py