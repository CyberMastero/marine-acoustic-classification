# Marine Acoustic Classification System

## Overview
This project presents a web-based system for underwater acoustic signal classification and event detection. It integrates audio signal processing, machine learning (Random Forest), and deep learning (CNN) to analyze marine sounds and identify significant acoustic events.

## Objectives
- Detect and classify marine acoustic signals
- Visualize waveform and spectrogram representations
- Identify acoustic events using RMS-based detection
- Provide interactive dashboard and analysis reports

## Methodology
1. Audio files are uploaded via a web interface
2. Preprocessing includes noise reduction, normalization, and segmentation
3. Feature extraction using MFCC and spectrograms
4. Classification using:
   - Random Forest (MFCC features)
   - CNN (spectrogram images)
5. Acoustic event detection using RMS energy
6. Results displayed with confidence scores and visualizations

## Technologies Used
- Python
- Flask
- TensorFlow
- Scikit-learn
- Librosa
- HTML, CSS, JavaScript

## Project Structure

## Features
- Marine sound classification
- Acoustic event detection
- Waveform and spectrogram visualization
- Interactive dashboard

## Future Enhancements
- Real-time audio monitoring
- Advanced deep learning models (CRNN, Transformers)
- Larger and diverse datasets
- Species-level classification

## Project Status
Completed and functional