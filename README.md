# Viralytics – Short-Form Video Engagement Prediction System

Viralytics is a machine-learning web application that predicts the relative engagement potential of short-form videos based on their audio-visual characteristics. It analyses an uploaded video's visual quality, motion dynamics, audio profile, and temporal pacing, then uses a trained regression model to produce an engagement continuation score.

**MCA AIML Academic Mini-Project**

---

## Table of Contents

1. [Project Objectives](#project-objectives)
2. [System Architecture](#system-architecture)
3. [Technology Stack](#technology-stack)
4. [Dataset Description](#dataset-description)
5. [Feature Extraction Methodology](#feature-extraction-methodology)
6. [Machine Learning Algorithms](#machine-learning-algorithms)
7. [Evaluation Metrics](#evaluation-metrics)
8. [Installation Instructions (Windows)](#installation-instructions-windows)
9. [Running the Project](#running-the-project)
10. [Project Structure](#project-structure)
11. [Troubleshooting](#troubleshooting)
12. [Known Limitations](#known-limitations)

---

## Project Objectives

- Extract meaningful audio-visual features from short-form video files using OpenCV and Librosa.
- Train and compare multiple regression models (Linear Regression, Decision Tree, Random Forest) to predict the Engagement Continuation Rate (ECR).
- Build a Flask web application where users can upload a video and receive an engagement score with feature-based recommendations.
- Demonstrate computer vision, audio signal processing, feature engineering, regression modelling, and web deployment.

---

## System Architecture

```
Browser (HTML/CSS/JavaScript)
        |
        v
Flask Application (app.py)
        |
        +---- Video Upload & Validation
        |
        +---- Feature Extraction (OpenCV + Librosa + FFmpeg)
        |
        +---- Trained ML Model (Scikit-learn)
        |
        +---- Prediction & Recommendations
        |
        v
Results displayed in browser
```

---

## Technology Stack

| Component            | Technology                                 |
|----------------------|--------------------------------------------|
| Backend              | Python 3.12, Flask, Werkzeug               |
| Machine Learning     | Scikit-learn, Pandas, NumPy, Joblib        |
| Computer Vision      | OpenCV (`opencv-python`)                   |
| Audio Processing     | Librosa, SoundFile                         |
| Video Processing     | FFmpeg (via `imageio-ffmpeg`)              |
| Frontend             | HTML5, CSS3, Vanilla JavaScript, Jinja2    |
| Charts               | Chart.js (frontend), Matplotlib (training) |
| Storage              | Local filesystem, CSV, Joblib (.pkl)       |

---

## Dataset Description

**Source:** [SnapUGC / VQualA Engagement Dataset](https://github.com/dasongli1/SnapUGC_Engagement)

- **Training CSV:** `train_data.csv` – 106,192 records with columns: `Id`, `Title`, `Description`, `Download_link`, `ECR`
- **Target:** `ECR` (Engagement Continuation Rate) – a value between 0 and 1 representing the proportion of viewers who continue watching through the video.
- **Local subset:** ~657 downloaded video files in `video_test/`, each named by their dataset `Id`.

---

## Feature Extraction Methodology

26 numerical features are extracted from each video:

### Visual Features (6)
- Average brightness, contrast, saturation, sharpness
- Color distribution standard deviation, warmth ratio

### Motion Features (3)
- Average motion intensity (Farneback optical flow)
- Motion variation (standard deviation)
- Maximum motion burst

### Scene Change Features (2)
- Scene change count
- Scene changes per minute

### Audio Features (7)
- Has audio flag
- RMS energy, zero-crossing rate
- Spectral centroid, bandwidth, rolloff
- Silence ratio

### Basic / Temporal Features (8)
- Duration, FPS, width, height, aspect ratio, total frames, resolution

**Important:** The same extraction code (`src/feature_extraction.py`) is used for both training and inference to ensure consistency.

---

## Machine Learning Algorithms

| Algorithm              | Description                                              |
|------------------------|----------------------------------------------------------|
| Linear Regression      | Baseline linear model                                    |
| Decision Tree          | Regularised tree (max_depth=10, min_samples_leaf=5)      |
| Random Forest          | Ensemble of 100 trees (max_depth=12, sqrt features)      |

- **Split:** 70% train / 15% validation / 15% test (stratified by ECR bins)
- **Selection:** Best model chosen by lowest validation MAE
- **Scaling:** StandardScaler applied to all features

---

## Evaluation Metrics

| Metric | Description                                                |
|--------|------------------------------------------------------------|
| MAE    | Mean Absolute Error – average prediction error magnitude   |
| MSE    | Mean Squared Error – penalises larger errors more heavily  |
| RMSE   | Root Mean Squared Error – MSE in original scale            |
| R²     | Coefficient of determination (1.0 = perfect, 0.0 = mean)  |

A mean-prediction baseline is included for comparison.

---

## Installation Instructions (Windows)

### Prerequisites

- **Python 3.10+** installed and in PATH ([python.org](https://www.python.org/downloads/))
- **Git** (optional, for cloning)

### Step 1: Open Command Prompt

```cmd
cd C:\Users\YourUsername\Desktop\Viralytics
```

### Step 2: Create Virtual Environment (Recommended)

```cmd
python -m venv venv
venv\Scripts\activate
```

### Step 3: Install Dependencies

```cmd
pip install -r requirements.txt
```

This installs Flask, Scikit-learn, OpenCV, Librosa, SoundFile, imageio-ffmpeg (bundles FFmpeg), Matplotlib, and all required dependencies.

> **Note on FFmpeg:** The `imageio-ffmpeg` package bundles an FFmpeg binary automatically. No separate FFmpeg installation is required.

### Step 4: Verify Installation

```cmd
python -c "import cv2, librosa, flask, sklearn; print('All OK')"
```

---

## Running the Project

### Step 1: Verify Videos

Scans `video_test/`, validates each video is openable, and matches IDs to `train_data.csv`:

```cmd
python scripts/verify_videos.py
```

### Step 2: Extract Features

Extracts all 26 features from each verified video and saves to `data/processed/video_features.csv`:

```cmd
python scripts/extract_dataset_features.py
```

> This takes **several minutes** (~2–8 seconds per video depending on hardware).

### Step 3: Train Models

Trains all three regression models, selects the best, and saves model + charts:

```cmd
python src/train_model.py
```

### Step 4: Start the Web Application

```cmd
python app.py
```

Open your browser and go to: **http://127.0.0.1:5000**

### Quick Reference (Full Pipeline)

```cmd
python -m venv venv
venv\Scripts\activate
pip install -r requirements.txt
python scripts/verify_videos.py
python scripts/extract_dataset_features.py
python src/train_model.py
python app.py
```

---

## Project Structure

```
Viralytics/
├── app.py                              # Flask web application
├── config.py                           # Centralised configuration
├── requirements.txt                    # Python dependencies
├── README.md                           # This file
├── train_data.csv                      # Original SnapUGC dataset CSV
│
├── data/
│   ├── raw/                            # (empty – videos are in video_test/)
│   └── processed/
│       └── video_features.csv          # Extracted features (generated)
│
├── models/
│   ├── best_model.pkl                  # Trained model bundle (generated)
│   └── model_metrics.json              # Training metrics (generated)
│
├── src/
│   ├── __init__.py
│   ├── feature_extraction.py           # Reusable feature extraction module
│   ├── dataset_preparation.py          # Dataset loading, matching, cleaning
│   ├── train_model.py                  # Model training and evaluation
│   ├── predict.py                      # Prediction pipeline
│   └── recommendations.py             # Feature-based recommendations
│
├── templates/
│   ├── base.html                       # Base template with navigation
│   ├── index.html                      # Dashboard / home page
│   ├── upload.html                     # Video upload page
│   ├── results.html                    # Analysis results page
│   ├── model_performance.html          # Model metrics and charts
│   └── about.html                      # Project information
│
├── static/
│   ├── css/style.css                   # Main stylesheet
│   ├── js/main.js                      # Client-side JavaScript
│   ├── uploads/                        # Uploaded videos (runtime)
│   └── results/                        # Training charts (generated)
│
├── scripts/
│   ├── verify_videos.py                # Video verification script
│   └── extract_dataset_features.py     # Batch feature extraction
│
├── tests/
│   ├── test_features.py                # Feature extraction tests
│   ├── test_prediction.py              # Prediction pipeline tests
│   └── test_upload.py                  # Upload validation tests
│
└── video_test/                         # ~657 downloaded dataset videos
    ├── <video_id_1>.mp4
    ├── <video_id_2>.mp4
    └── ...
```

---

## Troubleshooting

| Issue | Solution |
|-------|----------|
| `ModuleNotFoundError: No module named 'librosa'` | Run `pip install librosa soundfile imageio-ffmpeg` |
| `ModuleNotFoundError: No module named 'cv2'` | Run `pip install opencv-python` |
| Feature extraction is very slow | Reduce `target_sample_fps` in `config.py` to 2 or 1 |
| `Model not found` error on upload | Run `python src/train_model.py` first |
| `Feature CSV not found` error | Run `python scripts/extract_dataset_features.py` first |
| Videos not matched to CSV | Ensure video filenames match the `Id` column in `train_data.csv` |
| Upload rejected | Only MP4 and MOV files under 100 MB are accepted |
| Flask won't start | Ensure port 5000 is not in use, or change port in `app.py` |

---

## Known Limitations

1. **Small training set:** Only ~657 videos out of 106K+ are available locally. Model generalization is limited.
2. **No semantic understanding:** The system cannot understand video content, trends, creator reputation, or platform algorithms.
3. **Score interpretation:** The engagement score is a model-estimated metric, not a probability of virality or guaranteed view count.
4. **No deep learning:** Feature engineering uses handcrafted descriptors (brightness, motion, audio energy), not learned representations.
5. **Processing time:** Feature extraction takes 2–8 seconds per video depending on duration and hardware.
6. **Academic scope:** This application is designed for academic demonstration and is not production-hardened.

---

## License

Academic project – not for commercial distribution.

**Dataset Credit:** [SnapUGC / VQualA Engagement Dataset](https://github.com/dasongli1/SnapUGC_Engagement)

