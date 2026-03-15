# ML Deployment Crops

![Language](https://img.shields.io/badge/Language-Jupyter%20Notebook-DA5B0B?style=flat-square) ![Stars](https://img.shields.io/github/stars/Devanik21/ML-deployment-crops?style=flat-square&color=yellow) ![Forks](https://img.shields.io/github/forks/Devanik21/ML-deployment-crops?style=flat-square&color=blue) ![Author](https://img.shields.io/badge/Author-Devanik21-black?style=flat-square&logo=github) ![Status](https://img.shields.io/badge/Status-Active-brightgreen?style=flat-square)

> Precision agriculture meets predictive intelligence — forecast crop suitability and yield from soil, climate, and nutrient parameters.

---

**Topics:** `agricultural-ai` · `crop-prediction` · `deep-learning` · `docker` · `fastapi` · `machine-learning` · `mlops` · `model-deployment` · `neural-networks` · `production-ml`

## Overview

This application brings machine learning directly into the hands of farmers, agronomists, and agricultural researchers. By feeding a trained ensemble model with soil composition data, atmospheric conditions, and nutrient profiles, the system returns a ranked recommendation of the most suitable crops for the given environment — along with confidence scores and feature-level explanations that make the reasoning transparent and actionable.

The interface is built entirely in Streamlit, allowing zero-configuration deployment on any machine or cloud platform. The prediction pipeline is backed by a RandomForest or XGBoost model trained on the Crop Recommendation Dataset, with preprocessing handled by a scikit-learn Pipeline object that ensures consistent feature scaling between training and inference. A CSV batch mode allows agronomists to analyse entire fields in one operation rather than row by row.

Beyond prediction, the application includes an EDA module with interactive visualisations of the training dataset — heatmaps of feature correlations, box plots of nutrient distributions by crop type, and scatter plots of temperature vs rainfall coloured by crop category — giving users the context to interpret model outputs with domain knowledge.

---

## Motivation

Crop selection is one of the highest-stakes decisions in agriculture. A mismatch between crop requirements and soil or climatic conditions leads to yield loss, water waste, and financial hardship — particularly for smallholder farmers without access to agronomic advisory services. This project was built to democratise access to data-driven crop selection, putting a well-trained predictive model into a form that requires no technical expertise to use.

---

## Architecture

The system follows a classic ML inference pipeline:

```
User Input (N, P, K, pH, temp, humidity, rainfall)
        │
  Preprocessing Pipeline (StandardScaler / MinMaxScaler)
        │
  Trained Classifier (RandomForest / XGBoost / SVM)
        │
  Top-3 Crop Recommendations + Confidence Scores
        │
  Feature Importance Chart (SHAP / model.feature_importances_)
```

Model training is done offline in a Jupyter notebook; the serialised model (`joblib`) is loaded at app startup. All input validation is handled by Streamlit's built-in widget constraints before any inference is triggered.

---

## Features

### Multi-nutrient Input Interface
Sliders and number inputs for Nitrogen (N), Phosphorus (P), Potassium (K), soil pH, temperature, humidity, and rainfall — all with validated ranges derived from the training data distribution.

### Ensemble Classifier Inference
A trained RandomForest or XGBoost model outputs a ranked list of crop recommendations with per-class probability scores, giving users confidence in the top suggestion.

### SHAP Feature Explanations
Each prediction is accompanied by a SHAP waterfall chart showing which input features most strongly influenced the recommendation — making the model interpretable for non-technical users.

### Interactive EDA Dashboard
An embedded exploratory data analysis tab with correlation heatmaps, feature distribution plots, and crop-wise box plots for soil and climate parameters.

### CSV Batch Prediction
Upload a CSV file with multiple field samples; the app processes all rows through the same pipeline and returns a downloadable results table.

### Model Comparison Panel
Switch between trained models (RF, XGBoost, SVM, Logistic Regression) and compare their accuracy, precision, recall, and F1-score on the held-out test set.

### Seasonal Awareness Mode
Optional season input (Kharif / Rabi / Zaid) adjusts feature weighting to account for seasonal temperature and rainfall norms.

### Offline-First Architecture
No external API calls required at inference time — all model weights are bundled and loaded locally, ensuring the app works without internet access.

---

## Tech Stack

| Library / Tool | Role | Why This Choice |
|---|---|---|
| **Streamlit** | Web application framework | Zero-config Python-native UI with widget state management |
| **scikit-learn** | ML model training and pipeline | Pipeline, RandomForestClassifier, preprocessing transformers |
| **XGBoost** | Gradient-boosted tree classifier | High-accuracy alternative to RandomForest on tabular data |
| **SHAP** | Model explainability | TreeExplainer for feature importance visualisation |
| **pandas** | Data manipulation | CSV loading, filtering, batch processing |
| **NumPy** | Numerical operations | Array operations for feature vectors |
| **Plotly** | Interactive visualisation | Bar charts, scatter plots, heatmaps with zoom/hover |
| **joblib** | Model serialisation | Fast pickle-based model save/load |

> **Key packages detected in this repo:** `streamlit` · `pandas` · `numpy` · `scikit-learn` · `joblib` · `matplotlib` · `seaborn` · `xgboost` · `lightgbm` · `tensorflow`

---

## Getting Started

### Prerequisites

- Python 3.9+ (or Node.js 18+ for TypeScript/JS projects)
- `pip` or `npm` package manager
- Relevant API keys (see Configuration section)

### Installation

```bash
# 1. Clone and enter the repository
git clone https://github.com/Devanik21/ADV-crop-prediction-app.git
cd ADV-crop-prediction-app

# 2. Create and activate a virtual environment
python -m venv venv
source venv/bin/activate   # Windows: venv\Scripts\activate

# 3. Install dependencies
pip install streamlit scikit-learn xgboost shap pandas numpy plotly joblib

# 4. Launch the application
streamlit run app.py
```

---

## Usage

```bash
# Run the app
streamlit run app.py

# Batch prediction from CLI (if batch_predict.py is present)
python batch_predict.py --input field_samples.csv --output results.csv

# Retrain the model with new data
python train_model.py --data crop_data.csv --model rf --save model.pkl
```

---

## Configuration

| Variable | Default | Description |
|---|---|---|
| `MODEL_PATH` | `model.pkl` | Path to the serialised classifier |
| `SCALER_PATH` | `scaler.pkl` | Path to the fitted preprocessing scaler |
| `TOP_K_CROPS` | `3` | Number of top recommendations to display |
| `SHAP_ENABLED` | `True` | Toggle SHAP explanation computation |

> Copy `.env.example` to `.env` and populate all required values before running.

---

## Project Structure

```
ML-deployment-crops/
├── README.md
├── requirements.txt
├── app.py
├── crop prediction.ipynb
├── Crop_recommendation.csv
└── ...
```

---

## Roadmap

- [ ] Integration with real-time soil sensor APIs (IoT-connected field probes)
- [ ] Geo-aware recommendations using satellite NDVI data for the input coordinates
- [ ] Yield quantity prediction (regression) alongside crop-type classification
- [ ] Mobile-optimised Streamlit layout for use on smartphones in the field
- [ ] Multi-language UI support (Hindi, Bengali, Tamil) for regional accessibility

---

## Contributing

Contributions, issues, and feature requests are welcome. Please:

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/your-feature`)
3. Commit your changes (`git commit -m 'feat: add your feature'`)
4. Push to your branch (`git push origin feature/your-feature`)
5. Open a Pull Request

Please follow conventional commit messages and ensure any new code is documented.

---

## Notes

The model was trained on the publicly available Crop Recommendation Dataset. Performance metrics (Accuracy ~99% on test set for RF) reflect the dataset's characteristics and may vary on out-of-distribution field data. Always validate recommendations with local agronomic expertise.

---

## Author

**Devanik Debnath**  
B.Tech, Electronics & Communication Engineering  
National Institute of Technology Agartala

[![GitHub](https://img.shields.io/badge/GitHub-Devanik21-black?style=flat-square&logo=github)](https://github.com/Devanik21)
[![LinkedIn](https://img.shields.io/badge/LinkedIn-devanik-blue?style=flat-square&logo=linkedin)](https://www.linkedin.com/in/devanik/)

---

## License

This project is open source and available under the [MIT License](LICENSE).

---

*Crafted with curiosity, precision, and a belief that good software is worth building well.*
