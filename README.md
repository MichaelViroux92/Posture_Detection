# Posture Prediction using SpinePose, XGBoost VS CNN

This project focuses on predicting human posture from images using both classical machine learning and deep learning techniques. The system leverages **SpinePose** for extracting spine landmarks + **XGBoost** for tabular feature-based classification versus **CNNs** for end-to-end image-based prediction. Model experimentation and tracking are handled using **MLflow**.

---

## Project Overview

### Objectives
- Annotate posture classes on image data
- Extract spine-related pose features using SpinePose
- Apply classical ML models (e.g., XGBoost) on engineered features
- Train a CNN for direct image classification
- Track experiments and metrics using MLflow
- Compare results of XGBoost VS CNN

---

## Notebooks Explained

### `annotate_data.ipynb`
- Provides an interactive tool to annotate image data with posture labels.
- Saves annotations in a format compatible with other notebooks.

### `get_features.ipynb`
- Extracts pose landmarks using **SpinePose**.
- Performs additional feature engineering such as:
  - Calculating joint angles
  - Normalizing positions
  - Measuring distances between key landmarks
- Outputs a clean, tabular dataset ready for machine learning.

### `xgboost_model.ipynb`
- Loads the engineered features dataset.
- Trains a **XGBoost** classifier to predict posture classes.
- Performs hyperparameter tuning and evaluation.
- Logs:
  - Accuracy, precision, recall
  - Feature importances
  - Model parameters and artifacts  
  using **MLflow**.

### `cnn_model.ipynb`
- Preprocesses raw images (resizing, augmentations, normalization).
- Defines and trains a **Convolutional Neural Network** (CNN) for direct image classification.
- Logs all relevant training metrics and models via **MLflow**.

---