# Midterm – Machine Learning  
Hands-On End-to-End Models (Classification, Regression & Clustering)

## 1. Repository Purpose
This repository contains my individual submission for the Machine Learning Midterm Examination (UTS) with the theme:

“Hands-On End-to-End Models in Machine Learning and Deep Learning”

The focus in this repository is on implementing traditional machine learning techniques to solve three different tasks:

1. Fraud detection (binary classification)
2. Song year prediction (regression)
3. Customer segmentation (clustering)

Each task is developed as a complete end-to-end pipeline, starting from data loading, preprocessing, model development, evaluation, and interpretation.

---

## 2. Project Overview

### 2.1 Objectives
The objectives of this project are as follows:

- Build complete end-to-end pipelines for each task:
  - Binary classification for fraud detection
  - Regression to predict song release year
  - Unsupervised clustering for customer segmentation

- Practice key machine learning concepts:
  - Data cleaning and preprocessing
  - Handling missing values and outliers
  - Addressing class imbalance (fraud detection)
  - Feature engineering or feature selection
  - Training and evaluating multiple ML models
  - Basic hyperparameter tuning
  - Interpretation and comparison of model results

### 2.2 Task Implementations

#### (1) Fraud Detection – Binary Classification
- Task: Predict whether a transaction is fraudulent (`isFraud = 1`).
- Dataset includes features such as amount, transaction time, product code, card information, and address.
- Model output includes:
  - Performance metrics on validation data
  - Probability predictions for `test_transaction.csv`

#### (2) Song Year Prediction – Regression
- Task: Predict the release year of a song using numerical audio features.
- The dataset contains:
  - A target value (year of release)
  - Multiple numeric input features derived from audio characteristics

#### (3) Customer Segmentation – Unsupervised Clustering
- Task: Group customers based on credit card usage and payment behavior.
- Dataset includes balance information, purchase frequency, cash advances, payments, credit limits, and tenure.
- The final step includes interpreting the meaning of each cluster.

---

## 3. Datasets

The datasets used in this project were provided as part of the midterm examination.

### 3.1 Fraud Detection Dataset
Files included:
- `train_transaction.csv`
  - Contains labeled online transaction records
  - Target variable: `isFraud` (0 = normal, 1 = fraudulent)

- `test_transaction.csv`
  - Contains unlabeled transaction data with the same feature structure
  - Used to generate fraud probability predictions

### 3.2 Regression Dataset
File:
- `midterm-regresi-dataset.csv`
  - First column contains the target value (year of release)
  - Remaining columns are numeric audio features representing timbre, spectral data, and other signal-derived metrics

### 3.3 Clustering Dataset
File:
- `clusteringmidterm.csv`
  - Contains customer attributes related to spending, payment, and card usage
  - Includes features such as BALANCE, PURCHASES, CASH_ADVANCE, CREDIT_LIMIT, PAYMENTS, TENURE, and more

---

## 4. Project Structure
(Note: Filenames and folder organization may be adjusted depending on the final implementation.)

```text
midterm-machine-learning/
├── data/
│   ├── train_transaction.csv
│   ├── test_transaction.csv
│   ├── midterm-regresi-dataset.csv
│   └── clusteringmidterm.csv
├── notebooks/
│   ├── 01_fraud_detection_classification_ml.ipynb
│   ├── 02_song_year_regression_ml.ipynb
│   └── 03_customer_clustering_ml.ipynb
├── src/
│   ├── preprocessing.py
│   ├── models_classification.py
│   ├── models_regression.py
│   └── models_clustering.py
├── requirements.txt
└── README.md
