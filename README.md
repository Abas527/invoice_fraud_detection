# Invoice Fraud Detection — Explainable ML Pipeline

**Invoice Fraud Detection** — end-to-end system to detect suspicious invoices using OCR → structured extraction → hybrid anomaly + supervised models (One-Class SVM + XGBoost). Includes SHAP explanations, Streamlit dashboard, FastAPI inference, DVC for data versioning, and MLflow experiment tracking.

![demo-gif](docs/demo.gif)  <!-- replace with your actual gif -->

## Table of Contents

* [Project Overview](#project-overview)
* [Quickstart (2 minutes)](#quickstart-2-minutes)
* [Installation](#installation)
* [Usage](#usage)
* [Project Structure](#project-structure)
* [Dataset & Schema](#dataset--schema)
* [Modeling & Features](#modeling--features)
* [Evaluation & Results](#evaluation--results)
* [Explainability (SHAP)](#explainability-shap)
* [Reproducibility & MLOps](#reproducibility--mlops)
* [Deployment](#deployment)
* [Limitations & Future Work](#limitations--future-work)
* [Contributing](#contributing)
* [License & Contact](#license--contact)

---

## Project Overview

Invoices are a major fraud vector for companies. This project provides:

* Robust PDF text extraction (pdfplumber + Tesseract fallback)
* Structured CSV output for feature engineering
* Hybrid fraud detection: **One-Class SVM** (novel/unseen anomalies) + **XGBoost** (known fraud patterns)
* Explainability via **SHAP** for per-invoice explanations
* Streamlit dashboard for upload, inference, and visual explanations
* MLOps: **DVC** for data, **MLflow** for experiments, **Docker** for deployment

---

## Quickstart (2 minutes)

```bash
# 1) clone
git clone https://github.com/YOUR_USERNAME/invoice_fraud_detection.git
cd invoice_fraud_detection

# 2) create venv & install
python -m venv .venv
source .venv/bin/activate   # or .venv\Scripts\activate on Windows
pip install -r requirements.txt

# 3) run demo Streamlit app
streamlit run app/dashboard.py
# open http://localhost:8501
```

---

## Installation

### System deps

* **Tesseract OCR** (for scanned PDFs)

  * Ubuntu: `sudo apt install tesseract-ocr`
  * Windows: install from [https://github.com/tesseract-ocr/tesseract](https://github.com/tesseract-ocr/tesseract)

* **poppler-utils** (for pdf2image)

  * Ubuntu: `sudo apt install poppler-utils`

### Python deps

```bash
pip install -r requirements.txt
```

`requirements.txt` includes: `pdfplumber`, `pytesseract`, `pdf2image`, `pandas`, `scikit-learn`, `xgboost`, `shap`, `streamlit`, `fastapi`, `uvicorn`, `mlflow`, `dvc`, `reportlab`, `faker`

---

## Usage

### 1) Generate synthetic training data

```bash
python scripts/generate_invoice_rows.py --num 10000 --out synthetic/invoice_rows.csv
```

### 2) Train models

```bash
python src/train.py --config configs/train_config.yaml
# Trains OneClassSVM and XGBoost, saves models to models/
# Logs experiments to MLflow by default at ./mlruns
```

### 3) Run inference on a single PDF

```bash
python src/extract_text_from_pdf.py data/sample/invoice_0001.pdf --out parsed_row.csv
python src/predict.py parsed_row.csv --model models/ocsvm_pipeline.pkl
```

### 4) Web UI (Streamlit)

```bash
streamlit run app/dashboard.py
# Upload PDF → Shows prediction + SHAP explanation
```

### 5) API (FastAPI)

```bash
uvicorn app.api:app --reload --port 8000
# POST /predict with file upload or JSON
```

---

## Project Structure

```
invoice_fraud_detection/
├── app/                     # Streamlit + FastAPI apps
├── configs/                 # YAML configs (training, inference)
├── data/                    # raw/processed (dvc-tracked)
├── docs/                    # screenshots, demo gif
├── models/                  # saved models (joblib)
├── scripts/                 # data generation scripts
├── src/                     # core modules (extract, preprocess, train, predict, shap)
├── requirements.txt
├── Dockerfile
└── README.md
```

---

## Dataset & Schema

* **Source:** synthetic generator (`scripts/generate_invoice_rows.py`) + sample real invoices (if available).
* **CSV columns** (example):

  * `filename, invoice_no, issue_date, vendor_name, vendor_taxid, client_name, n_items, total_net, vat_percent, vat_amount, total_gross, payment_days, vendor_frequency, is_fraud, fraud_type`

If real data is sensitive, run the generator to reproduce a working dataset.

---

## Modeling & Features

* **Features used:** `total_net`, `vat_amount`, `total_gross`, `n_items`, `payment_days`, `vendor_frequency`, plus engineered features like `avg_amount_per_vendor`, `amount_ratio`, `days_between_issue_and_payment`.
* **Models:**

  * **One-Class SVM** (trained on `is_fraud == 0`) for anomaly detection of novel frauds.
  * **XGBoost** classifier (trained on labeled dataset) for known fraud patterns.
* **Explainability:** SHAP `TreeExplainer` for XGBoost, `KernelExplainer` for One-Class SVM.

---

## Evaluation & Results

* Example metrics (on synthetic test set):

  * XGBoost: Precision=0.86, Recall=0.79, F1=0.82, AUC=0.91
  * One-Class SVM (anomaly detection): Precision@top-k and recall vary; tuning `nu` helps control false positives.

(Include plots: ROC curve, confusion matrix, PR curve in `/docs`)

---

## Explainability (SHAP)

* For each prediction, the app shows:

  * **SHAP summary (bar)** for global feature importance
  * **SHAP force/waterfall plot** for single invoice explanation
* Example: a high `total_net` and low `vendor_frequency` increase the fraud score.

---

## MLOps

* **DVC** tracks large data files and models. To reproduce:

  ```bash
  dvc pull  # gets data & models from remote
  mlflow ui  # to browse experiment runs
  ```
* Experiments logged to MLflow — each run stores params, metrics, artifacts (model.pkl).

---

## Deployment

* Build Docker:

  ```bash
  docker build -t invoice-fraud-app:latest .
  docker run -p 8501:8501 invoice-fraud-app:latest
  ```
* Deployable on Railway, Heroku, or any container platform.

---

## Limitations & Future Work

* Current system focuses on **textual/numeric fraud** (amounts, duplicates, vendor mismatch). It does **not** detect advanced **visual** forgeries (image edits) — future work: vision models + forgery detection.
* Real-world performance requires more diverse, labeled fraud examples and domain adaptation per region.

---


## License & Contact

* License: MIT (or choose your license)
* Author: YOUR_NAME — [GitHub link / Email / LinkedIn]

---



