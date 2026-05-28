````md
<div align="center">

# Comparative Analysis of Dimensionality Reduction Techniques for GMM Clustering in Human Activity Recognition

<p align="center">
  Advanced Human Activity Recognition (HAR) pipeline using
  <strong>Gaussian Mixture Models (GMM)</strong>,
  <strong>PCA</strong>, and
  <strong>UMAP</strong>.
</p>

<p align="center">
  <img src="https://img.shields.io/badge/Python-3.10+-0EA5E9?style=for-the-badge&logo=python&logoColor=white" />
  <img src="https://img.shields.io/badge/Machine%20Learning-GMM-8B5CF6?style=for-the-badge" />
  <img src="https://img.shields.io/badge/Dimensionality%20Reduction-PCA%20%7C%20UMAP-10B981?style=for-the-badge" />
  <img src="https://img.shields.io/badge/Dataset-UCI%20HAR-F59E0B?style=for-the-badge" />
</p>

<p align="center">
  <a href="https://alihafezi.site/uci-har-gaussian-mixture-models/">
    <img src="https://img.shields.io/badge/View%20Full%20Analysis-111827?style=for-the-badge&logo=vercel&logoColor=white" />
  </a>
  
  <a href="https://www.kaggle.com/datasets/uciml/human-activity-recognition-with-smartphones">
    <img src="https://img.shields.io/badge/Download%20Dataset-2563EB?style=for-the-badge&logo=kaggle&logoColor=white" />
  </a>
</p>

</div>

---

# Overview

This project presents a comprehensive **Human Activity Recognition (HAR)** analysis pipeline using smartphone sensor data.

The workflow includes:

- Data preprocessing
- Exploratory Data Analysis (EDA)
- Feature engineering
- Outlier handling
- Dimensionality reduction
- Gaussian Mixture Model (GMM) clustering
- Visualization and comparative analysis

A major focus of this project is evaluating the effect of **dimensionality reduction** techniques on clustering performance in high-dimensional spaces.

The following methods are compared:

| Method | Purpose |
|---|---|
| **PCA** | Linear dimensionality reduction |
| **UMAP** | Non-linear manifold learning |
| **GMM** | Probabilistic clustering |

---

# Dataset Information

The dataset contains smartphone sensor data collected from **30 volunteers** aged between **19 and 48 years** performing six activities:

- WALKING
- WALKING_UPSTAIRS
- WALKING_DOWNSTAIRS
- SITTING
- STANDING
- LAYING

### Data Collection

- Samsung Galaxy S II mounted on the waist
- Accelerometer + gyroscope signals
- Sampling rate: **50Hz**
- Window size: **2.56 seconds**
- 70/30 train-test split

---

# Project Structure

```bash
.
├── Data
│   ├── processed
│   └── raw
├── main.py
├── modules
│   ├── clustering.py
│   ├── data_loading.py
│   ├── data_processing.py
│   ├── eda.py
│   ├── feature_engineering.py
│   └── utils.py
├── README.md
├── requirements.txt
└── results
    └── plots
````

---

# Installation

## 1. Clone Repository

```bash
git clone https://github.com/yourusername/har_analysis_pipeline.git
cd har_analysis_pipeline
```

## 2. Create Virtual Environment

```bash
python -m venv venv
```

### Linux / macOS

```bash
source venv/bin/activate
```

### Windows

```bash
venv\\Scripts\\activate
```

---

## 3. Install Dependencies

```bash
pip install -r requirements.txt
```

---

## 4. Run Pipeline

```bash
python main.py
```

---

# Pipeline Workflow

```text
Raw Data
   ↓
Preprocessing
   ↓
EDA
   ↓
Feature Engineering
   ↓
PCA / UMAP
   ↓
GMM Clustering
   ↓
Evaluation & Visualization
```

---

# Key Modules

| Module                   | Description                       |
| ------------------------ | --------------------------------- |
| `main.py`                | Main pipeline orchestration       |
| `data_loading.py`        | Loads raw and processed datasets  |
| `data_processing.py`     | Missing values + outlier handling |
| `eda.py`                 | Exploratory Data Analysis         |
| `feature_engineering.py` | Scaling + PCA/UMAP                |
| `clustering.py`          | GMM clustering + evaluation       |
| `utils.py`               | Visualization helpers             |

---

# Features

* Gaussian Mixture Model clustering
* PCA dimensionality reduction
* UMAP embedding visualization
* BIC/AIC model selection
* Outlier handling using:

  * IQR
  * Z-score
* Automated visual reporting
* Modular architecture

---

# Results

The project generates:

* Activity distribution plots
* PCA visualizations
* UMAP embeddings
* Cluster comparison charts
* BIC/AIC evaluation plots

---

# Future Improvements

* Real-time HAR pipeline
* Deep learning integration
* Additional clustering algorithms
* Hyperparameter optimization

---

# License

This project is licensed under the MIT License.

---

# Credits

<div align="center">

### Ali Hafezi

<a href="https://github.com/hafezi-ali">
  <img src="https://img.shields.io/badge/GitHub-111827?style=for-the-badge&logo=github&logoColor=white"/>
</a>

<a href="https://x.com/hafezi_alii">
  <img src="https://img.shields.io/badge/X-000000?style=for-the-badge&logo=x&logoColor=white"/>
</a>

<a href="https://www.linkedin.com/in/alihafezii/">
  <img src="https://img.shields.io/badge/LinkedIn-2563EB?style=for-the-badge&logo=linkedin&logoColor=white"/>
</a>

<a href="https://alihafezi.site">
  <img src="https://img.shields.io/badge/Website-0EA5E9?style=for-the-badge&logo=googlechrome&logoColor=white"/>
</a>

</div>
```

