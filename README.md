# Human Activity Recognition (HAR) Analysis Pipeline

This project presents a comprehensive analysis pipeline for Human Activity Recognition (HAR) using sensor data. The pipeline includes processing raw sensor data, feature engineering, and clustering, along with exploratory data analysis (EDA). Key results, such as visualizations and trained models, are provided for in-depth analysis. Furthermore, the project investigates and compares the impact of dimensionality reduction on the effectiveness of Gaussian Mixture Models (GMMs) in high-dimensional spaces. Two methods, Principal Component Analysis (PCA) and Uniform Manifold Approximation and Projection (UMAP), are employed to generate lower-dimensional representations of the data, thereby optimizing GMM performance while mitigating the curse of dimensionality. A comparative analysis of these techniques is provided.

For detailed results and insights, refer to the [GMM HAR Analysis Report](https://alihafezi.site/uci-har-gaussian-mixture-models/).

---

## Project Overview

### Directory Structure

```
.
├── Data
│   ├── processed
│   │   ├── data_uci_handled_outliers.csv
│   │   ├── x_uci_handled_outliers_iqr.csv
│   │   └── x_uci_handled_outliers_zs.csv
│   └── raw
│       └── (contains raw and processed datasets)
├── main.py
├── modules
│   ├── clustering.py
│   ├── data_loading.py
│   ├── data_processing.py
│   ├── eda.py
│   ├── feature_engineering.py
│   └── utils.py
├── README.md
├── Report.md
├── requirements.txt
└── results
    └── plots
        └── (contains various plot outputs from the analysis)
```

---

## Setup Instructions

### Step 1: Clone the Repository

```bash
git clone https://github.com/yourusername/har_analysis_pipeline.git
cd har_analysis_pipeline
```

### Step 2: Create and Activate a Virtual Environment

It is recommended to use a virtual environment to manage dependencies.

```bash
python3 -m venv venv
source venv/bin/activate
```

### Step 3: Install Dependencies

Install the required libraries listed in `requirements.txt`.

```bash
pip install -r requirements.txt
```

### Step 4: Run the Main Script

Execute the analysis pipeline by running the `main.py` script:

```bash
python main.py
```

---

## Data Details

### Raw Data
- **activity_labels.txt**: Maps activity IDs to activity names.
- **data_uci.csv**: Original HAR dataset.
- **subject_test.txt**, **subject_train.txt**: Identifiers for test/train subjects.
- **X_test.txt**, **X_train.txt**: Sensor readings for test/train datasets.
- **y_test.txt**, **y_train.txt**: Activity labels for test/train datasets.

### Processed Data
- **data_uci_handled_outliers.csv**: Data with outliers addressed.
- **x_uci_handled_outliers_iqr.csv**: Outliers handled using IQR.
- **x_uci_handled_outliers_zs.csv**: Outliers handled using Z-scores.

### Results
Contains visualization outputs such as activity distributions, comparison plots, and dimensionality reduction visualizations (e.g., PCA/UMAP).

---

## Key Components

### `main.py`

The primary script orchestrating the pipeline. It includes data loading, preprocessing, EDA, feature engineering, dimensionality reduction, clustering, and visualization.

#### Workflow Highlights:
1. **Setup:** Initializes directories for saving results.
2. **Data Loading & Preprocessing:**
   - Loads raw or processed data.
   - Handles missing values and outliers.
   - Saves cleaned datasets and visualizations.
3. **EDA:** Generates data summaries and visualizations.
4. **Feature Engineering:**
   - Scales data.
   - Applies dimensionality reduction techniques (UMAP, PCA).
5. **Clustering:**
   - Uses GMM for clustering with optimal cluster counts based on BIC/AIC criteria.
   - Visualizes clusters and class distributions.
6. **Completion:** Outputs results and a success message.

---

### Modular Design

#### `data_loading.py`
- **Purpose:** Loads raw and processed data into pandas DataFrames.
- **Key Functions:**
  - `load_data()`: Combines and prepares raw data.
  - `load_cleaned_data()`: Loads pre-cleaned datasets.

#### `clustering.py`
- **Purpose:** Implements GMM clustering and evaluates models.
- **Key Functions:**
  - `criteria_values()`: Calculates BIC/AIC values for different cluster counts.
  - `perform_gmm_clustering()`: Applies GMM clustering.
  - `plot_criteria_values()`: Visualizes BIC/AIC trends.

#### `data_processing.py`
- **Purpose:** Cleans and preprocesses data.
- **Key Functions:**
  - `detect_missing_values()`: Identifies missing data.
  - `handle_outliers()`: Manages outliers using IQR/Z-score methods.
  - `plot_outlier_handling()`: Visualizes outlier handling with boxplots, Q-Q plots, and distributions.

#### `eda.py`
- **Purpose:** Performs exploratory data analysis.
- **Key Functions:**
  - `perform_eda()`: Generates data summaries and distributions.

#### `feature_engineering.py`
- **Purpose:** Prepares data for analysis.
- **Key Functions:**
  - `feature_scaling()`: Normalizes features.
  - `apply_optimal_pca()`: Reduces dimensionality with PCA.
  - `umap_standard_embedding()`: Applies UMAP for embedding.

#### `utils.py`
- **Purpose:** Provides utility functions.
- **Key Functions:**
  - `plot_2d_scatter()`: Creates 2D scatter plots for visualization.

---

## License

This project is licensed under the [MIT License](./LICENSE). For more information, see the `LICENSE` file.

---

## Contributing

Contributions are welcome! Feel free to fork the repository and submit pull requests.

---

## Acknowledgements

- [UCI HAR Dataset](https://archive.ics.uci.edu/dataset/240/human+activity+recognition+using+smartphones) for the dataset used in this project.
- Thanks to contributors and researchers for making this project possible.

---

## Future Work

- Integration of additional machine learning models.
- Exploration of advanced clustering methods.
- Expansion of analysis to include real-time activity recognition<!-- CREDITS -->


<h2 id="credits"> :scroll: Credits</h2>

Ali Hafezi

[![GitHub Badge](https://img.shields.io/badge/GitHub-100000?style=for-the-badge&logo=github&logoColor=white)](https://github.com/hafezi-ali)
[![X Badge](https://img.shields.io/badge/X-000000?style=for-the-badge&logo=x&logoColor=white)](https://x.com/hafezi_alii)
[![LinkedIn Badge](https://img.shields.io/badge/LinkedIn-0077B5?style=for-the-badge&logo=linkedin&logoColor=white)](https://www.linkedin.com/in/alihafezii/)
[![Website Badge](https://img.shields.io/badge/Website-4285F4?style=for-the-badge&logo=world&logoColor=white)](https://alihafezi.site)

