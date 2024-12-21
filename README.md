
# HAR Analysis Pipeline

This project is a Human Activity Recognition (HAR) analysis pipeline that processes raw sensor data, performs feature engineering, clustering, and provides exploratory data analysis (EDA). It also includes various results, including plots and models used for the analysis.

## Project Structure

```
.
├── Data
│   ├── processed
│   │   ├── data_uci_handled_outliers.csv
│   │   ├── x_uci_handled_outliers_iqr.csv
│   │   └── x_uci_handled_outliers_zs.csv
│   └── raw
│       └── (contains raw and processed datasets)
├── main.py
├── modules
│   ├── clustering.py
│   ├── data_loading.py
│   ├── data_processing.py
│   ├── eda.py
│   ├── feature_engineering.py
│   └── utils.py
├── README.md
├── Report.md
├── requirements.txt
└── results
    └── plots
        └── (contains various plot outputs from the analysis)

```

## Setup Instructions

Follow these steps to set up and run the project on your local machine.

### 1. Clone the Repository

```bash
git clone https://github.com/yourusername/har_analysis_pipeline.git
cd har_analysis_pipeline
```

### 2. Create and Activate the Virtual Environment

It's recommended to use a virtual environment to manage dependencies.

```bash
python3 -m venv venv
source venv/bin/activate
```

### 3. Install Dependencies

Use the `requirements.txt` to install the necessary libraries.

```bash
pip install -r requirements.txt
```

### 4. Run the Main Script

To execute the pipeline and start the analysis, run the `main.py` script:

```bash
python main.py
```

## Data Structure

### Raw Data
- **activity_labels.txt**: Mapping of activity IDs to activity names.
- **data_uci.csv**: Original dataset for the HAR analysis.
- **subject_test.txt**, **subject_train.txt**: Identifiers for test/train subjects.
- **X_test.txt**, **X_train.txt**: Sensor readings for test/train data.
- **y_test.txt**, **y_train.txt**: Activity labels for test/train data.

### Processed Data
- **data_uci_handled_outliers.csv**: Processed data with outliers handled.
- **x_uci_handled_outliers_iqr.csv**: Data processed with IQR for outlier handling.
- **x_uci_handled_outliers_zs.csv**: Data processed with Z-scores for outlier handling.

### Results
- Contains various plot outputs that visualize the analysis, such as activity distributions, comparison plots, and PCA/UMAP visualizations.

## Modules

- **clustering.py**: Clustering algorithms and model evaluations.
- **data_loading.py**: Functions to load the raw data files.
- **data_processing.py**: Data cleaning and preprocessing steps.
- **eda.py**: Exploratory Data Analysis (EDA) functions.
- **feature_engineering.py**: Feature engineering functions for model inputs.
- **utils.py**: Utility functions for various tasks.

## License

This project is licensed under the [MIT License](./LICENSE). See the `LICENSE` file for details.

## Contributing

Feel free to fork this repository and submit pull requests. Contributions are welcome!

## Acknowledgements

- [UCI HAR Dataset](https://archive.ics.uci.edu/dataset/240/human+activity+recognition+using+smartphones): The dataset used in this project.
- Special thanks to contributors and researchers who made the dataset available.

## Future Work

- Future versions of this pipeline will include more advanced analysis, additional clustering techniques, and integration with other machine learning models.
