# Documentation: Advanced HAR Training Pipeline

## 1. Executive Summary

This document details an advanced machine learning pipeline for Human Activity Recognition (HAR), implemented in the `base_train.py` script. The primary goal is to enhance classification accuracy by moving beyond a single, monolithic model. It employs a sophisticated strategy involving data clustering to identify distinct patterns, followed by the training of specialized models for each cluster. This ensemble approach allows for a more nuanced understanding of the feature space, leading to potentially superior performance on the UCI HAR Dataset.

## 2. Configuration Options

The script includes the following configuration options at the top of the file, allowing you to control the execution of the pipeline:

- **`CLUSTERING_METHODS_TO_RUN`**: A list of strings that determines which clustering algorithms to use.
    - **Options**: `['gmm', 'kmeans']`
    - **Example**: `['gmm']` will run the pipeline only with Gaussian Mixture Models. `['gmm', 'kmeans']` will run it sequentially for both.

- **`GRID_SEARCH_MODE`**: A string that controls the depth of the hyperparameter search for the classifiers.
    - **Options**:
        - `'full'`: Performs an exhaustive search over a wide range of hyperparameters. This is computationally expensive but likely to yield better results.
        - `'fast'`: Uses a minimal, pre-defined set of parameters for a single training run per model. This is ideal for quickly testing the pipeline's integrity or getting baseline results.

## 3. The Dataset

- **Identifier**: UCI HAR Dataset
- **Domain**: Human Activity Recognition using smartphone sensors.
- **Characteristics**:
    - **Features**: 561 time and frequency domain features derived from accelerometer and gyroscope signals.
    - **Activities (Classes)**: 6 distinct activities are targeted for classification:
        1.  WALKING
        2.  WALKING_UPSTAIRS
        3.  WALKING_DOWNSTAIRS
        4.  SITTING
        5.  STANDING
        6.  LAYING
    - **Subjects**: Data collected from 30 volunteers.
- **File Used**: `Data/UCI_HAR_Dataset/data_uci_handled.csv`, a pre-cleaned version of the original dataset.

## 4. Pipeline Architecture

The script executes a sequential pipeline, where the output of each stage serves as the input for the next. The intermediate results are persisted to disk using `pickle` for modularity and reproducibility. The entire pipeline (from Stage 2 onwards) will run for each method specified in `CLUSTERING_METHODS_TO_RUN`.

### Step 1: Data Ingestion and Preparation

- **Action**: Loads the dataset and splits it into features (`X`) and labels (`y`).
- **Details**: A standard 80/20 train-test split is performed using a fixed `random_state=42` to ensure that the same data split is used in every run, which is crucial for reproducible results.

### Step 2: Preprocessing and Dimensionality Reduction

- **Action**: The training data undergoes normalization and Principal Component Analysis (PCA).
- **Rationale**:
    - **Normalization**: Standardizes features to have a mean of 0 and a standard deviation of 1, which is a prerequisite for many ML algorithms, including PCA and SVMs.
    - **PCA**: Reduces the high-dimensional feature space (561 features). This mitigates the "curse of dimensionality," reduces computational load, and can remove noise. The `n_components='auto'` setting lets scikit-learn determine the optimal number of components to retain sufficient variance.
- **Output**: A dictionary containing the transformed datasets, saved as `Results/single_preprocessing_res_grid_main.pkl`.

### Step 3: Cluster-Based Resampling (Core Innovation)

- **Action**: The preprocessed training data is partitioned into clusters using the selected clustering algorithm (GMM or KMeans). This step is iterated over a grid of parameters.
- **Rationale**: The core hypothesis is that the dataset is not uniform; instead, it contains distinct sub-groups of data (e.g., dynamic vs. static activities might form natural clusters). A single global model may struggle to learn these diverse patterns effectively. By clustering first, we can train specialized "expert" models for each data sub-group.
- **Key Parameters**:
    - **`n_c_list` (Number of Clusters)**: A hyperparameter (`[2, 3, 4, 5, 6]`) that determines how many sub-groups to partition the data into.
    - **`phi_list` (Sampling Ratio)**: A hyperparameter (`[0.1, 0.2, ..., 0.9]`) that controls what proportion of the most representative points (closest to the cluster centroid) are kept from each cluster for training. This acts as a sophisticated data-pruning technique.
- **Output**: A DataFrame containing the clustered and resampled data for each parameter combination, saved as `Results/df_single_resampling_res_grid_[METHOD]_main.pkl`, where `[METHOD]` is `gmm` or `kmeans`.

### Step 4: Grid Search and Base Model Training

- **Action**: For each clustered subset of data generated in the previous step, a suite of standard classifiers is trained. The hyperparameter search space is determined by the `GRID_SEARCH_MODE`.
- **Classifiers**:
    - Support Vector Machine (SVM)
    - Stochastic Gradient Descent (SGD)
    - Logistic Regression (LR)
    - K-Nearest Neighbors (KNN)
    - Decision Tree (DT)
- **Process**: A `GridSearchCV` is performed for each model on each cluster's data subset. This exhaustively searches a predefined `model_param_grids` to find the optimal hyperparameters for each "expert" model.

### Step 5: Model Evaluation and Selection

- **Action**: The results from the grid search are systematically collected and analyzed.
- **Process**: The script extracts the best parameters, the best cross-validation scores, and the full CV results for every model on every cluster. This creates a rich dataset for analysis.
- **Final Model Creation**: The best-performing hyperparameters are used to instantiate the final "expert" models for each cluster. These models are now ready for prediction.

### Step 6: Results Serialization

- **Action**: The final, comprehensive DataFrame containing all experimental results is saved.
- **Output**: `Results/df_single_resampling_res_grid_[METHOD]_main_full.pkl`. This file contains:
    - The original grid parameters (`phi`, `n_c`).
    - The resampled data for each experiment.
    - The full grid search objects for every model and cluster.
    - The extracted best parameters and scores.
    - The final, instantiated best models for each cluster.

## 5. Key Innovations and Rationale

1.  **Cluster-Based Resampling**: Instead of random sampling, this method uses data structure (clusters) to create smaller, more coherent training sets. This can lead to faster training and potentially better models by focusing on the most representative data points.
2.  **Ensemble of Experts**: The final output is not a single model, but an ensemble of models, where each model is an expert on a specific sub-region of the feature space. When making a prediction on a new data point, one could first determine its most likely cluster and then use that cluster's expert model for prediction.
3.  **Exhaustive Hyperparameter Search**: The pipeline automates the tedious process of tuning not only the machine learning models but also the data processing (clustering) steps, creating a highly optimized end-to-end system.

## 6. How to Run

1.  **Configure**: Set `CLUSTERING_METHODS_TO_RUN` and `GRID_SEARCH_MODE` at the top of `base_train.py`.
2.  **Dependencies**: Ensure all libraries listed in `requirements.txt` are installed.
3.  **Environment**: The script is designed for a Python environment (like Jupyter, Colab, or a standard IDE) where the working directory and `PYTHONPATH` can be set. The file paths are currently hardcoded for a specific environment and will need to be adjusted.
4.  **Execution**: Run the script sequentially. Due to the nested loops and grid searches, execution can be computationally intensive and time-consuming, especially in `'full'` mode.

## 7. Limitations

-   **Computational Cost**: The multi-level grid search is very resource-intensive, particularly when running both clustering methods in 'full' mode.
-   **Hardcoded Paths**: The script is not easily portable without modifying the file paths.
-   **Memory Usage**: Storing all results, including full `GridSearchCV` objects, in a single DataFrame can be memory-intensive. For larger-scale experiments, a different results storage strategy (e.g., a database or individual files) would be necessary. 