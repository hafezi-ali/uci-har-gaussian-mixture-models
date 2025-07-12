# %%
# =================================================================================================
#
#                    Advanced Machine Learning Pipeline for Human Activity Recognition
#
# =================================================================================================
# This script implements an end-to-end machine learning pipeline for Human Activity Recognition (HAR).
# The core methodology is based on clustering the training data to create specialized subsets,
# and then training an ensemble of "expert" classifiers on these subsets.
#
# Pipeline Stages:
# 1. Data Ingestion & Preprocessing: Load, split, and normalize the data. Apply PCA for dimensionality reduction.
# 2. Cluster-Based Resampling: Systematically partition the data using GMM or KMeans over a grid of parameters.
# 3. Hyperparameter-Tuned Model Training: For each data subset, train multiple classifiers using GridSearchCV.
# 4. Results Aggregation & Analysis: Consolidate all experimental results into a structured DataFrame.
# 5. Best Model Instantiation: Create final, optimized models based on the best-found parameters.
# 6. Serialization: Save all results to disk for later analysis and prediction.
# =================================================================================================

# =================================================================================================
#                                  CONFIGURATION
# =================================================================================================
# Select the clustering methods to run. Options: 'gmm', 'kmeans'. Can be a list of one or more.
CLUSTERING_METHODS_TO_RUN = ['gmm', 'kmeans']

# Select the grid search mode.
# 'full': Performs an exhaustive search over a wide range of hyperparameters.
# 'fast': Uses a minimal set of parameters for quick testing and validation.
GRID_SEARCH_MODE = 'fast'
# =================================================================================================


import copy
# Import necessary libraries
import os
import pickle
from sklearn.cluster import KMeans
from sklearn.mixture import GaussianMixture


# --- Environment Setup ---
# NOTE: These paths are hardcoded and must be changed to match your local environment.
os.chdir('/content/drive/MyDrive/Thp')
os.environ['PYTHONPATH'] = '/content/drive/MyDrive/Thp'

# --- Custom Module Imports ---
from Funcs.Functions import *
from Funcs.Model_functions import *
import Funcs
# from Lib.lib import *

# %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
#                                         STAGE 1: DATA PREPARATION
# %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

# Load the pre-cleaned UCI HAR dataset.
uci_data_handle_new = pd.read_csv('Data/UCI_HAR_Dataset/data_uci_handled.csv', index_col=0)
# The activity labels are 1-based, which is a common convention.
uci_data_handle_new['Activity'] = uci_data_handle_new['Activity'] + 1
database = {'uci': uci_data_handle_new}
# %%

# Separate features (X) from labels (y). 'Activity' and 'subject' are metadata.
X = database['uci'].drop(['Activity', 'ActivityName', 'subject'], axis=1)
y = database['uci']['ActivityName']
# Split the dataset into training and testing sets. A fixed random_state ensures reproducibility.
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
# %%

# --- Single, initial preprocessing step for the entire training dataset ---
# This involves normalization and PCA to reduce dimensionality and prepare for clustering.
n_components = 'auto'  # Let PCA decide the number of components to keep.
random_state_pca = None # For reproducibility, a seed could be set here.
single_preprocessing_res_grid = Funcs.Functions.single_preprocessing(
    X_train=X_train, X_test=X_test,
    y_train=y_train,
    n_components=n_components,
    random_state_pca=random_state_pca,
)
# %%

# Persist the preprocessed data. This allows skipping the above steps in future runs.
with open('Results/single_preprocessing_res_grid_main.pkl', 'wb') as f:
    pickle.dump(single_preprocessing_res_grid, f)
# %%


def resampling_grid(phi_list, n_c_list, random_state_clustering, single_preprocessing_res_grid, y_train,
                    distance_metrics, clustering_method_name):
    """
    Iterates through a grid of clustering parameters (phi, n_c) to generate various resampled datasets.
    This is the core of the data partitioning strategy.
    """
    df_single_resampling_res_grid = pd.DataFrame(columns=['phi', 'n_c', 'single_resampling_res'])
    # Outer loop: Iterate over the sampling ratio `phi`.
    for phi in tqdm(phi_list, position=0, leave=True, desc=f"Phi Loop ({clustering_method_name})"):
        # Inner loop: Iterate over the number of clusters `n_c`.
        for n_c in tqdm(n_c_list, position=1, leave=True, desc="N_c Loop"):
            if clustering_method_name == 'gmm':
                cluster_params = {'n_components': n_c, 'random_state': random_state_clustering}
                clustering_method = GaussianMixture().set_params(**cluster_params)
            elif clustering_method_name == 'kmeans':
                cluster_params = {'n_clusters': n_c, 'random_state': random_state_clustering, 'n_init': 'auto'}
                clustering_method = KMeans().set_params(**cluster_params)
            else:
                raise ValueError(f"Unknown clustering method: {clustering_method_name}")


            # The `resampling` function performs GMM clustering and then selects the top `phi` percent of samples.
            single_resampling_res = resampling(
                clusterig_method=clustering_method,
                X_train_dist=single_preprocessing_res_grid['X_train'],
                y_train_dist=y_train, phi=phi,
                random_state_cluster_samples_shuffling=random_state_cluster_samples_shuffling,
                distance_metrics=distance_metrics
            )
            # Store the results (including the data subsets) in a new row.
            new_row = pd.DataFrame({'phi': [phi], 'n_c': [n_c], 'single_resampling_res': [single_resampling_res]})
            df_single_resampling_res_grid = pd.concat([df_single_resampling_res_grid, new_row], ignore_index=True)
    return df_single_resampling_res_grid

# =================================================================================================
#                         MAIN PIPELINE LOOP - EXECUTES FOR EACH CLUSTERING METHOD
# =================================================================================================
for clustering_method in CLUSTERING_METHODS_TO_RUN:
    print(f"\n{'='*80}\nRUNNING PIPELINE FOR CLUSTERING METHOD: {clustering_method.upper()}\n{'='*80}")

    # %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    #                                   STAGE 2: CLUSTER-BASED RESAMPLING
    # %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

    # --- Define the hyperparameter grid for the clustering experiment ---
    # `phi`: The proportion of the most representative samples to keep from each cluster.
    phi_list = np.arange(0.1, 1, 0.1)
    # `n_c`: The number of clusters to partition the data into.
    n_c_list = np.arange(2, 7)

    # --- Clustering configuration ---
    random_state_clustering = None # For reproducibility, a seed could be set here.
    random_state_cluster_samples_shuffling = 42 # Ensures consistent shuffling of cluster samples.
    distance_metrics = 'Euclidean' # Metric used to find points closest to cluster centroids.

    # Execute the resampling grid search. This is a computationally intensive step.
    df_single_resampling_res_grid = resampling_grid(phi_list, n_c_list, random_state_clustering,
                                                 single_preprocessing_res_grid, y_train, distance_metrics,
                                                 clustering_method_name=clustering_method)
    # %%

    # Persist the resampled data subsets for the current clustering method.
    with open(f'Results/df_single_resampling_res_grid_{clustering_method}_main.pkl', 'wb') as f:
        pickle.dump(df_single_resampling_res_grid, f)
    # %%

    # %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    #                                   STAGE 3: MODEL TRAINING & GRID SEARCH
    # %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

    # Retrieve a dictionary of base model instances.
    models = get_models()
    normalization = False # Normalization is already done in the preprocessing stage.
    param_models_list = list() # This will store the grid search results for each experiment.

    # Define the hyperparameter search space for each classifier based on the selected mode.
    if GRID_SEARCH_MODE == 'full':
        model_param_grids = {
            'svm': {'C': [0.1, 1, 10, 100], 'kernel': ['linear', 'poly', 'rbf', 'sigmoid'], 'gamma': ['scale', 'auto']},
            'sgd': {'alpha': [0.0001, 0.001, 0.01], 'loss': ['log_loss']},
            'lr': {'C': [0.1, 1, 10], 'solver': ['lbfgs', 'liblinear','saga']},
            'knn': {'n_neighbors': [3, 5, 7, 9], 'weights': ['uniform', 'distance'], 'algorithm': ['auto', 'ball_tree', 'kd_tree']},
            'dt': {'criterion': ['gini', 'entropy'], 'splitter': ['best', 'random'], 'max_depth': [None, 10, 20, 30, 40, 50], 'min_samples_split': [2, 10, 20], 'min_samples_leaf': [1, 5, 10]}
        }
    else: # 'fast' mode
        model_param_grids = {
            'svm': {'C': [1], 'kernel': ['rbf']},
            'sgd': {'alpha': [0.001], 'loss': ['log_loss']},
            'lr': {'C': [1], 'solver': ['lbfgs']},
            'knn': {'n_neighbors': [5], 'weights': ['distance']},
            'dt': {'max_depth': [20], 'criterion': ['gini']}
        }

    # --- Main training loop ---
    # Iterate over each row of the resampling grid, where each row represents a unique (phi, n_c) experiment.
    for index, row in tqdm(df_single_resampling_res_grid.iterrows(), total=len(df_single_resampling_res_grid),
                           position=0, leave=True, desc=f"Grid Training ({clustering_method})"):
        # For each experiment, train models on its specific clustered data subsets.
        # The `grid_train_base_models` function handles the training for all classifiers on all clusters for this row.
        param_models = grid_train_base_models(
            cluster_samples=row['single_resampling_res']['rm_cluster_samples'],
            cluster_samples_lables=row['single_resampling_res']['rm_cluster_samples_labels'],
            model_param_grids=model_param_grids, models=models,
            normalization=normalization)
        param_models_list.append(param_models)
    # %%

    # Add the trained model results as a new column to the main DataFrame.
    # Each entry in this column is a nested dictionary containing the trained GridSearchCV objects.
    df_single_resampling_res_grid['param_models_list'] = param_models_list
    # %%

    # %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    #                                   STAGE 4: RESULTS AGGREGATION & ANALYSIS
    # %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

    # Exclude Gaussian Naive Bayes from the final analysis as it may not be a focus.
    models = get_models()
    models.pop('gn')
    # %%

    # --- Extract and structure the grid search results for easier analysis ---
    # `candidate_models_cluster_grid_nested`: A nested list to store the best params and scores.
    # Structure: [experiment] -> [cluster] -> {model_name: {'best_params_': ..., 'best_score_': ...}}
    candidate_models_cluster_grid_nested = list()
    for index, row in df_single_resampling_res_grid.iterrows():
        n_c = row['n_c']
        candidate_models_cluster_grid = list()
        for i_cluster in range(n_c): # Note: original code had a bug here using `n_c` as loop var. Corrected to `i_cluster`.
            candidate_best_models = dict()
            for model_name in models:
                grid_search_object = row['param_models_list'][i_cluster][model_name]['grid_' + model_name + '_cluster']
                candidate_best_models[model_name] = {
                    'best_params': grid_search_object.best_params_,
                    'best_score_': grid_search_object.best_score_
                }
            candidate_models_cluster_grid.append(candidate_best_models)
        candidate_models_cluster_grid_nested.append(candidate_models_cluster_grid)
    # %%

    # Add the structured best models to the DataFrame.
    df_single_resampling_res_grid['candidate_models_cluster_grid'] = candidate_models_cluster_grid_nested
    # %%

    # --- Extract the full cross-validation results for even deeper analysis ---
    # This allows examining the performance of all hyperparameter combinations, not just the best one.
    rm_duplicated_cv_results_model_sorteds_nested = list()
    cv_results_model_sorteds_nested = list()
    for index, row in df_single_resampling_res_grid.iterrows():
        n_c = row['n_c']
        cv_results_model_sorted_list = list()
        rm_duplicated_cv_results_model_sorted_list = list()
        for i_cluster in range(n_c):
            cv_results_models_sorted = dict()
            rm_duplicated_cv_results_models_sorted = dict()
            for model_name in models:
                grid_search_object = row['param_models_list'][i_cluster][model_name]['grid_' + model_name + '_cluster']
                cv_results_df = pd.DataFrame(grid_search_object.cv_results_)
                # Sort results by performance.
                cv_results_model_sorted = cv_results_df.sort_values(by='mean_test_score', ascending=False)
                # Remove duplicate scores to see unique performance tiers.
                rm_duplicated_cv_results_model_sorted = cv_results_model_sorted.drop_duplicates(subset=["mean_test_score"], keep="first")

                cv_results_models_sorted[model_name] = cv_results_model_sorted
                rm_duplicated_cv_results_models_sorted[model_name] = rm_duplicated_cv_results_model_sorted

            cv_results_model_sorted_list.append(cv_results_models_sorted)
            rm_duplicated_cv_results_model_sorted_list.append(rm_duplicated_cv_results_models_sorted)
        rm_duplicated_cv_results_model_sorteds_nested.append(rm_duplicated_cv_results_model_sorted_list)
        cv_results_model_sorteds_nested.append(cv_results_model_sorted_list)
    # %%

    # Add the detailed CV results to the DataFrame.
    df_single_resampling_res_grid['cv_results_model_sorteds'] = cv_results_model_sorteds_nested
    df_single_resampling_res_grid['rm_duplicated_cv_results_model_sorteds'] = rm_duplicated_cv_results_model_sorteds_nested
    # %%

    # %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    #                                   STAGE 5: BEST MODEL INSTANTIATION
    # %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

    i_experiment = 0
    best_models_clusters_nested = list()
    sorted_candidate_models_cluster_grid_nested = list()
    # Iterate over the CV results for each experiment.
    for experiment_cv_results in cv_results_model_sorteds_nested:
        n_c = df_single_resampling_res_grid.iloc[i_experiment]['n_c']
        best_models_clusters_dict = dict()
        # For each model type (svm, knn, etc.), create the set of expert models for the clusters.
        for model_name in models:
            # For each cluster, instantiate a new model with the best-found hyperparameters.
            best_models_for_clusters = [
                copy.deepcopy(models[model_name]).set_params(**experiment_cv_results[i_cluster][model_name].iloc[0]['params'])
                for i_cluster in range(n_c)
            ]
            best_models_clusters_dict[model_name] = best_models_for_clusters

        # Sort the dictionary of candidate models by score for this experiment.
        sorted_candidate_models_cluster_grid = sort_by_score(
            df_single_resampling_res_grid['candidate_models_cluster_grid'][i_experiment])

        sorted_candidate_models_cluster_grid_nested.append(sorted_candidate_models_cluster_grid)
        best_models_clusters_nested.append(best_models_clusters_dict)
        i_experiment += 1
    # %%

    # Add the final, instantiated models and sorted results to the DataFrame.
    df_single_resampling_res_grid['best_models_clusters'] = best_models_clusters_nested
    df_single_resampling_res_grid['sorted_candidate_models_cluster_grid'] = sorted_candidate_models_cluster_grid_nested
    # %%

    # %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    #                                   STAGE 6: FINAL SERIALIZATION
    # %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

    # Save the final, comprehensive DataFrame containing all experiments, results, and trained models.
    with open(f'Results/df_single_resampling_res_grid_{clustering_method}_main_full.pkl', 'wb') as f:
        pickle.dump(df_single_resampling_res_grid, f)
    # %%
    #  #################################### END OF PIPELINE FOR {clustering_method.upper()} ####################################

print(f"\n{'='*80}\nPIPELINE EXECUTION COMPLETE FOR ALL CONFIGURED CLUSTERING METHODS.\n{'='*80}")
