import warnings
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import math
from scipy.spatial.distance import mahalanobis
from functools import reduce

# ML imports
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.decomposition import PCA
from sklearn.model_selection import train_test_split
from sklearn.cluster import KMeans, DBSCAN
from sklearn.mixture import GaussianMixture
from sklearn.metrics import confusion_matrix, classification_report
from sklearn.exceptions import UndefinedMetricWarning

# PyTorch imports
import torch
from torch.utils.data import Dataset

from joblib import Parallel, delayed

# Suppress warnings
warnings.simplefilter(action='ignore', category=FutureWarning)
warnings.filterwarnings("ignore", category=UndefinedMetricWarning)


# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ Main functions ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
def fetch_null(data):
    null = data.isna().sum() / len(data) * 100
    null = null[null > 0]
    null.sort_values(inplace=True, ascending=False)
    null = pd.DataFrame({'percent': null})
    return null


def check_missing(data):
    miss_values = fetch_null(data)
    sns.heatmap(data.isnull(), cbar=False, cmap=sns.cm.rocket_r)
    plt.title('Heatmap of NaN; Nan visualization')
    plt.show()
    return miss_values


# use:
def handle_outlier(data, outlier_cols, st_ind, just_info, method, threshold, al_iqr):
    """
    Handle outliers in a given feature of a dataset.

    Parameters:
        - data (pandas dataframe): The dataset.
        - outlier_cols (list): The list of feature to handle outliers in.
        - st_ind (dict): The method to use for handling outliers ('median', 'mean', 'mode').
        - just_info (bool): If True, only returns the number of outliers and filtered samples.
        - method (str): Method to use for outlier detection ('iqr' or 'zs').
        - threshold (float): Z-score threshold.
        - al_iqr (float): Multiplier for IQR.

    Returns:
        - data_out (pandas dataframe): The dataset with outliers handled.
        - count_outliers (int): The number of outliers.
        - filtered_samples_outlier (pandas dataframe): The samples that contain outliers.
    """

    # instead of
    data_out_iqr = data.copy()
    data_out_zs = data.copy()
    count_outliers_iqr = dict()
    count_outliers_zs = dict()
    filtered_samples_outlier_iqr = dict()
    filtered_samples_outlier_zs = dict()

    for feature in outlier_cols:

        # IQR methode
        # Calculate the first and third quartiles
        Q1 = data[feature].quantile(0.25)
        Q3 = data[feature].quantile(0.75)

        # Calculate the inter quartile range
        IQR = Q3 - Q1

        if al_iqr:
            # Identify the outliers using the inter quartile range rule
            al = al_iqr
            v_iqr = (data[feature] < (Q1 - al * IQR)) | (data[feature] > (Q3 + al * IQR))

            # Count the number of outliers
            count_outliers_iqr[feature] = v_iqr.sum()

            # filter samples that have outliers in specific feature
            filtered_samples_outlier_iqr[feature] = data[v_iqr]

        # Z-score method
        # Identify the outliers using the Z-score threshold
        if threshold:
            # Calculate the Z-scores for the given column
            z_scores = np.abs(data[feature] - data[feature].mean()) / data[feature].std()

            # Count the number of outliers
            v_zs = z_scores > threshold
            count_outliers_zs[feature] = v_zs.sum()

            # filter samples that have outliers in specific feature
            filtered_samples_outlier_zs[feature] = data[v_zs]

        count_outliers = {'count_outliers_iqr': count_outliers_iqr, 'count_outliers_zs': count_outliers_zs}
        filtered_samples_outlier = {'filtered_samples_outlier_iqr': filtered_samples_outlier_iqr,
                                    'filtered_samples_outlier_zs': filtered_samples_outlier_zs}

        if not just_info:
            # Dictionary of statistical indicator for handling outliers
            st_inds = {
                'median': data[feature].median(),
                # 'mean_zs': data[~v_zs][feature].mean(), *
                'mean': data[feature].mean(),
                'mode': data[feature].mode()[0]
            }

            # Get the selected method for handling outliers
            val = st_inds.get(st_ind[feature])

            data_out_iqr[feature] = data_out_iqr[feature].where(~v_iqr, val)
            data_out_zs[feature] = data_out_zs[feature].where(~v_zs, val)
            data_out = {'IQR': data_out_iqr, 'ZS': data_out_zs}
        else:
            data_out = data
            count_outliers = count_outliers['count_outliers_' + method]
            filtered_samples_outlier = filtered_samples_outlier['filtered_samples_outlier_' + method]

    return data_out, count_outliers, filtered_samples_outlier


def up_memberc(phi, cluster_c, X_train, y_train, distance_metrics):
    # Calculate the number of data points
    num_data_points = X_train.shape[0]

    # Determine the number of closest points to keep for each centroid
    k = round(phi * num_data_points)

    # Initialize lists to store the results
    CS, CSY, Mindist = [], [], []

    # Loop through each centroid
    for centroid in cluster_c:
        if distance_metrics == 'Euclidean':
            # Calculate the Euclidean distance between each data point and the centroid
            # distances = [np.linalg.norm(np.abs(centroid - X_train[x])) for x in range(num_data_points)]
            distances = np.linalg.norm(X_train - centroid, axis=1)
        elif distance_metrics == 'mahalanobis':
            # diff = X_train - centroid
            # distances = np.sqrt(np.sum(np.dot(diff, inv_cov) * diff, axis=1))

            # Calculate the Mahalanobis distance between each data point and the centroid
            cov = np.cov(X_train.T)
            inv_cov = np.linalg.inv(cov)
            distances = [mahalanobis(X_train[x], centroid, inv_cov) for x in range(num_data_points)]

        # Sort the distances and keep the closest k data points
        # closest_points1 = pd.DataFrame(distances).sort_values(by=0).iloc[:k]
        closest_indices = np.argsort(distances)[:k]
        # Get the closest k data points and their corresponding labels
        closest_X = pd.DataFrame(X_train).iloc[closest_indices]
        closest_y = pd.DataFrame(y_train).iloc[closest_indices]

        # Add the results to the lists
        CS.append(closest_X)
        CSY.append(closest_y)
        Mindist.append(distances[closest_indices])

    # Return the results
    return CS, CSY, Mindist


def sampling(cluster_model, distance_metrics, phi, xtr, ytr):
    # If the clustering method is K_means, fit the KMeans model to the training data and get the cluster assignments
    # and centers
    if isinstance(cluster_model, KMeans):
        cluster_model.fit(xtr)
        cluster_assignments = cluster_model.labels_
        cluster_centers = cluster_model.cluster_centers_

    # If the clustering method is DB_scan, fit the DB_scan model to the training data and get the cluster assignments
    # and core sample indices
    elif isinstance(cluster_model, DBSCAN):
        cluster_model.fit(xtr)
        cluster_assignments = cluster_model.labels_
        cluster_centers = cluster_model.core_sample_indices_

    # If the clustering method is GMM, fit the GMM model to the training data and get the cluster assignments and means
    elif isinstance(cluster_model, GaussianMixture):
        cluster_model.fit(xtr)
        cluster_assignments = cluster_model.predict(xtr)
        cluster_centers = cluster_model.means_

    # Get the cluster samples, labels, and minimum distances using the up_memberc function
    cluster_samples, cluster_samples_labels, min_distances = up_memberc(phi=phi,
                                                                        cluster_c=cluster_centers,
                                                                        X_train=np.array(xtr),
                                                                        y_train=np.array(ytr),
                                                                        distance_metrics=distance_metrics)

    # Create a dictionary where the key is "cluster_i" and the value is the indexes of samples in cluster i
    clusters_indexes = {f"cluster_{i}": np.where(cluster_assignments == i)[0]
                        for i in range(len(cluster_samples))}

    # Create a dictionary where the key is "cluster_i" and the value is the uncommon indexes between
    # the samples in cluster i and the samples in cluster_samples[i]

    uncommon_indexes_clusters_cluster_samples = {f"cluster_{i}": list(set(clusters_indexes[f"cluster_{i}"].tolist()) ^
                                                                      set(cluster_samples[i].index.values.tolist()))
                                                 for i in range(len(cluster_samples))}

    # Return all the results
    Results = {'cluster_assignments': cluster_assignments, 'cluster_model': cluster_model,
               'cluster_centers': cluster_centers,
               'min_distances': min_distances, 'cluster_samples': cluster_samples,
               'cluster_samples_lables': cluster_samples_labels, 'clusters_indexes': clusters_indexes,
               'uncommon_indexes_clusters_cluster_samples': uncommon_indexes_clusters_cluster_samples}
    return Results


def remove_minority(cluster_samples, cluster_samples_lables):

    rm_cluster_samples = []
    rm_cluster_samples_lables = []
    for cs, csy in zip(cluster_samples, cluster_samples_lables):
            # Combine samples and labels for processing
            combined = cs.assign(label=csy)

            # Calculate label frequencies as a percentage
            label_frequencies = combined.label.value_counts(normalize=True)

            # Identify labels to remove based on frequency threshold
            labels_to_keep = label_frequencies[label_frequencies >= 0.01].index

            # Keep rows with labels above the frequency threshold
            filtered_combined = combined[combined.label.isin(labels_to_keep)]

            # Separate samples and labels
            rm_cluster_samples_lables.append(filtered_combined.label)
            filtered_combined = filtered_combined.drop(columns='label')
            rm_cluster_samples.append(filtered_combined)

    return rm_cluster_samples, rm_cluster_samples_lables

random_state_cluster_samples_shuffling = 42
def resampling(clusterig_method, X_train_dist, y_train_dist, phi, random_state_cluster_samples_shuffling,
               distance_metrics):
    sampling_results = sampling(cluster_model=clusterig_method, distance_metrics=distance_metrics, phi=phi, xtr=X_train_dist, ytr=y_train_dist)
    cluster_model = sampling_results['cluster_model']
    cluster_centers = sampling_results['cluster_centers'].copy()
    cluster_samples = sampling_results['cluster_samples'].copy()
    cluster_samples_labels = sampling_results['cluster_samples_lables'].copy()
    min_distances = sampling_results['min_distances'].copy()
    cluster_assignments = sampling_results['cluster_assignments']

    # ~~

    # remove minority samples from each resampled cluster samples
    rm_cluster_samples, rm_cluster_samples_labels = remove_minority(cluster_samples, cluster_samples_labels)
    rm_cluster_samples = [x.sample(frac=1, random_state=random_state_cluster_samples_shuffling)
                          for x in rm_cluster_samples]
    rm_cluster_samples_labels = [y.sample(frac=1, random_state=random_state_cluster_samples_shuffling)
                                 for y in rm_cluster_samples_labels]

    Results = {'rm_cluster_samples': rm_cluster_samples, 'rm_cluster_samples_labels': rm_cluster_samples_labels,
               'cluster_centers': cluster_centers, 'min_distances': min_distances,
               'cluster_assignments': cluster_assignments, 'cluster_model': cluster_model}
    return Results

n_jobs = -1  # Use all available cores
gmm = GaussianMixture()
# new for more efficient running
def process_phi_nc_pair(phi, n_c, random_state_clustering, single_preprocessing_res_grid, y_train, random_state_cluster_samples_shuffling, distance_metrics):
    """Process a single phi, n_c pair for resampling"""
    gmm_params = {
        'n_components': n_c,
        'random_state': random_state_clustering,
    }
    clusterig_method = gmm.set_params(**gmm_params)
    single_resampling_res = resampling(
        clusterig_method=clusterig_method,
        X_train_dist=single_preprocessing_res_grid['X_train'],
        y_train_dist=y_train, phi=phi,
        random_state_cluster_samples_shuffling=random_state_cluster_samples_shuffling,
        distance_metrics=distance_metrics)
    
    return {'phi': phi, 'n_c': n_c, 'single_resampling_res': single_resampling_res}

def resampling_grid(phi_list, n_c_list, random_state_clustering, single_preprocessing_res_grid, y_train, distance_metrics):
    print(f"Running parallel resampling with {n_jobs} jobs")
    
    # Prepare all parameter combinations
    param_combinations = [(phi, n_c, random_state_clustering, single_preprocessing_res_grid, 
                           y_train, random_state_cluster_samples_shuffling, distance_metrics) 
                          for phi in phi_list for n_c in n_c_list]
    
    # Run in parallel
    results = Parallel(n_jobs=n_jobs, verbose=10)(
        delayed(process_phi_nc_pair)(*params) for params in param_combinations
    )
    
    # Convert results to DataFrame
    df_single_resampling_res_grid = pd.DataFrame(results)
    return df_single_resampling_res_grid


class MyDataset(Dataset):
    def __init__(self, X, y):
        self.X = X
        self.y = y

    def __len__(self):
        return len(self.X)

    def __getitem__(self, index):
        X = torch.tensor(self.X[index], dtype=torch.float32)
        y = torch.tensor(self.y[index], dtype=torch.long)
        return X, y


def single_preprocessing(X_train, X_test, y_train, n_components, random_state_pca):
    # Normalizing
    main_scaler = StandardScaler(with_mean=True, with_std=True)
    main_scaler.fit(X_train)
    X_train_scaled = main_scaler.transform(X_train)
    X_test_scaled = main_scaler.transform(X_test)
    # ~~

    if n_components != 'auto':

        # Transformation by PCA
        # Initialize the PCA model
        pca = PCA(n_components=n_components, random_state=random_state_pca)

        # Fit the PCA model to the training data
        pca.fit(X_train_scaled)

        # Transform the training data
        X_train_transformed = pca.transform(X_train_scaled)

        # Transform the new data using the PCA model
        X_test_transformed = pca.transform(X_test_scaled)

    # ~~

    elif n_components == 'auto':

        # Fit PCA with all components
        pca = PCA()
        pca.fit(X_train_scaled)

        # Get the explained variance ratio
        variance_ratio = pca.explained_variance_ratio_

        # Get the cumulative sum of the explained variance ratio
        cumulative_variance_ratio = np.cumsum(variance_ratio)

        # Define the desired level of variance to keep
        variance_threshold = 0.95

        # Find the index where the cumulative variance ratio reaches or exceeds the threshold
        optimal_index = next((i for i, v in enumerate(cumulative_variance_ratio) if v >= variance_threshold), None)

        # Fit PCA with optimal number of components
        pca_optimal = PCA(n_components=optimal_index + 1, random_state=random_state_pca)  # Add 1 because index starts
        # from 0

        X_train_transformed = pca_optimal.fit_transform(X_train_scaled)

        # Transform the test data using the PCA model
        X_test_transformed = pca_optimal.transform(X_test_scaled)

    # ~~

    # Encode your labels
    le = LabelEncoder()
    encoded_y_train = le.fit_transform(y_train)

    Results = {'X_test': X_test_transformed, 'X_train': X_train_transformed}

    return Results


# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ End Main functions ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~


def load_activity_map():
    map = {0: 'transient', 1: 'lying', 2: 'sitting', 3: 'standing', 4: 'walking', 5: 'running', 6: 'cycling',
           7: 'Nordic_walking', 9: 'watching_TV', 10: 'computer_work', 11: 'car driving', 12: 'ascending_stairs',
           13: 'descending_stairs', 16: 'vacuum_cleaning', 17: 'ironing', 18: 'folding_laundry', 19: 'house_cleaning',
           20: 'playing_soccer', 24: 'rope_jumping'}
    return map


# clean pamap data
def generate_three_IMU(name):
    x = name + '_x'
    y = name + '_y'
    z = name + '_z'
    return [x, y, z]


def generate_four_IMU(name):
    x = name + '_x'
    y = name + '_y'
    z = name + '_z'
    w = name + '_w'
    return [x, y, z, w]


def generate_cols_IMU(name):
    # temp
    temp = name + '_temperature'
    output = [temp]
    # acceleration 16
    acceleration16 = name + '_3D_acceleration_16'
    acceleration16 = generate_three_IMU(acceleration16)
    output.extend(acceleration16)
    # acceleration 6
    acceleration6 = name + '_3D_acceleration_6'
    acceleration6 = generate_three_IMU(acceleration6)
    output.extend(acceleration6)
    # gyroscope
    gyroscope = name + '_3D_gyroscope'
    gyroscope = generate_three_IMU(gyroscope)
    output.extend(gyroscope)
    # magnometer
    magnometer = name + '_3D_magnetometer'
    magnometer = generate_three_IMU(magnometer)
    output.extend(magnometer)
    # oreintation
    oreintation = name + '_4D_orientation'
    oreintation = generate_four_IMU(oreintation)
    output.extend(oreintation)
    return output


def load_IMU():
    output = ['time_stamp', 'activity_id', 'heart_rate']
    hand = 'hand'
    hand = generate_cols_IMU(hand)
    output.extend(hand)
    chest = 'chest'
    chest = generate_cols_IMU(chest)
    output.extend(chest)
    ankle = 'ankle'
    ankle = generate_cols_IMU(ankle)
    output.extend(ankle)
    return output


def load_subjects(root='Data/PAMAP2/Protocol/subject'):
    output = pd.DataFrame()
    cols = load_IMU()

    for i in range(101, 110):
        path = root + str(i) + '.dat'
        subject = pd.read_table(path, header=None, sep=r'\s+')
        subject.columns = cols
        subject['id'] = i
        output = output.append(subject, ignore_index=True)
    output.reset_index(drop=True, inplace=True)
    return output


def confusion_mn(y_real: object, y_pred: object, classes: object, plot: object, title: object) -> object:
    cm = confusion_matrix(y_real, y_pred)
    df_cm = pd.DataFrame(cm, index=classes, columns=classes)

    if plot:
        # Create heatmap using seaborn
        fig, ax = plt.subplots(figsize=(10, 10))
        sns.heatmap(df_cm, annot=True, cmap="Blues", fmt="d", ax=ax)
        # Add title and axis labels
        plt.title(title)
        plt.xlabel("Predicted Label")
        plt.ylabel("Actual Label")
        plt.xticks(rotation=30)
        plt.yticks(rotation=60)

        # Show plot
        plt.show()
    return df_cm

# Function imported by other modules
def sort_by_score(lst):
    # Create a new list to store the sorted dictionaries
    sorted_lst = []
    # Loop through each dictionary in the original list
    for d in lst:
        # Convert the dictionary into a list of tuples, where each tuple contains the classifier name and the
        # best_score_
        items = list(d.items())
        # Sort the list of tuples by the best_score_ in descending order
        items.sort(key=lambda x: x[1]['best_score_'], reverse=True)
        # Convert the sorted list of tuples back into a dictionary and append it to the new list
        sorted_d = dict(items)
        sorted_lst.append(sorted_d)
    # Return the new list
    return sorted_lst

# $$$$$$$$$$$$$$$$$$$$$$$$$$$$
