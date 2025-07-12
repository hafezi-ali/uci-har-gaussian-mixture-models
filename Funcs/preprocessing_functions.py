import numpy as np
import pandas as pd
import scipy.stats as stats
import seaborn as sns
from matplotlib import pyplot as plt
from tqdm import tqdm
import re


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
    # filtered_samples_outlier_iqr = dict()
    # filtered_samples_outlier_zs = dict()

    # for feature in outlier_cols:
    for feature in tqdm(outlier_cols, desc="Processing Features"):

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
            # filtered_samples_outlier_iqr[feature] = data[v_iqr]

        # Z-score method
        # Identify the outliers using the Z-score threshold
        if threshold:
            # Calculate the Z-scores for the given column
            z_scores = np.abs(data[feature] - data[feature].mean()) / data[feature].std()

            # Count the number of outliers
            v_zs = z_scores > threshold
            count_outliers_zs[feature] = v_zs.sum()

            # filter samples that have outliers in specific feature
            # filtered_samples_outlier_zs[feature] = data[v_zs]

        count_outliers = {'count_outliers_iqr': count_outliers_iqr, 'count_outliers_zs': count_outliers_zs}
        # filtered_samples_outlier = {'filtered_samples_outlier_iqr': filtered_samples_outlier_iqr,
        #                             'filtered_samples_outlier_zs': filtered_samples_outlier_zs}

        if not just_info:
            # Dictionary of statistical indicator for handling outliers
            st_inds = {
                'median': data[feature].median(),
                # 'mean_zs': data[~v_zs][feature].mean(), *
                'mean': data[feature].mean(),
                'mode': data[feature].mode().iloc[0]
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

    return data_out, count_outliers


def plot_outlier_handling(df, no_df, column):
    # Determine global limits for consistency
    x_min = min(df[column].min(), no_df['IQR'][column].min(), no_df['ZS'][column].min())
    x_max = max(df[column].max(), no_df['IQR'][column].max(), no_df['ZS'][column].max())

    # Distribution plots
    fig, ax = plt.subplots(1, 3, figsize=(18, 6))

    sns.histplot(no_df['IQR'][column], ax=ax[0], kde=True)
    ax[0].set_title("IQR")
    ax[0].set_xlim(x_min, x_max)

    sns.histplot(no_df['ZS'][column], ax=ax[1], kde=True)
    ax[1].set_title("ZS")
    ax[1].set_xlim(x_min, x_max)

    sns.histplot(df[column], ax=ax[2], kde=True)
    ax[2].set_title("Original Data")
    ax[2].set_xlim(x_min, x_max)

    plt.tight_layout()
    plt.show()

    # Boxplots
    fig, ax = plt.subplots(1, 3, figsize=(18, 6))

    sns.boxplot(x=no_df['IQR'][column], ax=ax[0])
    ax[0].set_title(f"{column} Distribution: IQR")
    ax[0].set_xlim(x_min, x_max)

    sns.boxplot(x=no_df['ZS'][column], ax=ax[1])
    ax[1].set_title(f"{column} Distribution: ZS")
    ax[1].set_xlim(x_min, x_max)

    sns.boxplot(x=df[column], ax=ax[2])
    ax[2].set_title(f"{column} Distribution: Original Data")
    ax[2].set_xlim(x_min, x_max)

    plt.tight_layout()
    plt.show()

    # Q-Q Plots
    fig, ax = plt.subplots(1, 3, figsize=(18, 6))

    stats.probplot(df[column].values.ravel(), dist="norm", plot=ax[0])
    ax[0].set_title(f"Q-Q Plot: Original Data")
    ax[0].set_xlim(-3, 3)  # Standardized Q-Q plots usually range from -3 to 3

    stats.probplot(no_df['IQR'][column].values.ravel(), dist="norm", plot=ax[1])
    ax[1].set_title(f"Q-Q Plot: IQR")
    ax[1].set_xlim(-3, 3)

    stats.probplot(no_df['ZS'][column].values.ravel(), dist="norm", plot=ax[2])
    ax[2].set_title(f"Q-Q Plot: ZS")
    ax[2].set_xlim(-3, 3)

    plt.tight_layout()
    plt.show()
