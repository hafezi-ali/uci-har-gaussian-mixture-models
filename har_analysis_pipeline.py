# Import Necessary Libraries
import os.path
import numpy as np
import umap
import pandas as pd
from modules.utils import plot_2d_scatter
from modules.data_processing import handle_outliers, detect_missing_values, plot_outlier_handling
from modules.data_loading import load_data, load_cleaned_data
from modules.eda import perform_eda
from modules.clustering import perform_gmm_clustering, criteria_values, plot_criteria_values
from modules.feature_engineering import (feature_scaling, apply_optimal_pca,
                                         umap_standard_embedding, plot_variance_ratio)
import plotly.io as pio

base_dir = os.path.dirname(os.path.abspath(__file__))
def main():
    plot_save_path = os.path.join(base_dir, 'har_analysis_results', 'plots')
    if not os.path.exists(plot_save_path):
        os.makedirs(plot_save_path)
    # ***************************************** Data Loading and Preprocessing ****************************************
    files = ['Data/processed/data_uci_handled_outliers.csv', 'Data/processed/x_uci_handled_outliers_iqr.csv',
             'Data/processedx_uci_handled_outliers_zs.csv']

    if all(os.path.exists(file) for file in files):
        print("\nLoading cleaned data...")
        data_uci_path = os.path.join(base_dir, 'Data/raw/data_uci.csv')
        if os.path.exists(data_uci_path):
            df = pd.read_csv(os.path.join(base_dir, 'Data/raw/data_uci.csv'))
        else:
            df = load_data()
            df.to_csv(os.path.join(base_dir, 'Data/raw/data_uci.csv'), index=False)
        df_cleaned, x_cleaned_zs_iqr = load_cleaned_data()

    else:
        processed_data_save_path = os.path.join(base_dir, 'Data', 'processed')
        if not os.path.exists(processed_data_save_path):
            os.makedirs(processed_data_save_path)
        # Load and Preprocess Data
        print("Loading and Preprocessing Data...")

        data_uci_path = os.path.join(base_dir, 'Data/raw/data_uci.csv')
        if os.path.exists(data_uci_path):
            df = pd.read_csv(os.path.join(base_dir, 'Data/raw/data_uci.csv'))
        else:
            df = load_data()
            df.to_csv(os.path.join(base_dir, 'Data/raw/data_uci.csv'), index=False)

        #  Handle Missing Values
        print("\nHandling Missing Values...")
        miss_values, check_missing_fig = detect_missing_values(df)
        check_missing_fig.savefig(os.path.join(base_dir,'har_analysis_results/plots/check_missing.png'))

        # Handling Outliers
        print("\nHandling Outliers...")
        df_features = df.drop(['Activity', 'ActivityName', 'subject'], axis=1)
        # * supported imputation_methods: 'median', 'mean', 'mode'
        x_cleaned_zs_iqr = handle_outliers(df_features, methode='median')

        # Update DataFrame
        df_cleaned = x_cleaned_zs_iqr['ZS'].copy()
        df_cleaned[['Activity', 'ActivityName', 'subject']] = df[['Activity', 'ActivityName', 'subject']]

        df_cleaned.to_csv(os.path.join(processed_data_save_path, 'data_uci_handled_outliers.csv'), index=False)
        x_cleaned_zs_iqr['IQR'].to_csv(os.path.join(processed_data_save_path, 'x_uci_handled_outliers_iqr.csv'),
                                       index=False)
        x_cleaned_zs_iqr['ZS'].to_csv(os.path.join(processed_data_save_path, 'x_uci_handled_outliers_zs.csv'),
                                      index=False)

    # Distribution, box plots and qq plots for paring original and outlier handled data
    x_raw = df.drop(['Activity', 'ActivityName', 'subject'], axis=1)

    dist_fig, boxplot_fig, qqplot_fig = plot_outlier_handling(x_raw, x_cleaned_zs_iqr, 'tBodyAcc-mean()-X')

    dist_fig.savefig(os.path.join(base_dir, 'har_analysis_results/plots/compare_original_vs_outlier_handled_distributions.png'))
    boxplot_fig.savefig(os.path.join(base_dir, 'har_analysis_results/plots/compare_original_vs_outlier_handled_boxplots.png'))
    qqplot_fig.savefig(os.path.join(base_dir, 'har_analysis_results/plots/compare_original_vs_outlier_handled_qqplots.png'))

    #  ***************************************** Exploratory Data Analysis (EDA) ***************************************

    print("\nPerforming Exploratory Data Analysis...")
    bar_plot_fig, activity_distribution_plot, stationary_activities_fig = perform_eda(df_cleaned)
    bar_plot_fig.savefig(os.path.join(base_dir, 'har_analysis_results/plots/bar_plot.png'))
    activity_distribution_plot.savefig(os.path.join(base_dir, 'har_analysis_results/plots/activity_distribution.png'))
    stationary_activities_fig.savefig(os.path.join(base_dir, 'har_analysis_results/plots/stationary_activities.png'))

    #  ***************************************** Feature Engineering  *****************************************

    print("\nFeature Engineering...")
    x, yn, yc, x_scaled = feature_scaling(df_cleaned)

    # ** umap visualization plot 2d_density of embedded features by standard umap (by default parameters) for better
    # visualization pattern of data

    # title = 'standard_embedding'
    emb_type = 'Raw Features'
    title = f'Standard 2D Density UMAP Visualization of HAR Activities for {emb_type}'
    umap_2d_density_fig, x_umap_standard_embedded = umap_standard_embedding(x, title)
    umap_2d_density_fig.write_html(os.path.join(base_dir, 'har_analysis_results/plots/umap_2d_density.html'))
    pio.write_image(umap_2d_density_fig, os.path.join(base_dir, 'har_analysis_results/plots/umap_2d_density.png'))

    # umap_2d_density_fig.show()

    # title = 'scaled_standard_embedding'
    emb_type = 'Scaled Raw Features'
    title = f'Standard 2D Density UMAP Visualization of HAR Activities for {emb_type}'
    scaled_umap_2d_density_fig, x_scaled_umap_standard_embedded = umap_standard_embedding(x_scaled, title)
    scaled_umap_2d_density_fig.write_html(os.path.join(base_dir, 'har_analysis_results/plots/scaled_umap_2d_density.html'))
    pio.write_image(scaled_umap_2d_density_fig, os.path.join(base_dir, 'har_analysis_results/plots/scaled_umap_2d_density.png'))
    # scaled_umap_2d_density_fig.show()
    # ** plot 2d_scatter of embedded features and labels of data points by standard umap for better visualization
    # pattern of data

    emb_type = 'Raw Features'
    title = f'Standard 2D Scatter UMAP Visualization of HAR Activities for {emb_type}'
    umap_2d_scatter_fig = plot_2d_scatter(x=x_umap_standard_embedded, color=yc, title=title, symbol=None)
    umap_2d_scatter_fig.write_html(os.path.join(base_dir, 'har_analysis_results/plots/umap_2d_scatter.html'))
    pio.write_image(umap_2d_scatter_fig, os.path.join(base_dir, 'har_analysis_results/plots/umap_2d_scatter.png'))


    # ** Optimal Dimensionality Reduction by PCA
    print("\nDimensionality Reduction (PCA)...")
    x_scaled_pca_transformed, scaled_pca_parameters = apply_optimal_pca(x_scaled)
    title = 'Explained Variance Ratio by Principal Component for Scaled PCA-Transformed Features'
    variance_ratio_fig = plot_variance_ratio(pca_parameters=scaled_pca_parameters, title=title)
    variance_ratio_fig.savefig(os.path.join(base_dir, 'har_analysis_results/plots/variance_ratio.png'))

    # **** dimension reduction by umap
    x_scaled_umap_embedded = umap.UMAP(
        n_neighbors=30,
        min_dist=0.0,
        n_components=83,
        random_state=42,
    ).fit_transform(x_scaled)

    #  ***************************************** Modeling and Evaluation  *****************************************

    # ************ Clustering and Visualization on x_scaled_pca_transformed data
    # determine the number of clusters

    # Plot the criteria values against the number of clusters to determine number of clusters
    # Define a range of possible number of clusters
    n_clusters = np.arange(2, 15)
    scaled_cluster_criteria_values = criteria_values(x=x_scaled_pca_transformed, n_clusters=n_clusters)

    title = 'Cluster Evaluation Criteria for Scaled PCA-Transformed Features'
    cluster_evaluation_criteria_fig = plot_criteria_values(n_clusters=n_clusters,
                                                           cluster_criteria_values=scaled_cluster_criteria_values,
                                                           title=title)
    cluster_evaluation_criteria_fig.savefig(os.path.join(base_dir, 'har_analysis_results/plots/pca_cluster_evaluation_criteria.png'))

    # perform gmm clustering on scaled x that transformed by pca

    n_components = 4
    pca_scaled_cluster_assignments, pca_scaled_cluster_centers = perform_gmm_clustering(x=x_scaled_pca_transformed,
                                                                                        n_components=n_components)

    # (x_scaled_pca_transformed: scaled x that transformed by pca)

    # plot 2d scatter on embedded x_scaled_pca_transformed by standard umap and
    # showing datapoints classes and their cluster numbers that assigned by gmm clustering methode.

    emb_type = 'Scaled PCA-Transformed Features and Cluster Assignments'
    title = f'Standard 2D UMAP Visualization of HAR Activities for {emb_type}'
    pca_scaled_cluster_assignments_fig = plot_2d_scatter(x=x_umap_standard_embedded,
                                                         color=pca_scaled_cluster_assignments.astype('str'),
                                                         title=title,
                                                         symbol=yc)
    pca_scaled_cluster_assignments_fig.write_html(os.path.join(base_dir,
                                                               'har_analysis_results/plots/pca_scaled_cluster_assignments.html'))
    pio.write_image(pca_scaled_cluster_assignments_fig, os.path.join(base_dir,
                                                               'har_analysis_results/plots/pca_scaled_cluster_assignments.png'))

    # ************ Clustering and Visualization on x_scaled_umap_embedded data

    # determine the number of clusters

    # Plot the criteria values against the number of clusters to determine number of clusters
    # Define a range of possible number of clusters

    title = 'Cluster Evaluation Criteria for Scaled UMAP-Embedded Features'
    # Plot the criteria values against the number of clusters
    # Define a range of possible number of clusters
    n_clusters = np.arange(2, 15)
    x_scaled_umap_cluster_criteria_values = criteria_values(x=x_scaled_umap_embedded, n_clusters=n_clusters)

    cluster_evaluation_criteria_fig = plot_criteria_values(n_clusters=n_clusters,
                                                           cluster_criteria_values=x_scaled_umap_cluster_criteria_values,
                                                           title=title)
    cluster_evaluation_criteria_fig.savefig(os.path.join(base_dir, 'har_analysis_results/plots/umap_cluster_evaluation_criteria.png'))

    # perform gmm clustering on scaled x that embedded by umap
    # clustering by umap

    umap_scaled_cluster_assignments, umap_scaled_cluster_centers = perform_gmm_clustering(x=x_scaled_umap_embedded,
                                                                                          n_components=n_components)

    # (x_scaled_umap_embedded: scaled x that embedded by umap)

    # plot 2d scatter on embedded x_scaled_umap_embedded by standard umap and
    # showing datapoints classes and their cluster numbers that assigned by gmm clustering methode.
    emb_type = 'Scaled UMAP-Embedded Features and Cluster Assignments'
    title = f'Standard 2D UMAP Visualization of HAR Activities for {emb_type}'
    umap_scaled_cluster_assignments_fig = plot_2d_scatter(x=x_umap_standard_embedded,
                                                          color=umap_scaled_cluster_assignments.astype('str'),
                                                          title=title,
                                                          symbol=yc)
    umap_scaled_cluster_assignments_fig.write_html(os.path.join(base_dir, 'har_analysis_results/plots/umap_scaled_cluster_assignments.html'))
    pio.write_image(umap_scaled_cluster_assignments_fig, os.path.join(base_dir, 'har_analysis_results/plots/umap_scaled_cluster_assignments.png'))
    print("\nPipeline Completed Successfully!")


if __name__ == "__main__":
    main()
    
