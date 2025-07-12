import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib import pyplot as plt
from collections import Counter
from io import StringIO
import sys
from sklearn.manifold import TSNE
from sklearn.mixture import GaussianMixture
import plotly.express as px
from sklearn.metrics import silhouette_score, calinski_harabasz_score
import tqdm
# from yellowbrick.cluster import KElbowVisualizer
from sklearn.preprocessing import StandardScaler
from sklearn.datasets import fetch_openml
from sklearn.decomposition import PCA
# Dimension reduction and clustering libraries
import umap.umap_ as umap
import sklearn.cluster as cluster
# from sklearn.metrics import adjusted_rand_score, adjusted_mutual_info_score
from plotly.figure_factory import create_distplot, create_2d_density, create_facet_grid


def plot_2d_scatter(X, color, title, color_asign, symbol):
    # Create a scatter plot
    fig = px.scatter(
        x=X[:, 0],
        y=X[:, 1],
        color=color,  # Assign colors based on target values
        labels={'color': color_asign},
        title=title,
        opacity=0.8,
        symbol=symbol,
        symbol_sequence=['circle', 'square', 'diamond', 'cross', 'x', 'triangle-up', 'triangle-down', 'star'],
    )
    # Update the layout for better visualization
    fig.update_layout(
        template='plotly_white',
        xaxis_title='UMAP Component 1',
        yaxis_title='UMAP Component 2',
        coloraxis_showscale=True,
        coloraxis_colorbar=dict(title="HAR", x=1.05),
        # legend=dict(x=0, y=1, traceorder="normal"),
        legend=dict(
            x=1.05,  # Position to the right
            y=0.5,  # Centered vertically
            borderwidth=1,
            # itemsizing='constant',
            traceorder="normal"
        )
    )

    # Add marker border and update marker size
    fig.update_traces(
        marker=dict(size=15, line=dict(width=0.5, color='black'))  # Border color
    )

    # Show the plot
    fig.show()


def plot_variance_ratio(pca_parameters):
    plt.figure(figsize=(10, 5))
    plt.bar(range(1, len(pca_parameters['variance_ratio']) + 1), pca_parameters['variance_ratio'], alpha=0.5,
            align='center',
            label='Individual explained variance')
    plt.step(range(1, len(pca_parameters['cumulative_variance_ratio']) + 1),
             pca_parameters['cumulative_variance_ratio'], where='mid',
             label='Cumulative explained variance')
    plt.axhline(y=pca_parameters['variance_threshold'], color='r', linestyle='--',
                label=f'{pca_parameters['variance_threshold'] * 100}% variance threshold')
    plt.axvline(x=pca_parameters['optimal_index'] + 1, color='g', linestyle='--',
                label=f'Optimal number of components: {pca_parameters['optimal_index'] + 1}')
    plt.ylabel('Explained variance ratio')
    plt.xlabel('Principal component index')
    plt.legend(loc='best')
    plt.title('Explained Variance Ratio by Principal Component')
    plt.show()


def plt_pca_scatterplot(X_2d, title):
    plt.figure(figsize=(10, 7))
    sns.scatterplot(x=X_2d[:, 0], y=X_2d[:, 1])
    plt.xlabel('First Principal Component')
    plt.ylabel('Second Principal Component')
    plt.title(title)
    plt.show()


def optimized_pca(X, variance_threshold):
    # Assuming X is your dataset
    pca = PCA()
    pca.fit(X)

    # Get the explained variance ratio
    variance_ratio = pca.explained_variance_ratio_

    # Get the cumulative sum of the explained variance ratio
    cumulative_variance_ratio = np.cumsum(variance_ratio)

    # Find the index where the cumulative variance ratio reaches or exceeds the threshold
    optimal_index = next((i for i, v in enumerate(cumulative_variance_ratio) if v >= variance_threshold), None)

    # Fit PCA with the optimal number of components
    pca_optimal = PCA(n_components=optimal_index + 1, random_state=42)
    X_pca_transformed = pca_optimal.fit_transform(X)

    pca_outputs = {'X_pca_transformed': X_pca_transformed, 'variance_ratio': variance_ratio,
                   'cumulative_variance_ratio': cumulative_variance_ratio,
                   'variance_threshold': variance_threshold, 'optimal_index': optimal_index}
    return pca_outputs


def criteria_values(X, n_clusters):
    # Initialize empty lists to store the criteria values
    bic = []
    aic = []

    for k in tqdm.tqdm(n_clusters, position=0, leave=True, colour='green'):
        # Fit a Gaussian mixture model with k components
        gmm = GaussianMixture(n_components=k, random_state=None).fit(X)

        # Append the BIC and AIC values to the lists
        bic.append(gmm.bic(X))
        aic.append(gmm.aic(X))

    cluster_criteria_values = {'bic': bic, 'aic': aic}
    return cluster_criteria_values


def plot_criteria_values(n_clusters, cluster_criteria_values, title):
    # Plot the criteria values against the number of clusters
    plt.figure(figsize=(12, 8))
    plt.subplot(2, 1, 1)
    plt.plot(n_clusters, cluster_criteria_values['bic'], '-o')
    plt.xlabel('Number of clusters')
    plt.ylabel('bic')
    plt.subplot(2, 1, 2)
    plt.plot(n_clusters, cluster_criteria_values['aic'], '-o')
    plt.xlabel('Number of clusters')
    plt.ylabel('aic')
    plt.suptitle(title)
    plt.show()
