
# *** modules/clustering.py ***
from sklearn.mixture import GaussianMixture
from tqdm import tqdm
import matplotlib.pyplot as plt


def criteria_values(x, n_clusters):
    # Initialize empty lists to store the criteria values
    bic = []
    aic = []

    for k in tqdm(n_clusters, position=0, leave=True, colour='green', desc='Calculating BIC and AIC for GMM'):
        # Fit a Gaussian mixture model with k components
        gmm = GaussianMixture(n_components=k, random_state=None).fit(x)

        # Append the BIC and AIC values to the lists
        bic.append(gmm.bic(x))
        aic.append(gmm.aic(x))

    cluster_criteria_values = {'bic': bic, 'aic': aic}
    return cluster_criteria_values


def perform_gmm_clustering(x, n_components):
    gmm = GaussianMixture(n_components=n_components, random_state=42)
    gmm.fit(x)
    cluster_assignments = gmm.predict(x)
    cluster_centers = gmm.means_

    return cluster_assignments, cluster_centers


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
    # plt.savefig(f"{plot_save_path}/criteria_values.png")
    cluster_evaluation_criteria_fig = plt.gcf()
    # plt.show()
    return cluster_evaluation_criteria_fig

