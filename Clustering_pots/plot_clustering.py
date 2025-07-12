import os
import numpy as np
import matplotlib.pyplot as plt
from sklearn.mixture import GaussianMixture
from sklearn.datasets import make_blobs
from matplotlib.patches import Ellipse
from scipy.spatial import ConvexHull

# Data generation
def generate_complex_2d_data(n_samples=1000, random_state=42):
    np.random.seed(random_state)
    centers = np.array([[2, 2], [5, 4], [2, 6]])
    cluster_std = [0.7, 0.6, 0.6]
    X, y_true = make_blobs(
        n_samples=n_samples,
        centers=centers,
        cluster_std=cluster_std,
        random_state=random_state
    )
    noise = np.random.normal(0, 0.09, X.shape)
    X += noise
    return X, y_true

# Utility to select nearest members to a center
def up_memberc(X, center, n_select=None, phi=None, cov=None, metric='euclidean'):
    n_samples = X.shape[0]
    if n_select is None:
        if phi is None:
            raise ValueError("Either n_select or phi must be provided.")
        n_select = int(np.floor(phi * n_samples))
    if metric == 'euclidean':
        dists = np.linalg.norm(X - center, axis=1)
    elif metric == 'mahalanobis':
        if cov is None:
            raise ValueError("cov must be provided for Mahalanobis distance.")
        inv_cov = np.linalg.inv(cov)
        diffs = X - center
        dists = np.sqrt(np.einsum('ij,ij->i', diffs @ inv_cov, diffs))
    else:
        raise ValueError("Unsupported metric: choose 'euclidean' or 'mahalanobis'.")
    sorted_idx = np.argsort(dists)
    chosen_idx = sorted_idx[:n_select]
    return chosen_idx, X[chosen_idx]

# GMM fitting and sampling
def fit_gmm_and_sample(X, n_components=3, phi=0.5, metric='euclidean'):
    gmm = GaussianMixture(
        n_components=n_components,
        covariance_type='full',
        random_state=42,
        max_iter=100,
        tol=1e-6
    )
    gmm.fit(X)
    sampled_indices = {}
    sampled_subsets = {}
    n_select = int(X.shape[0] * phi)
    for i in range(n_components):
        center_i = gmm.means_[i]
        cov_i = gmm.covariances_[i] if metric == 'mahalanobis' else None
        idx, subset = up_memberc(
            X, center=center_i, n_select=n_select, cov=cov_i, metric=metric
        )
        sampled_indices[i] = idx
        sampled_subsets[i] = subset
    responsibilities = gmm.predict_proba(X)
    return gmm, sampled_indices, sampled_subsets, responsibilities

# Plotting utilities
def plot_confidence_ellipses(ax, means, covariances, colors, alpha=0.3, dashed=False, scale=3):
    for i, (mean, cov) in enumerate(zip(means, covariances)):
        eigvals, eigvecs = np.linalg.eigh(cov)
        angle = np.degrees(np.arctan2(eigvecs[1, 0], eigvecs[0, 0]))
        width = scale * np.sqrt(eigvals[0])
        height = scale * np.sqrt(eigvals[1])
        ls = '--' if dashed else '-'
        ellipse = Ellipse(
            mean, width, height, angle=angle,
            facecolor='none' if dashed else colors[i], alpha=alpha if not dashed else 1.0,
            edgecolor=colors[i], linewidth=2, linestyle=ls
        )
        ax.add_patch(ellipse)

# Comprehensive visualization
def create_comprehensive_gmm_visualization(X, n_components=3, phi=0.5, metric='euclidean'):
    gmm, sampled_indices, sampled_subsets, responsibilities = fit_gmm_and_sample(
        X, n_components, phi, metric
    )
    colors = ['#FF6B6B', '#4ECDC4', '#45B7D1']
    fig, axes = plt.subplots(1, 3, figsize=(18, 6))
    ax1, ax2, ax3 = axes

    # Plot (a)
    ax1.scatter(X[:, 0], X[:, 1], color='gray', alpha=0.6, s=30)
    ax1.set_title('(a) Original Dataset', y=-0.15)
    ax1.grid(True, alpha=0.3)

    # Plot (b)
    pred_labels = gmm.predict(X)
    for i in range(n_components):
        mask = pred_labels == i
        ax2.scatter(X[mask, 0], X[mask, 1], color=colors[i], alpha=0.6, s=30, label=f'Comp {i+1}')
    plot_confidence_ellipses(ax2, gmm.means_, gmm.covariances_, colors, scale=3)
    ax2.scatter(gmm.means_[:, 0], gmm.means_[:, 1], color='black', marker='x', s=200, linewidths=3)
    ax2.set_title('(b) GMM Fitting (K=3)', y=-0.15)
    ax2.legend(fontsize=10)
    ax2.grid(True, alpha=0.3)

    # Plot (c)
    sampled_all = np.concatenate(list(sampled_indices.values()))
    colors_pts = [colors[label] for label in pred_labels[sampled_all]]
    ax3.scatter(X[:, 0], X[:, 1], color='lightgray', alpha=0.3, s=20)
    ax3.scatter(X[sampled_all, 0], X[sampled_all, 1], color=colors_pts, alpha=0.6, s=30, linewidths=0.5)
    for i in range(n_components):
        subset = sampled_subsets[i]
        if subset.shape[0] > 2:
            hull = ConvexHull(subset)
            hull_pts = subset[hull.vertices]
            hull_pts = np.vstack([hull_pts, hull_pts[0]])
            ax3.plot(hull_pts[:, 0], hull_pts[:, 1], linestyle='--', linewidth=2, color=colors[i], label=f'Subset {i+1}')
    ax3.set_title('(c) Sampled Points by GMM Label', y=-0.15)
    ax3.legend(fontsize=10, loc='upper right')
    ax3.grid(True, alpha=0.3)

    plt.tight_layout()
    return fig, gmm, sampled_indices, sampled_subsets

# Main execution
if __name__ == "__main__":
    results_dir = os.path.join('Results', 'plots')
    os.makedirs(results_dir, exist_ok=True)
    X, y_true = generate_complex_2d_data()
    fig, gmm, sampled_indices, sampled_subsets = create_comprehensive_gmm_visualization(X)
    fig.savefig(os.path.join(results_dir, "gmm_visualization.png"))
    plt.show()
