import numpy as np
import matplotlib.pyplot as plt
from sklearn.mixture import GaussianMixture
from sklearn.datasets import make_blobs
from matplotlib.patches import Ellipse
import seaborn as sns

# Set style for publication-quality plots
plt.style.use('seaborn-v0_8-whitegrid')
sns.set_palette("husl")

def generate_complex_2d_data(n_samples=1000, random_state=42):
    """Generate 2D data with three distinct clusters"""
    np.random.seed(random_state)
    centers = np.array([[2, 2], [6, 4], [2, 6]])
    cluster_std = [0.7, 0.6, 0.6]
    X, y_true = make_blobs(n_samples=n_samples, centers=centers,
                           cluster_std=cluster_std, random_state=random_state)
    noise = np.random.normal(0, 0.09, X.shape)
    X += noise
    return X, y_true


def fit_gmm_and_sample(X, n_components=3, phi=0.5, random_state=42):
    gmm = GaussianMixture(n_components=n_components, random_state=random_state,
                          covariance_type='full', max_iter=100, tol=1e-6)
    gmm.fit(X)
    responsibilities = gmm.predict_proba(X)
    sampled_indices = {}
    sampled_subsets = {}
    n_select = int(X.shape[0] * phi)
    for i in range(n_components):
        sorted_idx = np.argsort(responsibilities[:, i])[::-1]
        selected = sorted_idx[:n_select]
        sampled_indices[i] = selected
        sampled_subsets[i] = X[selected]
    return gmm, sampled_indices, sampled_subsets, responsibilities


def plot_confidence_ellipses(ax, means, covariances, colors, alpha=0.3, dashed=False):
    for i, (mean, cov) in enumerate(zip(means, covariances)):
        eigvals, eigvecs = np.linalg.eigh(cov)
        angle = np.degrees(np.arctan2(eigvecs[1, 0], eigvecs[0, 0]))
        width = 2 * np.sqrt(2 * eigvals[0])
        height = 2 * np.sqrt(2 * eigvals[1])
        ls = '--' if dashed else '-'
        ellipse = Ellipse(mean, width, height, angle=angle,
                          facecolor='none' if dashed else colors[i],
                          alpha=alpha if not dashed else 1.0,
                          edgecolor=colors[i], linewidth=2, linestyle=ls)
        ax.add_patch(ellipse)


def create_comprehensive_gmm_visualization(X, n_components=3, phi=0.5):
    gmm, sampled_indices, sampled_subsets, responsibilities = fit_gmm_and_sample(X, n_components, phi)
    colors = ['#FF6B6B', '#4ECDC4', '#45B7D1']
    fig = plt.figure(figsize=(20, 12))

    # (a)
    ax1 = plt.subplot(2, 3, 1)
    ax1.scatter(X[:, 0], X[:, 1], color='gray', alpha=0.6, s=30)
    ax1.set(title='(a) Original Dataset', xlabel='Feature 1', ylabel='Feature 2')
    ax1.grid(True, alpha=0.3)

    # (b)
    ax2 = plt.subplot(2, 3, 2)
    pred_labels = gmm.predict(X)
    for i in range(n_components):
        mask = pred_labels == i
        ax2.scatter(X[mask, 0], X[mask, 1], color=colors[i], alpha=0.6, s=30, label=f'Comp {i+1}')
    plot_confidence_ellipses(ax2, gmm.means_, gmm.covariances_, colors)
    ax2.scatter(gmm.means_[:,0], gmm.means_[:,1], color='black', marker='x', s=200, linewidths=3)
    ax2.set(title='(b) GMM Fitting (K=3)', xlabel='Feature 1', ylabel='Feature 2')
    ax2.legend(fontsize=10)
    ax2.grid(True, alpha=0.3)

    # (c)
    ax3 = plt.subplot(2, 3, 3)
    x_min, x_max = X[:, 0].min()-1, X[:, 0].max()+1
    y_min, y_max = X[:, 1].min()-1, X[:, 1].max()+1
    xx, yy = np.meshgrid(np.linspace(x_min, x_max, 100), np.linspace(y_min, y_max, 100))
    grid = np.c_[xx.ravel(), yy.ravel()]
    dens = np.exp(gmm.score_samples(grid)).reshape(xx.shape)
    cf = ax3.contourf(xx, yy, dens, levels=20, alpha=0.8, cmap='viridis')
    ax3.scatter(X[:,0], X[:,1], color='white', alpha=0.7, s=20, edgecolors='black')
    plt.colorbar(cf, ax=ax3, label='Density')
    ax3.set(title='(c) GMM Density Distribution', xlabel='Feature 1', ylabel='Feature 2')

    # (d) colored by predicted labels
    ax4 = plt.subplot(2, 3, 4)
    ax4.scatter(X[:,0], X[:,1], color='lightgray', alpha=0.3, s=20)
    pred_labels = gmm.predict(X)
    # plot only sampled points, colored by pred_labels
    sampled_all = np.concatenate(list(sampled_indices.values()))
    colors_pts = [colors[label] for label in pred_labels[sampled_all]]
    ax4.scatter(X[sampled_all,0], X[sampled_all,1], color=colors_pts, alpha=0.6, s=30, edgecolor='black', linewidths=0.5)
    # dashed ellipses per subset
    for i in range(n_components):
        subset = sampled_subsets[i]
        if len(subset)>1:
            cov_s = np.cov(subset.T)
            mean_s = subset.mean(axis=0)
            plot_confidence_ellipses(ax4, [mean_s], [cov_s], [colors[i]], alpha=0.3, dashed=True)
    ax4.set(title='(d) Sampled Points Colored by GMM Label', xlabel='Feature 1', ylabel='Feature 2')
    ax4.grid(True, alpha=0.3)

    # (e)
    ax5 = plt.subplot(2, 3, 5)
    sc = ax5.scatter(X[:,0], X[:,1], c=responsibilities[:,0], cmap='Reds', alpha=0.7, s=30)
    plt.colorbar(sc, ax=ax5, label='Resp')
    ax5.set(title='(e) Resp for Component 1', xlabel='Feature 1', ylabel='Feature 2')
    ax5.grid(True, alpha=0.3)

    # (f)
    ax6 = plt.subplot(2, 3, 6); ax6.axis('off')
    info = f"""GMM Params:\nK={n_components}, φ={phi}, N={X.shape[0]}"""
    ax6.text(0.05,0.95, info, transform=ax6.transAxes, va='top', family='monospace')
    ax6.set(title='(f) GMM Stats')

    plt.tight_layout()
    return fig, gmm, sampled_indices, sampled_subsets

if __name__=="__main__":
    X, _ = generate_complex_2d_data(n_samples=800)
    fig, gmm, idx, subs = create_comprehensive_gmm_visualization(X,3,0.5)
    plt.show()