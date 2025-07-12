import os
import numpy as np
import matplotlib.pyplot as plt
from sklearn.mixture import GaussianMixture
from sklearn.datasets import make_blobs
from matplotlib.patches import Ellipse
from scipy.spatial import ConvexHull

# Data generation (unchanged)
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

# GMM fitting and sampling based on responsibilities (Aligns with Algorithm 1)
def fit_gmm_and_sample_by_responsibility(X, n_components=3, phi=0.5, random_state=42):
    gmm = GaussianMixture(
        n_components=n_components,
        covariance_type='full',
        random_state=random_state,
        max_iter=100,
        tol=1e-6
    )
    gmm.fit(X)

    responsibilities_all_points = gmm.predict_proba(X)

    sampled_indices_per_component = {}
    sampled_data_subsets = {}

    n_total_samples = X.shape[0]
    # Number of instances to select FOR EACH component's subset
    # As per Algorithm 1: "Select int(m * phi) instances with the highest responsibilities into partition C'_i"
    # Here, 'm' is the total number of samples in X, not the number of samples assigned to component i.
    n_select_for_each_subset = int(np.floor(phi * n_total_samples))

    if n_select_for_each_subset == 0 and phi > 0 and n_total_samples > 0:
        n_select_for_each_subset = 1
    if n_select_for_each_subset == 0:
        print(f"Warning: n_select_for_each_subset is 0 for phi={phi}, n_total_samples={n_total_samples}. Subsets will be empty.")


    for i in range(n_components):
        # Responsibilities of all data points for the i-th component
        current_component_responsibilities = responsibilities_all_points[:, i]
        # Sort points by their responsibility for this i-th component (descending)
        sorted_point_indices_for_comp_i = np.argsort(current_component_responsibilities)[::-1]
        # Select top 'n_select_for_each_subset' points
        chosen_point_indices_for_comp_i = sorted_point_indices_for_comp_i[:n_select_for_each_subset]

        sampled_indices_per_component[i] = chosen_point_indices_for_comp_i
        sampled_data_subsets[i] = X[chosen_point_indices_for_comp_i]

    return gmm, sampled_indices_per_component, sampled_data_subsets, responsibilities_all_points


# Plotting utilities for confidence ellipses (largely unchanged)
def plot_confidence_ellipses(ax, means, covariances, colors, alpha=0.3, dashed=False, scale=3):
    for i, (mean, cov_param) in enumerate(zip(means, covariances)):
        cov = cov_param # Assuming 'full' covariance from GMM fitting
        if cov.ndim == 1: # Should not happen if covariance_type='full' and n_features > 1
            if cov.shape[0] == means.shape[1]: cov = np.diag(cov) # diag
            else: cov = np.eye(means.shape[1]) * cov[0] # spherical
        try:
            eigvals, eigvecs = np.linalg.eigh(cov)
            eigvals = np.maximum(eigvals, 1e-9) # Ensure non-negative for sqrt, avoid zero
        except np.linalg.LinAlgError:
            print(f"Warning: Eigenvalue decomposition failed for component {i}. Skipping ellipse.")
            continue
        angle = np.degrees(np.arctan2(eigvecs[1, 0], eigvecs[0, 0]))
        width = scale * np.sqrt(eigvals[0])
        height = scale * np.sqrt(eigvals[1])
        ls = '--' if dashed else '-'
        ellipse = Ellipse(
            xy=mean, width=width, height=height, angle=angle,
            facecolor='none' if dashed else colors[i],
            alpha=alpha if not dashed else 1.0,
            edgecolor=colors[i],
            linewidth=2,
            linestyle=ls
        )
        ax.add_patch(ellipse)

# Comprehensive visualization (Revised for 1x3 layout)
def create_comprehensive_gmm_visualization(X, n_components=3, phi=0.5, random_state=42):
    gmm, sampled_indices, sampled_subsets, _ = fit_gmm_and_sample_by_responsibility(
        X, n_components, phi, random_state
    )

    colors_palette = ['#FF6B6B', '#4ECDC4', '#45B7D1', '#FFD166', '#06D6A0', '#118AB2']
    current_colors = colors_palette[:n_components]

    fig, axes = plt.subplots(1, 3, figsize=(22, 6.5)) # Changed to 1x3
    ax1, ax2, ax3 = axes

    # Subplot (a): Original Dataset
    ax1.scatter(X[:, 0], X[:, 1], color='gray', alpha=0.6, s=30)
    ax1.set_title(f'(a) Original Dataset (N={X.shape[0]})', pad=15)
    ax1.grid(True, linestyle=':', alpha=0.7)
    ax1.set_xlabel('Feature 1')
    ax1.set_ylabel('Feature 2')
    ax1.axis('equal')


    # Subplot (b): GMM Clustering
    pred_labels = gmm.predict(X)
    for i in range(n_components):
        mask = pred_labels == i
        ax2.scatter(X[mask, 0], X[mask, 1], color=current_colors[i], alpha=0.5, s=30, label=f'Data for Comp. {i+1}')
    plot_confidence_ellipses(ax2, gmm.means_, gmm.covariances_, current_colors, scale=2, alpha=0.35) # scale=2 for ~95%
    ax2.scatter(gmm.means_[:, 0], gmm.means_[:, 1], color='black', marker='x', s=100, linewidths=2, label='Comp. Means', zorder=5)
    ax2.set_title(f'(b) GMM Clustering (K={gmm.n_components})', pad=15)
    ax2.legend(fontsize=9)
    ax2.grid(True, linestyle=':', alpha=0.7)
    ax2.set_xlabel('Feature 1')
    ax2.set_ylabel('Feature 2')
    ax2.axis('equal')


    # Subplot (c): GMM-based Resampled Subsets
    ax3.scatter(X[:, 0], X[:, 1], color='lightgray', alpha=0.15, s=20, label='_Original Data (faded)')

    for i in range(n_components):
        subset_data = sampled_subsets[i]
        # Plot points for this subset, colored by GMM component
        ax3.scatter(subset_data[:, 0], subset_data[:, 1], color=current_colors[i], alpha=0.6, s=30, label=f'_Subset Pts {i+1}')
        if subset_data.shape[0] > 2:
            try:
                hull = ConvexHull(subset_data)
                hull_points = subset_data[hull.vertices]
                closed_hull_points = np.vstack([hull_points, hull_points[0]])
                ax3.plot(closed_hull_points[:, 0], closed_hull_points[:, 1],
                         linestyle='--', linewidth=2, color=current_colors[i],
                         label=f'Sampled Subset {i+1}')
            except Exception as e:
                # print(f"Could not compute ConvexHull for subset {i+1} (data shape: {subset_data.shape}): {e}")
                if subset_data.shape[0] > 0:
                     ax3.scatter(subset_data[:,0], subset_data[:,1], facecolors='none', edgecolors=current_colors[i],
                               s=40, marker='o', linewidth=1.5, label=f'Sampled Subset {i+1} (no hull)')
        elif subset_data.shape[0] > 0:
             ax3.scatter(subset_data[:,0], subset_data[:,1], facecolors='none', edgecolors=current_colors[i],
                           s=50, marker='s', linewidth=1.5, label=f'Sampled Subset {i+1} ({subset_data.shape[0]} pt(s))')

    ax3.set_title(f'(c) GMM-based Resampled Subsets ($\phi$={phi:.2f})', pad=15)
    ax3.legend(fontsize=9, loc='best')
    ax3.grid(True, linestyle=':', alpha=0.7)
    ax3.set_xlabel('Feature 1')
    ax3.set_ylabel('Feature 2')
    ax3.axis('equal')

    plt.tight_layout(pad=1.5)
    return fig, gmm, sampled_indices, sampled_subsets

# Density and weight demonstration (Revised to be 1x2, using GMM component parameters)
def demonstrate_density_and_weight_combined(X_data_full, fitted_gmm, test_point_coords=None):
    if test_point_coords is None:
        # Place test point somewhere interesting, e.g., near a boundary or a specific component
        test_point_coords = fitted_gmm.means_[0] + np.random.randn(X_data_full.shape[1]) * 0.3
        # test_point_coords = X_data_full.mean(axis=0) + np.random.randn(X_data_full.shape[1]) * 0.5
        test_point_coords = np.clip(test_point_coords, X_data_full.min(axis=0), X_data_full.max(axis=0))

    n_gmm_components = fitted_gmm.n_components
    colors_palette = ['#FF6B6B', '#4ECDC4', '#45B7D1', '#FFD166', '#06D6A0', '#118AB2']
    current_plot_colors = colors_palette[:n_gmm_components]

    # Create a single figure with 1 row and 2 subplots
    fig, (ax_density_visualization, ax_density_bar_chart) = plt.subplots(1, 2, figsize=(15, 6))

    # --- Plot 1: Test Point and GMM Components (previously fig1, ax1) ---
    gmm_predicted_labels = fitted_gmm.predict(X_data_full)
    for i in range(n_gmm_components):
        component_mask = gmm_predicted_labels == i
        ax_density_visualization.scatter(X_data_full[component_mask, 0], X_data_full[component_mask, 1],
                                         color=current_plot_colors[i], alpha=0.25, s=20, label=f'_Data (Comp. {i+1})')

    plot_confidence_ellipses(ax_density_visualization, fitted_gmm.means_, fitted_gmm.covariances_,
                             current_plot_colors, alpha=0.4, scale=2) # Scale 2 for ~95%

    ax_density_visualization.scatter(test_point_coords[0], test_point_coords[1], color='black', marker='P',
                                     s=150, edgecolors='white', linewidth=1.0, label='Test Point P', zorder=10)

    ax_density_visualization.set_title('(d) Test Point P and GMM Components', pad=15)
    ax_density_visualization.legend(loc='best', fontsize=9)
    ax_density_visualization.grid(True, linestyle=':', alpha=0.7)
    ax_density_visualization.set_xlabel('Feature 1')
    ax_density_visualization.set_ylabel('Feature 2')
    ax_density_visualization.axis('equal')


    # --- Density and Weight Calculation (for Plot 2) ---
    densities_for_test_point = []
    dimensionality = X_data_full.shape[1]

    for i in range(n_gmm_components):
        mean_i = fitted_gmm.means_[i]
        cov_i_mat = fitted_gmm.covariances_[i] # Assumes 'full' from GMM fitting

        diff_vec = (test_point_coords - mean_i).reshape(-1,1)
        try:
            inv_cov_i = np.linalg.inv(cov_i_mat)
            det_cov_i = np.linalg.det(cov_i_mat)
            if det_cov_i <= 1e-9: density_val = 0.0
            else:
                norm_factor = 1.0 / (np.power(2 * np.pi, dimensionality / 2) * np.sqrt(det_cov_i))
                exponent = -0.5 * diff_vec.T @ inv_cov_i @ diff_vec
                density_val = norm_factor * np.exp(exponent.item())
        except np.linalg.LinAlgError: density_val = 0.0
        densities_for_test_point.append(max(density_val, 1e-12))

    total_density_sum = sum(densities_for_test_point)
    if total_density_sum > 1e-9:
        calculated_weights = [d / total_density_sum for d in densities_for_test_point]
    else:
        calculated_weights = [1.0 / n_gmm_components for _ in range(n_gmm_components)]
        # print("Warning: Test point has near-zero density for all GMM components. Assigning uniform weights.")

    # --- Plot 2: Density and Weight Bar Chart (previously fig2, ax2) ---
    x_bar_indices = np.arange(n_gmm_components)
    bar_plot_width = 0.35

    bars1 = ax_density_bar_chart.bar(x_bar_indices - bar_plot_width/2, densities_for_test_point, bar_plot_width,
                                     label='Density $\\rho_i(P)$', alpha=0.8, color='skyblue')
    bars2 = ax_density_bar_chart.bar(x_bar_indices + bar_plot_width/2, calculated_weights, bar_plot_width,
                                     label='Normalized Weight $w_i(P)$', alpha=0.8, color='salmon')

    ax_density_bar_chart.set_xticks(x_bar_indices)
    ax_density_bar_chart.set_xticklabels([f'Comp. {i+1}' for i in range(n_gmm_components)])
    ax_density_bar_chart.set_title('(e) Density & Weight of P for GMM Components', pad=15)
    ax_density_bar_chart.set_xlabel('GMM Component')
    ax_density_bar_chart.set_ylabel('Value')
    ax_density_bar_chart.legend(fontsize=9)
    ax_density_bar_chart.grid(True, linestyle=':', alpha=0.7, axis='y')
    ax_density_bar_chart.ticklabel_format(style='sci', axis='y', scilimits=(-3,3), useMathText=True)


    for i, bar in enumerate(bars1):
        h = bar.get_height()
        ax_density_bar_chart.text(bar.get_x() + bar.get_width()/2., h if h > 1e-9 else 1e-9, f'{densities_for_test_point[i]:.1e}', ha='center', va='bottom', fontsize=8, rotation=0)
    for i, bar in enumerate(bars2):
        h = bar.get_height()
        ax_density_bar_chart.text(bar.get_x() + bar.get_width()/2., h if h > 1e-3 else 1e-3, f'{calculated_weights[i]:.2f}', ha='center', va='bottom', fontsize=8, rotation=0)

    plt.tight_layout(pad=1.5)
    return fig # Return the single figure object

# Main execution
if __name__ == "__main__":
    results_dir = os.path.join('Results', 'plots_final_layout')
    os.makedirs(results_dir, exist_ok=True)

    X_main_data, _ = generate_complex_2d_data(n_samples=350, random_state=10) # Changed random_state for variety

    gmm_n_components = 3
    gmm_sampling_phi = 0.25 # Adjusted phi

    # Create and save the 1x3 clustering/resampling plot
    fig_clustering_resampling, fitted_gmm_model, _, _ = create_comprehensive_gmm_visualization(
        X_main_data,
        n_components=gmm_n_components,
        phi=gmm_sampling_phi,
        random_state=42
    )
    fig_clustering_resampling.savefig(os.path.join(results_dir, 'figure_clustering_resampling.pdf'), format='pdf', dpi=300, bbox_inches='tight')
    fig_clustering_resampling.savefig(os.path.join(results_dir, 'figure_clustering_resampling.png'), dpi=300, bbox_inches='tight')
    plt.show()

    # Define an example test point (can be adjusted)
    # example_test_point_coords = np.array([4.0, 3.0])
    example_test_point_coords = fitted_gmm_model.means_[1] + np.array([0.5, -0.5]) # Near component 2 mean

    # Create and save the 1x2 density/weight plot
    fig_density_weight = demonstrate_density_and_weight_combined(
        X_main_data,
        fitted_gmm_model,
        test_point_coords=example_test_point_coords
    )

    fig_density_weight.savefig(os.path.join(results_dir, 'figure_density_weights_combined.pdf'), format='pdf', dpi=300, bbox_inches='tight')
    fig_density_weight.savefig(os.path.join(results_dir, 'figure_density_weights_combined.png'), dpi=300, bbox_inches='tight')
    plt.show()

    print(f"Revised plots saved in directory: {results_dir}")