import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Ellipse
import matplotlib.transforms as transforms # Not strictly needed for the final ellipse version, but good to have if using other ellipse funcs
from sklearn.mixture import GaussianMixture
from sklearn.datasets import make_blobs

# Helper function to draw confidence ellipses for GMM components
def plot_ellipse(ax, mean, cov, color, n_std=2.0, **kwargs):
    """Plots an ellipse representing a 2D Gaussian component."""
    eigenvalues, eigenvectors = np.linalg.eigh(cov)
    # Order eigenvalues and eigenvectors for consistent ellipse orientation
    order = eigenvalues.argsort()[::-1]
    eigenvalues, eigenvectors = eigenvalues[order], eigenvectors[:, order]

    # Get angle of rotation from the first principal eigenvector
    vx, vy = eigenvectors[:,0][0], eigenvectors[:,0][1]
    theta = np.arctan2(vy, vx)

    # Ellipse parameters: width and height are 2*n_std*sqrt(eigenvalue)
    width, height = 2 * n_std * np.sqrt(eigenvalues)
    ellipse = Ellipse(xy=mean, width=width, height=height,
                      angle=np.degrees(theta), # angle in degrees
                      edgecolor=color, facecolor='none', **kwargs)
    ax.add_patch(ellipse)
    return ellipse


# --- Parameters for the illustrative example ---
n_samples = 350         # Number of data points
n_components_gmm = 3    # Number of GMM components to fit / true clusters
cluster_std_true = 1.6  # Standard deviation of the clusters (controls overlap)
random_state = 42       # For reproducibility
sampling_ratio_phi = 0.7 # As mentioned in your Algorithm 1 (phi)

# --- 1. Generate Synthetic 2D Data ---
# We generate data with 'n_components_gmm' clusters to clearly show GMM's role.
X_data, y_true_labels = make_blobs(n_samples=n_samples, 
                                   centers=n_components_gmm, 
                                   cluster_std=cluster_std_true, 
                                   random_state=random_state)
# For GMM fitting, we only use X_data (unlabeled features)

# --- 2. Fit Gaussian Mixture Model ---
gmm_model = GaussianMixture(n_components=n_components_gmm, 
                            covariance_type='full', # 'full' allows elliptical shapes
                            random_state=random_state)
gmm_model.fit(X_data)

# Get GMM parameters and predictions/probabilities
gmm_means = gmm_model.means_
gmm_covariances = gmm_model.covariances_
gmm_responsibilities = gmm_model.predict_proba(X_data) # Posterior probabilities (responsibilities gamma_ji)

# --- 3. Prepare for Plotting ---
# Define colors for different components
component_colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd'] # Matplotlib default colors

# Create the 2x2 figure
fig, axs = plt.subplots(2, 2, figsize=(12, 11)) # Adjusted figsize for better layout
plt.subplots_adjust(hspace=0.35, wspace=0.25) 

# --- Panel (a): Original Unlabeled Training Data (X_data) ---
ax_a = axs[0, 0]
ax_a.scatter(X_data[:, 0], X_data[:, 1], s=10, color='dimgray', alpha=0.7)
ax_a.set_title('(a) Original Training Data ($X$)', fontsize=12)
ax_a.set_xlabel('Feature 1')
ax_a.set_ylabel('Feature 2')
ax_a.set_xticks([])
ax_a.set_yticks([])

# --- Panel (b): GMM Fitting and Component Identification ---
ax_b = axs[0, 1]
ax_b.scatter(X_data[:, 0], X_data[:, 1], s=10, color='lightgray', alpha=0.5) # Fainter original data
for i in range(n_components_gmm):
    plot_ellipse(ax_b, gmm_means[i], gmm_covariances[i],
                 color=component_colors[i % len(component_colors)], 
                 linewidth=2, n_std=2.0) # n_std=2.0 covers ~95% for a 1D Gaussian
    ax_b.scatter(gmm_means[i, 0], gmm_means[i, 1], marker='X', s=100, 
                 color=component_colors[i % len(component_colors)], edgecolor='black', zorder=3)
ax_b.set_title('(b) GMM Fit & Component Identification', fontsize=12)
ax_b.set_xlabel('Feature 1')
ax_b.set_ylabel('Feature 2')
ax_b.set_xticks([])
ax_b.set_yticks([])

# --- Panel (c): Data Points Assigned by Max Responsibility ---
ax_c = axs[1, 0]
# Assign points to the component with the highest responsibility for visualization
component_assignment_viz = np.argmax(gmm_responsibilities, axis=1)
for i in range(n_components_gmm):
    ax_c.scatter(X_data[component_assignment_viz == i, 0], 
                 X_data[component_assignment_viz == i, 1],
                 s=10, color=component_colors[i % len(component_colors)], alpha=0.8)
ax_c.set_title('(c) Data Points by Max Responsibility', fontsize=12)
ax_c.set_xlabel('Feature 1')
ax_c.set_ylabel('Feature 2')
ax_c.set_xticks([])
ax_c.set_yticks([])

# --- Panel (d): Subset Sampling based on Responsibilities (phi) ---
ax_d = axs[1, 1]
# Plot all original data points faintly in the background
ax_d.scatter(X_data[:, 0], X_data[:, 1], s=5, color='gainsboro', label='_Original Data (bg)') # Use _ to hide from auto-legend

num_total_samples_m = X_data.shape[0]
num_to_sample_per_component = int(num_total_samples_m * sampling_ratio_phi)

legend_handles_d = []

for i in range(n_components_gmm):
    # Responsibilities of all data points for the current component i
    responsibilities_for_comp_i = gmm_responsibilities[:, i]
    
    # Get indices of points sorted by their responsibility for component i (descending)
    sorted_indices_for_comp_i = np.argsort(-responsibilities_for_comp_i)
    
    # Select the top 'num_to_sample_per_component' indices
    selected_indices_for_this_subset = sorted_indices_for_comp_i[:num_to_sample_per_component]
    
    # Plot these selected points, colored by the component they were sampled for
    scatter_plot = ax_d.scatter(X_data[selected_indices_for_this_subset, 0],
                                X_data[selected_indices_for_this_subset, 1],
                                s=25, # Make sampled points more prominent
                                color=component_colors[i % len(component_colors)],
                                alpha=0.6,
                                edgecolor='black', linewidth=0.5)
    legend_handles_d.append(scatter_plot)

ax_d.set_title(f'(d) Sampled Subsets ($\phi={sampling_ratio_phi}$)', fontsize=12)
ax_d.set_xlabel('Feature 1')
ax_d.set_ylabel('Feature 2')
ax_d.set_xticks([])
ax_d.set_yticks([])
ax_d.legend(legend_handles_d, 
            [f'Sampled for Comp. {j+1}' for j in range(n_components_gmm)], 
            loc='best', fontsize='small', title="Subsets $D'_k$")


# --- Final Touches ---
fig.suptitle('Figure 1: GMM-based Data Sampling for Ensemble Creation', fontsize=16, y=0.99)
plt.tight_layout(rect=[0, 0.01, 1, 0.96]) # Adjust rect to make space for suptitle and bottom elements

# Save the figure
plt.savefig('gmm_sampling_figure.pdf', dpi=300, bbox_inches='tight')
plt.savefig('gmm_sampling_figure.png', dpi=300, bbox_inches='tight') 
print("Figure 'gmm_sampling_figure.pdf' and 'gmm_sampling_figure.png' saved.")
plt.show() # Display the plot