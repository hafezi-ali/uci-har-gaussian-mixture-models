
# *** modules/feature_engineering.py ***
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
import umap
from plotly.figure_factory import create_2d_density
from matplotlib import pyplot as plt


def feature_scaling(df):
    x = df.drop(['ActivityName', 'Activity', 'subject'], axis=1)
    yc = df.ActivityName
    yn = df.Activity

    # Scale Features
    scaler = StandardScaler(with_mean=True, with_std=True)
    x_scaled = scaler.fit_transform(x)

    return x, yn, yc, x_scaled


def apply_optimal_pca(x, variance_threshold=0.95):
    # Perform PCA
    pca = PCA()
    x_pca = pca.fit_transform(x)

    # Explained Variance
    cumulative_variance = pca.explained_variance_ratio_.cumsum()
    num_components = (cumulative_variance >= variance_threshold).argmax() + 1

    pca_parameters = {'variance_ratio': pca.explained_variance_ratio_,
                      'cumulative_variance_ratio': cumulative_variance,
                      'variance_threshold': variance_threshold,
                      'optimal_index': num_components - 1}

    return x_pca[:, :num_components], pca_parameters


def umap_standard_embedding(x, title):
    x_umap_standard_embedded = umap.UMAP(random_state=42).fit_transform(x)
    colorscale = ['#7A4579', '#D56073', 'rgb(236,158,105)', (1, 1, 0.2), (0.98, 0.98, 0.98)]
    fig = create_2d_density(
        x_umap_standard_embedded[:, 0],
        x_umap_standard_embedded[:, 1],
        colorscale=colorscale,
        hist_color='rgb(255, 237, 222)',
        point_size=3,
        title=title,
    )

    # Update the layout to set the width
    fig.update_layout(
        width=700,  # Set the desired width
        # height=700,  # Optionally set the desired height
        xaxis=dict(scaleanchor="y", scaleratio=1),  # Ensure x and y scales are the same
        yaxis=dict(scaleanchor="x")
    )

    return fig, x_umap_standard_embedded



def plot_variance_ratio(pca_parameters, title):
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
    plt.title(title)
    # plt.savefig(f"{plot_save_path}/variance_ratio.png")
    variance_ratio_fig = plt.gcf()
    # plt.show()

    return variance_ratio_fig
