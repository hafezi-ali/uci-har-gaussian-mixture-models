# Plot confusion matrix for 1d_CNN(Adagrad) Model
from Lib.lib import *


def plot_outlier_handling(df, no_df, column):
    # Create figures for dist plot before and after of outlier handling by two Z_Score and IQR methods
    fig, ax = plt.subplots(1, 3, figsize=(18, 6))

    # Plot the dist plot of column in data_out_iqr dataset
    sns.distplot(no_df['IQR'][column], ax=ax[0])
    ax[0].set_title("IQR")

    # Plot the dist plot of column in data_out_zs dataset
    sns.distplot(no_df['ZS'][column], ax=ax[1])
    ax[1].set_title("ZS")

    # Plot the dist plot of column in original dataset
    sns.distplot(df[column], ax=ax[2])
    ax[2].set_title("original_data")

    plt.plot()

    # Create figures for boxplot before and after of outlier handling by two Z_Score and IQR methods
    fig, ax = plt.subplots(1, 3, figsize=(18, 6))

    # Plot the boxplot of column in data_out_iqr dataset
    sns.boxplot(no_df['IQR'][column], ax=ax[0])
    ax[0].set_title(f"{column} Distribution: data_out_iqr")

    # Plot the boxplot of column in data_out_zs dataset
    sns.boxplot(no_df['ZS'][column], ax=ax[1])
    ax[1].set_title(f"{column} Distribution: ZS")

    # Plot the boxplot of column in original dataset
    sns.boxplot(df[column], ax=ax[2])
    ax[2].set_title(f"{column} Distribution: Original Data")

    plt.show()

    # Create figures for Q-Q plot before and after of outlier handling by two Z_Score and IQR methods
    plt.figure(figsize=(6, 6))

    # Create a new figure for Q-Q plot after outlier handling (original data)
    qqplot(df[column], line='s')
    plt.title(f'Q-Q Plot: {column} Distribution Before Outlier Handling')
    plt.show()

    # Create a new figure for Q-Q plot after outlier handling (data_out_iqr)
    plt.figure(figsize=(6, 6))
    qqplot(no_df['IQR'][column], line='s')
    plt.title(f'Q-Q Plot: {column} Distribution After Outlier Handling (IQR)')
    plt.show()

    # Create a new figure for Q-Q plot after outlier handling (data_out_zs)
    plt.figure(figsize=(6, 6))
    qqplot(no_df['ZS'][column], line='s')
    plt.title(f'Q-Q Plot: {column} Distribution After Outlier Handling (ZS)')
    plt.show()


def plot_cs(xtr, ytr, sampling_results, cluster_name):
    png_renderer = pio.renderers["png"]
    plt.figure(figsize=(10, 10))

    cluster_indexes = sampling_results['clusters_indexes']['cluster_' + cluster_name]
    data_c = xtr.loc[cluster_indexes]
    data_c['lable'] = ytr.loc[cluster_indexes]
    data_c['type'] = 'in_cluster'

    uncommon = sampling_results['uncommon_indexes_clusters_cluster_samples']['cluster_' + cluster_name]
    data_uc = xtr.loc[uncommon]
    data_uc['lable'] = ytr.loc[uncommon]
    data_uc['type'] = 'added'
    datam = pd.concat([data_c, data_uc])

    fig = px.scatter(datam, x=0, y=1, color=datam.lable, symbol=datam.type,
                     symbol_map={'in_cluster': 'circle', 'added': 'star'})

    cluster_centers = sampling_results['cluster_centers']
    fig.add_scatter(x=np.array(cluster_centers[:, 0]),
                    y=np.array(cluster_centers[:, 1]), mode='markers',
                    marker=dict(
                        size=30,
                        line=dict(width=8, color='pink'),
                    ),
                    name='Cluster_center'
                    )

    fig.update_traces(marker_size=5)
    fig.update_layout(xaxis=dict(range=[-70, 85]), yaxis=dict(range=[-60, 60]))
    fig.show()


def plot_all_cs(xtr, ytr, sampling_results):
    png_renderer = pio.renderers["png"]
    plt.figure(figsize=(10, 10))
    fig = px.scatter(xtr, x=0, y=1, color=ytr, symbol=sampling_results['cluster_assignments'],
                     symbol_sequence=['x', 'star', 'square']
                     )

    cluster_centers = sampling_results['cluster_centers']
    fig.add_scatter(x=np.array(cluster_centers[:, 0]), y=np.array(cluster_centers[:, 1]), mode='markers',
                    marker=dict(
                        size=30,
                        line=dict(width=8, color='pink'),
                    ),
                    name='Cluster_center'
                    )

    fig.update_traces(marker_size=5)
    fig.update_layout(xaxis=dict(range=[-70, 85]), yaxis=dict(range=[-60, 60]))
    fig.show()
