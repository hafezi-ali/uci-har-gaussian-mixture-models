
# *** modules/eda.py ***
import seaborn as sns
import matplotlib.pyplot as plt
from collections import Counter


def perform_eda(df):
    # Data Overview
    print("Data Size:", df.size)
    print("Data Shape:", df.shape)
    print('\nNumber of unique subjects in the data: ', len(df.subject.unique()))
    print('Number of unique activities in the data: ', len(df.ActivityName.unique()))
    print('\nData statistics info: \n', df.describe())
    corr_matrix = df.drop(['Activity', 'ActivityName'], axis=1).corr()
    # Get summary statistics of the correlation matrix
    corr_summary = corr_matrix.unstack().describe()
    print('\ncorr_summary: \n', corr_summary)

    # Subject Distribution
    plt.figure(figsize=(15, 10))
    sorted_counts = df.subject.value_counts().sort_values(ascending=True)
    sns.barplot(x=sorted_counts.index, y=sorted_counts.values, color='cyan')
    plt.title('Bar Plot of Unique Users')
    plt.xlabel('Names')
    plt.ylabel('Count')
    bar_plot_fig = plt.gcf()
    # plt.savefig(f"{plot_save_path}/barplot_unique_users.png")
    plt.title('Barplot for unique users activities')
    # plt.show()

    # Activity Distribution
    sns.set_style('whitegrid')
    plt.rcParams['font.family'] = 'Dejavu Sans'
    plt.figure(figsize=(18, 11))
    plt.title('Data provided by each user', fontsize=20)
    sns.countplot(x='subject', hue='ActivityName', data=df)
    activity_distribution_plot = plt.gcf()
    # plt.savefig(f"{plot_save_path}/data_provided_by_each_user.png")
    plt.title('Data provided by each user')
    # plt.show()

    # Class Distribution
    cn = Counter(df.ActivityName)
    for activity, count in cn.items():
        print(f'{activity}: {count} ({count / len(df) * 100:.2f}%)')

    # Set the palette
    sns.set_palette("Set1", desat=0.80)
    # Create the FacetGrid
    facetgrid = sns.FacetGrid(df, hue='ActivityName', height=5, aspect=2)
    facetgrid.map(sns.kdeplot, 'tBodyAcc-mean()-X').add_legend()
    # Get the current figure
    stationary_activities_fig = plt.gcf()
    # Set the title with adjusted position
    plt.title('Stationary Activities', y=1.05)  # Adjust the y parameter as needed
    # Use tight_layout to adjust padding
    plt.tight_layout()
    # Show the plot
    # plt.show()

    return bar_plot_fig, activity_distribution_plot, stationary_activities_fig
